markdown

# Jarvis World Model — Bootstrap (v2, Repo-Aware)

A complete spec for adding persistent entity tracking on top of the existing Jarvis codebase. This is the only document an agent (or human) should need to implement the World Model. It is grounded in the actual repository as it exists today — the existing IdentityManager, DatabaseManager, EventBus, CameraManager, persona system, and 5-room camera topology are all treated as authoritative; the World Model integrates with them, never duplicates them.

**The shift in one sentence:** Jarvis already sees, hears, recognizes faces, and routes audio per-room — but it sees the world as disconnected per-tick observations. The World Model gives it a notebook: persistent entities with state, location, and history, so disappearance can be reasoned about instead of just observed.

**Demo target:** Anna walks in. *"Where's Cole?"* Jarvis answers from persistent world state — *"He was in the office about an hour ago, near his desk. I never saw him leave the office."* And he is, in fact, still hiding under the desk.

---

## Table of Contents

1. [Architectural Spine](#1-architectural-spine)
2. [The Bounded House Model](#2-the-bounded-house-model)
3. [Where the World Model Fits in the Existing Repo](#3-where-the-world-model-fits-in-the-existing-repo)
4. [The Entity State Machine](#4-the-entity-state-machine)
5. [Data Model](#5-data-model)
6. [Event Taxonomy](#6-event-taxonomy)
7. [Camera Topology Config](#7-camera-topology-config)
8. [Storage Layer (Async, Linked to persons)](#8-storage-layer-async-linked-to-persons)
9. [Identity Integration: Who vs. Where](#9-identity-integration-who-vs-where)
10. [Auto-Enrollment Inside IdentityManager](#10-auto-enrollment-inside-identitymanager)
11. [The ArcFace Upgrade (Done From Day One)](#11-the-arcface-upgrade-done-from-day-one)
12. [The ObservationBuilder Adapter](#12-the-observationbuilder-adapter)
13. [Association Algorithm](#13-association-algorithm)
14. [Camera Health & Degraded Modes](#14-camera-health--degraded-modes)
15. [Full Code: types.py and geometry.py](#15-full-code-typespy-and-geometrypy)
16. [Full Code: WorldStore](#16-full-code-worldstore)
17. [Full Code: WorldModel](#17-full-code-worldmodel)
18. [Full Code: ObservationBuilder](#18-full-code-observationbuilder)
19. [Synthetic Test Harnesses](#19-synthetic-test-harnesses)
20. [Query Layer (Orchestrator Tools)](#20-query-layer-orchestrator-tools)
21. [Persona Alignment](#21-persona-alignment)
22. [Phase 4: Pets by Name](#22-phase-4-pets-by-name)
23. [Phase 4: Objects](#23-phase-4-objects)
24. [Phase 5: Interactions](#24-phase-5-interactions)
25. [Phase 5: Cross-Day Patterns & Anomalies](#25-phase-5-cross-day-patterns--anomalies)
26. [Build Order with Verification Gates](#26-build-order-with-verification-gates)
27. [Failure Modes & Defenses](#27-failure-modes--defenses)
28. [Tunables Reference](#28-tunables-reference)

---

## 1. Architectural Spine

Two single-line rules govern every other design choice in this document:

> **IdentityManager owns *who*. WorldModel owns *where*, *what state*, *continuity*, and *change events*.**
>
> **The house is a bounded space. Every exterior door is camera-covered. Disappearances are explained by the boundary, not by arbitrary timeouts.**

If a design decision in this doc seems to violate either rule, the design decision is wrong, not the rule.

### What this means concretely

- **No second face recognizer.** The repo's `modules/identity/identity_manager.py` already does cross-modal identity (face + voice + drift verification + pending clusters + sample banks). The World Model consumes its outputs (`person_id`, `confidence`) and never re-implements identity. ArcFace, when it lands, lands inside `face_recognizer.py` and feeds IdentityManager — not as a parallel system.
- **No second sample database.** Existing `face_samples` and `voice_samples` tables stay where they are. The World Model adds `world_entities` and `world_entity_events` tables that *link* to `persons.id` via a nullable foreign key.
- **No second config root.** The repo's `rooms:` block is the source of truth for cameras, mics, speakers, and FPS. The World Model adds an optional `world_model:` sub-block per room for geometry (exits, landmarks). It does not introduce a parallel `cameras:` block.
- **No second storage path.** The repo uses `aiosqlite` through `DatabaseManager`. The World Model's `WorldStore` is async and uses the same connection — not a separate sync `sqlite3` connection with its own lock.
- **Drivers are not the World Model's concern.** USB webcam, ESP32 HTTP, Wyze RTSP, Wyze SSH+aplay — `CameraManager` already handles these via per-driver classes. The World Model receives normalized observations from a layer above, never deals with raw streams.

### What the World Model is responsible for

- Maintaining a persistent registry of entities (people, cats, objects).
- Assigning each new observation to an existing entity (or creating a new one).
- Running the entity state machine (PRESENT, IN_ROOM_UNSEEN, etc. — Section 4).
- Emitting *change-events* on the bus, never repeating "still here" events.
- Backing the LLM's tool calls about location, occupancy, and recent history.
- Reasoning about disappearance using the bounded house model.

That's it. Identity, recognition, ingestion, and persistence-mechanics belong to other modules.

---

## 2. The Bounded House Model

The most important design choice in this rewrite. The house has a topology:

```
                    Outside
                       │
            ┌──────────┼──────────────────────────────┐
            │   [exterior exits — all camera-covered] │
            │                                         │
  ┌──────┐  │  ┌──────────┐  ┌─────────┐              │
  │Office│──┼──│Living rm │──│Kitchen  │              │
  └──────┘  │  └────┬─────┘  └─────────┘              │
            │       │                                 │
       ┌────┴────┐  │     ┌─────────────┐             │
       │ Bedroom │──┴─────│ unmonitored │ × 2         │
       │ (cam)   │        │ bedrooms    │             │
       └─────────┘        │ (no cams)   │             │
                          └─────────────┘             │
            └─────────────────────────────────────────┘
```

In your specific house:

- **Camera-covered indoor rooms:** office (USB cam, 30 fps), living_room (Wyze), bedroom — master (Wyze), kitchen (Wyze), laundry_room (ESP32-CAM, video works, audio TBD).
- **Camera-covered exterior exits:** every door that leads outside has a camera that can see it.
- **Unmonitored interior zones:** two additional bedrooms with no cameras facing their doors.

This topology has a powerful implication: **anyone inside the house at time T is still inside the house at time T+1 unless an exit camera saw them leave.** That fact replaces every arbitrary "MISSING after 15 minutes" timeout in the v1 spec. The system never claims someone has vanished — it claims either:

- they are present in a camera, or
- they are in the house but currently unobserved (probably in an unmonitored zone, or in a blind spot of a monitored room), or
- they left the house at a specific time via a specific exit.

Each is an actionable, accurate statement. *"They have been missing for 15 minutes"* is none of those.

### How disappearances are classified

When an entity stops being observed, the World Model asks two questions in order:

1. **Did an exterior-exit camera see them transit out within the last few seconds?** If yes → they DEPARTED.
2. **Were they last seen near an exit polygon (interior or exterior)?** If yes → expect them to surface in the camera on the other side of that exit (interior) or treat as DEPARTED with corroboration (exterior). Until they do, they are TRANSITIONING.
3. **Otherwise** → they are still in their last-seen room (IN_ROOM_UNSEEN), full stop. No timeout escalation.

The *one* condition that escalates IN_ROOM_UNSEEN to something stronger is if a positive observation elsewhere in the house resolves to the same person via IdentityManager (e.g., a confident face match in another room). That's a positive resolution, not a negative timeout.

### The "departed" state is genuinely useful

Anna asking *"is Cole home?"* should get a definitive answer:

- If any indoor camera saw Cole in the last 30 seconds → **yes, he's home, in [room]**.
- If the last event for Cole was DEPARTED via the front door 3 hours ago → **no, he left at 11:23 AM through the front door**.
- If no indoor camera has seen him for hours but he wasn't seen leaving → **he's home, but I don't know which room — he was last in [room] and probably moved into one of the unmonitored bedrooms**.

This is a much more honest answer than the v1 design produced.

---

## 3. Where the World Model Fits in the Existing Repo

```
jarvis/
├── core/
│   ├── orchestrator.py        EXISTING — wire WorldModel here
│   ├── event_bus.py           EXISTING — add new topics
│   └── config.py              EXISTING — extend Pydantic model for rooms[].world_model
│
├── modules/
│   ├── vision/
│   │   ├── camera_manager.py        EXISTING — UNCHANGED
│   │   ├── object_detector.py       EXISTING — UNCHANGED (its TODOs get answered by world_model)
│   │   ├── face_recognizer.py       EXISTING — patched to use ArcFace (Section 11)
│   │   ├── posture_analyzer.py      EXISTING — UNCHANGED
│   │   ├── scene_analyzer.py        EXISTING — gets richer person_states input
│   │   └── observation_builder.py   NEW (Section 12)
│   │
│   ├── identity/
│   │   └── identity_manager.py      EXISTING — extended with auto-enrollment (Section 10)
│   │
│   ├── voice/
│   │   └── speaker_id.py            EXISTING — UNCHANGED
│   │
│   ├── memory/
│   │   └── database.py              EXISTING — add world_entities, world_entity_events tables
│   │
│   ├── context/
│   │   └── state_fusion.py          EXISTING — migrate to consume world.entity_event
│   │
│   └── world_model/                 NEW
│       ├── __init__.py
│       ├── types.py                 (Section 15)
│       ├── geometry.py              (Section 15)
│       ├── store.py                 (Section 16)
│       ├── world_model.py           (Section 17)
│       ├── associator.py            (Section 13)
│       └── query_tools.py           (Section 20)
│
├── dashboard/
│   └── ...                          EXISTING — add entity/state cards
│
├── scripts/
│   ├── test_world_model_synthetic.py   NEW (Section 19)
│   ├── test_world_model_handoff.py     NEW
│   ├── test_world_model_camera_drop.py NEW
│   └── test_world_model_query.py       NEW
│
├── config.yaml                  EXISTING — extend rooms[].world_model
└── WORLD_MODEL_BOOTSTRAP.md     this file
```

### EventBus topology

The repo's existing topics (per `core/event_bus.py`): `vision.frame_processed`, `vision.posture`, `context.state_changed`, `node.status`, plus others as the system has grown.

The World Model adds three new topics:

| Topic | Producer | Payload | Consumers |
|---|---|---|---|
| `vision.observation` | `ObservationBuilder` | normalized `Observation` batch (Section 5) | `WorldModel` |
| `world.entity_event` | `WorldModel` | `EntityEvent` (Section 5) — only on **change** | `state_fusion`, `dashboard`, `brain`, `memory`, `interactions` |
| `world.state_snapshot` | `WorldModel` (every 30s) | full registry projection | `dashboard` |

**Backward compatibility:** existing topics keep emitting unchanged. State fusion can be migrated incrementally — first to *also* listen to `world.entity_event`, then to drop its `vision.frame_processed` subscription once the migration is verified. This avoids a flag-day rewrite.

### What does *not* change

- `core/orchestrator.py` adds a `WorldModel` instantiation alongside its existing managers. It does not absorb World Model logic.
- `modules/vision/camera_manager.py`, `object_detector.py`, `posture_analyzer.py` keep doing what they do. The new `observation_builder.py` reads their outputs and synthesizes the `Observation` payload.
- `modules/memory/database.py` keeps its existing tables. Two new tables get added (`world_entities`, `world_entity_events`); nothing existing is renamed or restructured.

---

## 4. The Entity State Machine

Five states plus a boot transient. No arbitrary timeouts that produce "MISSING" without justification.

```
                           ┌──── observation matched in same room
                           ▼      (no event emit; this is "still present")
                    ┌─────────────┐
                    │   PRESENT   │ ◄─── REAPPEARED ─── any new match
                    │  in <room>  │                        ▲     ▲     ▲
                    └──────┬──────┘                        │     │     │
                           │ no observation this tick      │     │     │
                           ▼                               │     │     │
                  ┌─────────────────────┐                  │     │     │
                  │  IN_ROOM_UNSEEN     │ ─────────────────┘     │     │
                  │   in <room>         │ stays here unless...   │     │
                  │   last_landmark=... │                        │     │
                  └────┬─────────┬──────┘                        │     │
                       │         │                               │     │
   last_bbox near      │         │  last_bbox near               │     │
   exterior exit       │         │  interior exit                │     │
   AND seen on         │         │  AND seen in                  │     │
   exit-cam crossing   │         │  neighbor camera              │     │
   threshold           │         │                               │     │
                       ▼         ▼                               │     │
               ┌──────────────┐  ┌──────────────────┐            │     │
               │  DEPARTED    │  │ TRANSITIONING    │ ───────────┘     │
               │  via <exit>  │  │ from <room A>    │ rematch in       │
               │  at <ts>     │  │ to <room B?>     │ neighbor cam     │
               └──────┬───────┘  └────────┬─────────┘                  │
                      │                   │ no rematch in any cam      │
                      │                   │ AND last seen near         │
                      │                   │ unmonitored-zone door      │
                      │                   ▼                            │
                      │           ┌─────────────────────┐              │
                      │           │ IN_HOUSE_UNMONITORED│ ─────────────┘
                      │           │ entered via <door>  │
                      │           │ from <room>         │
                      │           └─────────────────────┘
                      │
                      │ rematch on any indoor camera
                      ▼
                   ┌─────────┐
                   │ PRESENT │  (state cycle continues)
                   └─────────┘
```

### State definitions

| State | Meaning | When it applies |
|---|---|---|
| `PRESENT` | Currently visible in some camera. | The default while an entity is being observed. |
| `IN_ROOM_UNSEEN` | Was visible, now isn't. Last landmark recorded. No exit observed. | The under-desk case. The "stepped out of frame momentarily" case. The "no exit camera saw them transit" case. |
| `TRANSITIONING` | Last seen near a doorway, expected to surface in adjacent camera (or settle into IN_HOUSE_UNMONITORED). | Brief — usually resolves within seconds. |
| `IN_HOUSE_UNMONITORED` | Known to have entered an unmonitored zone (one of the two bedrooms without cameras). | When TRANSITIONING points to a `to_unmonitored_zone` exit. |
| `DEPARTED` | Last confirmed exit via an exterior-cam-covered door. | Only when the exit camera saw the transit, not just the doorway approach. |
| `UNKNOWN_AT_BOOT` | System just started; entity's true state is being resolved. | First 30 seconds after boot only. |

### Walking through your specific scenarios

**Scenario A: Cole hides under the desk.**
1. Cole `PRESENT` in office, last bbox center over `under_desk` landmark.
2. Vision tick: no detection. `_pair_cost` produces no match. Last bbox was over `under_desk`, which is *not* an exit polygon.
3. State: `PRESENT → IN_ROOM_UNSEEN(room=office, last_landmark=under_desk, near_exit=false)`.
4. Stays here. Indefinitely. No timeout, no escalation.
5. Anna asks "where's Cole?" → query returns the state. LLM phrases: *"He was in the office, near the under-desk area, about an hour ago. I never saw him leave the office."*

**Scenario B: Cole walks into a (camera-less) guest bedroom from the hallway/living-room area.**
1. Cole `PRESENT` in living_room.
2. Cole's bbox approaches a doorway polygon labeled `to_unmonitored_zone: guest_bedroom`.
3. Vision tick: no detection. Last bbox was over the doorway. State: `PRESENT → TRANSITIONING(from=living_room, to=guest_bedroom?)`.
4. Within `T_handoff_seconds` (8s default), no neighbor camera matches. The target was an unmonitored zone, not a camera-equipped one.
5. State: `TRANSITIONING → IN_HOUSE_UNMONITORED(entered_via=guest_bedroom_door, from=living_room)`.
6. Anna asks "where's Cole?" → *"He went into the guest bedroom. I don't have a camera there, so I can't tell you what he's doing."*

**Scenario C: Cole leaves the house through the front door.**
1. Cole `PRESENT` in living_room.
2. Cole moves toward the exterior exit polygon. Vision tick: bbox crosses the exit threshold.
3. The exterior-exit polygon is drawn at the *threshold of the door from the inside*. A sustained bbox across it (≥2 frames) that then disappears = a real departure.
4. State: `PRESENT → DEPARTED(via=front_door, at=14:23)`.
5. Anna asks "is Cole home?" → *"No, he left through the front door at 2:23 PM."*

**Scenario D: Cole walks office → living_room → kitchen.**
1. Cole `PRESENT` in office, near doorway. Vision tick: gone.
2. State: `PRESENT → TRANSITIONING(from=office, to=living_room?)`.
3. Within 2 seconds, living_room camera sees a person, IdentityManager confirms Cole.
4. State: `TRANSITIONING → PRESENT (in living_room)`. Emits `MOVED_TO(from=office, to=living_room)`.
5. A few seconds later, similar handoff to kitchen.
6. The event log reads as a clean narrative.

### What does NOT cause a state transition

- Brief flicker (one frame missed): IN_ROOM_UNSEEN for one frame is fine, returning to PRESENT on the next match doesn't even emit an event.
- Posture change (sitting → standing): different event, same state.
- Movement within a room: different event, same state.

### Special handling: cold start and IS_HOME

When Jarvis boots, the registry loads the last-known state from the event log. Every entity that was PRESENT/IN_ROOM_UNSEEN at the time of last shutdown is reset to `UNKNOWN_AT_BOOT` for the first 30 seconds. During those 30 seconds, observations resolve them to PRESENT or leave them as UNKNOWN_AT_BOOT.

After 30 seconds, any UNKNOWN_AT_BOOT entity that has not been observed transitions to its last-known-non-PRESENT state from the log: most often `IN_HOUSE_UNMONITORED` for residents who were home, or `DEPARTED` for residents who were last seen leaving.

The `IS_HOME` derived flag (used by the `who_is_home()` query) is computed live: a resident is home if their state is `PRESENT`, `IN_ROOM_UNSEEN`, `TRANSITIONING`, or `IN_HOUSE_UNMONITORED`. They are not home if `DEPARTED` or `UNKNOWN_AT_BOOT`.

---

## 5. Data Model

```python
# modules/world_model/types.py

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
import numpy as np


class EntityState(str, Enum):
    PRESENT = "present"
    IN_ROOM_UNSEEN = "in_room_unseen"
    TRANSITIONING = "transitioning"
    IN_HOUSE_UNMONITORED = "in_house_unmonitored"
    DEPARTED = "departed"
    UNKNOWN_AT_BOOT = "unknown_at_boot"


class EventType(str, Enum):
    FIRST_SEEN = "first_seen"
    NAME_LINKED = "name_linked"           # entity got linked to a persons.id
    ENTERED = "entered"                   # first observation in a session
    MOVED_TO = "moved_to"                 # confirmed room change
    MOVED_WITHIN_ROOM = "moved_within_room"
    POSTURE_CHANGED = "posture_changed"
    LOST_VISIBILITY = "lost_visibility"   # → IN_ROOM_UNSEEN
    REAPPEARED = "reappeared"             # ← from any unseen state
    ENTERED_UNMONITORED = "entered_unmonitored"  # → IN_HOUSE_UNMONITORED
    DEPARTED = "departed"                 # → DEPARTED via exterior exit
    INTERACTED_WITH = "interacted_with"
    PICKED_UP = "picked_up"
    PLACED_DOWN = "placed_down"
    STATIONARY_LONG = "stationary_long"
    CAMERA_DEGRADED = "camera_degraded"   # informational; affects state machine
    CAMERA_RESTORED = "camera_restored"


@dataclass
class WorldEntity:
    """
    A persistent thing the world model tracks.
    For people, person_id links to the existing persons table — IdentityManager
    is the source of truth for who this is. The World Model never re-asserts
    identity from its own embedding.
    """
    id: str                              # uuid, stable for life of entity
    entity_type: str                     # "person" | "cat" | "object"
    person_id: Optional[int] = None      # FK to persons.id (people only); None for cats/objects
    display_name: Optional[str] = None   # "Cole", "Mittens", "wallet" — denormalized cache
    state: EntityState = EntityState.PRESENT
    last_seen_ts: Optional[datetime] = None
    last_seen_room: Optional[str] = None
    last_seen_camera: Optional[str] = None
    last_seen_bbox: Optional[tuple] = None
    last_seen_landmark: Optional[str] = None
    last_state_change_ts: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 0.0              # current state confidence
    last_attribution_confidence: float = 0.0  # how sure we are this obs matched ent
    is_resident: bool = False
    metadata: dict = field(default_factory=dict)
    # metadata can include:
    #   posture_history, stable_posture, hand_overlap_frames
    #   cat-specific: color_class, color_histogram, behavioral_profile, seed
    #   object-specific: detected_class, last_clip_embedding, last_snapshot_path
    #   state-specific: entered_unmonitored_via, departed_via, departed_ts
    #   suspended_due_to_camera_health (bool)


@dataclass
class Observation:
    """
    Normalized output of the perception layer, produced by ObservationBuilder.
    World Model only reads Observations — never raw frames or detector outputs.
    """
    camera: str
    room: str
    obj_class: str                       # "person" | "cat" | "wallet" | ...
    bbox: tuple                          # (x1, y1, x2, y2) in camera frame
    confidence: float
    ts: datetime
    # Identity, if resolved by IdentityManager (people only):
    person_id: Optional[int] = None
    person_name: Optional[str] = None
    person_match_confidence: float = 0.0
    # For cats and objects, embedding is the visual fingerprint:
    visual_embedding: Optional[np.ndarray] = None
    # Auxiliary signals attached as available:
    metadata: dict = field(default_factory=dict)
    # metadata fields:
    #   "posture": str — e.g., "standing", "sitting"
    #   "hand_bboxes": list[tuple]
    #   "frame_width": int, "frame_height": int
    #   "crop_path": str — path to saved crop (used for enrollment)
    #   "color_histogram": np.ndarray (cats only)
    #   "color_class": str (cats only — "striped", "black", etc.)
    #   "size_normalized": float (cats only)
    #   "yaw", "pitch", "roll": float (faces only, from InsightFace)
    #   "blur_score": float (faces only)


@dataclass
class EntityEvent:
    """Append-only event log entry. Source of truth; entities table is a projection."""
    id: str
    ts: datetime
    entity_id: str
    person_id: Optional[int]
    entity_name: Optional[str]
    entity_type: str
    event_type: EventType
    room: Optional[str]
    camera: Optional[str]
    bbox: Optional[tuple]
    landmark: Optional[str]
    state: EntityState                   # state AFTER this event
    confidence: float
    snapshot_path: Optional[str]
    related_entity_id: Optional[str]
    metadata: dict = field(default_factory=dict)
```

### Why `person_id` matters so much

The repo already has a `persons` table managed by IdentityManager. Every named human is a row there with face/voice samples and identity metadata. The World Model's `world_entities.person_id` is a **nullable foreign key** to that table, with this contract:

- **For person entities:** `person_id` is set by IdentityManager. It can change over time as identity gets resolved (entity created as anonymous → IdentityManager later confirms it's Cole → `person_id` set, `display_name` updated, `NAME_LINKED` event emitted).
- **For cat and object entities:** `person_id` is always None. `display_name` carries the name (or stays None for unnamed items).
- **The World Model never writes to `persons` directly.** Naming a person, merging duplicates, modifying samples — all of that goes through IdentityManager's API.

This is the boundary that prevents identity rot.

---

## 6. Event Taxonomy

Every event downstream consumers will see:

| Event | Trigger | Debounce | Notes |
|---|---|---|---|
| `FIRST_SEEN` | New `WorldEntity` row created (anonymous or named). | None | Fires once per entity ever. |
| `NAME_LINKED` | An entity's `person_id` gets set or changed. | None | Fires when IdentityManager confirms identity. |
| `ENTERED` | Entity transitions into a room from outside the system's view. | None | Distinguished from FIRST_SEEN by entity already existing. |
| `MOVED_TO` | Confirmed room change (PRESENT in room A → PRESENT in room B). | None | Has `from_room` in metadata. |
| `MOVED_WITHIN_ROOM` | bbox center moved >8% of frame size, was previously stationary ≥1s. | spatial threshold | Has `approaching_landmark`. |
| `POSTURE_CHANGED` | Posture classification differs from stable for ≥3 consecutive frames. | 3-frame minimum | Comes from `modules/vision/posture_analyzer.py`. |
| `LOST_VISIBILITY` | PRESENT → IN_ROOM_UNSEEN. | None | Has `near_exit: bool`, `last_landmark`, `reason`. |
| `REAPPEARED` | Any unseen state → PRESENT via match. | None | Has `from_state` in metadata. |
| `ENTERED_UNMONITORED` | TRANSITIONING → IN_HOUSE_UNMONITORED. | None | Has `entered_via` (which doorway). |
| `DEPARTED` | PRESENT/TRANSITIONING → DEPARTED via exterior exit. | None | Has `via_exit`, exit camera footage reference. |
| `INTERACTED_WITH` | Hand bbox overlaps tracked-object bbox for ≥3 frames. | 3-frame minimum | Phase 5. |
| `PICKED_UP` | Object visible → hand overlap → object disappears. | sequence | Phase 5. |
| `PLACED_DOWN` | Hand visible at object-shaped region → object becomes visible there. | sequence | Phase 5. |
| `STATIONARY_LONG` | No MOVED_WITHIN_ROOM for >5min while PRESENT. | timer | Useful for *"Cole has been at the desk for 4 hours"*. |
| `CAMERA_DEGRADED` / `CAMERA_RESTORED` | Camera health subscriber detects state change. | None | Affects state machine — Section 14. |

Each event payload includes the full context (entity_id, entity_name, person_id, room, camera, bbox, landmark, state, confidence, metadata) so consumers don't need to do follow-up lookups.

---

## 7. Camera Topology Config

The repo's `config.yaml` already has a sophisticated `rooms:` block with per-room `video`, `mic`, `speaker`, `fps_active`, `fps_idle`. The World Model adds an optional `world_model:` sub-block per room. Existing fields are not modified.

### Schema additions

```yaml
# config.yaml — add per-room geometry. Existing fields untouched.

rooms:
  - id: office
    display_name: Office
    video: { type: usb_index, device_index: 0 }
    mic: { type: usb_device_mic, device_name: default, sample_rate_hz: 16000, channels: 1 }
    speaker: { type: usb_device_spk, device_name: default, sample_rate_hz: 22050, channels: 1 }
    fps_active: 30
    fps_idle: 30

    world_model:                      # NEW
      enabled: true
      frame_width: 640
      frame_height: 480
      exits:
        # to_room: handoff to another camera-equipped room
        - kind: to_room
          to: living_room
          polygon: [[600, 0], [640, 0], [640, 480], [600, 480]]
        # to_unmonitored_zone: door into a room without a camera
        - kind: to_unmonitored_zone
          to: guest_bedroom
          polygon: [[0, 200], [40, 200], [40, 400], [0, 400]]
      landmarks:
        # named regions inside the camera frame; used for "approaching X" events
        # and for "last seen at X" phrasing
        - name: desk
          polygon: [[200, 250], [450, 250], [450, 400], [200, 400]]
        - name: under_desk
          polygon: [[200, 380], [450, 380], [450, 480], [200, 480]]
        - name: bookshelf
          polygon: [[0, 100], [150, 100], [150, 350], [0, 350]]

  - id: living_room
    # ... existing config (Wyze RTSP video, audio, SSH speaker) ...
    world_model:
      enabled: true
      frame_width: 1920
      frame_height: 1080
      exits:
        - kind: to_room
          to: office
          polygon: [[0, 400], [200, 400], [200, 1080], [0, 1080]]
        - kind: to_room
          to: kitchen
          polygon: [[1700, 0], [1920, 0], [1920, 600], [1700, 600]]
        # exterior_exit: the door to outside. CRUCIAL for DEPARTED state.
        - kind: exterior_exit
          name: front_door
          polygon: [[800, 800], [1100, 800], [1100, 1080], [800, 1080]]
        - kind: to_unmonitored_zone
          to: spare_bedroom_1
          polygon: [[1500, 600], [1700, 600], [1700, 900], [1500, 900]]
      landmarks:
        - name: couch
          polygon: [[600, 600], [1400, 600], [1400, 950], [600, 950]]
        - name: tv
          polygon: [[700, 100], [1300, 100], [1300, 450], [700, 450]]

  - id: kitchen
    # existing ...
    world_model:
      enabled: true
      frame_width: 1920
      frame_height: 1080
      exits:
        - kind: to_room
          to: living_room
          polygon: [[0, 0], [200, 0], [200, 1080], [0, 1080]]
        # If kitchen has a back door:
        - kind: exterior_exit
          name: back_door
          polygon: [[1700, 800], [1920, 800], [1920, 1080], [1700, 1080]]
      landmarks:
        - name: stove
          polygon: [[800, 200], [1100, 200], [1100, 600], [800, 600]]
        - name: fridge
          polygon: [[1400, 100], [1700, 100], [1700, 700], [1400, 700]]

  - id: bedroom
    # existing (master bedroom, Wyze) ...
    world_model:
      enabled: true
      frame_width: 1920
      frame_height: 1080
      exits:
        - kind: to_room
          to: living_room      # or hallway-equivalent in your topology
          polygon: [[0, 400], [150, 400], [150, 1080], [0, 1080]]
      landmarks:
        - name: bed
          polygon: [[600, 500], [1400, 500], [1400, 950], [600, 950]]

  - id: laundry_room
    # existing (ESP32-CAM HTTP, audio TBD) ...
    world_model:
      enabled: true
      frame_width: 800
      frame_height: 600
      exits:
        - kind: to_room
          to: kitchen          # or wherever it connects
          polygon: [[0, 200], [80, 200], [80, 500], [0, 500]]
      landmarks:
        - name: washer
          polygon: [[200, 300], [450, 300], [450, 550], [200, 550]]
        - name: dryer
          polygon: [[450, 300], [700, 300], [700, 550], [450, 550]]
```

**Three exit kinds**, each with different state-machine consequences:

| Kind | State on disappearance near this exit | Resolution |
|---|---|---|
| `to_room` | TRANSITIONING(to=room) | rematch in named room's camera, or fall back to IN_ROOM_UNSEEN if handoff fails |
| `to_unmonitored_zone` | IN_HOUSE_UNMONITORED(entered_via=name) | rematch in any indoor camera |
| `exterior_exit` | DEPARTED(via=name) | rematch in any indoor camera (= "they came back") |

**Note on the two unmonitored bedrooms:** their *doors* don't have cameras directly facing them. That means there is no exit polygon labeled `to_unmonitored_zone: spare_bedroom_X` on a camera that sees those doors specifically. So entries into those bedrooms will most often be classified as IN_ROOM_UNSEEN with a `near_exit=false` reason — they vanish *somewhere* in the living_room (or wherever) without a known doorway crossing. That's fine. The state is still correct: "Cole was last seen in the living_room and I haven't seen him leave the house, so he's still home, just in a place I can't see." You can later add coarse `to_unmonitored_zone` polygons by inferring the rough region of the camera frame where those doors are visible.

### Pydantic validation

The repo already validates rooms with `core.config.expand_and_validate`. Extend the `Room` Pydantic model with an optional `world_model: Optional[RoomWorldModel]` field.

```python
# core/config.py — extension to existing typed_rooms validation

from pydantic import BaseModel, Field
from typing import Literal, Optional


class ExitDef(BaseModel):
    kind: Literal["to_room", "to_unmonitored_zone", "exterior_exit"]
    polygon: list[tuple[int, int]] = Field(min_length=3)
    to: Optional[str] = None         # required for to_room and to_unmonitored_zone
    name: Optional[str] = None       # required for exterior_exit (e.g., "front_door")

    def model_post_init(self, _ctx) -> None:
        if self.kind in ("to_room", "to_unmonitored_zone") and not self.to:
            raise ValueError(f"{self.kind} exit requires 'to' field")
        if self.kind == "exterior_exit" and not self.name:
            raise ValueError("exterior_exit requires 'name' field")


class LandmarkDef(BaseModel):
    name: str
    polygon: list[tuple[int, int]] = Field(min_length=3)


class RoomWorldModel(BaseModel):
    enabled: bool = True
    frame_width: int = Field(gt=0)
    frame_height: int = Field(gt=0)
    exits: list[ExitDef] = []
    landmarks: list[LandmarkDef] = []


# Extend the existing Room model:
# class Room(BaseModel):
#     ... existing fields ...
#     world_model: Optional[RoomWorldModel] = None
```

### Drawing the polygons

You'll do this once per camera. Two reasonable workflows:

1. **Quick & dirty:** snapshot a frame (`ffmpeg -i rtsp://... -vframes 1 frame.png` or grab from the dashboard), eyeball corners in any image editor, copy pixel coords into YAML.
2. **Proper:** add a small page to the dashboard that lets you click points on a frozen frame and exports the polygon list. ~50 lines of JS, worth the time given you've got 5+ rooms.

The dashboard polygon editor is parked in Phase 4; for the MVP, hand-drawing in YAML is fine.

---

## 8. Storage Layer (Async, Linked to persons)

The repo uses `aiosqlite` through `DatabaseManager` (in `modules/memory/database.py`). The World Model uses **the same connection**, the same transaction discipline, the same migrations approach — never opens its own sync sqlite3 connection.

### Schema additions to existing `data/jarvis.db`

```sql
-- Add to modules/memory/database.py, in the existing migration runner.
-- Existing tables (persons, face_samples, voice_samples, identity_pending,
-- notifications, memories, activity_log, etc.) are not modified.

CREATE TABLE IF NOT EXISTS world_entities (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,                      -- 'person' | 'cat' | 'object'
    person_id INTEGER REFERENCES persons(id),       -- NULLable; set for resolved people
    display_name TEXT,                              -- denormalized for queries
    state TEXT NOT NULL,
    last_seen_ts TEXT,
    last_seen_room TEXT,
    last_seen_camera TEXT,
    last_seen_bbox TEXT,                            -- JSON [x1,y1,x2,y2]
    last_seen_landmark TEXT,
    last_state_change_ts TEXT,
    confidence REAL,
    last_attribution_confidence REAL,
    is_resident INTEGER DEFAULT 0,
    metadata TEXT                                   -- JSON
);

CREATE INDEX IF NOT EXISTS idx_world_entities_person
    ON world_entities(person_id);
CREATE INDEX IF NOT EXISTS idx_world_entities_state
    ON world_entities(state);
CREATE INDEX IF NOT EXISTS idx_world_entities_room
    ON world_entities(last_seen_room);


CREATE TABLE IF NOT EXISTS world_entity_events (
    id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    entity_id TEXT NOT NULL REFERENCES world_entities(id),
    person_id INTEGER REFERENCES persons(id),
    entity_name TEXT,
    entity_type TEXT NOT NULL,
    event_type TEXT NOT NULL,
    room TEXT,
    camera TEXT,
    bbox TEXT,                                       -- JSON
    landmark TEXT,
    state TEXT,                                      -- state AFTER this event
    confidence REAL,
    snapshot_path TEXT,
    related_entity_id TEXT,
    metadata TEXT                                    -- JSON
);

CREATE INDEX IF NOT EXISTS idx_world_events_entity_ts
    ON world_entity_events(entity_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_world_events_room_ts
    ON world_entity_events(room, ts DESC);
CREATE INDEX IF NOT EXISTS idx_world_events_type_ts
    ON world_entity_events(event_type, ts DESC);
CREATE INDEX IF NOT EXISTS idx_world_events_person_ts
    ON world_entity_events(person_id, ts DESC);


-- Visual fingerprint for cats and objects (people use the existing face_samples).
-- Stored as a separate table to avoid bloating world_entities row size.
CREATE TABLE IF NOT EXISTS world_entity_embeddings (
    entity_id TEXT PRIMARY KEY REFERENCES world_entities(id),
    embedding BLOB NOT NULL,                         -- raw float32 bytes
    dimension INTEGER NOT NULL,                      -- 512 for ArcFace, 512 for CLIP, etc.
    updated_ts TEXT NOT NULL
);
```

**Notes on the schema:**

- `person_id` columns are nullable. Cat and object entities never set them. Person entities start NULL (`FIRST_SEEN` for an unrecognized human) and get populated when IdentityManager confirms identity (`NAME_LINKED` event).
- No `face_samples` or `voice_samples` tables here. Those exist in the IdentityManager's domain. The World Model only stores the *visual* embedding for cats and objects, where there's no IdentityManager involvement.
- `world_entity_events` is the source of truth. `world_entities` is a projection — if it ever gets corrupted, replay events from the log to rebuild.

### Migration strategy

Add the schema additions to `modules/memory/database.py`'s existing `_ensure_schema` (or whatever the migration runner is called). Idempotent CREATE TABLE IF NOT EXISTS — safe to run on every boot.

### Encryption later, not now

SQLCipher is a real Windows dependency pain. For now, plain SQLite via the existing `DatabaseManager` is fine. The `data/jarvis.db` file lives on a Windows-encrypted volume already (BitLocker or similar) — that's the right level for a household movement log. Application-level SQLCipher can be revisited if you ever want to e.g. share the DB across machines without re-encrypting the volume.

---

## 9. Identity Integration: Who vs. Where

The boundary again: **IdentityManager owns who. WorldModel owns where, state, continuity.**

### How the World Model receives identity

`ObservationBuilder` is the layer that bridges them. When it builds an `Observation` for a person detection, it asks IdentityManager to identify the face crop:

```python
# Sketch — full code in Section 18

async def build_person_observation(self, frame, bbox, room, camera) -> Observation:
    crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    
    face_results = self.face_recognizer.detect_and_embed(crop)
    if not face_results:
        # No face detectable in this person crop — still emit Observation for
        # spatial continuity, but no identity attached.
        return Observation(..., person_id=None, person_name=None)
    
    face = face_results[0]
    
    # IdentityManager is the only path to identity.
    identity_result = await self.identity_manager.identify_from_embedding_async(
        face["embedding"]
    )
    # identity_result is one of:
    #   IdentityMatch(person_id=42, person_name="Cole", confidence=0.91, status="match")
    #   IdentityMatch(person_id=None, status="unknown",   confidence=best_sim)
    #   IdentityMatch(person_id=None, status="ambiguous", confidence=best_sim)
    
    return Observation(
        camera=camera, room=room,
        obj_class="person", bbox=bbox,
        confidence=detection_confidence,
        ts=datetime.utcnow(),
        person_id=identity_result.person_id,
        person_name=identity_result.person_name,
        person_match_confidence=identity_result.confidence,
        visual_embedding=None,    # people don't need a separate world-model embedding
        metadata={
            "crop_path": saved_crop_path,
            "yaw": face["yaw"], "pitch": face["pitch"], "roll": face["roll"],
            "blur_score": _laplacian_var(crop),
        },
    )
```

The contract: **the World Model trusts whatever IdentityManager says.** If IdentityManager returns "unknown," the World Model creates an anonymous person entity (`person_id = None`). If IdentityManager returns "ambiguous," the World Model still creates an anonymous entity, but the Observation carries no identity for the association algorithm to lean on. If IdentityManager later resolves the same anonymous entity to a person, the World Model writes a `NAME_LINKED` event and updates the row.

### Resolving anonymous → named

When an anonymous person entity exists (FIRST_SEEN happened with no identity match) and a later observation matches it visually (in the World Model's association sense — same camera, plausible movement, etc.) AND that later observation has a confident IdentityManager match, the resolution flow is:

1. World Model emits `NAME_LINKED(entity_id, person_id, person_name)`.
2. World Model updates `world_entities.person_id` and `display_name`.
3. World Model does NOT touch `persons` or `face_samples` — that's IdentityManager's job, which already happened when it returned the confident match.

This keeps the responsibility clean: the World Model is just stamping the identity authority's verdict onto its own row.

### What if IdentityManager and the World Model's own continuity disagree?

Example: World Model has an active entity `WE-77` (anonymous, last seen 2 seconds ago in the office, near the desk). A new observation arrives, also in the office near the desk, and IdentityManager identifies it confidently as Cole — but the existing Cole entity (`WE-12`) is currently `IN_HOUSE_UNMONITORED` after walking into the guest bedroom 5 minutes ago.

This is a real conflict. The World Model's spatial-temporal continuity says "this is the anonymous WE-77." IdentityManager says "this is Cole, who is WE-12."

**Resolution rule: IdentityManager wins for identity.** The new observation gets attributed to `WE-12` (Cole). `WE-77` becomes an unresolved anonymous entity that may or may not get cleaned up later. `WE-12`'s state goes `IN_HOUSE_UNMONITORED → REAPPEARED → PRESENT (in office)`. An event is emitted with `metadata.identity_overrode_continuity: true` so the dashboard can flag it.

This is rare but real, and the rule is: identity is more reliable than spatial continuity for people, because faces don't change between bedrooms.

---

## 10. Auto-Enrollment Inside IdentityManager

Auto-enrollment lives in IdentityManager, not the World Model. The diversity-replacement coreset algorithm — the right algorithm — extends IdentityManager's existing sample-bank logic.

### The extension

The repo's IdentityManager already has multiple samples per person, match thresholds, stranger thresholds, margins, pending clusters, drift verification. The auto-enrollment changes:

1. **Diversity gate.** When a new candidate sample arrives for an existing person, reject it if its cosine similarity to any existing sample is ≥0.95. (Existing IdentityManager may already have something like this; if not, add it.)

2. **Bounded-capacity coreset.** Cap samples per person at 30 (configurable). When at cap and a new candidate would otherwise be added, find the existing sample with highest average cosine similarity to its peers (the "most redundant"). Only swap the new one in if it would *decrease* the average pairwise similarity of the bank — i.e., if it's more diverse than the most redundant sample.

3. **Quality gates.** Reject candidates with: face area below 80×80 pixels, head yaw above ±45°, pitch above ±35°, blur (Laplacian variance) below 100, association confidence below 0.85. The first three signals come from the InsightFace ArcFace pipeline (Section 11), which exposes them as part of detection.

4. **Pause during ambiguity.** If two persons in IdentityManager are flagged as merge candidates (centroid sim 0.7–0.85), pause enrollment for both until merge resolution (manual confirm or auto-merge at >0.85).

The algorithm:

```python
# Extension to IdentityManager. Pseudocode; integrate with the actual API names
# in modules/identity/identity_manager.py.

async def consider_new_sample_async(
    self, person_id: int, new_embedding: np.ndarray,
    crop_path: str, quality_metadata: dict
) -> bool:
    """
    Called by ObservationBuilder/WorldModel after every confident person obs.
    Decides whether the sample is good enough and diverse enough to add.
    Returns True if added, False if rejected.
    """
    # Quality gate
    if not self._passes_quality_gates(quality_metadata):
        return False

    # Pause check
    if person_id in self._merge_candidate_persons:
        return False

    existing = await self.get_face_samples_async(person_id)
    existing_embeddings = [s.embedding for s in existing]

    # Diversity gate: refuse near-duplicates
    if existing_embeddings:
        max_sim = max(_cosine(new_embedding, e) for e in existing_embeddings)
        if max_sim >= self.SAMPLES_DIVERSITY_THRESHOLD:    # 0.95
            return False

    # Capacity check
    if len(existing) < self.SAMPLES_PER_PERSON_MAX:        # 30
        await self.add_face_sample_async(person_id, crop_path, new_embedding,
                                          quality_metadata)
        await self._rebuild_centroid(person_id)
        return True

    # At cap: only swap if new is more diverse than the most-redundant existing
    redundant_idx = _most_redundant_index(existing_embeddings)
    if _would_increase_diversity(existing_embeddings, redundant_idx, new_embedding):
        await self.remove_face_sample_async(existing[redundant_idx].id)
        await self.add_face_sample_async(person_id, crop_path, new_embedding,
                                          quality_metadata)
        await self._rebuild_centroid(person_id)
        return True

    return False


def _most_redundant_index(embeddings: list[np.ndarray]) -> int:
    """Index whose average cosine similarity to its peers is highest."""
    n = len(embeddings)
    if n < 2:
        return 0
    M = np.stack([e / (np.linalg.norm(e) + 1e-9) for e in embeddings])
    S = M @ M.T
    np.fill_diagonal(S, 0.0)
    return int(np.argmax(S.sum(axis=1) / (n - 1)))


def _would_increase_diversity(
    existing: list[np.ndarray], evict_idx: int, candidate: np.ndarray
) -> bool:
    """Check if replacing existing[evict_idx] with candidate decreases avg sim."""
    replaced = list(existing)
    replaced[evict_idx] = candidate
    return _avg_pairwise_sim(replaced) < _avg_pairwise_sim(existing)


def _avg_pairwise_sim(embeddings: list[np.ndarray]) -> float:
    n = len(embeddings)
    if n < 2:
        return 0.0
    M = np.stack([e / (np.linalg.norm(e) + 1e-9) for e in embeddings])
    S = M @ M.T
    np.fill_diagonal(S, 0.0)
    return float(S.sum() / (n * (n - 1)))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
```

### When does World Model trigger enrollment?

After every confident person observation that's been associated to a non-anonymous entity:

```python
# In WorldModel._update_matched, after state update:

if (obs.person_id is not None
    and obs.person_match_confidence >= self.cfg["enrollment"]["min_observation_confidence"]
    and "crop_path" in obs.metadata):
    asyncio.create_task(
        self.identity_manager.consider_new_sample_async(
            person_id=obs.person_id,
            new_embedding=obs.metadata.get("face_embedding"),  # populated by ObservationBuilder
            crop_path=obs.metadata["crop_path"],
            quality_metadata={
                "yaw": obs.metadata.get("yaw"),
                "pitch": obs.metadata.get("pitch"),
                "blur_score": obs.metadata.get("blur_score"),
            },
        )
    )
```

The World Model fires the request; IdentityManager owns the decision and the storage. Fire-and-forget: enrollment failure doesn't block the hot path.

### Voice auto-enrollment

Same algorithm, different gates. Voice samples need: minimum 1.5s duration, minimum 15 dB SNR, VAD pass (speech actually detected), single-speaker check (no overlapping speech), music detector pass (YAMNet `music` category < 0.3 — your existing YAMNet integration provides this). Otherwise identical: diversity threshold 0.92, cap 20, evict-if-more-diverse.

Owned by IdentityManager (or its voice arm — likely the existing `modules/voice/speaker_id.py` integrating with IdentityManager).

---

## 11. The ArcFace Upgrade (Done From Day One)

This is the call: do this from the start, not as a deferred optional upgrade. It lives in `modules/vision/face_recognizer.py`, integrated with IdentityManager.

### Why ArcFace (InsightFace `buffalo_l`) over DeepFace+Facenet

| Property | DeepFace+Facenet (current) | InsightFace ArcFace `buffalo_l` |
|---|---|---|
| Embedding dim | 128 | 512 |
| Encoder training data | smaller, older | very large (millions of identities) |
| Inter-class separation (different people far apart) | OK | excellent |
| Intra-class tightness (same person across angles/lighting) | OK | excellent |
| Per-detection extras | bbox, similarity | bbox, similarity, **head pose (yaw/pitch/roll), age, gender** |
| GPU memory | low | ~700MB |
| Speed on RTX 4070 Ti | fast | fast (5–10ms per face) |

The head pose extras are the killer feature for auto-enrollment quality gates — knowing yaw before deciding whether to keep a sample is a free upgrade.

### The migration

`modules/vision/face_recognizer.py` is the only file that changes. Its public API stays the same as far as IdentityManager is concerned: `detect_and_embed(image) → list[face_dict]`. Internals swap from DeepFace+Facenet to InsightFace.

```python
# modules/vision/face_recognizer.py — REWRITE

import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceRecognizer:
    """
    ArcFace-based face recognition using InsightFace's buffalo_l model.
    Integrates with IdentityManager — provides detection + embedding + pose.
    Identity matching (centroid + margin gating) is IdentityManager's job;
    this class produces the embeddings IdentityManager compares against.
    """

    SIMILARITY_THRESHOLD = 0.5    # absolute floor (centroid match)
    MARGIN_THRESHOLD = 0.10       # gap from 2nd best required for confident match

    def __init__(self, use_gpu: bool = True, model_name: str = "buffalo_l"):
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu
            else ["CPUExecutionProvider"]
        )
        self.app = FaceAnalysis(name=model_name, providers=providers)
        self.app.prepare(ctx_id=0 if use_gpu else -1, det_size=(640, 640))

    def detect_and_embed(self, image: np.ndarray) -> list[dict]:
        """
        Run face detection + embedding on a frame.
        Returns list of detection dicts including embedding and head pose.
        """
        if image is None or image.size == 0:
            return []
        faces = self.app.get(image)
        return [
            {
                "bbox": tuple(int(c) for c in f.bbox),
                "embedding": f.normed_embedding,        # already L2-normalized, 512-dim
                "det_score": float(f.det_score),
                "yaw": float(f.pose[1]) if hasattr(f, "pose") else 0.0,
                "pitch": float(f.pose[0]) if hasattr(f, "pose") else 0.0,
                "roll": float(f.pose[2]) if hasattr(f, "pose") else 0.0,
                "age": int(f.age) if hasattr(f, "age") else None,
                "gender": int(f.gender) if hasattr(f, "gender") else None,
            }
            for f in faces
        ]

    def embed_largest_face(self, image: np.ndarray) -> dict | None:
        """Convenience: detect, return only the largest face."""
        faces = self.detect_and_embed(image)
        if not faces:
            return None
        return max(faces, key=lambda f: (f["bbox"][2]-f["bbox"][0])
                                       * (f["bbox"][3]-f["bbox"][1]))

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))   # both already normalized
```

### IdentityManager changes

IdentityManager's centroid storage moves from 128-dim to 512-dim. This is a one-time migration — existing 128-dim samples are unusable with ArcFace and need to be regenerated.

**Migration strategy:**
1. Add a `model_version` column to `face_samples` (`'facenet_v1'` vs `'arcface_buffalo_l_v1'`).
2. On boot, IdentityManager filters its centroids to samples matching the active model_version.
3. Auto-enrollment generates new ArcFace samples as observations come in.
4. Old Facenet samples are kept (for history / fallback) but not used in matching.
5. After ~2 weeks of running, you can drop the Facenet samples manually.

This means **the first day or two after upgrade, IdentityManager will under-recognize until enough ArcFace samples are collected.** You can accelerate by:
- Bulk-running ArcFace over existing labeled crop archives if any exist.
- Manually enrolling each resident with 5–10 photos through a one-time dashboard flow (ArcFace, not Facenet). 5 minutes per person.

### Centroid + margin gating in IdentityManager

```python
# Inside IdentityManager.identify_from_embedding_async — pseudocode

async def identify_from_embedding_async(
    self, embedding: np.ndarray
) -> IdentityMatch:
    if not self._centroids:
        return IdentityMatch(person_id=None, status="no_enrolled_persons")
    
    person_ids = list(self._centroids.keys())
    sims = np.array([
        FaceRecognizer.cosine_similarity(embedding, self._centroids[pid])
        for pid in person_ids
    ])
    order = np.argsort(sims)[::-1]
    best_sim = float(sims[order[0]])
    second_sim = float(sims[order[1]]) if len(sims) > 1 else -1.0
    margin = best_sim - second_sim

    if best_sim < FaceRecognizer.SIMILARITY_THRESHOLD:
        return IdentityMatch(person_id=None, status="unknown",
                             confidence=best_sim)
    if margin < FaceRecognizer.MARGIN_THRESHOLD:
        # Two persons both close — refuse to guess. Add to identity_pending.
        await self._record_ambiguous(embedding, top=person_ids[order[0]],
                                     second=person_ids[order[1]],
                                     best_sim=best_sim, margin=margin)
        return IdentityMatch(person_id=None, status="ambiguous",
                             confidence=best_sim)
    
    return IdentityMatch(
        person_id=person_ids[order[0]],
        person_name=self._person_name(person_ids[order[0]]),
        confidence=best_sim,
        margin=margin,
        status="match",
    )
```

The margin threshold (0.10) is the fix for the misidentification you reported. With it, the system refuses to guess between Cole-and-Anna when they're both close, and emits an `ambiguous` flag that the dashboard can surface for review. False unknowns are recoverable; false positives between residents corrupt the identity bank.

### Voice fusion

When both face and voice observations resolve to the same entity within a short window, the World Model upgrades confidence. When they disagree, the higher-confidence wins and the event is flagged `disputed=true`. **Don't** force them to agree — face on one camera and voice on another mic for *different people simultaneously* is normal household behavior, not a bug.

---

## 12. The ObservationBuilder Adapter

This is the missing piece. The repo already has multiple detector outputs running at different cadences — ObservationBuilder is the layer that synthesizes them into normalized `Observation` payloads for the World Model.

### Where it sits

```
[CameraManager: per-room frame stream]
            │
            ▼
┌───────────────────────────────────────┐
│  Per-room detection workers           │
│  (existing — unchanged):              │
│  - ObjectDetector (YOLOv8)            │
│  - PostureAnalyzer (MediaPipe pose)   │
│  - FaceRecognizer (ArcFace)           │
│  - SceneAnalyzer (LLM, slower cadence)│
└─────────────┬─────────────────────────┘
              │
              ▼ raw outputs at various rates
   ┌─────────────────────────┐
   │  ObservationBuilder     │ NEW
   │  (normalizes to         │
   │   Observation objects)  │
   └─────────────┬───────────┘
                 │
                 ▼
        bus.publish("vision.observation", batch)
                 │
                 ▼
            WorldModel
```

ObservationBuilder is *not* the same thing as the existing scene_analyzer. SceneAnalyzer is a slower, LLM-mediated semantic descriptor ("I see Cole sitting at his desk reading"). ObservationBuilder is a fast, structured aggregator that fires every detection tick.

### Per-room cadence

The repo has variable FPS per room: office at 30, others at 5 active / 1 idle, laundry_room at 5/1. ObservationBuilder runs at whatever cadence the camera produces. The World Model is observation-driven, not timer-driven, so different cadences are fine.

Full code in Section 18.

---

## 13. Association Algorithm

Hungarian assignment over a multi-signal cost matrix, adapted to the new state machine and the bounded-house priors.

```
INPUT:
  observations: [Observation, ...]   # this tick, this camera
  candidates:   subset of WorldEntity likely to be in this camera now

STEP 1 — Build cost matrix
  For each (observation, entity) pair:
    cost = _pair_cost(obs, entity)
    Cap above COST_REJECT to refuse impossible pairs.

STEP 2 — Solve assignment with Hungarian (scipy.optimize.linear_sum_assignment).

STEP 3 — Process matches (cost < COST_REJECT)
  For each (obs, ent):
    Update entity (last_seen_*, confidence).
    Compute deltas:
      Was state in (IN_ROOM_UNSEEN, TRANSITIONING, IN_HOUSE_UNMONITORED, DEPARTED)?
        → emit REAPPEARED, set state = PRESENT.
      Did room change?
        → emit MOVED_TO, with from_room.
    Compute granular events (movement, posture, interaction).

STEP 4 — Process unmatched observations
  Try widening the candidate pool (any same-type entity with strong embedding match).
  For people: if obs.person_id is set, look up that entity directly (identity wins
    over spatial continuity).
  Otherwise create a new anonymous entity, emit FIRST_SEEN.

STEP 5 — Process unmatched entities (entity expected here, no observation)
  ONLY for entities that were PRESENT in this camera last tick.
  Skip if camera is in unhealthy set (Section 14).
  Inspect last_bbox vs camera's exit polygons:
    - Over an exterior_exit polygon → DEPARTED
    - Over a to_room exit polygon → TRANSITIONING(to=room)
    - Over a to_unmonitored_zone exit polygon → TRANSITIONING(target_kind=zone)
    - Otherwise → IN_ROOM_UNSEEN, emit LOST_VISIBILITY with reason=in_frame_disappearance
  Emit appropriate events. NO timeout-driven escalation to a "MISSING" state.

STEP 6 — Periodic timer (every 2s)
  For each entity in TRANSITIONING:
    If it points to a to_unmonitored_zone exit AND elapsed > T_handoff_seconds:
      → IN_HOUSE_UNMONITORED, emit ENTERED_UNMONITORED.
    If it points to a to_room exit AND elapsed > T_handoff_seconds:
      → IN_ROOM_UNSEEN (the handoff failed; person didn't show up where expected).
        This is unusual; emit metadata.handoff_failed=true for dashboard inspection.

  No escalation from IN_ROOM_UNSEEN, IN_HOUSE_UNMONITORED, or DEPARTED via timer.
  These states are stable and resolve only by new observations.
```

### The cost function for people

```python
def _person_pair_cost(self, obs: Observation, ent: WorldEntity) -> float:
    """
    People: identity-first cost. If IdentityManager confirms identity, that
    dominates everything else.
    """
    # Hard reject: type mismatch
    if ent.entity_type != "person":
        return self.cfg["cost_reject"] * 2

    # Strongest signal: identity match.
    if obs.person_id is not None and ent.person_id is not None:
        if obs.person_id == ent.person_id:
            # Identity confirms it's the same person. Spatial cost only
            # disambiguates between hypothetical multiple Cole-entities.
            spatial = self._spatial_distance(obs, ent)
            return 0.05 * spatial  # very low cost, very lightly modulated
        else:
            # Identity says different person. Hard reject.
            return self.cfg["cost_reject"] * 2

    # If identity is unknown on the obs side, fall back to spatial-temporal continuity.
    if not ent.last_seen_ts:
        return self.cfg["cost_reject"] * 2
    seconds_gone = (obs.ts - ent.last_seen_ts).total_seconds()
    if seconds_gone > 60.0:
        return self.cfg["cost_reject"] * 2  # too long ago

    if obs.camera == ent.last_seen_camera:
        cam_cost = 0.0
    elif self._cameras_are_neighbors(obs.camera, ent.last_seen_camera):
        cam_cost = 0.3
    else:
        cam_cost = 0.7

    spatial = self._spatial_distance(obs, ent)
    time_cost = min(seconds_gone / 60.0, 1.0)

    return 0.5 * cam_cost + 0.3 * spatial + 0.2 * time_cost
```

The cat cost function (Phase 4) and object cost function (Phase 4) are added in Section 22 and 23.

---

## 14. Camera Health & Degraded Modes

Elevated to first-class concern because of how badly it goes wrong otherwise. Concrete failure: a Wyze drops connection for 30 seconds, every PRESENT entity in that camera's room transitions to IN_ROOM_UNSEEN, Anna asks where you are, Jarvis says *"I haven't seen Cole since this morning."* Dead system trust.

### How it works

1. **CameraManager already tracks failure counts and reconnect attempts.** Extend it to publish health events:
   - `camera.health` event with `{camera_id, status: 'healthy'|'degraded'|'down', reason, ts}`
   - Status transitions:
     - `healthy → degraded` after N consecutive failed frame reads (e.g., 3 in a 5-second window)
     - `degraded → down` after M seconds without successful read (e.g., 30 seconds)
     - `down → degraded → healthy` on reconnect (don't jump straight back to healthy; wait for sustained success)

2. **WorldModel subscribes to `camera.health`.** When a camera goes `degraded` or `down`:
   - **Suspend** state machine processing for entities whose `last_seen_camera == this_camera`. Don't transition them to IN_ROOM_UNSEEN just because the camera died.
   - Mark each affected entity with `metadata.suspended_due_to_camera_health = true`.
   - Emit `CAMERA_DEGRADED` events for each affected entity (informational).

3. When the camera comes back `healthy`:
   - Resume state machine processing.
   - Emit `CAMERA_RESTORED` events.
   - The next observations (or lack thereof) drive normal state transitions.

### Implementation sketch

```python
# modules/world_model/world_model.py — health subscription

async def _on_camera_health(self, payload: dict):
    camera_id = payload["camera_id"]
    status = payload["status"]
    
    async with self._lock:
        if status in ("degraded", "down"):
            self._unhealthy_cameras.add(camera_id)
            for ent in self.entities.values():
                if ent.last_seen_camera == camera_id and ent.state == EntityState.PRESENT:
                    ent.metadata["suspended_due_to_camera_health"] = True
                    await self._emit(EventType.CAMERA_DEGRADED, ent, obs=None,
                                     metadata={"camera": camera_id, "status": status})
                    await self.store.upsert_entity(ent)
        elif status == "healthy":
            self._unhealthy_cameras.discard(camera_id)
            for ent in self.entities.values():
                if ent.metadata.pop("suspended_due_to_camera_health", False):
                    await self._emit(EventType.CAMERA_RESTORED, ent, obs=None,
                                     metadata={"camera": camera_id})
                    await self.store.upsert_entity(ent)
```

And in `_handle_unmatched_entity`, check before transitioning:

```python
async def _handle_unmatched_entity(self, ent: WorldEntity, camera: str, ts: datetime):
    if camera in self._unhealthy_cameras:
        return                             # camera is down — don't penalize the entity
    if ent.metadata.get("suspended_due_to_camera_health"):
        return                             # still resuming
    # ... rest of the disappearance logic ...
```

**Net effect:** Wyze loses connection for 30s. Cole stays PRESENT for those 30s. Anna asks where Cole is. *"He's in the office."* Camera comes back. Next observation either confirms (still PRESENT, no event) or shows him gone (normal LOST_VISIBILITY transition with all the right reasoning).

This has to ship in Phase 1, not as polish. Without it, the system is loudly wrong at exactly the moments it most needs to be quiet.

---

## 15. Full Code: types.py and geometry.py

### `modules/world_model/types.py`

Already in Section 5 above — paste that content into the file.

### `modules/world_model/geometry.py`

```python
# modules/world_model/geometry.py
"""
Pure geometry helpers. No I/O, no imports from other world_model modules.
Used by WorldModel for exit/landmark detection.
"""

from typing import Sequence


def point_in_polygon(x: float, y: float, polygon: Sequence[tuple[float, float]]) -> bool:
    """
    Standard ray-casting algorithm. Returns True if (x, y) is inside the polygon.
    Polygon is a list of (x, y) tuples; the polygon closes implicitly.
    """
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def bbox_center(bbox: tuple) -> tuple[float, float]:
    """Return (cx, cy) for a bbox (x1, y1, x2, y2)."""
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def bbox_iou(a: tuple, b: tuple) -> float:
    """Intersection-over-union for two bboxes (x1, y1, x2, y2)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def bbox_area_normalized(bbox: tuple, frame_w: int, frame_h: int) -> float:
    """Bbox area as fraction of frame area. Used for cat size estimation."""
    return ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / (frame_w * frame_h)
```

---

## 16. Full Code: WorldStore

```python
# modules/world_model/store.py
"""
Async storage layer for the World Model. Uses the existing aiosqlite-based
DatabaseManager — does not open its own connection.

Why no separate sync sqlite3: the repo's storage discipline is async,
the orchestrator's lifecycle is async, and the existing DB connection has
its own transaction handling. Mixing sync and async DB access is a known
source of deadlocks under load.
"""

import json
import uuid
from datetime import datetime
from typing import Optional, Any

import numpy as np

from .types import WorldEntity, EntityEvent, EntityState, EventType


class WorldStore:
    """
    Thin async persistence layer over the existing DatabaseManager.
    Every method is async-safe; concurrent calls are fine — the DatabaseManager
    serializes writes per its existing discipline.
    """

    def __init__(self, db_manager):
        # db_manager is the existing modules/memory/database.py:DatabaseManager
        # which exposes async execute/fetchall/fetchone methods.
        self.db = db_manager

    async def ensure_schema(self) -> None:
        """
        Idempotent CREATE TABLE IF NOT EXISTS for world_* tables.
        Called once at startup. Schema mirrors Section 8.
        """
        statements = [
            """CREATE TABLE IF NOT EXISTS world_entities (
                id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                person_id INTEGER REFERENCES persons(id),
                display_name TEXT,
                state TEXT NOT NULL,
                last_seen_ts TEXT,
                last_seen_room TEXT,
                last_seen_camera TEXT,
                last_seen_bbox TEXT,
                last_seen_landmark TEXT,
                last_state_change_ts TEXT,
                confidence REAL,
                last_attribution_confidence REAL,
                is_resident INTEGER DEFAULT 0,
                metadata TEXT
            )""",
            "CREATE INDEX IF NOT EXISTS idx_world_entities_person ON world_entities(person_id)",
            "CREATE INDEX IF NOT EXISTS idx_world_entities_state ON world_entities(state)",
            "CREATE INDEX IF NOT EXISTS idx_world_entities_room ON world_entities(last_seen_room)",
            """CREATE TABLE IF NOT EXISTS world_entity_events (
                id TEXT PRIMARY KEY,
                ts TEXT NOT NULL,
                entity_id TEXT NOT NULL REFERENCES world_entities(id),
                person_id INTEGER REFERENCES persons(id),
                entity_name TEXT,
                entity_type TEXT NOT NULL,
                event_type TEXT NOT NULL,
                room TEXT,
                camera TEXT,
                bbox TEXT,
                landmark TEXT,
                state TEXT,
                confidence REAL,
                snapshot_path TEXT,
                related_entity_id TEXT,
                metadata TEXT
            )""",
            "CREATE INDEX IF NOT EXISTS idx_world_events_entity_ts ON world_entity_events(entity_id, ts DESC)",
            "CREATE INDEX IF NOT EXISTS idx_world_events_room_ts ON world_entity_events(room, ts DESC)",
            "CREATE INDEX IF NOT EXISTS idx_world_events_type_ts ON world_entity_events(event_type, ts DESC)",
            "CREATE INDEX IF NOT EXISTS idx_world_events_person_ts ON world_entity_events(person_id, ts DESC)",
            """CREATE TABLE IF NOT EXISTS world_entity_embeddings (
                entity_id TEXT PRIMARY KEY REFERENCES world_entities(id),
                embedding BLOB NOT NULL,
                dimension INTEGER NOT NULL,
                updated_ts TEXT NOT NULL
            )""",
        ]
        for stmt in statements:
            await self.db.execute(stmt)
        await self.db.commit()

    # ------------------------------------------------------------------ entities

    async def upsert_entity(self, ent: WorldEntity) -> None:
        await self.db.execute(
            """
            INSERT INTO world_entities (
                id, entity_type, person_id, display_name, state,
                last_seen_ts, last_seen_room, last_seen_camera, last_seen_bbox,
                last_seen_landmark, last_state_change_ts, confidence,
                last_attribution_confidence, is_resident, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                entity_type=excluded.entity_type,
                person_id=excluded.person_id,
                display_name=excluded.display_name,
                state=excluded.state,
                last_seen_ts=excluded.last_seen_ts,
                last_seen_room=excluded.last_seen_room,
                last_seen_camera=excluded.last_seen_camera,
                last_seen_bbox=excluded.last_seen_bbox,
                last_seen_landmark=excluded.last_seen_landmark,
                last_state_change_ts=excluded.last_state_change_ts,
                confidence=excluded.confidence,
                last_attribution_confidence=excluded.last_attribution_confidence,
                is_resident=excluded.is_resident,
                metadata=excluded.metadata
            """,
            (
                ent.id, ent.entity_type, ent.person_id, ent.display_name,
                ent.state.value,
                _iso(ent.last_seen_ts), ent.last_seen_room, ent.last_seen_camera,
                json.dumps(ent.last_seen_bbox) if ent.last_seen_bbox else None,
                ent.last_seen_landmark,
                _iso(ent.last_state_change_ts), ent.confidence,
                ent.last_attribution_confidence,
                int(ent.is_resident), json.dumps(_clean_metadata(ent.metadata)),
            ),
        )

    async def upsert_embedding(self, entity_id: str, embedding: np.ndarray) -> None:
        await self.db.execute(
            """
            INSERT INTO world_entity_embeddings (entity_id, embedding, dimension, updated_ts)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(entity_id) DO UPDATE SET
                embedding=excluded.embedding,
                dimension=excluded.dimension,
                updated_ts=excluded.updated_ts
            """,
            (
                entity_id,
                embedding.astype(np.float32).tobytes(),
                int(embedding.shape[0]),
                datetime.utcnow().isoformat(),
            ),
        )

    async def load_entities(self) -> list[WorldEntity]:
        rows = await self.db.fetchall("SELECT * FROM world_entities")
        emb_rows = await self.db.fetchall(
            "SELECT entity_id, embedding FROM world_entity_embeddings"
        )
        emb_map = {r["entity_id"]: np.frombuffer(r["embedding"], dtype=np.float32)
                   for r in emb_rows}

        entities = []
        for row in rows:
            ent = WorldEntity(
                id=row["id"],
                entity_type=row["entity_type"],
                person_id=row["person_id"],
                display_name=row["display_name"],
                state=EntityState(row["state"]),
                last_seen_ts=_parse_iso(row["last_seen_ts"]),
                last_seen_room=row["last_seen_room"],
                last_seen_camera=row["last_seen_camera"],
                last_seen_bbox=tuple(json.loads(row["last_seen_bbox"]))
                    if row["last_seen_bbox"] else None,
                last_seen_landmark=row["last_seen_landmark"],
                last_state_change_ts=_parse_iso(row["last_state_change_ts"])
                    or datetime.utcnow(),
                confidence=row["confidence"] or 0.0,
                last_attribution_confidence=row["last_attribution_confidence"] or 0.0,
                is_resident=bool(row["is_resident"]),
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            )
            # Stash visual embedding (cats/objects only) on the entity for fast access
            if ent.id in emb_map:
                ent.metadata["_visual_embedding"] = emb_map[ent.id]
            entities.append(ent)
        return entities

    # ------------------------------------------------------------------ events

    async def append_event(self, payload: dict) -> None:
        await self.db.execute(
            """
            INSERT INTO world_entity_events (
                id, ts, entity_id, person_id, entity_name, entity_type,
                event_type, room, camera, bbox, landmark, state, confidence,
                snapshot_path, related_entity_id, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["id"], payload["ts"], payload["entity_id"],
                payload.get("person_id"),
                payload.get("entity_name"), payload.get("entity_type"),
                payload["event_type"], payload.get("room"), payload.get("camera"),
                json.dumps(payload.get("bbox")) if payload.get("bbox") else None,
                payload.get("landmark"),
                payload.get("state"), payload.get("confidence", 0.0),
                payload.get("snapshot_path"), payload.get("related_entity_id"),
                json.dumps(payload.get("metadata", {})),
            ),
        )

    async def search_events(
        self,
        entity_id: Optional[str] = None,
        person_id: Optional[int] = None,
        room: Optional[str] = None,
        event_types: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 50,
    ) -> list[dict]:
        q = "SELECT * FROM world_entity_events WHERE 1=1"
        params: list[Any] = []
        if entity_id:
            q += " AND entity_id = ?"
            params.append(entity_id)
        if person_id is not None:
            q += " AND person_id = ?"
            params.append(person_id)
        if room:
            q += " AND room = ?"
            params.append(room)
        if event_types:
            placeholders = ",".join("?" for _ in event_types)
            q += f" AND event_type IN ({placeholders})"
            params.extend(event_types)
        if since:
            q += " AND ts >= ?"
            params.append(since.isoformat())
        if until:
            q += " AND ts <= ?"
            params.append(until.isoformat())
        q += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        rows = await self.db.fetchall(q, params)
        return [dict(r) for r in rows]


def _iso(ts: Optional[datetime]) -> Optional[str]:
    return ts.isoformat() if ts else None


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    return datetime.fromisoformat(s) if s else None


def _clean_metadata(metadata: dict) -> dict:
    """Strip non-JSON-serializable values (numpy arrays, etc.) before persisting."""
    return {k: v for k, v in metadata.items()
            if not (k.startswith("_") or isinstance(v, np.ndarray))}
```

The key lines: this WorldStore takes the existing `DatabaseManager` in its constructor. It does not import `sqlite3` directly. It does not maintain its own connection. It does not have its own lock. The DB serialization is whatever the existing DatabaseManager already does.

---

## 17. Full Code: WorldModel

```python
# modules/world_model/world_model.py
"""
The WorldModel orchestrator. Receives Observations, holds the entity registry,
runs the association algorithm and state machine, emits change events.

Single-writer discipline: all mutations of self.entities happen under self._lock.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from loguru import logger
from scipy.optimize import linear_sum_assignment

from .types import WorldEntity, EntityState, EventType, Observation
from .store import WorldStore
from .geometry import point_in_polygon, bbox_center, bbox_iou


class WorldModel:
    """
    Stateful tracker. Subscribes to vision.observation and camera.health.
    Publishes world.entity_event and world.state_snapshot.
    """

    def __init__(
        self,
        bus,                       # core.event_bus.EventBus
        store: WorldStore,
        rooms_config: list[dict],  # the typed rooms list from config.yaml
        identity_manager,          # modules.identity.identity_manager.IdentityManager
        config: dict,              # config['world_model'] block
    ):
        self.bus = bus
        self.store = store
        self.identity_manager = identity_manager
        self.cfg = config

        # Build per-camera topology lookup from rooms config
        self.cameras = self._build_camera_topology(rooms_config)

        self.entities: dict[str, WorldEntity] = {}
        self._lock = asyncio.Lock()
        self._unhealthy_cameras: set[str] = set()

    @staticmethod
    def _build_camera_topology(rooms_config: list[dict]) -> dict:
        """
        Build a per-camera topology dict from the rooms[] config.
        Camera ID is derived as room ID (one camera per room in the current config).
        """
        topology = {}
        for room in rooms_config:
            wm = room.get("world_model")
            if not wm or not wm.get("enabled", True):
                continue
            cam_id = room["id"]                # one camera per room currently
            topology[cam_id] = {
                "room": room["id"],
                "frame_width": wm["frame_width"],
                "frame_height": wm["frame_height"],
                "exits": wm.get("exits", []),
                "landmarks": wm.get("landmarks", []),
            }
        return topology

    async def start(self) -> None:
        """Subscribe to bus topics and load persistent state."""
        await self.store.ensure_schema()
        await self._load_from_store()
        await self.bus.subscribe("vision.observation", self._on_observation_batch)
        await self.bus.subscribe("camera.health", self._on_camera_health)
        asyncio.create_task(self._timer_loop())
        asyncio.create_task(self._snapshot_loop())
        logger.info(
            f"[WorldModel] started — {len(self.entities)} entities loaded, "
            f"{len(self.cameras)} cameras configured"
        )

    async def _load_from_store(self) -> None:
        for ent in await self.store.load_entities():
            self.entities[ent.id] = ent
        # On boot, every PRESENT entity becomes UNKNOWN_AT_BOOT for the first 30s
        boot_ts = datetime.utcnow()
        for ent in self.entities.values():
            if ent.state == EntityState.PRESENT:
                ent.state = EntityState.UNKNOWN_AT_BOOT
                ent.last_state_change_ts = boot_ts
                await self.store.upsert_entity(ent)

    # =========================================================================
    # MAIN ENTRY POINTS
    # =========================================================================

    async def _on_observation_batch(self, payload: dict):
        """payload: {camera, room, ts, observations: [Observation, ...]}"""
        async with self._lock:
            camera = payload["camera"]
            ts = payload["ts"] if isinstance(payload["ts"], datetime) \
                 else datetime.fromisoformat(payload["ts"])
            observations: list[Observation] = payload["observations"]

            # Skip if camera is unhealthy — entities should already be suspended
            if camera in self._unhealthy_cameras:
                return

            candidates = self._candidate_entities_for_camera(camera)
            matched, unmatched_obs, unmatched_ents = self._associate(
                observations, candidates
            )

            for obs, ent, attribution_conf in matched:
                await self._update_matched(ent, obs, ts, attribution_conf)

            for obs in unmatched_obs:
                await self._handle_unmatched_observation(obs, ts)

            for ent in unmatched_ents:
                await self._handle_unmatched_entity(ent, camera, ts)

    async def _on_camera_health(self, payload: dict):
        camera_id = payload["camera_id"]
        status = payload["status"]
        async with self._lock:
            if status in ("degraded", "down"):
                self._unhealthy_cameras.add(camera_id)
                for ent in self.entities.values():
                    if (ent.last_seen_camera == camera_id
                            and ent.state == EntityState.PRESENT):
                        ent.metadata["suspended_due_to_camera_health"] = True
                        await self._emit(EventType.CAMERA_DEGRADED, ent, obs=None,
                                         metadata={"camera": camera_id, "status": status})
                        await self.store.upsert_entity(ent)
            elif status == "healthy":
                self._unhealthy_cameras.discard(camera_id)
                for ent in self.entities.values():
                    if ent.metadata.pop("suspended_due_to_camera_health", False):
                        await self._emit(EventType.CAMERA_RESTORED, ent, obs=None,
                                         metadata={"camera": camera_id})
                        await self.store.upsert_entity(ent)

    # =========================================================================
    # ASSOCIATION
    # =========================================================================

    def _associate(
        self, observations: list[Observation], candidates: list[WorldEntity]
    ) -> tuple[list, list, list]:
        if not observations or not candidates:
            return [], list(observations), list(candidates)

        n_obs, n_ent = len(observations), len(candidates)
        cost = np.full((n_obs, n_ent), self.cfg["cost_reject"] * 2)
        for i, obs in enumerate(observations):
            for j, ent in enumerate(candidates):
                cost[i, j] = self._pair_cost(obs, ent)

        row_idx, col_idx = linear_sum_assignment(cost)

        matched = []
        matched_obs_idx, matched_ent_idx = set(), set()
        for i, j in zip(row_idx, col_idx):
            if cost[i, j] < self.cfg["cost_reject"]:
                # Attribution confidence: how much better is this match than 2nd best?
                row_costs = np.sort(cost[i])
                margin = row_costs[1] - row_costs[0] if len(row_costs) > 1 else 1.0
                attribution_conf = float(np.clip(margin / 0.5, 0.0, 1.0))
                matched.append((observations[i], candidates[j], attribution_conf))
                matched_obs_idx.add(i)
                matched_ent_idx.add(j)

        unmatched_obs = [o for i, o in enumerate(observations) if i not in matched_obs_idx]
        unmatched_ents = [e for j, e in enumerate(candidates) if j not in matched_ent_idx]
        return matched, unmatched_obs, unmatched_ents

    def _pair_cost(self, obs: Observation, ent: WorldEntity) -> float:
        if ent.entity_type != obs.obj_class:
            return self.cfg["cost_reject"] * 2
        if ent.entity_type == "person":
            return self._person_pair_cost(obs, ent)
        if ent.entity_type == "cat":
            return self._cat_pair_cost(obs, ent)              # Section 22
        if ent.entity_type == "object":
            return self._object_pair_cost(obs, ent)            # Section 23
        return self.cfg["cost_reject"] * 2

    def _person_pair_cost(self, obs: Observation, ent: WorldEntity) -> float:
        # Identity wins if both sides have it
        if obs.person_id is not None and ent.person_id is not None:
            if obs.person_id == ent.person_id:
                return 0.05 * self._spatial_distance(obs, ent)
            else:
                return self.cfg["cost_reject"] * 2  # different people, hard reject

        # Fallback to spatial-temporal continuity
        if not ent.last_seen_ts:
            return self.cfg["cost_reject"] * 2
        seconds_gone = (obs.ts - ent.last_seen_ts).total_seconds()
        if seconds_gone > 60.0:
            return self.cfg["cost_reject"] * 2

        if obs.camera == ent.last_seen_camera:
            cam_cost = 0.0
        elif self._cameras_are_neighbors(obs.camera, ent.last_seen_camera):
            cam_cost = 0.3
        else:
            cam_cost = 0.7

        spatial = self._spatial_distance(obs, ent)
        time_cost = min(seconds_gone / 60.0, 1.0)
        return 0.5 * cam_cost + 0.3 * spatial + 0.2 * time_cost

    def _spatial_distance(self, obs: Observation, ent: WorldEntity) -> float:
        """Normalized 0–1 spatial distance between obs bbox and ent's last bbox."""
        if not ent.last_seen_bbox:
            return 0.5
        cam_cfg = self.cameras.get(obs.camera, {})
        fw = cam_cfg.get("frame_width") or obs.metadata.get("frame_width", 640)
        fh = cam_cfg.get("frame_height") or obs.metadata.get("frame_height", 480)
        cx_o, cy_o = bbox_center(obs.bbox)
        cx_e, cy_e = bbox_center(ent.last_seen_bbox)
        dx = abs(cx_o - cx_e) / fw
        dy = abs(cy_o - cy_e) / fh
        return min((dx + dy) / 2.0, 1.0)

    # =========================================================================
    # MATCHED ENTITY UPDATE
    # =========================================================================

    async def _update_matched(self, ent: WorldEntity, obs: Observation,
                              ts: datetime, attribution_conf: float):
        was_unseen = ent.state in (
            EntityState.IN_ROOM_UNSEEN, EntityState.TRANSITIONING,
            EntityState.IN_HOUSE_UNMONITORED, EntityState.DEPARTED,
            EntityState.UNKNOWN_AT_BOOT,
        )
        prior_state = ent.state
        prior_room = ent.last_seen_room
        room_changed = prior_room is not None and prior_room != obs.room

        # Identity resolution: anonymous entity got recognized
        if obs.person_id is not None and ent.person_id is None:
            ent.person_id = obs.person_id
            ent.display_name = obs.person_name
            ent.is_resident = True
            await self._emit(EventType.NAME_LINKED, ent, obs,
                             metadata={"person_id": obs.person_id,
                                       "person_name": obs.person_name})

        # Compute granular events before mutating spatial state
        movement_event = self._classify_movement(ent, obs)
        posture_event = self._classify_posture(ent, obs)
        interaction_event = self._classify_interaction(ent, obs)

        # Update last-seen fields
        new_landmark = self._nearest_landmark(obs.camera, bbox_center(obs.bbox))
        ent.last_seen_ts = obs.ts
        ent.last_seen_camera = obs.camera
        ent.last_seen_room = obs.room
        ent.last_seen_bbox = obs.bbox
        ent.last_seen_landmark = new_landmark
        ent.confidence = obs.confidence
        ent.last_attribution_confidence = attribution_conf

        # Track posture history
        if "posture" in obs.metadata:
            hist = ent.metadata.setdefault("posture_history", [])
            hist.append((obs.ts.isoformat(), obs.metadata["posture"]))
            ent.metadata["posture_history"] = hist[-10:]

        # State transition: any unseen state → PRESENT
        if was_unseen:
            ent.state = EntityState.PRESENT
            ent.last_state_change_ts = ts
            ent.metadata.pop("transitioning_target", None)
            ent.metadata.pop("transitioning_kind", None)
            await self._emit(EventType.REAPPEARED, ent, obs,
                             metadata={"from_state": prior_state.value})

        # Room change
        if room_changed and ent.state == EntityState.PRESENT and not was_unseen:
            await self._emit(EventType.MOVED_TO, ent, obs,
                             metadata={"from_room": prior_room})

        # Granular events
        if movement_event:
            await self._emit(EventType.MOVED_WITHIN_ROOM, ent, obs, metadata=movement_event)
        if posture_event:
            await self._emit(EventType.POSTURE_CHANGED, ent, obs, metadata=posture_event)
        if interaction_event:
            await self._emit(EventType.INTERACTED_WITH, ent, obs, metadata=interaction_event)

        await self.store.upsert_entity(ent)

        # Trigger auto-enrollment for confident person matches
        if (ent.entity_type == "person"
                and obs.person_id is not None
                and obs.person_match_confidence >= self.cfg.get("enrollment_min_conf", 0.85)
                and "crop_path" in obs.metadata
                and obs.metadata.get("face_embedding") is not None):
            asyncio.create_task(self._enroll_async(obs))

    async def _enroll_async(self, obs: Observation):
        """Hand off to IdentityManager — fire and forget."""
        try:
            await self.identity_manager.consider_new_sample_async(
                person_id=obs.person_id,
                new_embedding=obs.metadata["face_embedding"],
                crop_path=obs.metadata["crop_path"],
                quality_metadata={
                    "yaw": obs.metadata.get("yaw"),
                    "pitch": obs.metadata.get("pitch"),
                    "blur_score": obs.metadata.get("blur_score"),
                },
            )
        except Exception as e:
            logger.debug(f"[WorldModel] enrollment hand-off failed: {e}")

    # ------------------------------------------------------------------ delta classifiers

    def _classify_movement(self, ent: WorldEntity, obs: Observation) -> Optional[dict]:
        if ent.last_seen_bbox is None or ent.last_seen_room != obs.room:
            return None
        cam_cfg = self.cameras.get(obs.camera, {})
        fw = cam_cfg.get("frame_width", 640)
        fh = cam_cfg.get("frame_height", 480)
        old = bbox_center(ent.last_seen_bbox)
        new = bbox_center(obs.bbox)
        dx = abs(new[0] - old[0]) / fw
        dy = abs(new[1] - old[1]) / fh
        thresh = self.cfg.get("movement_jitter_threshold", 0.08)
        if dx < thresh and dy < thresh:
            return None
        return {
            "from_bbox": ent.last_seen_bbox,
            "to_bbox": obs.bbox,
            "delta_normalized": [dx, dy],
            "approaching_landmark": self._nearest_landmark(obs.camera, new),
        }

    def _classify_posture(self, ent: WorldEntity, obs: Observation) -> Optional[dict]:
        new_posture = obs.metadata.get("posture")
        if not new_posture:
            return None
        history = ent.metadata.get("posture_history", [])
        stable = ent.metadata.get("stable_posture", "unknown")
        recent = [p for _, p in history[-2:]] + [new_posture]
        n = self.cfg.get("posture_debounce_frames", 3)
        if len(recent) < n or len(set(recent[-n:])) > 1:
            return None
        if new_posture == stable:
            return None
        ent.metadata["stable_posture"] = new_posture
        return {"from": stable, "to": new_posture}

    def _classify_interaction(self, ent: WorldEntity, obs: Observation) -> Optional[dict]:
        hand_bboxes = obs.metadata.get("hand_bboxes", [])
        if not hand_bboxes:
            return None
        for obj_ent in self.entities.values():
            if obj_ent.entity_type != "object" or obj_ent.last_seen_room != obs.room:
                continue
            if obj_ent.last_seen_bbox is None:
                continue
            for hand_bbox in hand_bboxes:
                if bbox_iou(hand_bbox, obj_ent.last_seen_bbox) > 0.1:
                    cnt = obj_ent.metadata.get("hand_overlap_frames", 0) + 1
                    obj_ent.metadata["hand_overlap_frames"] = cnt
                    if cnt >= self.cfg.get("interaction_debounce_frames", 3):
                        obj_ent.metadata["hand_overlap_frames"] = 0
                        return {
                            "object_id": obj_ent.id,
                            "object_name": obj_ent.display_name
                                or obj_ent.metadata.get("detected_class", "object"),
                            "hand_bbox": list(hand_bbox),
                        }
        return None

    # =========================================================================
    # NEW OR RETURNING DETECTION
    # =========================================================================

    async def _handle_unmatched_observation(self, obs: Observation, ts: datetime):
        # If person_id is known, look up the existing entity directly
        # (identity wins over spatial continuity for people)
        if obs.obj_class == "person" and obs.person_id is not None:
            existing = self._find_entity_by_person_id(obs.person_id)
            if existing:
                attribution_conf = obs.person_match_confidence
                # Flag the override for the dashboard
                if existing.last_seen_camera and existing.last_seen_camera != obs.camera:
                    existing.metadata["identity_overrode_continuity"] = True
                await self._update_matched(existing, obs, ts, attribution_conf)
                return

        # Try the wider pool for cats/objects: same-type entity with strong embedding match
        if obs.obj_class in ("cat", "object") and obs.visual_embedding is not None:
            best, best_sim = None, 0.0
            for ent in self.entities.values():
                if ent.entity_type != obs.obj_class:
                    continue
                emb = ent.metadata.get("_visual_embedding")
                if emb is None:
                    continue
                sim = float(np.dot(obs.visual_embedding, emb)
                            / (np.linalg.norm(obs.visual_embedding)
                               * np.linalg.norm(emb) + 1e-9))
                if sim > best_sim:
                    best, best_sim = ent, sim
            if best and best_sim >= self.cfg.get("cosine_match_strong", 0.6):
                await self._update_matched(best, obs, ts, attribution_conf=best_sim)
                return

        # Genuinely new entity
        new_ent = WorldEntity(
            id=str(uuid.uuid4()),
            entity_type=obs.obj_class,
            person_id=obs.person_id,
            display_name=obs.person_name,
            state=EntityState.PRESENT,
            last_seen_ts=ts,
            last_seen_camera=obs.camera,
            last_seen_room=obs.room,
            last_seen_bbox=obs.bbox,
            last_seen_landmark=self._nearest_landmark(obs.camera, bbox_center(obs.bbox)),
            last_state_change_ts=ts,
            confidence=obs.confidence,
            is_resident=(obs.person_id is not None),  # named persons are residents
        )
        self.entities[new_ent.id] = new_ent
        if obs.visual_embedding is not None:
            new_ent.metadata["_visual_embedding"] = obs.visual_embedding
            await self.store.upsert_embedding(new_ent.id, obs.visual_embedding)
        await self.store.upsert_entity(new_ent)
        await self._emit(EventType.FIRST_SEEN, new_ent, obs)

    def _find_entity_by_person_id(self, person_id: int) -> Optional[WorldEntity]:
        for ent in self.entities.values():
            if ent.entity_type == "person" and ent.person_id == person_id:
                return ent
        return None

    # =========================================================================
    # ENTITY EXPECTED, NOT SEEN — the bounded house disappearance logic
    # =========================================================================

    async def _handle_unmatched_entity(self, ent: WorldEntity, camera: str, ts: datetime):
        # Only act on entities that were PRESENT in this camera
        if ent.last_seen_camera != camera or ent.state != EntityState.PRESENT:
            return

        # Skip if camera is suspended
        if (camera in self._unhealthy_cameras
                or ent.metadata.get("suspended_due_to_camera_health")):
            return

        # Classify the disappearance based on last bbox vs. exit polygons
        exit_match = self._classify_exit(ent.last_seen_bbox, camera)

        if exit_match is None:
            # In-frame disappearance — went under desk, behind couch, etc.
            ent.state = EntityState.IN_ROOM_UNSEEN
            ent.last_state_change_ts = ts
            await self._emit(
                EventType.LOST_VISIBILITY, ent, obs=None,
                metadata={
                    "reason": "in_frame_disappearance",
                    "near_exit": False,
                    "last_landmark": ent.last_seen_landmark,
                },
            )
        elif exit_match["kind"] == "to_room":
            # Possible handoff to neighboring camera
            ent.state = EntityState.TRANSITIONING
            ent.last_state_change_ts = ts
            ent.metadata["transitioning_target"] = exit_match["to"]
            ent.metadata["transitioning_kind"] = "to_room"
            await self._emit(
                EventType.LOST_VISIBILITY, ent, obs=None,
                metadata={
                    "reason": "near_exit",
                    "near_exit": True,
                    "exit_kind": "to_room",
                    "exit_to": exit_match["to"],
                    "last_landmark": ent.last_seen_landmark,
                },
            )
        elif exit_match["kind"] == "to_unmonitored_zone":
            # Quick TRANSITIONING then settle into IN_HOUSE_UNMONITORED via timer
            ent.state = EntityState.TRANSITIONING
            ent.last_state_change_ts = ts
            ent.metadata["transitioning_target"] = exit_match["to"]
            ent.metadata["transitioning_kind"] = "to_unmonitored_zone"
            await self._emit(
                EventType.LOST_VISIBILITY, ent, obs=None,
                metadata={
                    "reason": "near_exit",
                    "near_exit": True,
                    "exit_kind": "to_unmonitored_zone",
                    "exit_to": exit_match["to"],
                    "last_landmark": ent.last_seen_landmark,
                },
            )
        elif exit_match["kind"] == "exterior_exit":
            # Crossing an external door — treat as DEPARTED
            ent.state = EntityState.DEPARTED
            ent.last_state_change_ts = ts
            ent.metadata["departed_via"] = exit_match["name"]
            ent.metadata["departed_ts"] = ts.isoformat()
            await self._emit(
                EventType.DEPARTED, ent, obs=None,
                metadata={
                    "via_exit": exit_match["name"],
                    "last_landmark": ent.last_seen_landmark,
                },
            )

        await self.store.upsert_entity(ent)

    def _classify_exit(self, bbox: Optional[tuple], camera: str) -> Optional[dict]:
        """Return the matching exit dict if bbox center is inside any exit polygon."""
        if not bbox or camera not in self.cameras:
            return None
        cx, cy = bbox_center(bbox)
        for exit_def in self.cameras[camera].get("exits", []):
            if point_in_polygon(cx, cy, exit_def["polygon"]):
                return exit_def
        return None

    # =========================================================================
    # PERIODIC TIMERS
    # =========================================================================

    async def _timer_loop(self):
        while True:
            try:
                await asyncio.sleep(2.0)
                now = datetime.utcnow()
                async with self._lock:
                    for ent in list(self.entities.values()):
                        elapsed = now - ent.last_state_change_ts

                        # UNKNOWN_AT_BOOT → resolve after 30s based on last log state
                        if (ent.state == EntityState.UNKNOWN_AT_BOOT
                                and elapsed > timedelta(seconds=30)):
                            if ent.is_resident:
                                ent.state = EntityState.IN_HOUSE_UNMONITORED
                                ent.metadata["entered_unmonitored_via"] = "boot"
                            ent.last_state_change_ts = now
                            await self.store.upsert_entity(ent)

                        # TRANSITIONING → resolve based on target kind
                        if ent.state == EntityState.TRANSITIONING:
                            if elapsed > timedelta(seconds=self.cfg.get("T_handoff_seconds", 8)):
                                target_kind = ent.metadata.get("transitioning_kind")
                                target = ent.metadata.get("transitioning_target")
                                if target_kind == "to_unmonitored_zone":
                                    ent.state = EntityState.IN_HOUSE_UNMONITORED
                                    ent.metadata["entered_unmonitored_via"] = target
                                    ent.last_state_change_ts = now
                                    await self._emit(
                                        EventType.ENTERED_UNMONITORED, ent, obs=None,
                                        metadata={"entered_via": target},
                                    )
                                elif target_kind == "to_room":
                                    # Handoff failed — neighbor camera didn't see them.
                                    ent.state = EntityState.IN_ROOM_UNSEEN
                                    ent.last_state_change_ts = now
                                    await self._emit(
                                        EventType.LOST_VISIBILITY, ent, obs=None,
                                        metadata={
                                            "reason": "handoff_failed",
                                            "expected_room": target,
                                        },
                                    )
                                ent.metadata.pop("transitioning_target", None)
                                ent.metadata.pop("transitioning_kind", None)
                                await self.store.upsert_entity(ent)

                        # STATIONARY_LONG detection
                        if ent.state == EntityState.PRESENT:
                            mins = self.cfg.get("stationary_long_minutes", 5)
                            if elapsed > timedelta(minutes=mins):
                                already_fired = ent.metadata.get("stationary_fired_at")
                                marker = ent.last_state_change_ts.isoformat()
                                if already_fired != marker:
                                    ent.metadata["stationary_fired_at"] = marker
                                    await self.store.upsert_entity(ent)
                                    await self._emit(EventType.STATIONARY_LONG, ent, obs=None)
            except Exception:
                logger.exception("[WorldModel] timer loop iteration failed")

    async def _snapshot_loop(self):
        while True:
            try:
                await asyncio.sleep(30.0)
                async with self._lock:
                    snap = [
                        {
                            "id": e.id,
                            "type": e.entity_type,
                            "person_id": e.person_id,
                            "name": e.display_name,
                            "state": e.state.value,
                            "room": e.last_seen_room,
                            "landmark": e.last_seen_landmark,
                            "last_seen_ts": e.last_seen_ts.isoformat() if e.last_seen_ts else None,
                            "confidence": e.confidence,
                            "attribution_confidence": e.last_attribution_confidence,
                            "is_resident": e.is_resident,
                        }
                        for e in self.entities.values()
                    ]
                await self.bus.publish("world.state_snapshot", {"entities": snap})
            except Exception:
                logger.exception("[WorldModel] snapshot loop iteration failed")

    # =========================================================================
    # HELPERS
    # =========================================================================

    async def _emit(self, event_type: EventType, ent: WorldEntity,
                    obs: Optional[Observation], metadata: Optional[dict] = None):
        ts = obs.ts if obs else datetime.utcnow()
        payload = {
            "id": str(uuid.uuid4()),
            "ts": ts.isoformat(),
            "entity_id": ent.id,
            "entity_name": ent.display_name or f"unknown_{ent.entity_type}_{ent.id[:6]}",
            "entity_type": ent.entity_type,
            "person_id": ent.person_id,
            "event_type": event_type.value,
            "room": ent.last_seen_room,
            "camera": ent.last_seen_camera,
            "bbox": list(ent.last_seen_bbox) if ent.last_seen_bbox else None,
            "landmark": ent.last_seen_landmark,
            "state": ent.state.value,
            "confidence": ent.confidence,
            "snapshot_path": (obs.metadata.get("crop_path") if obs else None),
            "metadata": metadata or {},
        }
        await self.store.append_event(payload)
        await self.bus.publish("world.entity_event", payload)

    def _candidate_entities_for_camera(self, camera: str) -> list[WorldEntity]:
        cutoff = datetime.utcnow() - timedelta(minutes=2)
        return [
            e for e in self.entities.values()
            if e.last_seen_ts and e.last_seen_ts > cutoff and (
                e.last_seen_camera == camera
                or self._cameras_are_neighbors(e.last_seen_camera, camera)
            )
        ]

    def _cameras_are_neighbors(self, a: Optional[str], b: Optional[str]) -> bool:
        if not a or not b:
            return False
        if a == b:
            return True
        a_neighbors = {ex["to"] for ex in self.cameras.get(a, {}).get("exits", [])
                       if ex["kind"] == "to_room"}
        b_room = self.cameras.get(b, {}).get("room")
        return b_room in a_neighbors

    def _nearest_landmark(self, camera: str, point: tuple) -> Optional[str]:
        for lm in self.cameras.get(camera, {}).get("landmarks", []):
            if point_in_polygon(point[0], point[1], lm["polygon"]):
                return lm["name"]
        return None

    # =========================================================================
    # PUBLIC QUERY API
    # =========================================================================

    def find_entity_by_name(self, name: str) -> Optional[WorldEntity]:
        if not name:
            return None
        for e in self.entities.values():
            if e.display_name and e.display_name.lower() == name.lower():
                return e
        return None

    def find_entity_by_person_id(self, person_id: int) -> Optional[WorldEntity]:
        return self._find_entity_by_person_id(person_id)

    async def most_recent_event(self, entity_id: str) -> Optional[dict]:
        events = await self.store.search_events(entity_id=entity_id, limit=1)
        return events[0] if events else None

    # Cat and object cost functions — defined in pets.py / objects.py extensions
    # Stubs here so the dispatch in _pair_cost compiles:
    def _cat_pair_cost(self, obs, ent):
        # Filled in by Phase 4 — Section 22
        return self.cfg["cost_reject"] * 2

    def _object_pair_cost(self, obs, ent):
        # Filled in by Phase 4 — Section 23
        return self.cfg["cost_reject"] * 2
```

That's the full WorldModel for Phase 1–3. Phase 4 adds the cat and object cost functions (Section 22, 23). Phase 5 adds the InteractionMonitor as a separate subscriber (Section 24).

---

## 18. Full Code: ObservationBuilder

```python
# modules/vision/observation_builder.py
"""
Adapter from raw detector outputs to normalized Observation objects.
Subscribes to per-frame detector outputs, calls IdentityManager for face crops,
emits vision.observation batches consumed by WorldModel.
"""

import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from loguru import logger

from modules.world_model.types import Observation


class ObservationBuilder:
    """
    Per-room observation pipeline. Reads frames from CameraManager, runs detections,
    enriches person detections with IdentityManager identity, and emits normalized
    Observation batches on vision.observation.
    """

    TRACKED_OBJECT_CLASSES = {
        # Phase 4 starting set — cheap stuff, expand later
        "cell phone", "cup", "book", "laptop", "bottle", "remote",
    }

    def __init__(
        self,
        bus,
        camera_manager,
        object_detector,
        face_recognizer,            # the new ArcFace one
        identity_manager,
        posture_analyzer,
        rooms_config: list[dict],
        snapshot_dir: Path,
    ):
        self.bus = bus
        self.cm = camera_manager
        self.detector = object_detector
        self.face = face_recognizer
        self.identity = identity_manager
        self.posture = posture_analyzer
        self.rooms = {r["id"]: r for r in rooms_config}
        self.snapshot_dir = snapshot_dir
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    async def start(self):
        """Spawn one loop per camera-equipped room."""
        for room in self.rooms.values():
            wm = room.get("world_model")
            if wm and wm.get("enabled", True):
                asyncio.create_task(self._loop_for_room(room["id"]))
        logger.info(f"[ObservationBuilder] started on {len(self.rooms)} rooms")

    async def _loop_for_room(self, room_id: str):
        """One loop per room. Pulls frames at the room's effective FPS."""
        # Adapt to the actual CameraManager API in the repo. Patterns:
        #   - async-iterator: `async for frame, ts in self.cm.iter_frames(room_id):`
        #   - callback: `self.cm.on_frame(room_id, self._handle_frame)`
        #   - polling: `frame, ts = await self.cm.get_latest_frame(room_id)`
        # The exact wiring adjusts to whichever the repo uses.
        async for frame, ts in self.cm.iter_frames(room_id):
            try:
                observations = await self._build_for_frame(room_id, frame, ts)
                if observations:
                    await self.bus.publish("vision.observation", {
                        "camera": room_id,             # one cam per room currently
                        "room": room_id,
                        "ts": ts,
                        "observations": observations,
                    })
            except Exception as e:
                logger.exception(f"[ObservationBuilder] frame error in {room_id}: {e}")

    async def _build_for_frame(self, room: str, frame: np.ndarray, ts: datetime) -> list[Observation]:
        observations = []

        # 1. Object detection (YOLO)
        detections = await self.detector.detect_async(frame)

        # 2. Optional: hand bboxes (Phase 5; for now empty list)
        hand_bboxes: list[tuple] = []

        # 3. Optional: room-wide posture (Phase 5; for now None)
        posture: Optional[str] = None

        frame_h, frame_w = frame.shape[:2]

        for det in detections:
            cls = det.class_name
            if cls == "person":
                obs = await self._build_person_obs(
                    frame, det, room, ts, frame_w, frame_h, hand_bboxes, posture
                )
            elif cls == "cat":
                obs = self._build_cat_obs(frame, det, room, ts, frame_w, frame_h)
            elif cls in self.TRACKED_OBJECT_CLASSES:
                obs = self._build_object_obs(frame, det, room, ts, frame_w, frame_h)
            else:
                continue
            observations.append(obs)

        return observations

    async def _build_person_obs(self, frame, det, room, ts, fw, fh,
                                 hand_bboxes, posture) -> Observation:
        bbox = det.bbox
        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Save crop for enrollment / dashboard
        crop = frame[y1:y2, x1:x2]
        crop_path = (self.snapshot_dir
                     / f"person_{room}_{ts.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg")
        try:
            cv2.imwrite(str(crop_path), crop)
        except Exception:
            crop_path = None

        # Face detection on the person crop
        face_results = self.face.detect_and_embed(crop) if crop.size > 0 else []
        face = face_results[0] if face_results else None

        person_id, person_name, identity_conf = None, None, 0.0
        face_metadata = {}
        if face is not None:
            # IdentityManager handles centroid + margin
            id_match = await self.identity.identify_from_embedding_async(face["embedding"])
            person_id = id_match.person_id
            person_name = id_match.person_name
            identity_conf = id_match.confidence
            face_metadata = {
                "face_embedding": face["embedding"],
                "yaw": face["yaw"],
                "pitch": face["pitch"],
                "roll": face["roll"],
                "blur_score": _laplacian_var(crop),
            }

        return Observation(
            camera=room,
            room=room,
            obj_class="person",
            bbox=tuple(bbox),
            confidence=det.confidence,
            ts=ts,
            person_id=person_id,
            person_name=person_name,
            person_match_confidence=identity_conf,
            visual_embedding=None,
            metadata={
                "crop_path": str(crop_path) if crop_path else None,
                "frame_width": fw,
                "frame_height": fh,
                "hand_bboxes": hand_bboxes,
                "posture": posture,
                **face_metadata,
            },
        )

    def _build_cat_obs(self, frame, det, room, ts, fw, fh) -> Observation:
        # Phase 4 — full implementation in Section 22
        return Observation(
            camera=room, room=room, obj_class="cat",
            bbox=tuple(det.bbox), confidence=det.confidence, ts=ts,
            metadata={"frame_width": fw, "frame_height": fh},
        )

    def _build_object_obs(self, frame, det, room, ts, fw, fh) -> Observation:
        # Phase 4 — full implementation in Section 23
        return Observation(
            camera=room, room=room, obj_class="object",
            bbox=tuple(det.bbox), confidence=det.confidence, ts=ts,
            metadata={
                "detected_class": det.class_name,
                "frame_width": fw, "frame_height": fh,
            },
        )


def _laplacian_var(image: np.ndarray) -> float:
    """Blur metric: lower = blurrier."""
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
```

The `iter_frames` API on CameraManager is illustrative — adapt to whatever the actual repo's CameraManager exposes. Given the existing CameraManager already has per-room locks, RTSP drainer threads, and reconnect behavior, ObservationBuilder is consumer-only and can plug into whichever pattern the repo uses (async iterator, callback, or pull-style).

---

## 19. Synthetic Test Harnesses

**Debug the state machine on synthetic observations before fighting OpenCV.** Live RTSP at 5 FPS with reconnects will mask state machine bugs as flaky cameras.

Each test script feeds hand-crafted Observation sequences directly into a WorldModel instance and asserts the resulting events.

### `scripts/test_world_model_synthetic.py`

```python
"""
Phase 1 verification: feed synthetic observations, assert the state machine.
No cameras involved.
"""
import asyncio
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


# Stub minimal versions of the dependencies
class StubBus:
    def __init__(self):
        self.events = []
    async def publish(self, topic, payload):
        self.events.append((topic, payload))
    async def subscribe(self, topic, handler):
        pass


class StubStore:
    async def ensure_schema(self): pass
    async def load_entities(self): return []
    async def upsert_entity(self, ent): pass
    async def upsert_embedding(self, eid, emb): pass
    async def append_event(self, p): pass
    async def search_events(self, **kw): return []


class StubIdentityMatch:
    def __init__(self, person_id=None, person_name=None, confidence=0.0, status="unknown"):
        self.person_id = person_id
        self.person_name = person_name
        self.confidence = confidence
        self.status = status


class StubIdentityManager:
    async def identify_from_embedding_async(self, emb):
        return StubIdentityMatch()
    async def consider_new_sample_async(self, **kw):
        pass


from modules.world_model.world_model import WorldModel
from modules.world_model.types import Observation, EntityState


CONFIG = {
    "cost_reject": 1.5,
    "cosine_match_strong": 0.6,
    "T_handoff_seconds": 8,
    "movement_jitter_threshold": 0.08,
    "posture_debounce_frames": 3,
    "interaction_debounce_frames": 3,
    "stationary_long_minutes": 5,
    "enrollment_min_conf": 0.85,
}


def office_only_rooms_config():
    return [{
        "id": "office",
        "world_model": {
            "enabled": True,
            "frame_width": 640,
            "frame_height": 480,
            "exits": [
                {"kind": "to_room", "to": "living_room",
                 "polygon": [[600, 0], [640, 0], [640, 480], [600, 480]]},
            ],
            "landmarks": [
                {"name": "desk",
                 "polygon": [[200, 250], [450, 250], [450, 400], [200, 400]]},
                {"name": "under_desk",
                 "polygon": [[200, 380], [450, 380], [450, 480], [200, 480]]},
            ],
        },
    }]


async def test_under_desk_scenario():
    """Cole at desk → under_desk → disappears.
    Expected: PRESENT → IN_ROOM_UNSEEN(reason=in_frame_disappearance, last_landmark=under_desk).
    """
    bus = StubBus()
    wm = WorldModel(
        bus=bus, store=StubStore(),
        rooms_config=office_only_rooms_config(),
        identity_manager=StubIdentityManager(),
        config=CONFIG,
    )
    await wm.start()

    t0 = datetime.utcnow()

    # Tick 1: Cole at desk (face would have been recognized by IdentityManager)
    await wm._on_observation_batch({
        "camera": "office", "room": "office", "ts": t0,
        "observations": [Observation(
            camera="office", room="office", obj_class="person",
            bbox=(280, 300, 380, 400),     # over desk landmark
            confidence=0.95, ts=t0,
            person_id=42, person_name="Cole", person_match_confidence=0.91,
        )],
    })

    # Tick 2: Cole at under_desk
    t1 = t0 + timedelta(seconds=1)
    await wm._on_observation_batch({
        "camera": "office", "room": "office", "ts": t1,
        "observations": [Observation(
            camera="office", room="office", obj_class="person",
            bbox=(280, 400, 380, 470),     # over under_desk landmark
            confidence=0.93, ts=t1,
            person_id=42, person_name="Cole", person_match_confidence=0.89,
        )],
    })

    # Tick 3: no detection
    t2 = t0 + timedelta(seconds=2)
    await wm._on_observation_batch({
        "camera": "office", "room": "office", "ts": t2, "observations": [],
    })

    # Assertions
    cole = wm.find_entity_by_person_id(42)
    assert cole is not None, "Cole entity not created"
    assert cole.state == EntityState.IN_ROOM_UNSEEN, f"expected IN_ROOM_UNSEEN, got {cole.state}"
    assert cole.last_seen_landmark == "under_desk", f"expected under_desk, got {cole.last_seen_landmark}"
    assert cole.last_seen_room == "office"

    lost = [e for t, e in bus.events
            if t == "world.entity_event" and e.get("event_type") == "lost_visibility"]
    assert len(lost) == 1, f"expected 1 lost_visibility event, got {len(lost)}"
    assert lost[0]["metadata"]["reason"] == "in_frame_disappearance"
    assert lost[0]["metadata"]["near_exit"] is False

    print("PASS: under-desk scenario")


async def test_handoff_to_living_room():
    """Cole approaches doorway in office, then appears in living_room.
    Expected: TRANSITIONING → REAPPEARED + MOVED_TO event.
    """
    rooms = [
        office_only_rooms_config()[0],
        {
            "id": "living_room",
            "world_model": {
                "enabled": True,
                "frame_width": 1920, "frame_height": 1080,
                "exits": [{"kind": "to_room", "to": "office",
                           "polygon": [[0, 400], [200, 400], [200, 1080], [0, 1080]]}],
                "landmarks": [],
            },
        },
    ]
    bus = StubBus()
    wm = WorldModel(bus=bus, store=StubStore(), rooms_config=rooms,
                    identity_manager=StubIdentityManager(), config=CONFIG)
    await wm.start()

    t0 = datetime.utcnow()

    # Tick 1: Cole in office, away from door
    await wm._on_observation_batch({
        "camera": "office", "room": "office", "ts": t0,
        "observations": [Observation(
            camera="office", room="office", obj_class="person",
            bbox=(280, 300, 380, 400), confidence=0.95, ts=t0,
            person_id=42, person_name="Cole", person_match_confidence=0.91,
        )],
    })

    # Tick 2: Cole at the doorway in office (over to_room exit polygon)
    t1 = t0 + timedelta(seconds=1)
    await wm._on_observation_batch({
        "camera": "office", "room": "office", "ts": t1,
        "observations": [Observation(
            camera="office", room="office", obj_class="person",
            bbox=(610, 200, 635, 400), confidence=0.9, ts=t1,
            person_id=42, person_name="Cole", person_match_confidence=0.88,
        )],
    })

    # Tick 3: no detection in office
    t2 = t0 + timedelta(seconds=2)
    await wm._on_observation_batch({
        "camera": "office", "room": "office", "ts": t2, "observations": [],
    })

    cole = wm.find_entity_by_person_id(42)
    assert cole.state == EntityState.TRANSITIONING

    # Tick 4: Cole appears in living_room
    t3 = t0 + timedelta(seconds=3)
    await wm._on_observation_batch({
        "camera": "living_room", "room": "living_room", "ts": t3,
        "observations": [Observation(
            camera="living_room", room="living_room", obj_class="person",
            bbox=(150, 600, 250, 800), confidence=0.92, ts=t3,
            person_id=42, person_name="Cole", person_match_confidence=0.90,
        )],
    })

    cole = wm.find_entity_by_person_id(42)
    assert cole.state == EntityState.PRESENT
    assert cole.last_seen_room == "living_room"

    moved = [e for t, e in bus.events
             if t == "world.entity_event" and e.get("event_type") == "moved_to"]
    reappeared = [e for t, e in bus.events
                  if t == "world.entity_event" and e.get("event_type") == "reappeared"]
    assert len(reappeared) >= 1, "expected REAPPEARED event"
    assert len(moved) >= 1, "expected MOVED_TO event"
    assert reappeared[-1]["metadata"]["from_state"] == "transitioning"

    print("PASS: handoff to living_room")


async def test_camera_drop():
    """Cole PRESENT in office, camera goes degraded, then healthy.
    Expected: state remains PRESENT throughout.
    """
    bus = StubBus()
    wm = WorldModel(bus=bus, store=StubStore(),
                    rooms_config=office_only_rooms_config(),
                    identity_manager=StubIdentityManager(), config=CONFIG)
    await wm.start()

    t0 = datetime.utcnow()
    await wm._on_observation_batch({
        "camera": "office", "room": "office", "ts": t0,
        "observations": [Observation(
            camera="office", room="office", obj_class="person",
            bbox=(280, 300, 380, 400), confidence=0.95, ts=t0,
            person_id=42, person_name="Cole", person_match_confidence=0.91,
        )],
    })

    cole = wm.find_entity_by_person_id(42)
    assert cole.state == EntityState.PRESENT

    # Camera goes down
    await wm._on_camera_health({"camera_id": "office", "status": "down"})

    # Empty observation arrives — should NOT transition Cole to IN_ROOM_UNSEEN
    t1 = t0 + timedelta(seconds=2)
    await wm._on_observation_batch({
        "camera": "office", "room": "office", "ts": t1, "observations": [],
    })

    cole = wm.find_entity_by_person_id(42)
    assert cole.state == EntityState.PRESENT, \
           f"expected PRESENT during camera down, got {cole.state}"

    # Camera comes back
    await wm._on_camera_health({"camera_id": "office", "status": "healthy"})

    cole = wm.find_entity_by_person_id(42)
    assert cole.state == EntityState.PRESENT

    print("PASS: camera drop scenario")


async def test_unmonitored_zone():
    """Cole approaches to_unmonitored_zone polygon and disappears.
    Expected: TRANSITIONING → IN_HOUSE_UNMONITORED after T_handoff.
    """
    # Add a to_unmonitored_zone exit to the office config for this test
    rooms = office_only_rooms_config()
    rooms[0]["world_model"]["exits"].append({
        "kind": "to_unmonitored_zone", "to": "guest_bedroom",
        "polygon": [[0, 200], [40, 200], [40, 400], [0, 400]],
    })

    config = dict(CONFIG)
    config["T_handoff_seconds"] = 1   # speed up the test

    bus = StubBus()
    wm = WorldModel(bus=bus, store=StubStore(), rooms_config=rooms,
                    identity_manager=StubIdentityManager(), config=config)
    await wm.start()

    t0 = datetime.utcnow()
    # Cole near the unmonitored-zone door
    await wm._on_observation_batch({
        "camera": "office", "room": "office", "ts": t0,
        "observations": [Observation(
            camera="office", room="office", obj_class="person",
            bbox=(10, 250, 35, 380), confidence=0.9, ts=t0,
            person_id=42, person_name="Cole", person_match_confidence=0.88,
        )],
    })

    # Disappears
    t1 = t0 + timedelta(seconds=1)
    await wm._on_observation_batch({
        "camera": "office", "room": "office", "ts": t1, "observations": [],
    })

    cole = wm.find_entity_by_person_id(42)
    assert cole.state == EntityState.TRANSITIONING

    # Wait for T_handoff
    await asyncio.sleep(2.5)
    cole = wm.find_entity_by_person_id(42)
    assert cole.state == EntityState.IN_HOUSE_UNMONITORED, \
           f"expected IN_HOUSE_UNMONITORED, got {cole.state}"
    assert cole.metadata.get("entered_unmonitored_via") == "guest_bedroom"

    print("PASS: unmonitored zone scenario")


async def main():
    await test_under_desk_scenario()
    await test_handoff_to_living_room()
    await test_camera_drop()
    await test_unmonitored_zone()
    print("\nAll synthetic tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
```

These test scripts run in well under a second and cover the four critical state-machine paths. They must all pass before any live-camera integration.

---

## 20. Query Layer (Orchestrator Tools)

The orchestrator already has a tool registry (it recently gained `get_room_snapshot`). World Model query tools register there alongside existing tools.

```python
# modules/world_model/query_tools.py
"""
LLM-facing tool functions for World Model queries.
Registered into the orchestrator's tool registry; called by the brain LLM.
"""

from datetime import datetime, timedelta
from typing import Optional

from .world_model import WorldModel


class WorldQueryTools:
    def __init__(self, world: WorldModel):
        self.world = world

    async def get_entity_status(self, name: str) -> dict:
        """
        Where is X right now? Use for 'where is Cole', 'is Anna home', etc.
        """
        ent = self.world.find_entity_by_name(name)
        if not ent:
            return {"found": False, "message": f"No entity named {name} in registry."}
        elapsed = datetime.utcnow() - ent.last_state_change_ts
        return {
            "found": True,
            "name": ent.display_name,
            "type": ent.entity_type,
            "state": ent.state.value,
            "last_seen_room": ent.last_seen_room,
            "last_seen_camera": ent.last_seen_camera,
            "last_seen_landmark": ent.last_seen_landmark,
            "last_seen_ts": ent.last_seen_ts.isoformat() if ent.last_seen_ts else None,
            "duration_in_state_seconds": int(elapsed.total_seconds()),
            "confidence": ent.confidence,
            "attribution_confidence": ent.last_attribution_confidence,
            "is_resident": ent.is_resident,
            # State-specific extras the LLM can use for natural phrasing:
            "departed_via": ent.metadata.get("departed_via"),
            "departed_ts": ent.metadata.get("departed_ts"),
            "entered_unmonitored_via": ent.metadata.get("entered_unmonitored_via"),
            "last_event": await self.world.most_recent_event(ent.id),
        }

    async def list_entities_in_room(self, room: str) -> list[dict]:
        """Roster of who's in a room right now (PRESENT only)."""
        return [
            {"name": e.display_name or f"unknown_{e.entity_type}",
             "type": e.entity_type, "state": e.state.value,
             "confidence": e.confidence}
            for e in self.world.entities.values()
            if e.last_seen_room == room and e.state.value == "present"
        ]

    async def who_is_home(self) -> list[dict]:
        """List residents currently considered 'home' (any in-house state)."""
        in_house_states = {"present", "in_room_unseen", "transitioning",
                           "in_house_unmonitored"}
        return [
            {"name": e.display_name, "state": e.state.value,
             "last_room": e.last_seen_room}
            for e in self.world.entities.values()
            if e.is_resident and e.display_name
               and e.state.value in in_house_states
        ]

    async def search_recent_events(
        self,
        entity_name: Optional[str] = None,
        room: Optional[str] = None,
        event_types: Optional[list[str]] = None,
        hours_ago: int = 24,
        limit: int = 20,
    ) -> list[dict]:
        """Search the entity event log."""
        entity_id = None
        if entity_name:
            ent = self.world.find_entity_by_name(entity_name)
            if not ent:
                return []
            entity_id = ent.id
        since = datetime.utcnow() - timedelta(hours=hours_ago)
        return await self.world.store.search_events(
            entity_id=entity_id, room=room, event_types=event_types,
            since=since, limit=limit,
        )
```

Register these in the orchestrator's tool registry as a group. The brain LLM picks them naturally; the persona's existing presence-reasoning rules now have *structured ground truth* to call instead of inferring from raw frames.

---

## 21. Persona Alignment

This is the bonus that comes free from the World Model existing.

The repo's `default` persona prompt (verified from current `config.yaml`) already says:

> *"If a known person disappears from a camera but didn't leave through a known door, assume they're still in the room. A blank frame is not an exit. You can ask if you're curious, but don't assume they left."*

> *"Sleep/nap state is per-person. If anyone is asleep or napping in a room, stay quiet there even when responding to someone else who is awake — defer to the sleeper."*

That instruction was previously the LLM's *only* source of presence reasoning — it had to infer disappearance type from raw frame text. With the World Model, the same instruction is now *backed by structured truth*:

- `state == IN_ROOM_UNSEEN, near_exit=false` → the prompt's "still in the room" assertion is now a queryable fact.
- `state == DEPARTED, departed_via=front_door` → "left through a known door" is a recorded event with a timestamp.
- `state == IN_HOUSE_UNMONITORED, entered_unmonitored_via=guest_bedroom_door` → an explicit positive state, not a guess.

The persona prompt should be updated minimally to *use* the tools rather than infer:

```
Presence reasoning:
- For 'where is X' or 'is X home', call get_entity_status(name).
- The state field tells you what happened: 'present', 'in_room_unseen' (still
  in their last room, didn't leave through a door), 'transitioning' (briefly
  between rooms), 'in_house_unmonitored' (in a room with no camera),
  'departed' (left the house through an exterior door).
- Phrase the answer naturally based on state. Don't recite the state name.
- For 'in_room_unseen', mention the last_seen_landmark if available
  ("near the desk", "by the couch").
- For 'departed', mention departed_via and departed_ts.
- For 'in_house_unmonitored', mention entered_unmonitored_via.
- If attribution_confidence is below 0.5, soften the answer with hedge words.
```

### The privacy directive also benefits

The persona overlay's privacy directive (verified from `config.yaml`):

> *"Cole has private modes that are not to be referenced... except when ALL of: (a) Cole is alone in his current room (no other person present or audible), (b) Cole brought up the topic himself in this conversation, (c) the topic is directly relevant to what's being discussed."*

Condition (a) — *"Cole is alone in his current room"* — was previously a fragile inference. With the World Model it becomes a single tool call:

```python
others = await list_entities_in_room(cole_room)
cole_alone = (len([e for e in others if e["type"] == "person"]) == 1
              and others[0]["name"] == "Cole")
```

Plus a voice-presence check from the existing speaker ID pipeline (no other resident's voice heard in the room within the last ~30 seconds). This becomes a deterministic guard, not a creative inference. The privacy directive is now safer to enforce.

---


## 22. Phase 4: Pets by Name

Phase 4 adds cats. Architecturally nothing new — cats are `WorldEntity(entity_type="cat", person_id=None)` and ride through the same association → state-machine → event-emit pipeline as people. What changes is the cost function (Section 13's `_cat_pair_cost` stub gets filled in), the ObservationBuilder enrichment (color, histogram, size), the bootstrap flow (config-declared cats become entity rows on first run), and a cluster-based cold-start protocol for when there isn't yet enough behavioral data to disambiguate.

### 22.1 The lineup, and why disambiguation is hard

Your specific four cats:

| Name | Color class | Coat | Size | Personality | Home room |
|---|---|---|---|---|---|
| **Officer** | striped (serval-look) | normal | medium | skittish | office |
| **Bandit** | striped (serval-look) | normal | XL | lazy | living_room |
| **Smudge** | black | shiny smooth, slightly darker | medium | social | bedroom |
| **Onyx** | black | slightly fluffier, slightly less black | medium | social | living_room |

Honest disambiguation analysis pair-by-pair:

- **Officer vs. Bandit (both striped):** easy. Bandit is XL — a medium-vs-XL bbox-area ratio at the same camera distance is a 2–3× difference, well above any noise floor. Personality differential (skittish vs. lazy) shows up as motion patterns once enough data accumulates: Officer flees humans approaching within ~2m, Bandit doesn't move when you sit on him. Plus location prior — Officer is in the office 95% of the time, which alone is enough most of the time.
- **Smudge vs. Onyx (both black, both medium, both social):** **hard, possibly unsolvable visually on Wyze v3 RTSP.** The visible differences (shininess, fluffiness, darkness) are below the resolution of a 1920×1080 stream at the typical 6–10ft viewing distance, especially under mixed lighting. Color histograms will look near-identical. ArcFace-style identity embeddings don't exist for cats. The signal that *does* discriminate them is **behavioral and contextual**, especially location prior (Smudge bedroom-leaning, Onyx living-room-leaning) and short-term continuity ("I just saw Onyx in the bedroom, so this black cat in the kitchen is more likely Smudge").
- **Officer/Bandit vs. Smudge/Onyx (striped vs. black):** trivially separable by color class. The hard pairs are within-color-class.

The system should be honest about this. When the cost-function attribution confidence for a black cat falls below ~0.6, the answer to "where's Smudge?" should come back as *"There's a black cat on the couch — probably Smudge but could be Onyx, I'm not certain."* Hedging is correct; guessing wrong corrupts the behavioral profile and makes future answers worse.

### 22.2 Pet declaration in config

```yaml
# config.yaml — new top-level section. Cats are residents (is_resident=True).

pets:
  cats:
    - name: Officer
      color_class: striped
      expected_size: medium
      home_room: office
      personality: skittish
    - name: Bandit
      color_class: striped
      expected_size: xl
      home_room: living_room
      personality: lazy
    - name: Smudge
      color_class: black
      expected_size: medium
      home_room: bedroom
      personality: social
    - name: Onyx
      color_class: black
      expected_size: medium
      home_room: living_room
      personality: social
```

### 22.3 Cat enrichment in ObservationBuilder

When a YOLO `cat` detection arrives, `ObservationBuilder._build_cat_obs` enriches the bbox with three signals before emitting the Observation: color class, color histogram, and size-normalized.

```python
# modules/vision/observation_builder.py — replace the stub _build_cat_obs

import cv2
import numpy as np

def _build_cat_obs(self, frame, det, room, ts, fw, fh) -> Observation:
    bbox = det.bbox
    x1, y1, x2, y2 = [int(v) for v in bbox]
    crop = frame[y1:y2, x1:x2]

    color_class = self._classify_cat_color(crop)              # 'striped' | 'black' | 'unknown'
    color_hist = self._cat_color_histogram(crop)              # normalized HSV histogram
    size_norm = ((x2 - x1) * (y2 - y1)) / (fw * fh)          # bbox area as fraction of frame
    visual_emb = self._cat_visual_embedding(crop)             # CLIP-based, 512-dim

    return Observation(
        camera=room, room=room, obj_class="cat",
        bbox=tuple(bbox), confidence=det.confidence, ts=ts,
        visual_embedding=visual_emb,
        metadata={
            "frame_width": fw, "frame_height": fh,
            "color_class": color_class,
            "color_histogram": color_hist,
            "size_normalized": size_norm,
        },
    )

# ---- helpers ----

def _classify_cat_color(self, crop: np.ndarray) -> str:
    """
    Crude but effective: classify a cat crop as 'striped', 'black', or 'unknown'.
    'striped' detection: high local-contrast variance in grayscale (the stripes).
    'black' detection: low mean luminance + low saturation in HSV.
    Tunable thresholds; verify with a labeled batch first.
    """
    if crop is None or crop.size == 0:
        return "unknown"
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Stripe heuristic: local contrast variance.
    # Striped cats have high-frequency luminance variation; solid-coat cats don't.
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Black cat heuristic: low mean V (value/luminance) and low mean S (saturation).
    mean_v = float(hsv[..., 2].mean())
    mean_s = float(hsv[..., 1].mean())

    # Thresholds — tune against your specific cameras and lighting.
    if mean_v < 70 and mean_s < 40:
        return "black"
    if laplacian_var > 350:
        return "striped"
    return "unknown"


def _cat_color_histogram(self, crop: np.ndarray) -> np.ndarray:
    """
    Normalized HSV color histogram, 16x16x16 bins, flattened.
    Used by _hist_cost in the cat cost function. Returns float32 array, sums to 1.
    """
    if crop is None or crop.size == 0:
        return np.zeros(16 * 16 * 16, dtype=np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 16, 16],
                        [0, 180, 0, 256, 0, 256])
    hist = hist.flatten().astype(np.float32)
    hist /= (hist.sum() + 1e-9)
    return hist


def _cat_visual_embedding(self, crop: np.ndarray) -> np.ndarray:
    """
    CLIP-based visual embedding. Same encoder as objects (Section 23).
    Stored in world_entity_embeddings; used as a soft signal in _cat_pair_cost.
    """
    if crop is None or crop.size == 0:
        return np.zeros(512, dtype=np.float32)
    return self.clip_encoder.encode_image(crop).astype(np.float32)
```

A note on the color classifier: the stripe-detection threshold (Laplacian variance > 350) and the black-cat thresholds (V < 70, S < 40) are starting values. Spend an afternoon collecting ~30 labeled crops per cat from your actual cameras and tune them. The dashboard polygon-editor page can be extended into a "label this crop" page for this purpose.

### 22.4 Bootstrap from config — `pets.py`

```python
# modules/world_model/pets.py
"""
Pet bootstrap. Reads config.pets.cats, creates a WorldEntity per cat on first run.
Idempotent — running on every boot is safe. Existing entities (matched by display_name)
are not duplicated, but their seed metadata is refreshed so config edits take effect.
"""

import uuid
from datetime import datetime
from typing import Any
from .types import WorldEntity, EntityState
from .store import WorldStore


async def bootstrap_pets_from_config(
    store: WorldStore, pet_config: dict
) -> list[WorldEntity]:
    existing = await store.load_entities()
    by_name = {e.display_name: e for e in existing if e.entity_type == "cat"}
    out = []
    for cat in pet_config.get("cats", []):
        seed = {
            "color_class": cat.get("color_class"),
            "expected_size": cat.get("expected_size"),
            "home_room": cat.get("home_room"),
            "personality": cat.get("personality"),
        }
        if cat["name"] in by_name:
            ent = by_name[cat["name"]]
            ent.metadata["seed"] = seed
            await store.upsert_entity(ent)
        else:
            ent = WorldEntity(
                id=str(uuid.uuid4()),
                entity_type="cat",
                person_id=None,
                display_name=cat["name"],
                state=EntityState.IN_ROOM_UNSEEN,
                last_seen_room=cat.get("home_room"),
                last_state_change_ts=datetime.utcnow(),
                metadata={
                    "seed": seed,
                    "behavioral_profile": {},     # built nightly; empty until then
                },
                is_resident=True,
            )
            await store.upsert_entity(ent)
        out.append(ent)
    return out
```

### 22.5 The cluster-based cold-start protocol

This is the part the v1 doc had that v2 was missing. Here's the situation: on day 1, the system has four declared cats and zero behavioral data. A `cat` detection comes in. The cost function looks at color_class — if striped, candidates are {Officer, Bandit}; if black, candidates are {Smudge, Onyx}. Within each pair, location prior is undefined (no profile yet), size discrimination only works for Officer vs. Bandit, and visual embedding is uninformative. The system *can't disambiguate Smudge from Onyx on day 1*. So what should it do?

The cold-start protocol:

**Phase 1 of cold start (days 0–3): collect, don't claim.** Every cat detection creates an unattributed observation cluster — a `WorldEntity(display_name=None, entity_type="cat", metadata={"cluster_seed": True})` instead of being attributed to a named cat. The four declared cats exist as entities but stay in `IN_ROOM_UNSEEN` state with no observations attached. Meanwhile every `cat` detection accumulates into the unattributed cluster pool.

**Phase 2 of cold start (day 3–4): cluster + label.** When the unattributed cluster pool reaches a configurable threshold (default 200 observations), kick off K-means clustering on the visual embeddings + color_class one-hot + room one-hot. K=4 (one cluster per declared cat). The dashboard surfaces a small page showing the four clusters as collages of crops — you click through and label which cluster is which cat. (Black cluster A in the bedroom most often → "Smudge"; black cluster B in the living room → "Onyx"; striped cluster XL → "Bandit"; striped cluster mostly-office → "Officer".)

**Phase 3 of cold start (day 4 onward): seed the profiles.** When labels are submitted, every unattributed observation in cluster *k* gets attributed to the corresponding declared cat. Their behavioral profiles are now seeded with real data. The system is now warm.

Implementation sketch:

```python
# modules/world_model/pets_cluster.py
"""
Cold-start cluster builder for cat disambiguation.
"""

import numpy as np
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
from .store import WorldStore


class CatClusterBuilder:
    def __init__(self, store: WorldStore, config: dict):
        self.store = store
        self.cfg = config

    async def cluster_unattributed_cats(self) -> dict:
        """
        Run after cluster_min_observations have been collected.
        Returns a dict: {cluster_id: [event_ids...]} for the dashboard to label.
        """
        # Pull recent unattributed cat observations
        events = await self.store.search_events(
            event_types=["first_seen", "moved_within_room", "reappeared"],
            since=datetime.utcnow() - timedelta(days=7),
            limit=10000,
        )
        cat_events = [
            e for e in events if e["entity_type"] == "cat"
            and (e.get("entity_name") is None or e.get("entity_name", "").startswith("unknown_"))
        ]
        if len(cat_events) < self.cfg.get("cluster_min_observations", 200):
            return {}

        features = []
        for e in cat_events:
            meta = e.get("metadata", {})
            color_oh = self._color_one_hot(meta.get("color_class"))
            room_oh = self._room_one_hot(e.get("room"))
            visual = meta.get("visual_embedding") or np.zeros(512)
            feat = np.concatenate([visual, color_oh * 5.0, room_oh * 2.0])
            features.append(feat)
        X = np.stack(features)

        km = KMeans(n_clusters=4, n_init=10, random_state=42).fit(X)
        clusters: dict[int, list[str]] = {0: [], 1: [], 2: [], 3: []}
        for label, e in zip(km.labels_, cat_events):
            clusters[int(label)].append(e["id"])
        return clusters

    def _color_one_hot(self, color_class: str | None) -> np.ndarray:
        return {
            "striped": np.array([1.0, 0.0, 0.0]),
            "black":   np.array([0.0, 1.0, 0.0]),
            "unknown": np.array([0.0, 0.0, 1.0]),
        }.get(color_class or "unknown", np.array([0.0, 0.0, 1.0]))

    def _room_one_hot(self, room: str | None) -> np.ndarray:
        rooms = ["office", "living_room", "bedroom", "kitchen", "laundry_room"]
        v = np.zeros(len(rooms))
        if room in rooms:
            v[rooms.index(room)] = 1.0
        return v


async def apply_cluster_labels(
    store: WorldStore, cluster_to_cat_name: dict[int, str], clusters: dict[int, list[str]]
):
    """
    Dashboard submits {cluster_id: cat_name}. Apply by re-attributing every event
    in that cluster to the named cat entity, then trigger a profile rebuild.
    """
    cat_entities = {e.display_name: e for e in await store.load_entities()
                    if e.entity_type == "cat"}
    for cluster_id, cat_name in cluster_to_cat_name.items():
        target = cat_entities.get(cat_name)
        if not target:
            continue
        event_ids = clusters.get(cluster_id, [])
        # The actual SQL UPDATE is straightforward — event_id is the PK
        await store.db.execute(
            f"UPDATE world_entity_events SET entity_id=?, entity_name=? "
            f"WHERE id IN ({','.join('?' * len(event_ids))})",
            (target.id, cat_name, *event_ids),
        )
```

### 22.6 BehavioralProfileBuilder — full implementation

The profile is what makes Smudge/Onyx disambiguation work after enough data accumulates. It has five components, each derived from `world_entity_events` over a rolling 30-day window:

| Component | What it measures | How it's used |
|---|---|---|
| **`room_distribution`** | overall % of observations in each room | base location prior |
| **`room_distribution_by_hour`** | per-hour % of observations in each room | hour-conditioned location prior (heavy hitter for Smudge/Onyx) |
| **`bbox_size_per_room`** | mean + stddev of size_normalized in each room | room-specific size cost (a cat closer to the kitchen camera looks larger than the same cat in the living room — this normalizes that out) |
| **`stationary_fraction`** | fraction of time the cat doesn't trigger MOVED_WITHIN_ROOM | personality signal — Bandit will be high, Officer will be low |
| **`human_avoidance_score`** | correlation between cat's PRESENT state and humans being absent from the same room | personality signal — Officer will be high (avoids humans), Bandit/Smudge/Onyx low |
| **`co_occurrence_partners`** | which other cats this cat is most often co-located with | weak disambiguation signal — if Smudge and Onyx are usually together, knowing one's location boosts/suppresses the other's |

```python
# modules/world_model/pets.py — add BehavioralProfileBuilder

from datetime import datetime, timedelta
from collections import defaultdict


class BehavioralProfileBuilder:
    """
    Runs nightly. For each cat entity, queries last 30 days of events,
    builds a profile dict, writes back to entity.metadata['behavioral_profile'].
    """

    async def rebuild_for(self, world, ent, days_back: int = 30) -> dict:
        since = datetime.utcnow() - timedelta(days=days_back)
        events = await world.store.search_events(
            entity_id=ent.id, since=since, limit=50000,
        )
        if not events:
            return ent.metadata.get("behavioral_profile", {})

        profile = {
            "room_distribution": self._room_distribution(events),
            "room_distribution_by_hour": self._room_distribution_by_hour(events),
            "bbox_size_per_room": self._bbox_size_per_room(events),
            "stationary_fraction": self._stationary_fraction(events),
            "human_avoidance_score": await self._human_avoidance(world, ent, since),
            "co_occurrence_partners": await self._co_occurrence(world, ent, since),
            "n_observations": len(events),
            "window_start": since.isoformat(),
            "window_end": datetime.utcnow().isoformat(),
        }
        ent.metadata["behavioral_profile"] = profile
        await world.store.upsert_entity(ent)
        return profile

    def _room_distribution(self, events: list[dict]) -> dict[str, float]:
        counts: dict[str, int] = defaultdict(int)
        for e in events:
            r = e.get("room")
            if r:
                counts[r] += 1
        total = sum(counts.values()) or 1
        return {r: c / total for r, c in counts.items()}

    def _room_distribution_by_hour(self, events: list[dict]) -> dict[int, dict[str, float]]:
        per_hour: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for e in events:
            r = e.get("room")
            if not r:
                continue
            h = datetime.fromisoformat(e["ts"]).hour
            per_hour[h][r] += 1
        out: dict[int, dict[str, float]] = {}
        for h, counts in per_hour.items():
            total = sum(counts.values()) or 1
            out[h] = {r: c / total for r, c in counts.items()}
        return out

    def _bbox_size_per_room(self, events: list[dict]) -> dict[str, dict[str, float]]:
        per_room: dict[str, list[float]] = defaultdict(list)
        for e in events:
            r = e.get("room")
            sz = (e.get("metadata") or {}).get("size_normalized")
            if r and sz is not None:
                per_room[r].append(float(sz))
        return {
            r: {"mean": float(np.mean(sizes)), "std": float(np.std(sizes)),
                "n": len(sizes)}
            for r, sizes in per_room.items() if len(sizes) >= 5
        }

    def _stationary_fraction(self, events: list[dict]) -> float:
        movements = sum(1 for e in events if e["event_type"] == "moved_within_room")
        appearances = sum(1 for e in events
                          if e["event_type"] in ("first_seen", "reappeared", "moved_to"))
        if appearances == 0:
            return 0.5
        # Heuristic: low movement-per-appearance = stationary cat
        ratio = movements / max(appearances, 1)
        return float(max(0.0, min(1.0, 1.0 - ratio / 5.0)))

    async def _human_avoidance(self, world, ent, since) -> float:
        """
        For each cat PRESENT event in a room, check whether a person was also
        PRESENT in the same room within ±60 seconds. Avoidance = 1 - (cohabitation rate).
        """
        cat_events = await world.store.search_events(
            entity_id=ent.id, event_types=["reappeared", "moved_to", "first_seen"],
            since=since, limit=10000,
        )
        if not cat_events:
            return 0.5
        cohab = 0
        for ce in cat_events:
            ts = datetime.fromisoformat(ce["ts"])
            window = await world.store.search_events(
                room=ce.get("room"),
                event_types=["reappeared", "moved_to", "first_seen"],
                since=ts - timedelta(seconds=60),
                until=ts + timedelta(seconds=60),
                limit=50,
            )
            if any(w["entity_type"] == "person" for w in window):
                cohab += 1
        return float(1.0 - (cohab / len(cat_events)))

    async def _co_occurrence(self, world, ent, since) -> dict[str, float]:
        """
        For each cat PRESENT event, find other cats also PRESENT in the same room
        within ±60s. Returns {other_cat_name: co_occurrence_rate}.
        """
        cat_events = await world.store.search_events(
            entity_id=ent.id, event_types=["reappeared", "moved_to", "first_seen"],
            since=since, limit=10000,
        )
        if not cat_events:
            return {}
        partner_counts: dict[str, int] = defaultdict(int)
        for ce in cat_events:
            ts = datetime.fromisoformat(ce["ts"])
            window = await world.store.search_events(
                room=ce.get("room"),
                event_types=["reappeared", "moved_to", "first_seen"],
                since=ts - timedelta(seconds=60),
                until=ts + timedelta(seconds=60),
                limit=50,
            )
            for w in window:
                if (w["entity_type"] == "cat" and w["entity_id"] != ent.id
                        and w.get("entity_name")):
                    partner_counts[w["entity_name"]] += 1
        n = len(cat_events) or 1
        return {name: c / n for name, c in partner_counts.items()}
```

The `human_avoidance_score` and `co_occurrence_partners` are mostly informational on day 30 but become real signals on day 90 when the profile is dense — they're how the system would eventually notice *"Officer is in the living room while Cole is also in the living room — that's unusual for Officer."*

Schedule the rebuild via the existing scheduler that runs the rest of Jarvis's nightly jobs (per repo convention — `core/orchestrator.py` likely has a daily-tasks loop). One call per cat per night.

### 22.7 The cat cost function (full)

This expands the stub from Section 13. Same algorithm as v1 but adapted to the new data model — `obs.visual_embedding` rather than `obs.embedding`, `_visual_embedding` in metadata rather than `embedding` field.

```python
# modules/world_model/world_model.py — replace the _cat_pair_cost stub

import numpy as np


def _cat_pair_cost(self, obs: Observation, ent: WorldEntity) -> float:
    """
    Cats: behavioral and contextual signals dominate, visual corroborates.
    Five-component cost; weights tuned from v1 deployment experience.
    """
    # Color class hard filter — striped/black mismatch is a hard reject
    obs_color = obs.metadata.get("color_class", "unknown")
    ent_color = ent.metadata.get("seed", {}).get("color_class", "unknown")
    if obs_color != "unknown" and ent_color != "unknown" and obs_color != ent_color:
        return self.cfg["cost_reject"] * 2

    # ---------- VISUAL: histogram + embedding (weak; both can be similar within color class) ----------
    hist_cost = _hist_bhattacharyya(
        obs.metadata.get("color_histogram"),
        ent.metadata.get("color_histogram"),
    )

    ent_emb = ent.metadata.get("_visual_embedding")
    obs_emb = obs.visual_embedding
    if ent_emb is not None and obs_emb is not None:
        sim = float(np.dot(obs_emb, ent_emb)
                    / (np.linalg.norm(obs_emb) * np.linalg.norm(ent_emb) + 1e-9))
        emb_cost = 1.0 - sim
    else:
        emb_cost = 0.5

    # ---------- SIZE ----------
    profile = ent.metadata.get("behavioral_profile", {})
    obs_size = obs.metadata.get("size_normalized")
    expected = profile.get("bbox_size_per_room", {}).get(obs.room, {})
    if expected and obs_size is not None and expected.get("n", 0) >= 5:
        # Use learned room-specific size statistics
        mean_sz, std_sz = expected["mean"], max(expected["std"], 1e-3)
        z = abs(obs_size - mean_sz) / std_sz
        size_cost = float(min(z / 3.0, 1.0))   # 3 std deviations = max cost
    else:
        # Fall back to seed-based expected size tier
        size_cost = _size_cost_from_seed(obs_size, ent.metadata.get("seed", {}))

    # ---------- LOCATION PRIOR (the heavy hitter for Smudge/Onyx) ----------
    hour = obs.ts.hour
    by_hour = profile.get("room_distribution_by_hour", {})
    hour_dist = by_hour.get(hour) or by_hour.get(str(hour)) or {}
    p_room = hour_dist.get(obs.room) or profile.get("room_distribution", {}).get(obs.room, 0.05)
    location_cost = float(min(-np.log(p_room + 0.01), 2.0))

    # ---------- CONTINUITY ----------
    if ent.last_seen_ts:
        seconds_gone = (obs.ts - ent.last_seen_ts).total_seconds()
    else:
        seconds_gone = 1e6
    if obs.camera == ent.last_seen_camera and seconds_gone < 5:
        continuity = -0.3
    elif obs.room == ent.last_seen_room and seconds_gone < 30:
        continuity = -0.15
    else:
        continuity = 0.0

    # ---------- CO-OCCURRENCE TIE-BREAKER (Smudge/Onyx specifically) ----------
    # If the OTHER black cat was just seen in a different room very recently,
    # bias the cost down (this is more likely to be us).
    co_partners = profile.get("co_occurrence_partners", {})
    other_recently_seen_elsewhere = self._other_cat_seen_recently(
        ent, obs.ts, exclude_room=obs.room, lookback_seconds=30
    )
    if other_recently_seen_elsewhere:
        co_bonus = -0.2 if other_recently_seen_elsewhere in co_partners else 0.0
    else:
        co_bonus = 0.0

    # ---------- COMBINE ----------
    cost = max(0.0, (
        0.20 * emb_cost +
        0.20 * hist_cost +
        0.15 * size_cost +
        0.30 * location_cost +
        0.05 * (min(seconds_gone / 60.0, 1.0)) +
        continuity +
        co_bonus
    ))
    return cost


def _other_cat_seen_recently(self, ent, ts, exclude_room: str,
                             lookback_seconds: int = 30) -> str | None:
    """
    Did any OTHER cat get observed in a different room in the last N seconds?
    Returns that cat's display_name, or None.
    """
    cutoff = ts - timedelta(seconds=lookback_seconds)
    for other in self.entities.values():
        if (other.entity_type == "cat" and other.id != ent.id
                and other.last_seen_ts and other.last_seen_ts >= cutoff
                and other.last_seen_room and other.last_seen_room != exclude_room):
            return other.display_name
    return None


def _hist_bhattacharyya(h1, h2) -> float:
    """Bhattacharyya distance, normalized 0–1. Lower = more similar."""
    if h1 is None or h2 is None:
        return 0.5
    h1 = np.asarray(h1, dtype=np.float32)
    h2 = np.asarray(h2, dtype=np.float32)
    bc = float(np.sum(np.sqrt(h1 * h2)))
    return float(min(1.0, np.sqrt(max(0.0, 1.0 - bc))))


def _size_cost_from_seed(obs_size, seed: dict) -> float:
    if obs_size is None:
        return 0.5
    expected_size = seed.get("expected_size", "medium")
    targets = {"small": 0.02, "medium": 0.04, "large": 0.07, "xl": 0.10}
    target = targets.get(expected_size, 0.04)
    return float(min(abs(np.log(max(obs_size, 1e-4) / target)) / 2.0, 1.0))
```

### 22.8 Day-1 → day-30 disambiguation walk-through

To make sure the timeline is concrete:

**Day 1 (zero data):** The system sees a black cat in the kitchen. Both Smudge and Onyx have empty profiles. `_cat_pair_cost` returns nearly identical costs for both. Hungarian picks one arbitrarily. Attribution confidence is low (margin ≈ 0). The `query_tools` layer marks the response as hedged → LLM phrases it *"a black cat is in the kitchen — I can't tell yet whether it's Smudge or Onyx, give me a few weeks of observations."* This is the honest answer.

**Day 3 (cluster threshold reached):** `CatClusterBuilder` runs, produces 4 clusters from the 200+ unattributed observations. Dashboard surfaces them. You label them. `apply_cluster_labels` reattributes every event. `BehavioralProfileBuilder` runs against the freshly-labeled data. Profiles now have ~50 events per cat — sparse but workable. `room_distribution` is meaningful; `room_distribution_by_hour` is still noisy.

**Day 7:** Profiles have ~150 events per cat. Smudge's `room_distribution` is something like `{bedroom: 0.55, living_room: 0.20, kitchen: 0.15, office: 0.10}`; Onyx's is `{living_room: 0.50, bedroom: 0.20, kitchen: 0.20, office: 0.10}`. The Smudge/Onyx cost gap on a kitchen observation is now ~0.05–0.10 (small but meaningful). Attribution confidence comes back at 0.5–0.7 most of the time. Answers are still occasionally hedged but mostly committed.

**Day 30:** Profiles are dense. `room_distribution_by_hour` is well-populated. The co-occurrence-tie-breaker has real co_partners data. A black cat in the kitchen at 7 PM gets a confident attribution because *"at 7 PM, Smudge is in the bedroom 60% of the time and Onyx is in the kitchen 35% of the time, AND Smudge was just seen on the bed 20 seconds ago — so this is Onyx, attribution_confidence=0.78."* The hedge falls off.

**Day 90:** Anomaly scoring (Section 25) starts giving useful results because the profile is dense enough that "unusual" is actually defined.

### 22.9 Cat-specific events worth firing

Beyond the universal events from Section 6, two cat-specific events are worth firing if you want a richer event log. Both gated on landmarks declared in the room config:

```yaml
# config.yaml — example litterbox landmark
rooms:
  - id: laundry_room
    world_model:
      landmarks:
        - name: litterbox
          polygon: [[100, 400], [300, 400], [300, 550], [100, 550]]
        - name: food_dish
          polygon: [[400, 450], [550, 450], [550, 550], [400, 550]]
```

When a cat's bbox center is over `litterbox` for ≥3 frames → emit a `LITTERBOX_VISIT` event (a specialization of `INTERACTED_WITH` with `metadata.landmark="litterbox"`). Same for `food_dish` → `FOOD_DISH_VISIT`. These are useful for:

- *"Is anyone using the litterbox abnormally often?"* (signal of UTI / stress)
- *"Has anyone eaten today?"* (visible feeding rhythm, useful for vet visits)
- *"Did Officer come out for food?"* (he's skittish; if he didn't eat in 24h, that matters)

These don't need to be in the EventType enum; they ride as `INTERACTED_WITH` events with metadata. The dashboard and LLM can filter on metadata.landmark.

---

## 23. Phase 4: Objects

Objects are `WorldEntity(entity_type="object", person_id=None)`. They get CLIP visual embeddings stored in `world_entity_embeddings`, the cost function from Section 13's stub gets implemented, and a query-by-description path (`find_object("wallet")`) gets added on top. The architecture is the same as cats; what's new is the open-vocabulary vs. closed-vocabulary detection question and the CLIP encoder bootstrap.

### 23.1 Detection: closed-vocab YOLO vs. open-vocab OWL/GDINO

The repo's existing ObjectDetector uses YOLOv8, which is closed-vocab — fixed list of 80 COCO classes. That covers the obvious targets:

```
cell phone, cup, book, laptop, bottle, remote, keyboard, mouse,
backpack, handbag, scissors, clock, vase, wine glass, fork, knife,
spoon, bowl, banana, apple, orange, sandwich, pizza, donut, cake
```

Plenty for *"where's my phone?"* and *"is there food on the counter?"*. **Not** plenty for *"where's my wallet?"* (not a COCO class), *"where are my keys?"* (also not), *"where's the dog leash?"* (no), *"where's the brown box that came yesterday?"* (definitely not).

For open-vocab — anything describable that's not in YOLO — there are two viable models:

| Model | Speed on RTX 4070 Ti | Recall on novel objects | Localization | License |
|---|---|---|---|---|
| **OWLv2 (`google/owlv2-base-patch16-ensemble`)** | ~120ms/frame at 640×640 | high | accurate bboxes | Apache 2.0 |
| **Grounding DINO (`IDEA-Research/grounding-dino-base`)** | ~80ms/frame at 800×800 | very high | accurate bboxes, slightly tighter | Apache 2.0 |

Grounding DINO is the better default — faster, slightly higher recall — but OWLv2 is fine and has a simpler integration story. **Recommendation:** start with OWLv2 because the HuggingFace integration is a single import; switch to Grounding DINO if recall on your specific household objects is unsatisfactory.

Either way, **don't run open-vocab detection at full frame rate.** It's too expensive. The right design is hybrid:

```
YOLO @ full FPS  (always running, finds the 80 known classes)
     ↓
OWL/GDINO @ low FPS (every 30s, finds the user's declared open-vocab list)
     ↓
Once an OWL detection lands, that bbox is registered as a tracked region.
For subsequent frames, lighter-weight tracking (or just re-detection at the
same coarse rate) updates the entity. Object motion is rare enough that
30s polling is fine for objects users actually want to find.
```

### 23.2 Open-vocab declaration in config

```yaml
# config.yaml — declare what open-vocab objects the system should track.
# Each entry is a free-text description that gets fed to OWL/GDINO as a query.

tracked_objects:
  open_vocabulary:
    - name: wallet
      description: "a small leather wallet"
      typical_rooms: [office, bedroom, living_room]
    - name: keys
      description: "house keys on a keyring"
      typical_rooms: [kitchen, living_room, office]
    - name: glasses
      description: "eyeglasses with black frames"
      typical_rooms: [bedroom, office]
    - name: medication_bottle
      description: "an orange prescription pill bottle"
      typical_rooms: [bedroom, kitchen]
  closed_vocabulary:
    # Leave empty to track ALL of YOLO's 80 classes; otherwise restrict
    # to specific ones to reduce noise.
    - cell phone
    - cup
    - book
    - laptop
    - bottle
    - remote
    - backpack
  detection_interval_seconds:
    open_vocabulary: 30      # OWL/GDINO every 30s
    closed_vocabulary: 0     # YOLO every frame
```

The `typical_rooms` list is used by the cost function to apply a room prior — a wallet is *much* more likely to be in the office than in the bathroom, so a YOLO-detected-rectangle that *might* be a wallet gets a higher prior in the office.

### 23.3 The CLIP encoder

Used for two things: (1) computing the visual embedding of every tracked-object Observation, and (2) text-embedding the query in `find_object("wallet")` so we can match by similarity.

```python
# modules/vision/clip_encoder.py
"""
CLIP encoder for object visual embeddings + text-query matching.
Singleton — instantiate once in the orchestrator and pass to ObservationBuilder
and the world-model query layer.
"""

import torch
import numpy as np
from PIL import Image
import open_clip
from loguru import logger


class CLIPEncoder:
    """
    OpenCLIP wrapper. ViT-B/32 chosen as default — good balance of speed
    (~3ms image embed on 4070 Ti) and quality. Bump to ViT-L/14 if recall
    on visually-similar objects (multiple wallets, multiple cups) is poor.
    """

    def __init__(self, model_name: str = "ViT-B-32",
                 pretrained: str = "laion2b_s34b_b79k",
                 device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        logger.info(f"[CLIPEncoder] {model_name}/{pretrained} loaded on {self.device}")

    @torch.no_grad()
    def encode_image(self, image_bgr: np.ndarray) -> np.ndarray:
        """Encode a BGR numpy image (e.g., a YOLO crop) to a 512-dim L2-normalized vector."""
        if image_bgr is None or image_bgr.size == 0:
            return np.zeros(512, dtype=np.float32)
        rgb = image_bgr[..., ::-1]
        pil = Image.fromarray(rgb)
        x = self.preprocess(pil).unsqueeze(0).to(self.device)
        emb = self.model.encode_image(x)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().squeeze(0).astype(np.float32)

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        """Encode a text query to a 512-dim L2-normalized vector."""
        toks = self.tokenizer([text]).to(self.device)
        emb = self.model.encode_text(toks)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().squeeze(0).astype(np.float32)

    @torch.no_grad()
    def encode_text_batch(self, texts: list[str]) -> np.ndarray:
        toks = self.tokenizer(texts).to(self.device)
        emb = self.model.encode_text(toks)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().astype(np.float32)
```

### 23.4 Open-vocab detector — `OpenVocabDetector`

```python
# modules/vision/openvocab_detector.py
"""
OWLv2 wrapper for low-frequency open-vocabulary detection. Used to find
user-declared objects (wallet, keys, glasses) that aren't in YOLO's COCO classes.
"""

import torch
import numpy as np
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from loguru import logger


class OpenVocabDetector:
    def __init__(self, model_name: str = "google/owlv2-base-patch16-ensemble",
                 device: str = "cuda", score_threshold: float = 0.20):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.score_threshold = score_threshold
        logger.info(f"[OpenVocabDetector] {model_name} loaded on {self.device}")

    @torch.no_grad()
    def detect(self, image_bgr: np.ndarray, queries: list[str]) -> list[dict]:
        """
        Run OWLv2 with a list of text queries on a frame. Returns list of dicts:
        [{name, bbox: (x1,y1,x2,y2), score}, ...]
        """
        if image_bgr is None or image_bgr.size == 0 or not queries:
            return []
        rgb = image_bgr[..., ::-1]
        pil = Image.fromarray(rgb)
        inputs = self.processor(text=[queries], images=pil,
                                return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([pil.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs, threshold=self.score_threshold,
            target_sizes=target_sizes,
        )[0]
        out = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            x1, y1, x2, y2 = [int(c) for c in box.tolist()]
            out.append({
                "name": queries[int(label)],
                "bbox": (x1, y1, x2, y2),
                "score": float(score),
            })
        return out
```

### 23.5 ObservationBuilder enrichment for objects

Two paths into the object pipeline — the high-frequency YOLO path and the low-frequency OWL path. Both produce `Observation(obj_class="object")` payloads with a CLIP embedding attached.

```python
# modules/vision/observation_builder.py — replace _build_object_obs

def _build_object_obs(self, frame, det, room, ts, fw, fh) -> Observation:
    bbox = det.bbox
    x1, y1, x2, y2 = [int(v) for v in bbox]
    crop = frame[y1:y2, x1:x2]
    visual_emb = self.clip_encoder.encode_image(crop)
    return Observation(
        camera=room, room=room, obj_class="object",
        bbox=tuple(bbox), confidence=det.confidence, ts=ts,
        visual_embedding=visual_emb,
        metadata={
            "detected_class": det.class_name,
            "frame_width": fw, "frame_height": fh,
            "source": "yolo",
        },
    )

# New: low-frequency open-vocab loop, separate from the per-frame loop.

async def _open_vocab_loop_for_room(self, room_id: str):
    """
    Runs every detection_interval_seconds.open_vocabulary (default 30s).
    Pulls a single fresh frame, runs OWLv2 with the user-declared queries,
    emits Observations for any hits.
    """
    queries = [t["description"] for t in self.tracked_objects_open_vocab]
    if not queries:
        return
    interval = self.cfg.get("openvocab_detection_interval_seconds", 30)
    while True:
        try:
            await asyncio.sleep(interval)
            frame, ts = await self.cm.get_latest_frame(room_id)
            if frame is None:
                continue
            results = self.openvocab.detect(frame, queries)
            if not results:
                continue
            fh, fw = frame.shape[:2]
            observations = []
            for r in results:
                # Map description → user-friendly name
                cfg_entry = next(
                    (t for t in self.tracked_objects_open_vocab
                     if t["description"] == r["name"]), None
                )
                friendly_name = cfg_entry["name"] if cfg_entry else r["name"]
                x1, y1, x2, y2 = r["bbox"]
                crop = frame[y1:y2, x1:x2]
                emb = self.clip_encoder.encode_image(crop)
                observations.append(Observation(
                    camera=room_id, room=room_id, obj_class="object",
                    bbox=r["bbox"], confidence=r["score"], ts=ts,
                    visual_embedding=emb,
                    metadata={
                        "detected_class": friendly_name,
                        "openvocab_query": r["name"],
                        "frame_width": fw, "frame_height": fh,
                        "source": "owlv2",
                    },
                ))
            if observations:
                await self.bus.publish("vision.observation", {
                    "camera": room_id, "room": room_id, "ts": ts,
                    "observations": observations,
                })
        except Exception:
            logger.exception(f"[ObservationBuilder] open-vocab loop error in {room_id}")
```

Spawn one open-vocab loop per camera-equipped room in `start()`.

### 23.6 Object cost function with class-and-room priors

The stub from Section 13 needs to grow a class-and-room prior. A `cell phone` detection in the kitchen should be more likely to match the existing `Cole's cell phone` entity (last seen in kitchen) than to match a different `cell phone` entity that was last in the bedroom — even when the visual embeddings happen to be similar.

```python
# modules/world_model/world_model.py — replace _object_pair_cost stub

def _object_pair_cost(self, obs: Observation, ent: WorldEntity) -> float:
    # Class hard filter — a "wallet" obs cannot match a "cell phone" entity
    obs_class = obs.metadata.get("detected_class")
    ent_class = ent.metadata.get("detected_class")
    if obs_class and ent_class and obs_class != ent_class:
        return self.cfg["cost_reject"] * 2

    # Visual: CLIP embedding similarity
    ent_emb = ent.metadata.get("_visual_embedding")
    if ent_emb is None or obs.visual_embedding is None:
        return self.cfg["cost_reject"] * 2
    sim = float(np.dot(obs.visual_embedding, ent_emb)
                / (np.linalg.norm(obs.visual_embedding)
                   * np.linalg.norm(ent_emb) + 1e-9))
    emb_cost = 1.0 - sim

    # Spatial: same room is essentially free, different room is expensive
    # (objects don't move on their own — if it's in a different room, it's likely
    # a different object of the same class, or someone moved it)
    if ent.last_seen_room == obs.room:
        room_cost = 0.0
    elif obs_class in self._typical_rooms_for_object_class.get(obs_class, []):
        # The new room is one of this object's typical rooms — softer penalty
        room_cost = 0.25
    else:
        room_cost = 0.5

    # Time decay — an entity not seen in 7 days is less likely to be matched against
    if ent.last_seen_ts:
        days_gone = (obs.ts - ent.last_seen_ts).total_seconds() / 86400.0
        time_cost = float(min(days_gone / 14.0, 0.5))   # up to +0.5 over 2 weeks
    else:
        time_cost = 0.5

    return 0.55 * emb_cost + 0.30 * room_cost + 0.15 * time_cost
```

The `_typical_rooms_for_object_class` lookup is built once at startup from the config:

```python
# In WorldModel.__init__:
self._typical_rooms_for_object_class = {
    t["name"]: t.get("typical_rooms", [])
    for t in (config.get("tracked_objects", {}).get("open_vocabulary", []) or [])
}
```

### 23.7 The find_object query

```python
# modules/world_model/query_tools.py — add this method

async def find_object(self, description: str, k: int = 3) -> dict:
    """
    Text-query for an object. Embeds the description with CLIP, finds the top-k
    most similar tracked objects, returns the best match (or all top-k if user
    wants to see alternatives).
    """
    text_emb = self.world.clip_encoder.encode_text(description)
    candidates = []
    for e in self.world.entities.values():
        if e.entity_type != "object":
            continue
        emb = e.metadata.get("_visual_embedding")
        if emb is None:
            continue
        sim = float(np.dot(text_emb, emb)
                    / (np.linalg.norm(text_emb) * np.linalg.norm(emb) + 1e-9))
        candidates.append((sim, e))
    candidates.sort(key=lambda x: x[0], reverse=True)
    if not candidates or candidates[0][0] < self.cfg.get("clip_match_threshold", 0.25):
        return {"found": False,
                "message": f"I don't have a tracked object that looks like '{description}'.",
                "checked_entities": len(candidates)}
    top = candidates[:k]
    primary_sim, primary = top[0]
    return {
        "found": True,
        "name": primary.display_name or primary.metadata.get("detected_class"),
        "last_seen_room": primary.last_seen_room,
        "last_seen_landmark": primary.last_seen_landmark,
        "last_seen_ts": primary.last_seen_ts.isoformat() if primary.last_seen_ts else None,
        "match_similarity": primary_sim,
        "alternatives": [
            {
                "name": e.display_name or e.metadata.get("detected_class"),
                "room": e.last_seen_room,
                "similarity": s,
            }
            for s, e in top[1:]
        ],
        # If primary similarity is borderline, the LLM should hedge
        "hedge": primary_sim < 0.32,
    }
```

### 23.8 Entity dedup for objects (the "every cup is a new cup" problem)

Without care, every YOLO `cell phone` detection creates a new `WorldEntity(detected_class="cell phone")`. After a week you'd have 10,000 cell-phone entities, almost all stale.

The dedup discipline:

1. **Strong embedding match wins** — when an unmatched `object` observation arrives, the existing `_handle_unmatched_observation` (in `world_model.py`) already searches all object entities for the highest-similarity embedding match before creating a new entity. The `cosine_match_strong` threshold (default 0.6) is what gates this.
2. **Same-room-same-class match wins more easily.** Lower the threshold for same-room-same-class:

```python
# In _handle_unmatched_observation, before creating a new object entity:

if obs.obj_class == "object" and obs.visual_embedding is not None:
    best, best_sim = None, 0.0
    obs_class = obs.metadata.get("detected_class")
    for ent in self.entities.values():
        if ent.entity_type != "object":
            continue
        if ent.metadata.get("detected_class") != obs_class:
            continue
        emb = ent.metadata.get("_visual_embedding")
        if emb is None:
            continue
        sim = _cos(obs.visual_embedding, emb)
        # Same-room-same-class: lower threshold
        threshold = (self.cfg.get("cosine_match_strong_same_room", 0.45)
                     if ent.last_seen_room == obs.room
                     else self.cfg.get("cosine_match_strong", 0.6))
        if sim > best_sim and sim >= threshold:
            best, best_sim = ent, sim
    if best:
        await self._update_matched(best, obs, ts, attribution_conf=best_sim)
        return
```

3. **Periodic stale-object pruning.** Once nightly, drop object entities that haven't been seen in N days AND have no `INTERACTED_WITH` events in their history. (Don't drop ones the user has touched — those have story value even if invisible right now.)

```python
async def prune_stale_objects(self, max_age_days: int = 30):
    now = datetime.utcnow()
    cutoff = now - timedelta(days=max_age_days)
    async with self._lock:
        for ent in list(self.entities.values()):
            if ent.entity_type != "object":
                continue
            if (ent.last_seen_ts is None or ent.last_seen_ts < cutoff):
                interactions = await self.store.search_events(
                    entity_id=ent.id,
                    event_types=["interacted_with", "picked_up", "placed_down"],
                    limit=1,
                )
                if not interactions:
                    del self.entities[ent.id]
                    # Optionally: hard-delete row, or just let it persist as
                    # historical record. Recommended: keep the row for query
                    # history but mark it metadata.pruned=True so it's excluded
                    # from candidate pools.
                    ent.metadata["pruned"] = True
                    await self.store.upsert_entity(ent)
```

---

## 24. Phase 5: Interactions

Interactions are *narrative events*: PICKED_UP, PLACED_DOWN, INTERACTED_WITH. These are what turn the system from "I see a wallet on the desk" into "Cole picked up the wallet from the desk at 3:45 PM and placed it on the kitchen counter at 3:47 PM." They depend on hand detection (MediaPipe Hands), the existing object entity tracking from Section 23, and a temporal correlation engine that watches the event bus for sequences.

### 24.1 MediaPipe Hands integration

Where it sits: inside ObservationBuilder, called per-frame for every camera-equipped room. Output: a list of hand bboxes per frame. Cost: ~3–5ms per frame on RTX 4070 Ti at 640×640 input. Cheap enough to run at full FPS in the office (30 fps), but for the Wyze rooms (5 fps active / 1 fps idle) it's nearly free.

```python
# modules/vision/hand_detector.py
"""
MediaPipe Hands wrapper. Returns hand bboxes per frame. Each bbox is the
axis-aligned bounding box around the 21 keypoints MediaPipe produces per hand.
"""

import cv2
import numpy as np
import mediapipe as mp
from loguru import logger


class HandDetector:
    def __init__(self, max_num_hands: int = 4, min_detection_confidence: float = 0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )
        logger.info(f"[HandDetector] MediaPipe Hands loaded (max={max_num_hands})")

    def detect(self, image_bgr: np.ndarray) -> list[dict]:
        """
        Returns list of dicts:
        [{
            'bbox': (x1, y1, x2, y2),
            'handedness': 'Left' | 'Right',
            'confidence': float,
            'wrist_xy': (x, y)   # useful for "object held in hand" reasoning
        }, ...]
        """
        if image_bgr is None or image_bgr.size == 0:
            return []
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_bgr.shape[:2]
        results = self.hands.process(rgb)
        if not results.multi_hand_landmarks:
            return []
        out = []
        for landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness or [],
        ):
            xs = [lm.x * w for lm in landmarks.landmark]
            ys = [lm.y * h for lm in landmarks.landmark]
            bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
            wrist_xy = (int(landmarks.landmark[0].x * w),
                        int(landmarks.landmark[0].y * h))
            out.append({
                "bbox": bbox,
                "handedness": handedness.classification[0].label
                              if handedness.classification else "Unknown",
                "confidence": float(handedness.classification[0].score)
                              if handedness.classification else 0.0,
                "wrist_xy": wrist_xy,
            })
        return out
```

### 24.2 hand_bboxes flow into person Observations

In `ObservationBuilder._build_for_frame`, run hand detection once per frame, then attach the hand bboxes that geometrically correspond to each person to that person's Observation.

```python
# modules/vision/observation_builder.py — adapt _build_for_frame

async def _build_for_frame(self, room: str, frame: np.ndarray, ts: datetime) -> list[Observation]:
    observations = []
    detections = await self.detector.detect_async(frame)

    # Run hand detection once per frame (cheap)
    all_hands = self.hand_detector.detect(frame)

    frame_h, frame_w = frame.shape[:2]

    for det in detections:
        cls = det.class_name
        if cls == "person":
            # Find hands that are inside or adjacent to this person's bbox
            person_hands = [
                h for h in all_hands
                if _bbox_overlaps_or_within(h["bbox"], det.bbox, slack=20)
            ]
            obs = await self._build_person_obs(
                frame, det, room, ts, frame_w, frame_h,
                hand_bboxes=[h["bbox"] for h in person_hands],
                hand_details=person_hands,
                posture=None,
            )
        elif cls == "cat":
            obs = self._build_cat_obs(frame, det, room, ts, frame_w, frame_h)
        elif cls in self.TRACKED_OBJECT_CLASSES:
            obs = self._build_object_obs(frame, det, room, ts, frame_w, frame_h)
        else:
            continue
        observations.append(obs)
    return observations


def _bbox_overlaps_or_within(small: tuple, large: tuple, slack: int = 0) -> bool:
    """Returns True if `small` bbox is mostly inside `large` (with slack pixels)."""
    sx1, sy1, sx2, sy2 = small
    lx1, ly1, lx2, ly2 = large
    return (sx1 >= lx1 - slack and sy1 >= ly1 - slack
            and sx2 <= lx2 + slack and sy2 <= ly2 + slack)
```

The `hand_details` go into observation metadata for the InteractionMonitor to reason about handedness later (useful for *"Cole picked it up with his right hand"* if you ever care).

### 24.3 The pickup/placement state machine

Detecting *"Cole picked up the wallet"* is a temporal-correlation problem, not a per-frame classification. The states:

```
S0  object visible at L, no hand overlap
       │   hand overlaps object bbox
       ▼
S1  object visible at L, hand overlap (≥1 frame)
       │   ≥3 consecutive frames of overlap
       ▼
S2  INTERACTED_WITH event fired
       │   object stops being detected at L (next ≤T_pickup_seconds)
       ▼
S3  hand still in scene, object gone → PICKED_UP candidate
       │   hand leaves scene OR person leaves room
       ▼
S4  PICKED_UP event fired (with person_id, object_id, source_room)


Reverse:

S0  hand visible in scene, no object at the spot
       │   hand activity at coordinates (x,y) for ≥3 frames, no object detected
       ▼
S1  hand activity zone in scene (object placement candidate)
       │   object becomes visible at (x±slack, y±slack) within next T_place_seconds
       ▼
S2  PLACED_DOWN event fired (with person_id, object_id, dest_room)
```

The trick is that the S2→S3 and S2→S0 reverse path both depend on *bus events from other entities* — the object's own `LOST_VISIBILITY` event for pickup, the object's own `FIRST_SEEN` event for placement. So the InteractionMonitor is a separate subscriber that correlates events across entities.

```python
# modules/world_model/interactions.py
"""
InteractionMonitor: subscribes to world.entity_event, watches for pickup and
place-down patterns by correlating events across entities and time.
"""

import asyncio
from datetime import datetime, timedelta
from collections import deque
from typing import Optional
from loguru import logger


class InteractionMonitor:
    """
    Detects PICKED_UP and PLACED_DOWN events by correlating:
      - INTERACTED_WITH events (hand-object overlap, fired by WorldModel)
      - Object entity LOST_VISIBILITY events (object disappeared)
      - Object entity FIRST_SEEN events (new object appeared in a room)
      - Person entity events in the same room (for "who did it")
    """

    def __init__(self, bus, world, config: dict):
        self.bus = bus
        self.world = world
        self.cfg = config
        # Recent event buffers — bounded; older entries fall off naturally
        self.recent_interactions: deque = deque(maxlen=500)
        self.recent_object_losses: deque = deque(maxlen=500)
        self.recent_object_appearances: deque = deque(maxlen=500)
        self.recent_hand_activity: deque = deque(maxlen=500)

    async def start(self):
        await self.bus.subscribe("world.entity_event", self._on_event)
        logger.info("[InteractionMonitor] started")

    async def _on_event(self, event: dict):
        et = event["event_type"]
        ts = datetime.fromisoformat(event["ts"])
        if et == "interacted_with":
            self.recent_interactions.append((ts, event))
            asyncio.create_task(self._check_for_pickup(event, ts))
        elif et == "lost_visibility" and event["entity_type"] == "object":
            self.recent_object_losses.append((ts, event))
        elif et == "first_seen" and event["entity_type"] == "object":
            self.recent_object_appearances.append((ts, event))
            asyncio.create_task(self._check_for_placedown(event, ts))

    async def _check_for_pickup(self, interaction_event: dict, ts: datetime):
        """
        After an INTERACTED_WITH fires, wait T_pickup_seconds. If the object
        in question disappears (LOST_VISIBILITY) within that window, fire PICKED_UP.
        """
        wait_s = self.cfg.get("pickup_settle_seconds", 3)
        await asyncio.sleep(wait_s)
        obj_id = interaction_event["metadata"].get("object_id")
        if not obj_id:
            return
        # Look for a LOST_VISIBILITY for that object within ±T_pickup_seconds
        for loss_ts, loss in list(self.recent_object_losses):
            if (loss["entity_id"] == obj_id
                and abs((loss_ts - ts).total_seconds()) <= wait_s + 1):
                # Fire PICKED_UP, attributing to the person from the interaction
                payload = {
                    **interaction_event,
                    "id": str(__import__("uuid").uuid4()),
                    "ts": loss_ts.isoformat(),
                    "event_type": "picked_up",
                    "metadata": {
                        **interaction_event.get("metadata", {}),
                        "object_id": obj_id,
                        "object_name": interaction_event["metadata"].get("object_name"),
                        "object_lost_at": loss_ts.isoformat(),
                        "source_room": interaction_event.get("room"),
                    },
                }
                await self.world.store.append_event(payload)
                await self.bus.publish("world.entity_event", payload)
                return

    async def _check_for_placedown(self, appearance_event: dict, ts: datetime):
        """
        When an object FIRST_SEEN fires (or REAPPEARED in a different room than its
        last known location), check whether a person had hand-overlap activity in
        the same room within the last T_place_seconds. If yes, fire PLACED_DOWN.
        """
        wait_s = self.cfg.get("place_window_seconds", 4)
        room = appearance_event.get("room")
        if not room:
            return
        # Walk recent_interactions backwards looking for any same-room hand activity
        for inter_ts, inter in reversed(list(self.recent_interactions)):
            if (inter.get("room") == room
                and abs((inter_ts - ts).total_seconds()) <= wait_s):
                payload = {
                    **appearance_event,
                    "id": str(__import__("uuid").uuid4()),
                    "ts": ts.isoformat(),
                    "event_type": "placed_down",
                    "metadata": {
                        **appearance_event.get("metadata", {}),
                        "person_id": inter.get("person_id"),
                        "person_name": inter.get("entity_name"),
                        "dest_room": room,
                    },
                }
                await self.world.store.append_event(payload)
                await self.bus.publish("world.entity_event", payload)
                return
```

### 24.4 Edge cases — what makes this hard

The above is the happy path. Real scenes break it in several ways. The defenses:

| Edge case | What goes wrong | Defense |
|---|---|---|
| **Drop without contact** | You drop your phone on the couch. There's no INTERACTED_WITH frame because the phone was never *held* in the camera's view, just released into it. | Allow PLACED_DOWN to fire purely on FIRST_SEEN-near-recent-hand-activity, even without a prior INTERACTED_WITH. The cost is more false positives, accept it. |
| **Occluded handoff** | Cole hands the wallet to Anna without the wallet ever being out of contact with a hand. The wallet might never trigger LOST_VISIBILITY — it just appears next to a different hand. | Track *which person* the INTERACTED_WITH was attributed to; if the next INTERACTED_WITH for the same object is a different person, fire `HANDED_OFF(from=Cole, to=Anna, object=wallet)`. (New event type, adds to the EventType enum.) |
| **Object visible briefly during transit** | Cole carries the wallet from the office to the kitchen. The wallet appears in living_room camera for one frame on the way through. | The wallet entity gets a LOST_VISIBILITY in office, then a brief PRESENT in living_room, then another LOST_VISIBILITY, then FIRST_SEEN in kitchen. The InteractionMonitor sees the kitchen FIRST_SEEN and fires PLACED_DOWN. The transit is recoverable in the event log via timestamp inspection. |
| **Multiple objects in a hand** | Cole picks up phone and wallet at once. Two INTERACTED_WITH events fire, both with the same hand bbox. | This is fine — both PICKED_UP events fire correctly. Each object's LOST_VISIBILITY is independent. |
| **Object placed near, not on, a landmark** | Phone placed on the *arm* of the couch, not the couch surface. The landmark "couch" polygon doesn't include the arm. | The PLACED_DOWN event still fires; the metadata records bbox coordinates, not just landmark name. The LLM, when asked, says *"on the couch — well, technically on the arm of the couch."* |
| **Hand without object overlap, object disappears anyway** | Cole bumps the wallet off the desk; it falls and is no longer detected. No INTERACTED_WITH ever fires. | LOST_VISIBILITY fires for the wallet entity with `reason=in_frame_disappearance`; no PICKED_UP. The wallet just goes IN_ROOM_UNSEEN. Eventually it's found again (by FIRST_SEEN of a wallet-class object in the same or different room) and the user can correlate. Honest failure — the system doesn't know what happened. |
| **Two hands, one object** | Cole holds the phone with both hands. Two hand bboxes overlap one object bbox. | Increment hand_overlap_frames once per *frame*, not once per hand. Otherwise the debounce trips on frame 2 instead of frame 3. |
| **Person bbox encompasses multiple objects, one hand** | Cole is at a cluttered desk. His person bbox contains the phone, the cup, the keyboard. His hand only overlaps the phone. | Already correct — `_classify_interaction` iterates objects whose bbox overlaps the *hand* bbox, not the person bbox. Only the phone fires INTERACTED_WITH. |

### 24.5 Querying interactions

Add to `query_tools.py`:

```python
async def what_did_someone_do_with(self, person_name: str, object_name: str,
                                    hours_ago: int = 24) -> list[dict]:
    """
    'What did Cole do with the wallet?' — returns chronological sequence of
    INTERACTED_WITH, PICKED_UP, PLACED_DOWN events involving both.
    """
    person_ent = self.world.find_entity_by_name(person_name)
    if not person_ent:
        return []
    obj_ent = self.world.find_entity_by_name(object_name)
    obj_id = obj_ent.id if obj_ent else None

    since = datetime.utcnow() - timedelta(hours=hours_ago)
    events = await self.world.store.search_events(
        person_id=person_ent.person_id,
        event_types=["interacted_with", "picked_up", "placed_down"],
        since=since, limit=200,
    )
    if obj_id:
        events = [e for e in events
                  if (e.get("metadata") or {}).get("object_id") == obj_id]
    return list(reversed(events))   # oldest first for narrative phrasing


async def who_last_touched(self, object_name: str) -> dict:
    """
    'Who last touched my wallet?' — find the most recent PICKED_UP / PLACED_DOWN
    for this object, return the person involved.
    """
    obj_ent = self.world.find_entity_by_name(object_name)
    if not obj_ent:
        return {"found": False}
    events = await self.world.store.search_events(
        entity_id=obj_ent.id,
        event_types=["picked_up", "placed_down", "interacted_with"],
        limit=1,
    )
    if not events:
        return {"found": False, "message": f"No interaction events for {object_name}."}
    e = events[0]
    return {
        "found": True,
        "object_name": object_name,
        "event_type": e["event_type"],
        "ts": e["ts"],
        "person_name": (e.get("metadata") or {}).get("person_name"),
        "room": e.get("room"),
    }
```

The LLM uses these to phrase narratives: *"Cole picked up the wallet from the desk at 3:45, walked through the living room, and placed it on the kitchen counter at 3:47."*

### 24.6 Dashboard timeline of interactions

Worth building a small dashboard card that shows recent INTERACTED_WITH / PICKED_UP / PLACED_DOWN events as a chronological list. Useful for:

- Debugging false positives (you can see what the system *thought* you did)
- Manual correction (a "this didn't happen" button that adds a `metadata.user_invalidated=true` flag, which the InteractionMonitor's tuning loop reads to adjust thresholds over time)
- Story value (*"what did I do this morning?"* — a feed of your own interactions)

The card is a SQL query against `world_entity_events` filtered to those three event types, ordered by ts DESC, limit 50, with crop thumbnails from `snapshot_path` if available.

---

## 25. Phase 5: Cross-Day Patterns & Anomalies

This is where the system stops just *recording* and starts noticing. PatternMiner builds behavioral profiles for each resident from the rolling event log; AnomalyScorer scores live events against those profiles and fires `world.anomaly` when something is unusual enough to merit attention.

The reason to build this is grounded: a household assistant that knows *"Anna usually gets home by 6 PM, it's 11 PM and she hasn't"* is meaningfully more useful than one that knows only *"Anna is not home."* The second is data; the first is concern.

### 25.1 What patterns to mine

For each resident (people only — pets get their own behavioral profile via `BehavioralProfileBuilder` in Section 22), the PatternMiner builds:

| Pattern | Source data | Used for |
|---|---|---|
| **Daily arrival distribution** | DEPARTED followed by REAPPEARED, per weekday | "Anna usually gets home by 6 PM on Tuesdays" |
| **Daily departure distribution** | REAPPEARED followed by DEPARTED | "Cole usually leaves between 8:30 and 9:00 AM" |
| **Room occupancy by hour** | All time-stamped PRESENT-state events, per weekday-hour bucket | "Cole is in the office from 9 AM to noon on weekdays" |
| **Co-presence frequencies** | Pairs of residents PRESENT in same room | "Cole and Anna eat dinner together at 7 PM most evenings" |
| **Routine sequences** | First-seen-room-of-day, sequence of MOVED_TO events through morning | "Cole's morning is: bedroom → kitchen → office, in that order" |
| **Long-stay durations per room** | STATIONARY_LONG events bucketed by room | "Cole is stationary in the office in 3-4 hour blocks; 30+ min stationary in the bedroom is unusual" |

These all live as histograms — not single numbers but distributions, because anomaly scoring needs the full shape (KL-divergence-style) rather than just the mean.

### 25.2 PatternMiner — full implementation

Storage: profiles live in `entity.metadata['pattern_profile']` for now. If they grow large or get queried independently of the entity row, split into a `world_pattern_profiles` table later.

```python
# modules/world_model/patterns.py
"""
PatternMiner: nightly job that builds behavioral profiles for each resident
from the rolling 30-day event log.
"""

import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any
from .types import EventType
from loguru import logger


class PatternMiner:
    def __init__(self, world, days_back: int = 30):
        self.world = world
        self.days_back = days_back

    async def run_nightly(self):
        async with self.world._lock:
            residents = [e for e in self.world.entities.values()
                         if e.is_resident and e.entity_type == "person" and e.person_id]
        for ent in residents:
            try:
                profile = await self._build_profile_for(ent)
                ent.metadata["pattern_profile"] = profile
                ent.metadata["pattern_profile_updated_ts"] = datetime.utcnow().isoformat()
                await self.world.store.upsert_entity(ent)
                logger.info(f"[PatternMiner] rebuilt profile for {ent.display_name}: "
                            f"{profile.get('n_events', 0)} events analyzed")
            except Exception:
                logger.exception(f"[PatternMiner] profile build failed for {ent.display_name}")

    async def _build_profile_for(self, ent) -> dict[str, Any]:
        since = datetime.utcnow() - timedelta(days=self.days_back)
        events = await self.world.store.search_events(
            person_id=ent.person_id, since=since, limit=100000,
        )
        if not events:
            return {"n_events": 0, "window_start": since.isoformat()}

        return {
            "n_events": len(events),
            "window_start": since.isoformat(),
            "window_end": datetime.utcnow().isoformat(),
            "arrival_by_weekday": self._arrival_distribution(events),
            "departure_by_weekday": self._departure_distribution(events),
            "room_by_weekday_hour": self._room_by_weekday_hour(events),
            "co_presence": await self._co_presence(ent, events, since),
            "morning_routine": self._morning_routine(events),
            "long_stays_by_room": self._long_stays_by_room(events),
            "weekly_active_hours": self._weekly_active_hours(events),
        }

    # ----- arrivals / departures -----

    def _arrival_distribution(self, events: list[dict]) -> dict[int, dict[int, int]]:
        """
        Returns {weekday: {hour: count}} for REAPPEARED-from-DEPARTED events.
        I.e., when does this person come home, broken out by weekday.
        """
        out: dict[int, dict[int, int]] = {wd: defaultdict(int) for wd in range(7)}
        prev_state = None
        for e in sorted(events, key=lambda x: x["ts"]):
            cur = e.get("state")
            if (prev_state == "departed"
                and e["event_type"] == "reappeared"):
                t = datetime.fromisoformat(e["ts"])
                out[t.weekday()][t.hour] += 1
            prev_state = cur
        return {wd: dict(d) for wd, d in out.items()}

    def _departure_distribution(self, events: list[dict]) -> dict[int, dict[int, int]]:
        out: dict[int, dict[int, int]] = {wd: defaultdict(int) for wd in range(7)}
        for e in events:
            if e["event_type"] == "departed":
                t = datetime.fromisoformat(e["ts"])
                out[t.weekday()][t.hour] += 1
        return {wd: dict(d) for wd, d in out.items()}

    # ----- room-by-time -----

    def _room_by_weekday_hour(self, events: list[dict]) -> dict:
        """
        Returns {weekday: {hour: {room: probability}}}.
        Counts how often this person was observed in each room per
        weekday-hour bucket, normalizes per-bucket.
        """
        bucket: dict[int, dict[int, dict[str, int]]] = {
            wd: {h: defaultdict(int) for h in range(24)} for wd in range(7)
        }
        for e in events:
            r = e.get("room")
            if not r:
                continue
            t = datetime.fromisoformat(e["ts"])
            bucket[t.weekday()][t.hour][r] += 1
        out = {}
        for wd, by_hour in bucket.items():
            out[wd] = {}
            for h, counts in by_hour.items():
                total = sum(counts.values()) or 1
                out[wd][h] = {r: c / total for r, c in counts.items()}
        return out

    # ----- co-presence -----

    async def _co_presence(self, ent, my_events, since) -> dict[str, float]:
        """
        For each other resident, count overlapping PRESENT windows. Returns
        {other_name: fraction_of_my_present_time_overlapping_with_them}.
        """
        my_present_pairs = []
        for e in my_events:
            if e["event_type"] in ("reappeared", "moved_to", "first_seen"):
                my_present_pairs.append((datetime.fromisoformat(e["ts"]),
                                          e.get("room")))
        if not my_present_pairs:
            return {}
        out: dict[str, int] = defaultdict(int)
        for ts, room in my_present_pairs:
            window = await self.world.store.search_events(
                room=room,
                event_types=["reappeared", "moved_to", "first_seen"],
                since=ts - timedelta(minutes=2),
                until=ts + timedelta(minutes=2),
                limit=50,
            )
            for w in window:
                if (w["entity_type"] == "person" and w["entity_id"] != ent.id
                        and w.get("entity_name")):
                    out[w["entity_name"]] += 1
        n = len(my_present_pairs)
        return {name: c / n for name, c in out.items()}

    # ----- morning routine -----

    def _morning_routine(self, events: list[dict]) -> dict[str, Any]:
        """
        Find the typical sequence of rooms visited between first-presence-of-day
        and ~3 hours later. Returns the most common ordered sequence.
        """
        by_day: dict[str, list[tuple[datetime, str]]] = defaultdict(list)
        for e in events:
            if e["event_type"] in ("reappeared", "moved_to", "first_seen"):
                t = datetime.fromisoformat(e["ts"])
                day_key = t.strftime("%Y-%m-%d")
                if e.get("room"):
                    by_day[day_key].append((t, e["room"]))
        sequences: list[tuple[str, ...]] = []
        for day, entries in by_day.items():
            entries.sort(key=lambda x: x[0])
            if not entries:
                continue
            first = entries[0][0]
            cutoff = first + timedelta(hours=3)
            seq = []
            seen_rooms = set()
            for t, r in entries:
                if t > cutoff:
                    break
                if r not in seen_rooms:
                    seq.append(r)
                    seen_rooms.add(r)
            if len(seq) >= 2:
                sequences.append(tuple(seq))
        if not sequences:
            return {}
        # Most common sequence
        from collections import Counter
        most_common = Counter(sequences).most_common(3)
        return {
            "most_common_sequences": [
                {"sequence": list(s), "count": c} for s, c in most_common
            ],
            "n_days_analyzed": len(by_day),
        }

    def _long_stays_by_room(self, events: list[dict]) -> dict[str, dict[str, float]]:
        per_room: dict[str, list[float]] = defaultdict(list)
        for e in events:
            if e["event_type"] == "stationary_long":
                r = e.get("room")
                # stationary_long fires every minute it's still stationary; we want
                # the duration accumulated. Approximation: one entry per stationary
                # episode = T_stationary + duration, but the event log captures the
                # firing point. Better: compute from state-change-ts pairs in a
                # follow-up pass. For now, just count occurrences per room.
                if r:
                    per_room[r].append(1.0)
        return {
            r: {"n": len(samples), "rate_per_day": len(samples) / 30.0}
            for r, samples in per_room.items()
        }

    def _weekly_active_hours(self, events: list[dict]) -> dict[int, set]:
        """Set of hours per weekday in which this person was observed at all."""
        out: dict[int, set] = {wd: set() for wd in range(7)}
        for e in events:
            t = datetime.fromisoformat(e["ts"])
            out[t.weekday()].add(t.hour)
        return {wd: sorted(list(s)) for wd, s in out.items()}
```

### 25.3 AnomalyScorer — full implementation

This is the live-scoring part. Subscribes to `world.entity_event`, scores each event against the entity's behavioral profile, fires `world.anomaly` when the score exceeds threshold. The score is a weighted combination of components, each comparing the live event's properties to the profile's distribution.

```python
# modules/world_model/anomaly.py
"""
AnomalyScorer: subscribes to world.entity_event, scores live events against
the entity's pattern_profile, publishes world.anomaly when over threshold.
"""

import math
from datetime import datetime, timedelta
from collections import deque
from loguru import logger


class AnomalyScorer:
    def __init__(self, bus, world, config: dict):
        self.bus = bus
        self.world = world
        self.cfg = config
        self.threshold = config.get("anomaly_threshold", 6.0)
        self.cooldown_seconds = config.get("anomaly_cooldown_seconds", 600)
        self.min_history_days = config.get("min_history_days", 14)
        # Per-entity cooldown timestamps
        self._last_alert_ts: dict[str, datetime] = {}

    async def start(self):
        await self.bus.subscribe("world.entity_event", self._on_event)
        logger.info("[AnomalyScorer] started")

    async def _on_event(self, event: dict):
        ent = self.world.entities.get(event["entity_id"])
        if not ent or not ent.is_resident or ent.entity_type != "person":
            return

        profile = ent.metadata.get("pattern_profile")
        if not profile or profile.get("n_events", 0) == 0:
            return

        # Require enough history before we trust the profile
        window_start = profile.get("window_start")
        if window_start:
            wstart = datetime.fromisoformat(window_start)
            if datetime.utcnow() - wstart < timedelta(days=self.min_history_days):
                return

        # Compute score
        score, components = self._score(event, ent, profile)
        if score < self.threshold:
            return

        # Cooldown
        last = self._last_alert_ts.get(ent.id)
        if last and (datetime.utcnow() - last).total_seconds() < self.cooldown_seconds:
            return
        self._last_alert_ts[ent.id] = datetime.utcnow()

        await self.bus.publish("world.anomaly", {
            "entity_id": ent.id,
            "entity_name": ent.display_name,
            "event": event,
            "score": score,
            "components": components,
            "ts": event["ts"],
        })
        logger.info(f"[AnomalyScorer] anomaly for {ent.display_name}: "
                    f"score={score:.2f} ({components})")

    def _score(self, event: dict, ent, profile: dict) -> tuple[float, dict]:
        """
        Compute anomaly score from multiple components. Each component is
        roughly 0–10, weighted, summed.
        """
        components = {}

        ts = datetime.fromisoformat(event["ts"])
        weekday = ts.weekday()
        hour = ts.hour
        room = event.get("room")
        et = event["event_type"]

        # ---- Time-of-day component ----
        # How unusual is this hour for any activity at all?
        active_hours = profile.get("weekly_active_hours", {}).get(weekday, [])
        if active_hours and hour not in active_hours:
            # Find distance to nearest active hour
            distances = [min(abs(hour - h), 24 - abs(hour - h)) for h in active_hours]
            time_score = float(min(min(distances) * 2.0, 10.0))
        else:
            time_score = 0.0
        components["time_of_day"] = time_score

        # ---- Room-given-time component ----
        # How unusual is this room at this weekday-hour?
        room_dist = (profile.get("room_by_weekday_hour", {})
                     .get(weekday, {}).get(hour, {}))
        if room and room_dist:
            p_room = room_dist.get(room, 0.0)
            if p_room > 0:
                # Surprise: -log(p)
                room_score = float(min(-math.log(p_room + 0.01), 8.0))
            else:
                # Never seen here at this time — high surprise
                room_score = 8.0
        else:
            room_score = 0.0
        components["room_at_time"] = room_score

        # ---- Departure/arrival lateness component ----
        if et == "departed":
            dep_dist = profile.get("departure_by_weekday", {}).get(weekday, {})
            arr_score = self._distribution_outlier_score(hour, dep_dist)
            components["departure_time"] = arr_score
        elif et == "reappeared":
            arr_dist = profile.get("arrival_by_weekday", {}).get(weekday, {})
            arr_score = self._distribution_outlier_score(hour, arr_dist)
            components["arrival_time"] = arr_score
        else:
            components["arrival_time"] = 0.0

        # ---- Combine ----
        weights = {
            "time_of_day": 0.30,
            "room_at_time": 0.45,
            "arrival_time": 0.15,
            "departure_time": 0.15,
        }
        score = sum(weights.get(k, 0.0) * v for k, v in components.items())
        return score, components

    def _distribution_outlier_score(self, value: int, dist: dict[int, int]) -> float:
        """Score an integer hour against a histogram distribution (hour → count)."""
        if not dist:
            return 0.0
        total = sum(dist.values()) or 1
        p = dist.get(value, 0) / total
        if p == 0:
            # Hour never observed in this distribution
            # Find distance to nearest observed hour
            observed = sorted(dist.keys())
            distances = [min(abs(value - h), 24 - abs(value - h)) for h in observed]
            return float(min(min(distances) * 1.5, 8.0))
        return float(min(-math.log(p + 0.01), 6.0))
```

### 25.4 Anomaly handling — the user feedback loop

A scorer that fires every time you do something slightly unusual is a scorer that gets ignored. Two protections:

**Cooldown.** Per-entity cooldown of 10 minutes default — after firing an anomaly for Cole, suppress further Cole-anomalies for 10 minutes regardless of score. Keeps the dashboard from spamming when Cole is mid-unusual-day (e.g., on vacation, doing a bunch of unusual stuff).

**Severity gating.** Threshold defaults to 6.0 on the 0–10 scale. *Tunable up if you find the dashboard flooded; tunable down if it never fires.* Tune over the first 2 weeks of operation against the false-positive rate you observe.

**User feedback loop.** Every anomaly published to the dashboard has a "this isn't actually unusual" button. Clicking it writes a `world.anomaly_invalidated` event with the original event's id and the user's reason (free-text). A nightly job reads invalidations from the last week and adjusts `anomaly_threshold` upward if the FP rate exceeds 30%. Implementation:

```python
# modules/world_model/anomaly.py — add tuning method

async def auto_tune(self, days_back: int = 7):
    """
    Look at recent anomalies and their invalidation rate.
    If FP rate > 0.3, raise threshold by 0.5. If FP rate < 0.05 and we had ≥10
    anomalies in the window, lower threshold by 0.25. Otherwise leave it alone.
    """
    since = datetime.utcnow() - timedelta(days=days_back)
    anomalies = await self.world.store.search_events(
        event_types=["world.anomaly"], since=since, limit=10000,
    )
    invalidations = await self.world.store.search_events(
        event_types=["world.anomaly_invalidated"], since=since, limit=10000,
    )
    if len(anomalies) < 10:
        return
    fp_rate = len(invalidations) / max(len(anomalies), 1)
    if fp_rate > 0.3:
        self.threshold += 0.5
        logger.info(f"[AnomalyScorer] auto-tune: FP rate {fp_rate:.2f}, "
                    f"raising threshold to {self.threshold:.2f}")
    elif fp_rate < 0.05 and len(anomalies) >= 10:
        self.threshold = max(self.threshold - 0.25, 3.0)
        logger.info(f"[AnomalyScorer] auto-tune: FP rate {fp_rate:.2f}, "
                    f"lowering threshold to {self.threshold:.2f}")
```

### 25.5 Examples — what the system should and shouldn't notice

**Should notice (true positives):**

- Cole walks into kitchen at 3 AM when his weekly_active_hours[2] (Tuesday) doesn't include 3.
- Anna comes home at 11 PM when her arrival_by_weekday[4] (Friday) is concentrated 17:00–18:30.
- A new person FIRST_SEEN during work hours when nobody's normally home.
- Cole is in the bedroom at 2 PM on a workday when the weekday-hour distribution heavily favors office.
- Cole's morning routine deviates significantly from the typical sequence (e.g., goes straight from bedroom to office, skipping the usual kitchen stop).

**Should NOT notice (avoid these false positives):**

- Cole is in a slightly unusual room at a usual time (e.g., kitchen at 11 AM when 11 AM is usually office) — modest anomaly score, probably below threshold.
- New residents (children growing into a recognizable face) building up profile from scratch — `min_history_days` gate prevents premature scoring.
- Vacation returns (everything looks unusual after 2 weeks away) — cooldown prevents spam.
- Cat events — `is_resident` gate plus `entity_type == "person"` explicitly excludes cats from anomaly scoring (cats get their own behavioral_profile but no scorer for now).

### 25.6 Cross-correlation patterns (deferred to "if you want it later")

The mining can go further: pair-correlations like *"when Cole is gaming, Anna is usually in the living_room"*, sequence-correlations like *"the front door opens between 5–6 PM and within 2 minutes Anna is in the kitchen"*. These are interesting but cost real engineering time to do well, and the AnomalyScorer above gets you 80% of the value without them. Defer until the simpler version is running and you've identified specific cross-patterns worth modeling.

### 25.7 Dashboard cards

Three new cards once Phase 5b is running:

- **Behavioral profile visualization** — heatmap of room × weekday-hour for each resident, showing where they typically are. Useful for sanity-checking the profile builder.
- **Anomaly review queue** — list of recent anomalies, highest-score first, each with the contributing components and a "this isn't unusual" button (which produces the invalidation event for auto-tuning).
- **Pattern Miner status** — last run timestamp, n_events processed per resident, profile freshness.

---

## 26. Build Order with Verification Gates

Each step has an explicit pass/fail gate. Don't move forward until the previous step is verified.

### Phase 1 — World Model spine + bounded house state machine (5–7 days)

| Step | Files | Verify |
|---|---|---|
| 1.1 Schema + types | `types.py`, `geometry.py`, `store.py`, schema additions in `database.py` | DB migrations run cleanly on a fresh + on existing DB. `WorldStore.upsert_entity` round-trips. |
| 1.2 ArcFace migration | `face_recognizer.py` rewrite, IdentityManager extended for 512-dim centroids, `model_version` column added | Manual face enrollment test: enroll Cole + Anna with 5 photos each. `identify_face` returns correct person_id with confidence ≥0.6 on test crops. Margin gating refuses to guess between Cole + Anna on ambiguous test images. |
| 1.3 ObservationBuilder (people only) | `observation_builder.py`, integrated with existing object_detector + face_recognizer + IdentityManager | Frame-rate test: 5fps room produces ~5 obs/s on bus. Person obs include person_id when face is recognized. |
| 1.4 WorldModel core | `world_model.py`, all delta classifiers, exit/landmark logic | Synthetic test scripts (Section 19) all pass. |
| 1.5 Camera health | health subscription + suspension logic, plus CameraManager publishing `camera.health` | Synthetic test: simulate camera_health=down event during PRESENT state; entity stays PRESENT, doesn't transition to IN_ROOM_UNSEEN. |
| 1.6 Wire into orchestrator | `core/orchestrator.py` instantiates WorldModel + ObservationBuilder, subscribes them to bus | Boot Jarvis with one real camera (office). Walk in, walk out toward door, walk back. Event log shows MOVED_WITHIN_ROOM, LOST_VISIBILITY (near_exit or in_frame), REAPPEARED. |
| 1.7 Verify under-desk | live test in office | Crouch under desk. Event log shows IN_ROOM_UNSEEN with last_landmark=under_desk and reason=in_frame_disappearance. State stays for at least 10 minutes. |

**Gate to Phase 2:** under-desk and in-frame-doorway scenarios produce correct state in live single-camera testing. Ask Jarvis "where am I?" — it answers from world state correctly even after you crouch under the desk.

### Phase 2 — Multi-room + topology (3–5 days)

| Step | Files | Verify |
|---|---|---|
| 2.1 Annotate other rooms | `config.yaml` exits + landmarks for living_room, kitchen, bedroom, laundry_room | Polygons render correctly on dashboard polygon viewer (build that small page first if you haven't). |
| 2.2 Exterior exits | `exterior_exit` polygons on rooms with outside doors (front_door on living_room, back_door on kitchen if applicable) | Walk from living_room toward front door. Last detection over exterior_exit polygon → DEPARTED state. Walk back in → REAPPEARED. |
| 2.3 Unmonitored zones | `to_unmonitored_zone` polygons drawn on the rooms whose cameras can see the unmonitored bedroom doorways (best-effort — your two unmonitored bedrooms don't have direct camera coverage, so these polygons are coarse) | Walk into a guest bedroom from living_room. After T_handoff, state is IN_HOUSE_UNMONITORED. Walk back out → REAPPEARED + state = PRESENT. If polygon coverage is missing, fallback is IN_ROOM_UNSEEN — also acceptable. |
| 2.4 Room handoffs | neighbor relationships derived from `to_room` exits | Walk office → living_room → kitchen. Three MOVED_TO events with correct from_room. |

**Gate to Phase 3:** all four state-machine paths (PRESENT, IN_ROOM_UNSEEN, IN_HOUSE_UNMONITORED, DEPARTED) demonstrate correctly in live testing across at least 3 cameras. The fallback path (IN_ROOM_UNSEEN when no exit polygon matches) also demonstrates correctly when you walk into one of the unmonitored bedrooms from a region that lacks `to_unmonitored_zone` coverage.

### Phase 3 — LLM tools + persona alignment (2–3 days)

| Step | Files | Verify |
|---|---|---|
| 3.1 query_tools.py (people-only) | implement `get_entity_status`, `list_entities_in_room`, `who_is_home`, `search_recent_events` | Unit tests pass against live world model. |
| 3.2 Tool registration | hook the four tools into whatever tool-calling layer the orchestrator already uses for the persona/LLM (the existing repo has a tool registry — register the world tools alongside it). Define JSON schemas for each tool's args and return type. | Conversation harness: ask the LLM "is anyone home?" → it picks `who_is_home`, returns the resident list. Ask "where's Cole?" → it picks `get_entity_status` with `entity_name="Cole"`. The tool calls succeed end-to-end and the response uses the data. |
| 3.3 Persona context augmentation | extend the persona prompt builder to inject a small "world snapshot" prefix before the user turn — top-N currently-PRESENT entities with rooms, optionally last 3 change-events. Token budget: ≤200 tokens for the snapshot. | Token-count test: snapshot stays under budget for 5 residents + 5 cats. Manual test: persona answers "what's going on right now?" without needing to call a tool — it pulls from the snapshot directly. Verify the snapshot is *deduplicated against tool-call results* so the LLM doesn't see the same fact twice. |
| 3.4 Polygon viewer dashboard page | small page that takes a room id and renders camera frame + exit/landmark polygons overlay. Uses existing dashboard auth + camera frame endpoint. | Visual: pull up office page, see `under_desk` and `door_to_hallway` polygons drawn on a recent frame. Use this same page in Phase 2 to draw the other rooms' polygons; building it now pays for itself fast. |

**Gate to Phase 4:** the demo target works. Anna asks *"where's Cole?"* through the persona; Cole is hiding under the desk. The persona, using either the world-snapshot prefix or a tool call, answers correctly. Specifically: it does **not** say "I don't know," it does **not** hallucinate a location, and it includes the *last seen* timestamp + landmark. Run the demo three times in different conditions (Cole present, Cole genuinely DEPARTED, Cole IN_HOUSE_UNMONITORED) to verify all three branches read naturally.

### Phase 4 — Pets + objects (4–6 days)

This phase is *optional for the boss demo* but high-value for the long-running system. Skip if the deadline is tight; come back after the demo.

| Step | Files | Verify |
|---|---|---|
| 4.1 Cat enrollment scaffolding | extend `IdentityManager` with `cat_samples` table mirror, `enroll_cat(name, image_crops)` method. Per Section 22, store color/pattern descriptors as the primary identity signal — no cat ArcFace, just a coarse color histogram + size-bin classifier. | Manual enroll: 5 photos per cat × 5 cats. Each cat gets a `persons.id`-equivalent in a `cats` table (or `persons` row with `entity_type='cat'` if you collapsed the schema; Section 22 picks one — be consistent). Re-id test: pass 10 held-out cat photos, ≥70% correct match on top-1. |
| 4.2 Cat observations through ObservationBuilder | YOLOv8 already detects `cat`. Add the cat-recognition path: detection → crop → color/size descriptor → IdentityManager.identify_cat → world observation with `entity_type='cat'`. | One cat in office: WorldModel emits `cat.entered_room` events with the right name. Two cats in same frame: associations don't swap. |
| 4.3 Cat-specific delta heuristics | cats teleport more than people (jumping, climbing). Loosen `T_handoff` for cats and skip the "near exit before disappearance" requirement — a cat can vanish behind furniture without that being suspicious. Configurable per-`entity_type`. | Synthetic test: cat detection gap of 30s with no near-exit observation. Person → IN_ROOM_UNSEEN with reason=`in_frame_disappearance`. Cat → IN_ROOM_UNSEEN with reason=`cat_typical_concealment` and a longer timeout before any anomaly fires. |
| 4.4 Object detection (low-priority, lower confidence) | per Section 23, only persistent named objects get tracked (e.g., `cole_phone`, `keys`). Initial set is hand-enrolled by photo + label. Use YOLOv8 + a simple per-object color/shape classifier; do **not** try to re-id arbitrary objects. | Place phone on the desk, leave room. Object state = PRESENT, room=office, last_seen recent. Move phone to bedroom. Both `object.left_room` (office) and `object.entered_room` (bedroom) fire within `T_handoff`. |
| 4.5 Object query tools | `find_object(name)` returning current room + last_seen + confidence. Confidence floor is lower than for people — be explicit in the response (`"I think your keys are in the kitchen, but I'm only ~60% sure — last positive ID was 25 minutes ago."`). | Ask the persona "where are my keys?" → tool fires, persona reads back the location with appropriate hedging. Move keys, ask again, persona reflects new room. |

**Gate to Phase 5:** at least one cat re-id'd correctly across rooms over a full day of operation; at least one object reliably queryable through the persona; no false-positive cat anomalies in the dashboard. *If you've blown the deadline budget, ship Phase 3 and do Phase 4 post-demo — the demo target is people, not cats and keys.*

### Phase 5 — Interactions + cross-day patterns (5–7 days)

This phase requires real history to be meaningful. Don't start until you have at least 7 days of Phase 1–3 data accumulated. The pattern miner needs density; running it against 24 hours of events produces garbage profiles and trains your gut to distrust the system.

| Step | Files | Verify |
|---|---|---|
| 5.1 Interaction inference | `modules/world_model/interactions.py` per Section 24. Subscribes to `entity.state_changed` and `world.observation`, runs the proximity-and-duration windowed reducer, emits `world.interaction` events. | Synthetic: two PRESENT entities in the same room ≥`min_interaction_seconds` (default 60s) → one interaction event with both entity_ids and a duration. Three entities in the same room produce three pairwise interactions, not one trio (or one trio if you implemented the n-ary version — be consistent with Section 24). |
| 5.2 Interaction storage | `world_interactions` table; FK to both entities; indexed on (entity_a, entity_b, started_at). | Round-trip test: insert, query by entity, query by date range. Pagination works at 10k+ rows. |
| 5.3 Pattern miner (offline batch) | `modules/world_model/pattern_miner.py` — nightly job that builds `behavioral_profile` rows per resident. Per Section 25, profile shape = weekly_active_hours[7][24] + per-room time distribution + arrival/departure histograms. Built from `world_entity_events` and `world_interactions`. | Run against 7 days of synthetic data with a known pattern (Cole in office 9–17 weekdays). Resulting profile shows the expected histogram concentration. Diff between successive runs is small for stable behavior, large after a real schedule change. |
| 5.4 Anomaly scorer | `modules/world_model/anomaly.py` — listens to `entity.state_changed`, scores against the resident's `behavioral_profile`, publishes `world.anomaly` if score ≥ threshold and not in cooldown. `auto_tune` runs nightly per Section 25.4. | Synthetic: Cole event at 3 AM kitchen against a profile that says Cole is never in the kitchen at 3 AM → score ≥6, anomaly fires. Same event with 7-min cooldown still active → suppressed. Invalidation feedback loop: invalidate 4 of 10 anomalies → next nightly auto-tune raises threshold by 0.5. |
| 5.5 Dashboard cards | three cards per Section 25.7: behavioral profile heatmap, anomaly review queue with invalidate button, pattern miner status. | Cards render with real data. Heatmap colors map to expected concentration. Invalidate button writes the right event type to the bus. Refresh propagates. |
| 5.6 Persona augmentation | optional: include the *last unhandled high-severity anomaly* in the persona context (one line, max 80 tokens) — so if the persona is asked "anything weird going on?" it has the answer. | Conversation test: trigger an anomaly via synthetic event. Within 60s, ask the persona an open question; if context injection is enabled, a careful follow-up like "by the way, anything unusual?" gets a substantive answer. |

**Final gate (system done):** 14 consecutive days of operation in which (a) the system answers location queries correctly ≥95% of the time as judged by you and Anna, (b) the anomaly dashboard fires no more than 2 false positives per week after auto-tuning settles, (c) no manual identity correction has been required for ≥7 days. At this point you have a real, working ambient home AI — not a demo.

---

## 27. Failure Modes & Defenses

A consolidated catalog of the realistic failure modes, ranked by likelihood × blast radius. Each entry has the failure, the symptom you'd actually observe, and the concrete defense. *None of these are theoretical* — every one of them either has bitten a similar system in the wild or is one config typo away from biting yours.

### 27.1 Identity collision — two people compress into one entity

**Failure.** ArcFace embeddings for two genuinely different people land within the recognition threshold (siblings, twins, lighting-similar strangers). The system picks one `person_id` and assigns *both* humans' observations to it. Cole gets a worldline that's actually Cole + his cousin who's visiting.

**Symptom.** `entity.state_changed` fires room transitions that don't match physics — Cole "teleports" from office to living_room without an intermediate transition because both people were observed simultaneously. Audit log shows the same `person_id` in two cameras at the same timestamp.

**Defense.** Three layers, in order of cost:
1. **Margin gating in IdentityManager** (Section 11) — if the top-1 ArcFace match is within 0.05 cosine distance of the top-2, refuse to identify. Tag the observation as `person_id=NULL, suggested=[id_a, id_b]`. WorldModel treats it as an unidentified person and accumulates observations until the margin clears.
2. **Co-presence detection in WorldModel** — if `entity.observed` events for the same `person_id` appear in two non-adjacent rooms within `T_handoff`, raise an `identity.collision_suspected` event and force the next observation through the verification path (voice cross-check, multi-frame averaging).
3. **Manual reconciliation tool** in the dashboard — show co-presence events grouped by suspected `person_id`; let you split them with one click. The split rewrites the affected `world_entities` rows and republishes corrected events.

### 27.2 Auto-enrollment poisoning

**Failure.** A misclassification gets used as a positive sample for a known person. Over weeks, the centroid drifts toward someone else. Eventually the system stops recognizing the *real* person.

**Symptom.** Recognition confidence for Cole steadily drops over weeks despite no visual change. Or worse, Cole-the-person starts being assigned to Anna's `person_id` because Anna's centroid drifted toward Cole.

**Defense.**
1. **Diversity-bounded sample replacement** (already in Section 10) — new auto-enrolled samples must increase intra-cluster diversity, not collapse it. Reject samples that are within 0.02 cosine of the cluster mean (low information).
2. **Confidence floor on auto-enrollment** — `auto_enroll_confidence_threshold = 0.85` (vs `recognition_threshold = 0.6`). Only *very confident* observations feed the enrollment pipeline. Tunable per-person if a specific resident's appearance is unusually variable.
3. **Centroid drift watchdog** — store a snapshot of every centroid weekly. If a centroid moves more than `drift_threshold = 0.15` cosine in any 7-day window, freeze auto-enrollment for that person and surface a dashboard alert. Manually verify with a fresh enrollment photo.
4. **Versioned model migration** — when ArcFace itself is upgraded, do **not** carry centroids forward. Every face in `face_samples` is re-embedded under the new model version, and centroids are recomputed from scratch. The `model_version` column on `face_samples` makes this a one-script migration.

### 27.3 Topology config drift

**Failure.** You move your desk to the other side of the office. The `under_desk` polygon now sits over empty floor. Cole's "under-desk hiding" demo silently breaks because the landmark's coordinates no longer match reality.

**Symptom.** Events tag `last_landmark` values that don't make sense given where Cole actually was. Or, more subtly: the system stops emitting `near_exit_at_disappearance` reasons and falls back to `in_frame_disappearance` for everyone, even people who clearly walked through doors.

**Defense.**
1. **Polygon viewer dashboard page is a first-class debugging tool** — Phase 3 step 3.4 isn't just for setup; you should pull it up the first time *anything* feels off. Visual mismatch between polygon and reality jumps out immediately.
2. **Camera-resolution invariance** — store polygons in *normalized coordinates* (0–1 range). When the camera resolution changes (Wyze firmware update, ESP32-CAM hardware swap), polygons follow without rewriting config.
3. **Polygon-coverage fraction monitoring** — nightly job computes `(events_with_resolved_landmark / total_events_with_position) per camera`. If a camera's coverage drops below 80% week-over-week, surface a dashboard warning. The polygons either drifted or the camera moved.
4. **Landmark "smoke test" probe** in `world_model.health()` — emit a synthetic observation centered on each declared landmark; verify the resolver returns the expected name. Run on boot. If your `under_desk` polygon now resolves to nothing, you find out at startup, not when Anna is asking where Cole is.

### 27.4 Camera health degraded but not down

**Failure.** A camera's frame-rate drops to 0.5 fps because the Wyze RTSP stream is throttling. Detections still arrive — but with massive temporal gaps. The state machine misinterprets the gaps as `IN_ROOM_UNSEEN` even though the person is plainly walking around.

**Symptom.** A camera's room generates an unusual ratio of `IN_ROOM_UNSEEN → REAPPEARED → IN_ROOM_UNSEEN` cycles, particularly during periods you know you were in the room. Effective FPS at the observation level is much lower than the configured FPS.

**Defense.**
1. **Effective-FPS calculation in CameraManager** — running average of detection inter-arrival times, exposed as `camera.health.fps_effective`. WorldModel suspends `T_lost_visibility` countdowns when `fps_effective < 0.5 × fps_configured`.
2. **Health-degraded events are first-class** (Section 14) — `camera.health` events with `state=degraded` (in addition to up/down) get the same suspension treatment as `state=down`, with a smaller grace period.
3. **Observation-rate alarm** — dashboard card listing each camera's effective vs configured FPS over the last hour. Wyze quirks become visible in 5 seconds.

### 27.5 Bus event reordering / out-of-order observations

**Failure.** Your async event bus delivers `entity.observed` events out of timestamp order — typically because two cameras emit observations at the same wall-clock instant and the bus serializes them in arrival order, which differs from observation order.

**Symptom.** Brief "rewind" events: an entity transitions PRESENT(office) → PRESENT(living_room) → PRESENT(office) within 200ms because the office observation arrived after the living_room one despite being earlier. Anna's history page looks like she has Parkinson's.

**Defense.**
1. **All world model logic uses `observation.timestamp`, never `event.received_at`.** Reordering at the bus layer is then irrelevant — the WorldModel sorts by the observation timestamp before evaluating the state machine.
2. **Small reorder window** in WorldModel — buffer observations for `reorder_window_ms = 250` before evaluating. Trades 250ms of demo latency for correctness. The buffer flushes synchronously when an entity is queried directly (so `get_entity_status` is still real-time), only state-machine evaluation waits.
3. **Late-arriving observation handling** — if an observation arrives more than `reorder_window_ms` after `now()`, log a `world.late_observation` event with the lateness in ms; if these are common, raise the window. Don't try to retroactively rewrite state — that way lies madness.

### 27.6 Database lock contention

**Failure.** Phase 5's pattern miner kicks off at 2 AM and holds a write lock on `world_entity_events` while scanning a week of data. ObservationBuilder is still running during the night (someone's awake) and writes pile up behind the lock. The bus backlogs, observations get dropped, the world model loses state continuity.

**Symptom.** Bus queue depth alerts at night. Gaps in `world_entity_events` for the early-morning hours. Pattern miner runs eventually complete but produce profiles that make 4 AM look like the dead zone (because that's when the system was choking, not when residents were inactive).

**Defense.**
1. **Pattern miner uses a read-replica or a snapshot copy.** SQLite's snapshot model (with `aiosqlite` and proper `journal_mode=WAL`) makes concurrent readers cheap. Make sure WAL mode is on and verify with `PRAGMA journal_mode;` after migration.
2. **Pattern miner runs in chunks** — process one resident at a time, commit between residents, never hold a long-running transaction.
3. **Bus backpressure visibility** — `bus.queue_depth` metric in the dashboard. If you see it spike at night, that's your signal to investigate the miner *before* you start losing observations.

### 27.7 Persona token bloat

**Failure.** Phase 3.3 injects world snapshot context into every persona turn. Over time, well-meaning additions accumulate: snapshot + last 3 events + last anomaly + behavioral note + cat status + unread notifications. The persona's first-token-budget for *user input* shrinks until the LLM starts truncating actual conversation history.

**Symptom.** The persona forgets things from earlier in the same conversation. Long replies feel cut off. Token usage per call climbs week over week without features being added.

**Defense.**
1. **Hard token budget on the world snapshot** — `world_snapshot_max_tokens = 200`. If the snapshot exceeds it, drop the lowest-priority lines (events older than 10 minutes go first, then non-resident entities, then keep only the 3 most-PRESENT residents).
2. **Snapshot-vs-tool-call deduplication** — when the persona calls `get_entity_status`, the snapshot for that entity is suppressed from the next turn's prefix. No fact appears twice in the prompt.
3. **Per-feature opt-in flags in config** — `persona.world_snapshot.include_anomalies: true|false` and so on. New features default to `false`; you have to explicitly turn them on, which forces a token-budget thought before flipping the bit.

### 27.8 Config-error class — the silent bad default

**Failure.** A new room is added to `config.yaml` but its `exits:` block is omitted. The room loads fine. Detections in it work fine. But every disappearance from that room falls into the `in_frame_disappearance` bucket because no `to_room` polygon exists, and the state machine never produces `MOVED_TO` events out of that room.

**Symptom.** One specific room's residents seem to vanish frequently. Heatmaps show the room as a "black hole" — entries happen, exits never do.

**Defense.**
1. **Schema validation on boot** — every room must have at least one `exit` polygon, even if it's just a single `to_room` declaration. `Room.__post_init__` raises if this is missing. If you genuinely have an island room (no doors? — unlikely) you can declare `exits: [{type: island}]` to opt out, but the default is *fail loud*.
2. **Topology graph render** — generate a rooms-as-nodes / exits-as-edges Graphviz diagram on every boot. Visual disconnection jumps out.
3. **Per-room health metric** — entries-vs-exits ratio per room over last 24h. Should be ~1.0 for any room a human uses regularly. If it's 2.0, that's a black-hole signal.

### 27.9 The catch-all: every threshold is wrong eventually

The rest of the failure modes I'd enumerate boil down to: a threshold default that worked great in your house in November doesn't fit Anna's brother visiting at Christmas, doesn't fit the schedule change after you start a new job, doesn't fit the cat that started napping on top of the bookshelf. The defense is a single discipline: **every tunable in this doc has a default, a unit, and a one-line description of when to change it** — and the dashboard surfaces the current values plus the recent values that would have changed scoring outcomes if applied. See Section 28.

---

## 28. Tunables Reference

Every threshold and parameter mentioned anywhere in this document, in one table. The default is what the system ships with; the range is the region you can sweep without rewriting logic; the lever is the symptom that should make you reach for it.

### 28.1 Recognition (Sections 9–11)

| Param | Default | Range | When to change |
|---|---|---|---|
| `face_recognition_threshold` (cosine) | 0.40 | 0.30 – 0.55 | Lower if known residents go unrecognized in good lighting. Raise if strangers are being silently matched to residents. |
| `face_margin_threshold` (cosine) | 0.05 | 0.03 – 0.10 | Raise if siblings/twins are getting confused. Lower if too many observations land in `unidentified` for known residents. |
| `voice_recognition_threshold` | 0.65 | 0.50 – 0.80 | Tune higher if voice is being used as a tiebreaker and you're getting wrong tiebreaks. |
| `auto_enroll_confidence_threshold` | 0.85 | 0.75 – 0.95 | Raise if you see centroid drift. Lower only if a resident's face genuinely never gets observed at 0.85+ (rare — usually a camera quality issue). |

### 28.2 Auto-enrollment (Section 10)

| Param | Default | Range | When to change |
|---|---|---|---|
| `enrollment_min_initial_samples` | 3 | 2 – 10 | Raise for higher-stakes residents (admins). 3 is fine for family. |
| `enrollment_max_samples_per_person` | 50 | 20 – 200 | Raise if recognition is unstable across lighting/angle conditions and you have storage to burn. |
| `enrollment_diversity_threshold` (cosine) | 0.02 | 0.01 – 0.05 | Raise to make the bank more diverse (slower but more robust). Lower if your bank is filling with near-identical photos. |
| `enrollment_min_face_size_px` | 80 | 60 – 120 | Raise if low-quality crops are degrading centroids. Lower if a far camera never produces qualifying samples. |

### 28.3 Association & state machine (Sections 4, 13)

| Param | Default | Range | When to change |
|---|---|---|---|
| `T_handoff_seconds` (people) | 8 | 4 – 15 | Lower if residents teleport in event logs (handoff window swallowing real transitions). Raise if MOVED_TO events are being missed. |
| `T_handoff_seconds` (cats) | 30 | 15 – 60 | Cats vanish behind couches. Tune up if cat events are noisy. |
| `T_lost_visibility_to_unseen_seconds` | 4 | 2 – 10 | Lower for higher-FPS cameras. Raise if state oscillates PRESENT ↔ IN_ROOM_UNSEEN noisily. |
| `T_unseen_to_unmonitored_seconds` | 90 | 30 – 300 | Lower if your house is small and an unmonitored bedroom stay is short. Raise if residents nap in unmonitored rooms. |
| `position_iou_threshold` | 0.30 | 0.15 – 0.60 | Lower if same-person tracks are splitting across frames. Raise if two close-together people are merging. |
| `near_exit_distance_normalized` | 0.15 | 0.05 – 0.30 | The "how close to a door before disappearance counts as plausible exit" parameter. Tune by walking and watching the event log. |
| `reorder_window_ms` | 250 | 0 – 1000 | Raise if `world.late_observation` events are common. Set to 0 only if you've fully eliminated the bus reorder cause. |

### 28.4 Camera health (Section 14, 27.4)

| Param | Default | Range | When to change |
|---|---|---|---|
| `camera_health_grace_period_seconds` | 30 | 10 – 120 | Time after `camera.health=down` before forcing `IN_ROOM_UNSEEN` for entities that were PRESENT in that camera. Raise for cameras with frequent brief blips. |
| `fps_degraded_ratio` | 0.5 | 0.3 – 0.8 | If `fps_effective / fps_configured` drops below this, mark as degraded. Lower if your cameras have legitimately variable FPS. |
| `camera_health_check_interval_seconds` | 5 | 1 – 30 | Health publish cadence. Don't go below 1; CameraManager will spend more cycles checking than seeing. |

### 28.5 Pattern miner & anomaly (Section 25)

| Param | Default | Range | When to change |
|---|---|---|---|
| `min_history_days` | 14 | 7 – 30 | How many days of data a resident needs before they get a `behavioral_profile`. Lower for faster ramp-up at the cost of false positives early. |
| `pattern_miner_run_interval_hours` | 24 | 6 – 168 | Nightly is fine for most setups. Run more often only if your schedule changes weekly (kids' summer break, etc). |
| `anomaly_threshold` (0–10 score) | 6.0 | 3.0 – 9.0 | Auto-tuned per Section 25.4. Floor at 3.0 — below that you're in pure-noise territory. |
| `anomaly_cooldown_seconds` (per entity) | 600 | 60 – 3600 | After an anomaly fires for X, suppress further X-anomalies. Raise if the dashboard is being spammed during legitimately unusual periods (vacation returns). |
| `anomaly_min_history_days` | 14 | 7 – 30 | Don't score against profiles built from too-little data. Same gate as `min_history_days` but explicit at the scorer level too. |
| `anomaly_auto_tune_window_days` | 7 | 3 – 30 | Window over which auto-tune evaluates FP rate. Shorter window reacts faster but is noisier. |
| `anomaly_auto_tune_fp_high` | 0.30 | 0.10 – 0.50 | Above this FP rate, threshold raises. |
| `anomaly_auto_tune_fp_low` | 0.05 | 0.01 – 0.15 | Below this, threshold lowers (but only with ≥10 anomalies in the window). |

### 28.6 Storage (Sections 5, 8, 27.6)

| Param | Default | Range | When to change |
|---|---|---|---|
| `world_events_retention_days` | 90 | 30 – 730 | Pattern miner needs ≥`min_history_days × 2` to be safe. Lower only if disk pressure forces it. |
| `world_observations_retention_days` | 7 | 1 – 30 | Raw per-tick observations are voluminous and replaceable. Keep just enough for debugging. |
| `interactions_retention_days` | 365 | 90 – ∞ | Interactions are summary-grade — keep them long. They're how the persona answers "when did Anna and Cole last spend an evening together?" and the storage cost is trivial. |
| `pragma_journal_mode` | WAL | WAL | Don't change. Don't even think about it. |
| `pragma_synchronous` | NORMAL | NORMAL / FULL | FULL is paranoid-grade and slower; NORMAL is fine for this workload. |

### 28.7 Persona augmentation (Section 21, 27.7)

| Param | Default | Range | When to change |
|---|---|---|---|
| `world_snapshot_max_tokens` | 200 | 0 – 800 | 0 disables snapshot entirely (persona must always tool-call for world state). Raise only when you're sure the persona has token headroom — it usually doesn't. |
| `world_snapshot_max_residents` | 5 | 1 – 20 | Top-N most-recently-active residents in the snapshot. Tune down before tuning tokens — fewer residents at full detail beats many residents at minimal detail. |
| `world_snapshot_include_anomalies` | false | false / true | Off by default. Turn on once anomaly FP rate is genuinely under control; otherwise the persona starts every conversation with "by the way, something might be weird." |
| `world_snapshot_event_lookback_minutes` | 10 | 1 – 60 | How far back recent change-events are pulled into the snapshot. |

### 28.8 Interaction inference (Section 24)

| Param | Default | Range | When to change |
|---|---|---|---|
| `min_interaction_seconds` | 60 | 30 – 600 | Two entities co-located for less than this don't count as "interacting." Raise if you're getting noise from people walking past each other. |
| `interaction_room_match_strictness` | strict | strict / proximity | Strict = same room only; proximity = same room OR adjacent rooms within 3m by polygon centroid. Switch to proximity if your room boundaries are noisier than your interactions. |
| `interaction_break_seconds` | 120 | 30 – 600 | If both parties were co-located, then one leaves for less than this and returns, count it as one continuous interaction rather than two. |

---

That closes the loop on every cross-reference in this document. Every threshold in the prose has a row in §28; every defense in §27 cites the section it builds on; every phase in §26 has an explicit gate. If you find a parameter mentioned anywhere that isn't in §28, that's a doc bug — open an issue against this file before you start coding around it.

Build well. The demo is a stepping stone; the real prize is a system you and Anna stop noticing because it just works.

— end of bootstrap —