"""
JARVIS — World Model
====================
Mission: Pure-data dataclasses + enums for the persistent entity layer.
         No I/O. Imported by every other world_model module.

Modules: modules/world_model/types.py
Spec:    new 2.md §5 (Data Model) and §15 (Full Code: types.py).
"""

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
