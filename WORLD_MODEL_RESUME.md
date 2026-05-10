<!-- markdownlint-disable -->
# World Model integration — resume notes

Last updated: **2026-05-10** (Phase 4 fully complete — §22 pets, §23 objects, §24 interactions, §29 + v4.1 alarms; only §25 + §30 remain)

## Resume protocol if you're picking up after auto-compaction

1. Read this file end-to-end first.
2. Run `git log --oneline -15` to see exactly what's landed.
3. Run all six synthetic suites — should all be green (44 scenarios total):
   - `python scripts/test_world_model_synthetic.py` (4 scenarios — Phase 1 state machine)
   - `python scripts/test_pets_synthetic.py` (13 scenarios — §22 pets)
   - `python scripts/test_review_fixes_synthetic.py` (6 scenarios — chatgpt review bugs)
   - `python scripts/test_interactions_synthetic.py` (7 scenarios — §24 hands + pickup/placement/handoff)
   - `python scripts/test_clown_alarm_synthetic.py` (8 scenarios — v4.1 §29.8 clown)
   - `python scripts/test_objects_synthetic.py` (6 scenarios — §23 CLIP/objects)
4. Check the "Where things stand" section below.
5. Resume from the queue (§25 next, then §30, plus operational items).

## Where things stand (2026-05-10, post-§23)

**Phase 4 is fully code-complete.** Six commits this session covered the
build queue Cole ranked first:

| Section | Commit | Status |
|---|---|---|
| §22 Phase 4 pets-by-name | several earlier | ✅ |
| §22.5 cluster builder UI | bcba1f8 | ✅ |
| §22.9 species-specific events | 2dbe09a | ✅ |
| §10 IdentityManager auto-enrollment | c7887dd | ✅ |
| §29 alarms framework + cat-escape | 76dc89e | ✅ |
| §29.3 DoorOpenAlarm | 76dc89e | ✅ |
| §29.4 FireAlarm (with override / visual confirm / unattended rearm) | 76dc89e | ✅ |
| §24 interactions (HandDetector + pickup/placement) | 65af7c4 | ✅ |
| §24.4 HANDED_OFF + §24.6 timeline | e0d43b3 | ✅ |
| v4.1 §29.8 ClownAlarm + §29 dispatcher wired into orchestrator | 010c3e4 | ✅ |
| §23 CLIPEncoder + OpenVocabDetector + find_object + dedup + prune | 8361979 | ✅ |
| Six chatgpt-review correctness bugs | 161cb2a | ✅ |
| Pydantic validation + §32 schema | c7887dd | ✅ |

**Remaining build queue (Cole's ordering):**
1. **§25 PatternMiner + AnomalyScorer** — high long-term value, but *needs ~30 days of accumulated event data before it's useful*. Heavy chapter (~5h). The classes can be written now and just sit idle until enough data exists.
2. **§30 multi-persona wake words** — low-effort cosmetic. Existing repo has wake words AND personas; §30 is just the routing function tying them. ~2h.

**Operational gaps (Cole owns these — not code tasks):**
- ArcFace face re-enrollment for residents (DB has Facenet + 14 valid ArcFace rows; centroid bank only loads ArcFace).
- Polygon tuning via `/polygons` viewer (every coord in `config.yaml` is a placeholder).
- Pet identity bootstrapping — let the system run a few days, then label the clusters via `/clusters` UI.

## Klaxon files in modules/safety/alarms/

- `catescapealarm.mp3` → cat_escape (loaded ✅)
- `dooralarm.mp3` → door_open (loaded ✅)
- `firealarm.wav` → fire (loaded ✅)
- `clownalarm.wav` → clown (horns; loaded ✅)
- `Circus Calliope.mp3` → calliope (loaded ✅ — token added in 010c3e4)

## Recent commit chain (2026-05-10 session, newest first)

```
8361979 feat(world_model+vision): §23 Phase 4 Objects — CLIP + OWLv2 + find_object
010c3e4 feat(safety+dashboard): v4.1 Clown Alarm — §29.8 + §29 wired into orchestrator
e0d43b3 feat(world_model+dashboard): §24 close-outs — HANDED_OFF + interactions timeline
65af7c4 feat(world_model+vision): §24 Phase 5 — interactions (MediaPipe Hands + pickup/placement)
bcba1f8 feat(dashboard+vision): §22.5 cluster-builder UI + cat/dog crop persistence
76dc89e feat(safety): §29 alarms — DoorOpen, Fire, klaxon audio, alarm_fires
c7887dd feat(world_model+identity+config): §10 auto-enroll, §32 schema, Pydantic validation
161cb2a fix(world_model+identity): six review-flagged correctness bugs
9ccffd1 fix(camera) + feat(dashboard): living_room visibility + pets / world-events panels
2dbe09a feat(world_model): §22.9 landmark events + §23 closed-vocab object cost + pet query tools
1217184 feat(world_model): §22 wire-up — orchestrator bootstrap + nightly loop + config
03043fc feat(world_model): §22.5 cold-start cluster protocol + animal event metadata
```

## Recently-installed deps (mid-session)

Cole installed `open_clip_torch` mid-§23 implementation: `ftfy-6.3.1
open_clip_torch-3.3.0 timm-1.0.27`. CLIP encoder now uses real
weights when constructed; `try_load()` falls back to NullCLIPEncoder
on any future failure (weights download issue, CUDA OOM, etc.).

OWLv2 weights download to the HF cache on first construction.
`transformers` was already installed; OpenVocabDetector should work
without any further pip work.

This file is the bridge between sessions. The bible is
`scripts/massive_new_integration/new 2.md`; the **TOC comments at the
top of that doc** mark every section that's been completed (search for
`DONE 2026-05-10`). When picking up, read the TOC first.

---

## Where we are right now

**Code-complete, synthetic-test verified:**

| Section | File(s) | Verify |
|---|---|---|
| §5, §6, §8, §15, §16 | `modules/world_model/{types,geometry,store}.py` | DB roundtrip ✅ |
| §11 (ArcFace) | `modules/vision/face_recognizer.py` (rewritten); `face_samples.model_version` migration | InsightFace `buffalo_l` loads on CUDA, 512-dim embeddings ✅ |
| §12, §18 | `modules/vision/observation_builder.py` + `IdentityManager.identify_from_embedding_async` | 5 fps room → 10 obs / 2 s ✅ |
| §13, §17 | `modules/world_model/world_model.py` | All 4 §19 synthetic tests pass — see `scripts/test_world_model_synthetic.py` ✅ |
| §14 (both sides) | WorldModel `_on_camera_health`; `CameraManager.attach_bus` + `_health_loop` | Hysteresis healthy → degraded → down → degraded → healthy ✅ |
| Phase 1.6 wiring | `core/orchestrator.py` constructs WorldStore + WorldModel + ObservationBuilder + WorldQueryTools after IdentityManager | Imports clean, pyright clean ✅ |
| §20 (query layer) | `modules/world_model/query_tools.py` | Unit tests against synthetic WorldModel ✅ |
| Phase 3.2 (tool registry) | `_WORLD_TOOLS` schemas + `_world_tool_handlers` in orchestrator | Schema well-formed; live LLM tool-pick is hands-on |
| §21 (prompt snapshot) | `WorldModel.build_snapshot_for_prompt` injected as `extras["world_snapshot"]` | 5 residents + 5 cats + 3 events = ~117 tokens ✅ |
| Phase 2 (config polygons) | `config.yaml` — `world_model:` block on all 5 rooms | Pydantic validates; **polygons are placeholders, REPLACE_ME comments throughout** |
| Phase 3.4 (polygon viewer) | `dashboard/static/polygon_viewer.html` + `/api/world_model/rooms{,/.../polygons}` + `/polygons` page | Endpoints + page serve via TestClient ✅ |
| §31 (notifications) | `modules/notifications/{dispatcher,channels}.py` | Routing + parallel dispatch + delivery-log roundtrip ✅ |
| §29 (alarm framework + cat-escape) | `modules/safety/alarms/{state,alarm,audio,dispatcher,cat_escape}.py` | 7-scenario synthetic test ✅ |
| §22 (Phase 4 — pets by name) | `modules/world_model/pets.py`, `modules/world_model/cluster_builder.py`; `_animal_pair_cost` + `_build_cat_obs`/`_build_dog_obs` rewrite; `pet_affinities` schema; orchestrator nightly profile loop | 11-scenario synthetic test (`scripts/test_pets_synthetic.py`) ✅ |
| §22.9 species-specific events | `WorldModel._classify_landmark_dwell` + `_LANDMARK_INTERACTION_KIND` map; landmark scaffolding in `config.yaml` (laundry: litterbox + food_dish; living_room: dog_food_dish, dog_water_bowl, leash_hook) | Debounced INTERACTED_WITH(metadata.interaction_kind=...) emits once after 3 frames ✅ |
| §23 closed-vocab object cost | `WorldModel._object_pair_cost` — same-class hard filter + same-room continuity + bbox IoU/center distance + 15-min staleness window. CLIP/open-vocab portion (§23.4-§23.6) deferred. | Class-match + cross-room + class-mismatch all behave per spec ✅ |
| Pet-aware query tools (§20+§22) | `WorldQueryTools.where_is_pet` (with unmonitored_home fallback), `list_pets`, `pet_care_summary`; orchestrator `_WORLD_TOOLS` registers all three for the LLM tool registry | Synthetic tests cover unmonitored fallback + interaction-kind aggregation ✅ |

**Hands-on verify gates still pending** (need Cole + running Jarvis):
- Phase 1.2 — re-enroll Cole + Anna with 5 ArcFace photos each, confirm `identify ≥0.6` + margin-gate refusal.
- Phase 1.6 / 1.7 — boot Jarvis, walk to/from desk, crouch under desk, verify event log shows `MOVED_WITHIN_ROOM` / `LOST_VISIBILITY(reason=in_frame_disappearance, last_landmark=under_desk)` / `REAPPEARED`.
- Phase 2 — visit `/polygons` in dashboard, walk past each room's exit and landmark, tune polygon coordinates in `config.yaml`.
- Phase 3 demo gate — ask Jarvis "where am I?" while crouching under desk; confirm correct answer from world state.
- §31 — fire dashboard test buttons (still TODO, see "what's next") to confirm phone alerts arrive.
- §29 — unleash a cat near an exterior door (or simulate via observation injection) and verify klaxon path.

---

## Things that ARE NOT in the code — keep me from re-discovering them

### 1. `.venv/Lib/site-packages/insightface/app/__init__.py` is patched

InsightFace's `app/__init__.py` does `from .mask_renderer import *` which pulls
`albumentations` → `albucore` → `numkong` (a heavy aug-only dep chain we don't
need). I patched the file in-place to:

```python
try:
    from .mask_renderer import *
except ImportError:
    pass
```

**Pip reinstall of insightface will clobber this.** If FaceAnalysis suddenly
fails with `ModuleNotFoundError: No module named 'albumentations'`, re-apply
the patch (or `pip install albumentations albucore numkong --no-deps` —
heavier but doesn't get clobbered).

### 2. Dep-pin chain for InsightFace + TensorFlow co-existence

DeepFace/Facenet (legacy face stack) and InsightFace/ArcFace (new) co-exist.
Their transitive deps fight; the working pin chain is:

- `numpy<2.0` (TF 2.17 needs 1.x; InsightFace can use either)
- `onnx==1.16.2` (1.21+ requires `ml-dtypes>=0.5`, which TF 2.17 forbids)
- `ml-dtypes<0.5.0,>=0.3.1` (TF 2.17's pin)
- `insightface 0.7.3` (the version with `FaceAnalysis` / `buffalo_l`)
- `onnxruntime-gpu 1.26.0` (CUDA + Tensorrt providers verified working)
- `albumentations` deliberately NOT installed — see above patch

If `pip install` ever upgrades `onnx` past 1.20, tensorflow imports break
with `module 'ml_dtypes' has no attribute 'float4_e2m1fn'`. Pin onnx back
down: `pip install "onnx==1.16.2" --no-deps`.

### 3. MSVC build-tools-via-vcvarsall workflow

Visual Studio Build Tools 2022 is installed at
`C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\` but
`cl.exe` is NOT in PATH for normal shells. To rebuild insightface from
source (the only way to get a Python 3.11 / Windows wheel of 0.7.3):

```cmd
cmd /c "\"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat\" x64 && cd /d C:\Users\Cole\CodeStuff\Ai_ccountabilibuddy && .venv\Scripts\python.exe -m pip install insightface --no-deps --no-cache-dir"
```

The `vcvarsall.bat x64` prefix sets up MSVC env vars (INCLUDE, LIB, etc.)
that pip's setuptools needs to find the C++ compiler and headers.

### 4. Pydantic v2 silently accepts `world_model:` blocks

`core/config.py:RoomConfig` has no `world_model` field. Pydantic v2 default
is `extra="ignore"` so the YAML `world_model:` block survives via the raw
dict (`config["rooms"][i]["world_model"]`) but is dropped from the typed
`RoomConfig` model. **The orchestrator reads from the raw dict, not the
typed model**, so this works — but if anyone tightens RoomConfig to
`extra="forbid"`, world_model breaks silently.

If you want strict typing for `world_model:`, add a `RoomWorldModelCfg`
Pydantic model and an `Optional[RoomWorldModelCfg]` field on `RoomConfig`.
The doc's §7 has the schema sketch.

### 5. Phase 1.6 partial-init degrades cleanly

The orchestrator's WorldModel construction is gated on **all** of:
- `self.db is not None`
- `self.identity is not None`
- `self.cameras is not None`
- `self.object_detector is not None`
- `self.face_recognizer is not None`

If any of those failed to init (e.g. CUDA not found, a Wyze cam offline at
boot, IdentityManager DB error), `world_model` stays `None` and Jarvis
runs without persistent presence tracking — **no crash, no warning beyond
a debug log**. Look for `[Init] World Model + ObservationBuilder started`
in startup logs to confirm it came up. If it's missing, check earlier init
logs for which dep failed.

---

## What's next per the doc's build order

**§22 (Phase 4 — pets by name) is now CODE-COMPLETE** as of 2026-05-10.
The build artifacts are:

- `modules/world_model/pets.py` — `bootstrap_pets_from_config` +
  `BehavioralProfileBuilder` (cats AND dogs).
- `modules/world_model/cluster_builder.py` — `AnimalClusterBuilder` +
  `apply_cluster_labels` (cold-start protocol).
- `modules/vision/observation_builder.py` — full `_build_cat_obs` +
  new `_build_dog_obs` + `_build_animal_obs` dispatcher; descriptor
  helpers (`_classify_cat_color`, `_classify_dog_color`,
  `_color_histogram`, `_coat_texture_descriptor`,
  `_coarse_breed_class`).
- `modules/world_model/world_model.py` — `_animal_pair_cost(species)`
  with cat/dog weight tables; archived-pet hard reject; cat/dog event
  metadata blends in observation descriptors so the cluster builder
  has signal to work with.
- `modules/world_model/store.py` — schema + ALTER-based forward
  migration for `household_owner_id`, `unmonitored_home_room`,
  `archived_at`; new `pet_affinities` table; `replace_affinities`.
- `modules/world_model/types.py` — `WorldEntity` exposes the three
  new fields.
- `core/orchestrator.py` — bootstrap_pets_from_config call after
  `world_model.start()`; `_world_model_nightly_loop` periodic task
  for §22.6 BehavioralProfileBuilder.
- `config.yaml` — full `world_model:`, `residents:`, `pets:`, and
  `outdoor_pets:` blocks declaring the 6 cats + 2 dogs + Scooter.

**Hands-on verify gates for §22:**
- Boot Jarvis; tail logs for `[pets] bootstrap complete: N cat(s), M dog(s)`.
- Run `python scripts/test_pets_synthetic.py` → 7 scenarios pass.
- Day-1 → day-30 disambiguation walk (§22.8) — needs real cat
  observations accumulating; the cluster-builder dashboard page is
  not yet built (Phase 3.4 polygon viewer pattern is the model).

**Remaining doc chunks** (no order dependency):
- §22.10 OutdoorObserver (Scooter) — Phase 6, deferred.
- §23 objects (CLIP encoder + open-vocab detector + cost function).
- §29.3 / §29.4 (door-open + fire alarms — pending detector signals).
- §30 wake words / multi-persona scaffold.
- Cluster-builder dashboard page (§22.5) for human-in-the-loop labeling.

**Other sections that could go in parallel** (no order dependency on §22):
- §32 schema migrations — finish the `alarm_state` / `alarm_fires` /
  `door_state` / `pet_affinities` tables so §29 + §22 can land without
  schema work later.
- §30 wake words / multi-persona — orthogonal scaffold.
- §29.3 door-open + §29.4 fire alarms — need door-state vision detector
  and fire detector to be useful; framework is ready for them.

---

## How to verify Phase 1 yourself when you boot

1. Boot Jarvis with the existing config.
2. Watch for `[Init] World Model + ObservationBuilder started` in the
   startup log. If missing, see "Phase 1.6 partial-init" above.
3. Re-enroll Cole's face via the dashboard (Phase 1.2 verify) — old
   Facenet samples are in the DB but ignored by IM's centroid bank now.
4. Walk into the office from elsewhere, crouch under your desk for 30 s.
   Tail `data/jarvis_2026-MM-DD.log` for `world.entity_event` lines —
   you should see `moved_within_room`, then `lost_visibility` with
   `reason=in_frame_disappearance` and `last_landmark=under_desk`.
5. Visit `http://localhost:7070/polygons` (or whatever port the dashboard
   runs on). Pick a room from the dropdown, see the configured polygons
   overlaid on the live frame. The polygon coords are placeholders —
   tune them in `config.yaml` and refresh the page.
6. Run `python scripts/test_world_model_synthetic.py` to confirm the
   state machine still passes after any future changes.

---

## Resume protocol for a fresh Claude session

1. Read this file (`WORLD_MODEL_RESUME.md`).
2. Read the TOC of `scripts/massive_new_integration/new 2.md` and look
   for `DONE 2026-05-10` markers.
3. Read `modules/world_model/__init__.py` to confirm what's exported.
4. Run `python scripts/test_world_model_synthetic.py` to confirm the
   state machine still works (proves the env is healthy).
5. Pick the next chunk from "What's next" above, or whatever Cole points
   you at.

If something seems broken on first run, the dep-pin chain (#2 above) is
the most likely cause. Check `pip show numpy onnx ml-dtypes
insightface tensorflow` and compare against the pinned versions.
