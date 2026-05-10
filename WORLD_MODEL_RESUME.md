# World Model integration — resume notes

Last updated: **2026-05-10**

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

The strict next code-only chunk is **§22 (Phase 4 — pets by name)**. Per
§26, Phase 4 is "optional for the boss demo" but high-value for the
long-running system. Implementation lives in §22.3-§22.7:

- `modules/world_model/pets.py` — bootstrap from `config.pets`
- `modules/vision/observation_builder.py::_build_cat_obs` — replace stub
  with full color/pattern/coat-texture enricher (§22.3)
- `modules/world_model/world_model.py::_cat_pair_cost` — replace stub with
  the full §22.7 cost function
- `modules/world_model/cluster_builder.py` — cold-start cluster protocol
  (§22.5)
- `modules/world_model/behavioral_profile_builder.py` — nightly profile
  rebuild (§22.6)
- Schema additions per §22.0a (household_owner_id, unmonitored_home_room,
  pet_affinities table)

The verify gate for §22 is "5 photos × 5 cats enrolled, ≥70% top-1 match
on held-out crops". That's hands-on (photos) but the code is fully
writable in advance.

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
