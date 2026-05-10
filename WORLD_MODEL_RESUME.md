<!-- markdownlint-disable -->
# World Model integration — resume notes

Last updated: **2026-05-10** (mid-session: §24 fully closed; mid-implementing v4.1 clown alarm; §23 next)

## Resume protocol if you're picking up after auto-compaction

1. Read this file end-to-end first.
2. Run `git log --oneline -15` to see exactly what's landed.
3. Run all four synthetic suites — should all be green:
   - `python scripts/test_world_model_synthetic.py` (4 scenarios)
   - `python scripts/test_pets_synthetic.py` (13 scenarios)
   - `python scripts/test_review_fixes_synthetic.py` (6 scenarios)
   - `python scripts/test_interactions_synthetic.py` (7 scenarios)
4. Check the "What I'm currently in the middle of" section below.
5. Resume from there.

## What I'm currently in the middle of (2026-05-10 12:50)

**Implementing v4.1 Clown Alarm.** Spec docs:
- `v4_1_clown_alarm_micropatch.md` (Cole pasted as conversation message — not a file in the repo)
- `clown_alarm_patch_patch.md` (correction to the micropatch's `generate_improv` function — Cole pasted too)

**Apply patch-patch into micropatch first**, then implement. The patch-patch fixes:
- Removes the broken `count_curated_in_style(style_seed)` reference (it was undefined; just use `len(in_style)`)
- Adds 3-layer fallback for sparse pools:
  - Layer 1: ≥3 in-style examples → standard generation
  - Layer 2: 1-2 in-style → supplement with cross-style examples + marked prompt
  - Layer 3: 0 in-style → log warning + fall back to curated entry (no LLM call)
- Adds explicit try/except around the LLM call (timeout/error → curated fallback)
- Adds `record_improv_event()` at every branch for dashboard observability
- Adds `_select_curated_fallback()` helper with hardcoded last-resort response

**Files to create:**
- `modules/safety/alarms/clown.py` — ClownAlarm class (subclasses Alarm, PRIORITY=40)
- `assets/clown_responses.yaml` — curated response pool from §29.8.3 verbatim
- `prompts/clown_improv.md` — standard generation template
- `prompts/clown_improv_supplement.md` — variant for thin-pool case
- `scripts/test_clown_alarm_synthetic.py` — tests for the state machine + selection + fallback

**Files to modify:**
- `modules/safety/alarms/__init__.py` — export ClownAlarm
- `modules/safety/alarms/state.py` — add `AlarmType.CLOWN = "clown"` (priority 40)
- `modules/safety/alarms/dispatcher.py` — extend priority_order with `AlarmType.CLOWN` at the end
- `modules/safety/alarms/audio.py` — extend repeat-cadence + add 3-stage clown sequence (horns ×3 → TTS → calliope) — needs a custom path because clown is "play once, end naturally" not the standard "loop forever"
- `modules/safety/alarms/klaxon.py` — add `(calliope, "calliope")` to `_NAME_TOKENS` so `Circus Calliope.mp3` gets classified
- `core/orchestrator.py` — wire ClownAlarm at boot, register with dispatcher
- `dashboard/server.py` — `POST /api/world_model/clown_test` for test fire button
- `dashboard/static/index.html` + `app.js` + `style.css` — clown alarm card with status + test fire button + recent fires + response pool browser

**After clown alarm → §23 Objects.** Cole picked it (highest daily-use UX value) — CLIP encoder + open-vocab detector (OWLv2 or GroundingDINO) + object cost function + dedup + `find_object` query. Heavy ML deps (~600MB-1GB of new weights). Closed-vocab cost function already done (committed earlier).

## Recent commit chain (2026-05-10 session)

```
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
