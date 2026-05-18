"""
JARVIS — World Model
====================
Mission: The WorldModel orchestrator. Receives Observations, holds the
         entity registry, runs the association algorithm + state
         machine, emits change events. Single-writer discipline:
         every mutation of `self.entities` happens under `self._lock`.

         Subscriptions:
            - vision.observation  → _on_observation_batch
            - camera.health        → _on_camera_health

         Publications:
            - world.entity_event   (every state change / movement / etc.)
            - world.state_snapshot (~every 30 s, full registry view)

Modules: modules/world_model/world_model.py
Classes: WorldModel
Spec:    new 2.md §13 (Association Algorithm), §17 (Full Code).

#todo: Phase 4 object cost function fills `_object_pair_cost`. Today the
       stub hard-rejects every object candidate so objects become
       FIRST_SEEN per observation — fine for Phase 1 (no objects
       tracked yet), wrong once §23 lands.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np
from loguru import logger
from scipy.optimize import linear_sum_assignment

from core.async_utils import TrackedTaskSet
from modules.world_model.geometry import bbox_center, bbox_iou, point_in_polygon
from modules.world_model.store import WorldStore
from modules.world_model.types import (
    EntityState,
    EventType,
    Observation,
    WorldEntity,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


# Cap on per-entity visibility history. ~40 entries covers ~8 seconds of
# 5fps observation + plenty of headroom for the seen-fraction window. Old
# entries roll off the back so the metadata blob stays bounded.
_VIS_HISTORY_MAX = 40


# Runtime polygon overrides written by the dashboard polygon editor.
# Lives next to data/jarvis.db so the editor doesn't need to touch
# config.yaml (which is git-tracked and has REPLACE_ME placeholder
# comments we don't want to clobber). Map shape:
#   { camera_id: {frame_width, frame_height, exits, landmarks} }
import json as _wm_json
from pathlib import Path as _wm_Path

_POLYGON_OVERRIDES_PATH = _wm_Path("data/polygon_overrides.json")


def _load_polygon_overrides() -> dict:
    """Return the polygon overrides dict, or {} if the file doesn't
    exist or is malformed. Never raises — overrides are best-effort
    and the config.yaml values are always a safe fallback."""
    p = _POLYGON_OVERRIDES_PATH
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as fh:
            data = _wm_json.load(fh)
        if not isinstance(data, dict):
            return {}
        return data
    except Exception as e:
        logger.warning(f"[WorldModel] polygon overrides load failed: {e}")
        return {}


def save_polygon_overrides(overrides: dict) -> None:
    """Atomically write polygon overrides to disk. Caller owns
    validation — we just persist what's given. The dashboard endpoint
    should validate shape before calling this."""
    p = _POLYGON_OVERRIDES_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        _wm_json.dump(overrides, fh, indent=2)
    tmp.replace(p)


def _record_visibility(ent: WorldEntity, ts: datetime, seen: bool) -> None:
    """Append a (ts_iso, seen) pair to the entity's rolling visibility
    history. Used by the smoothing logic in _handle_unmatched_entity so
    LOST_VISIBILITY only fires when the seen-fraction over a recent
    window drops below threshold — i.e. real disappearance, not a
    single bad frame."""
    hist = ent.metadata.setdefault("vis_history", [])
    hist.append([ts.isoformat(), bool(seen)])
    if len(hist) > _VIS_HISTORY_MAX:
        del hist[: len(hist) - _VIS_HISTORY_MAX]


def _seen_fraction(
    ent: WorldEntity, ts: datetime, window_seconds: float
) -> tuple[float, int]:
    """Compute (fraction seen, sample count) over the last
    `window_seconds` of this entity's visibility history. Returns
    (1.0, 0) when there's no usable history — the caller should treat
    insufficient data as 'don't fire yet'."""
    hist = ent.metadata.get("vis_history") or []
    if not hist:
        return 1.0, 0
    cutoff = ts - timedelta(seconds=window_seconds)
    seen = 0
    total = 0
    for entry in hist:
        try:
            entry_ts = datetime.fromisoformat(entry[0])
        except (ValueError, TypeError, IndexError):
            continue
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.replace(tzinfo=timezone.utc)
        if entry_ts < cutoff:
            continue
        total += 1
        if entry[1]:
            seen += 1
    if total == 0:
        return 1.0, 0
    return seen / total, total


class WorldModel:
    """
    Stateful tracker. Subscribes to vision.observation and camera.health.
    Publishes world.entity_event and world.state_snapshot.
    """

    def __init__(
        self,
        bus: Any,                       # core.event_bus.EventBus
        store: WorldStore,
        rooms_config: list[dict],       # the typed rooms list from config.yaml
        identity_manager: Any,          # modules.identity.identity_manager.IdentityManager
        config: dict,                   # config['world_model'] block
    ) -> None:
        self.bus = bus
        self.store = store
        self.identity_manager = identity_manager
        # Tunables. Defaults track §28.3 / §17 of the spec; callers can
        # override any subset via the `world_model:` block in config.yaml.
        self.cfg: dict[str, Any] = {
            "cost_reject": 1.0,
            "enrollment_min_conf": 0.85,
            "movement_jitter_threshold": 0.15,
            "posture_debounce_frames": 3,
            "interaction_debounce_frames": 3,
            "T_handoff_seconds": 8,
            "stationary_long_minutes": 5,
            "cosine_match_strong": 0.6,
            "candidate_lookback_minutes": 2,
            "snapshot_interval_seconds": 30.0,
            "timer_tick_seconds": 2.0,
            "boot_resolution_seconds": 30,
            # Visibility-flicker hysteresis. An entity must miss
            # `visibility_grace_misses` consecutive observation batches
            # OR `visibility_grace_seconds` of wall time before we
            # transition state out of PRESENT and emit LOST_VISIBILITY.
            # Without this, YOLO false-negatives produce ~25 lost/
            # reappeared events per second per entity.
            # Wall-time floor + smoothing parameters for visibility
            # transitions. See _handle_unmatched_entity.
            #   - visibility_grace_seconds: minimum elapsed since the
            #     last confirmed match before LOST_VISIBILITY can fire.
            #   - visibility_window_seconds: width of the rolling
            #     seen/missed window for the smoothing.
            #   - visibility_min_samples: need this many observations
            #     in the window before we trust the seen fraction.
            #   - visibility_seen_fraction_floor: hold until the seen
            #     fraction drops below this. 0.25 means "we have to
            #     have been mostly missing for the window's duration."
            # Legacy visibility_grace_misses is no longer consulted but
            # left in cfg for backwards-compat with any read-only tooling.
            "visibility_grace_seconds": 3.0,
            "visibility_window_seconds": 6.0,
            "visibility_min_samples": 4,
            "visibility_seen_fraction_floor": 0.25,
            "visibility_grace_misses": 5,
            "landmark_dwell_frames": 3,
            # Spatial continuity for person obs that lost face
            # attribution. Within this window, an unattributed person
            # detection in the same room is merged into the nearest
            # known person entity instead of spawning a duplicate
            # unknown_person_*. See _handle_unmatched_observation.
            "person_continuity_seconds": 5.0,
            **(config or {}),
        }

        # Per-camera topology lookup from rooms config.
        self.cameras: dict[str, dict] = self._build_camera_topology(rooms_config)

        self.entities: dict[str, WorldEntity] = {}
        self._lock = asyncio.Lock()
        self._unhealthy_cameras: set[str] = set()
        # §23 — optional CLIP encoder for find_object text-query
        # similarity. None / NullCLIPEncoder → text-query path
        # short-circuits to "no match" (correct behavior; nothing
        # crashes).
        self.clip_encoder: Optional[Any] = None
        # Background tasks the model owns; cancelled in stop().
        self._timer_task: Optional[asyncio.Task] = None
        self._snapshot_task: Optional[asyncio.Task] = None
        # Fire-and-forget enrollment hand-offs to IdentityManager. Tracked
        # so they survive GC, log their exceptions, and shut down cleanly.
        self._bg_tasks = TrackedTaskSet(label="WorldModel")
        self._stopped = False

    # ── Topology ───────────────────────────────────────────────────────────

    @staticmethod
    def _build_camera_topology(rooms_config: list[dict]) -> dict:
        """
        Build a per-camera topology dict from the rooms[] config, then
        overlay any runtime polygon edits from data/polygon_overrides.json
        (written by the dashboard polygon editor). Overrides win; missing
        keys fall back to config.yaml.

        Camera ID derives from room ID — one camera per room in the
        current config. Multi-cam-per-room is a future expansion.
        """
        # Load runtime overrides up-front so per-room merging can read
        # them without re-loading the file on every iteration.
        overrides = _load_polygon_overrides()

        topology: dict[str, dict] = {}
        for room in rooms_config:
            wm = room.get("world_model")
            if not wm or not wm.get("enabled", True):
                continue
            cam_id = room["id"]
            override = overrides.get(cam_id) or {}
            topology[cam_id] = {
                "room": room["id"],
                "frame_width": override.get(
                    "frame_width", wm.get("frame_width", 640)
                ),
                "frame_height": override.get(
                    "frame_height", wm.get("frame_height", 480)
                ),
                "exits": override.get("exits", wm.get("exits", [])),
                "landmarks": override.get(
                    "landmarks", wm.get("landmarks", [])
                ),
                # Per-camera ignore zones — polygons over static false
                # positives (a framed painting, a TV). Consumed by
                # modules/vision/ignore_zones.py at detection time.
                "ignore_zones": override.get(
                    "ignore_zones", wm.get("ignore_zones", [])
                ),
            }
        return topology

    def reload_polygons(self) -> None:
        """Re-read polygon overrides from disk and rebuild camera
        topology. Called by the dashboard editor after a save so the
        live world model picks up the new polygons without restart."""
        rooms_config = [
            {"id": cam_id, "world_model": {"enabled": True, **cam}}
            for cam_id, cam in self.cameras.items()
        ]
        # Fresh build replaces the existing cameras dict in place so
        # in-flight references to self.cameras get the new values.
        new_topo = self._build_camera_topology(rooms_config)
        self.cameras = new_topo
        logger.info(
            f"[WorldModel] polygons reloaded for "
            f"{len(self.cameras)} cameras"
        )

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Subscribe to bus topics, load persistent state, spawn timers."""
        await self.store.ensure_schema()
        await self._load_from_store()
        await self.bus.subscribe("vision.observation", self._on_observation_batch)
        await self.bus.subscribe("camera.health", self._on_camera_health)
        self._timer_task = asyncio.create_task(
            self._timer_loop(), name="world_model:timer"
        )
        self._snapshot_task = asyncio.create_task(
            self._snapshot_loop(), name="world_model:snapshot"
        )
        logger.info(
            f"[WorldModel] started — {len(self.entities)} entities loaded, "
            f"{len(self.cameras)} cameras configured"
        )

    async def stop(self) -> None:
        """Cancel background loops. Idempotent; safe to call before start()."""
        self._stopped = True
        for t in (self._timer_task, self._snapshot_task):
            if t is not None:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
        self._timer_task = None
        self._snapshot_task = None
        try:
            await self._bg_tasks.shutdown(timeout=2.0)
        except Exception as e:
            logger.debug(f"[WorldModel] bg-task drain failed: {e}")

    async def _load_from_store(self) -> None:
        """Hydrate entities from disk. Every PRESENT entity becomes
        UNKNOWN_AT_BOOT for the first 30s — observations resolve, the
        timer demotes survivors to IN_HOUSE_UNMONITORED."""
        for ent in await self.store.load_entities():
            if ent.last_seen_ts is not None:
                ent.last_seen_ts = _as_utc(ent.last_seen_ts)
            if ent.last_state_change_ts is not None:
                ent.last_state_change_ts = _as_utc(ent.last_state_change_ts)
            if ent.archived_at is not None:
                ent.archived_at = _as_utc(ent.archived_at)
            self.entities[ent.id] = ent
        boot_ts = _utcnow()
        for ent in self.entities.values():
            if ent.state == EntityState.PRESENT:
                ent.state = EntityState.UNKNOWN_AT_BOOT
                ent.last_state_change_ts = boot_ts
                await self.store.upsert_entity(ent)

    # ────────────────────────────────────────────────────────────────────────
    # MAIN ENTRY POINTS
    # ────────────────────────────────────────────────────────────────────────

    async def _on_observation_batch(self, payload: dict) -> None:
        """payload: {camera, room, ts, observations: [Observation, ...]}"""
        from modules.context.perf_tracker import perf
        with perf().timeit("world_model.observation_batch"):
            await self._on_observation_batch_inner(payload)

    async def _on_observation_batch_inner(self, payload: dict) -> None:
        async with self._lock:
            camera = payload["camera"]
            ts = payload["ts"] if isinstance(payload["ts"], datetime) \
                 else datetime.fromisoformat(payload["ts"])
            ts = _as_utc(ts)
            observations: list[Observation] = payload["observations"]
            for obs in observations:
                obs.ts = _as_utc(obs.ts)

            # Skip if camera is unhealthy — entities should already be
            # suspended by _on_camera_health.
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

    async def _on_camera_health(self, payload: dict) -> None:
        camera_id = payload["camera_id"]
        status = payload["status"]
        async with self._lock:
            if status in ("degraded", "down"):
                self._unhealthy_cameras.add(camera_id)
                for ent in self.entities.values():
                    if (ent.last_seen_camera == camera_id
                            and ent.state == EntityState.PRESENT):
                        ent.metadata["suspended_due_to_camera_health"] = True
                        await self._emit(
                            EventType.CAMERA_DEGRADED, ent, obs=None,
                            metadata={"camera": camera_id, "status": status},
                        )
                        await self.store.upsert_entity(ent)
            elif status == "healthy":
                self._unhealthy_cameras.discard(camera_id)
                for ent in self.entities.values():
                    if ent.metadata.pop("suspended_due_to_camera_health", False):
                        await self._emit(
                            EventType.CAMERA_RESTORED, ent, obs=None,
                            metadata={"camera": camera_id},
                        )
                        await self.store.upsert_entity(ent)

    # ────────────────────────────────────────────────────────────────────────
    # ASSOCIATION
    # ────────────────────────────────────────────────────────────────────────

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

        matched: list[tuple[Observation, WorldEntity, float]] = []
        matched_obs_idx: set[int] = set()
        matched_ent_idx: set[int] = set()
        for i, j in zip(row_idx, col_idx):
            if cost[i, j] < self.cfg["cost_reject"]:
                # Attribution confidence — how much better is this match
                # than the second-best? Mapped 0..1 via /0.5 clip.
                row_costs = np.sort(cost[i])
                margin = row_costs[1] - row_costs[0] if len(row_costs) > 1 else 1.0
                attribution_conf = float(np.clip(margin / 0.5, 0.0, 1.0))
                matched.append((observations[i], candidates[j], attribution_conf))
                matched_obs_idx.add(int(i))
                matched_ent_idx.add(int(j))

        unmatched_obs = [
            o for i, o in enumerate(observations) if i not in matched_obs_idx
        ]
        unmatched_ents = [
            e for j, e in enumerate(candidates) if j not in matched_ent_idx
        ]
        return matched, unmatched_obs, unmatched_ents

    def _pair_cost(self, obs: Observation, ent: WorldEntity) -> float:
        obs.ts = _as_utc(obs.ts)
        if ent.entity_type != obs.obj_class:
            return self.cfg["cost_reject"] * 2
        # Archived pet entities never match — they're soft-deleted.
        if ent.archived_at is not None:
            return self.cfg["cost_reject"] * 2
        if ent.entity_type == "person":
            return self._person_pair_cost(obs, ent)
        if ent.entity_type == "cat":
            return self._animal_pair_cost(obs, ent, species="cat")  # §22
        if ent.entity_type == "dog":
            return self._animal_pair_cost(obs, ent, species="dog")  # §22
        if ent.entity_type == "object":
            return self._object_pair_cost(obs, ent)            # §23
        return self.cfg["cost_reject"] * 2

    def _person_pair_cost(self, obs: Observation, ent: WorldEntity) -> float:
        obs.ts = _as_utc(obs.ts)
        # Identity wins if both sides have it.
        if obs.person_id is not None and ent.person_id is not None:
            if obs.person_id == ent.person_id:
                return 0.05 * self._spatial_distance(obs, ent)
            else:
                return self.cfg["cost_reject"] * 2  # different people, hard reject

        # Fallback to spatial-temporal continuity.
        if not ent.last_seen_ts:
            return self.cfg["cost_reject"] * 2
        ent.last_seen_ts = _as_utc(ent.last_seen_ts)
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

    # ────────────────────────────────────────────────────────────────────────
    # MATCHED ENTITY UPDATE
    # ────────────────────────────────────────────────────────────────────────

    # Landmark name → interaction kind. The kind feeds into event metadata
    # so the dashboard / LLM can filter (e.g. last_litterbox_visit) without
    # re-deriving from the landmark name. Add new pairs as landmarks land.
    _LANDMARK_INTERACTION_KIND: dict[str, str] = {
        "litterbox": "litterbox_visit",
        "food_dish": "food_dish_visit",
        "dog_food_dish": "dog_food_visit",
        "dog_water_bowl": "dog_water_visit",
        "leash_hook": "leash_interaction",
    }

    async def _update_matched(
        self,
        ent: WorldEntity,
        obs: Observation,
        ts: datetime,
        attribution_conf: float,
    ) -> None:
        was_unseen = ent.state in (
            EntityState.IN_ROOM_UNSEEN, EntityState.TRANSITIONING,
            EntityState.IN_HOUSE_UNMONITORED, EntityState.DEPARTED,
            EntityState.UNKNOWN_AT_BOOT,
        )
        prior_state = ent.state
        prior_room = ent.last_seen_room
        room_changed = prior_room is not None and prior_room != obs.room

        # Identity resolution: anonymous entity got recognized.
        if obs.person_id is not None and ent.person_id is None:
            ent.person_id = obs.person_id
            ent.display_name = obs.person_name
            ent.is_resident = True
            await self._emit(
                EventType.NAME_LINKED, ent, obs,
                metadata={
                    "person_id": obs.person_id,
                    "person_name": obs.person_name,
                },
            )

        # Compute granular events BEFORE mutating spatial state.
        movement_event = self._classify_movement(ent, obs)
        posture_event = self._classify_posture(ent, obs)
        interaction_event = self._classify_interaction(ent, obs)

        # Update last-seen fields.
        new_landmark = self._nearest_landmark(obs.camera, bbox_center(obs.bbox))
        ent.last_seen_ts = obs.ts
        ent.last_seen_camera = obs.camera
        ent.last_seen_room = obs.room
        ent.last_seen_bbox = obs.bbox
        ent.last_seen_landmark = new_landmark
        ent.confidence = obs.confidence
        ent.last_attribution_confidence = attribution_conf
        # Reset the visibility-flicker counter on every confirmed match —
        # see _handle_unmatched_entity for the grace-window logic.
        if "miss_streak" in ent.metadata:
            ent.metadata.pop("miss_streak", None)
        # Sliding-window visibility history. Each entry is a [ts_iso,
        # seen_bool] pair; the grace logic below uses the seen fraction
        # over a recent window to ride out single-frame flickers without
        # firing LOST_VISIBILITY. See _handle_unmatched_entity.
        _record_visibility(ent, ts, seen=True)

        # Track posture history.
        posture_val = obs.metadata.get("posture")
        if posture_val:
            hist = ent.metadata.setdefault("posture_history", [])
            hist.append((obs.ts.isoformat(), posture_val))
            ent.metadata["posture_history"] = hist[-10:]

        # State transition: any unseen state → PRESENT.
        if was_unseen:
            ent.state = EntityState.PRESENT
            ent.last_state_change_ts = ts
            ent.metadata.pop("transitioning_target", None)
            ent.metadata.pop("transitioning_kind", None)
            await self._emit(
                EventType.REAPPEARED, ent, obs,
                metadata={"from_state": prior_state.value},
            )

        # Room change without an unseen-state intermediate.
        if room_changed and ent.state == EntityState.PRESENT and not was_unseen:
            await self._emit(
                EventType.MOVED_TO, ent, obs,
                metadata={"from_room": prior_room},
            )

        # Granular events.
        if movement_event:
            await self._emit(
                EventType.MOVED_WITHIN_ROOM, ent, obs, metadata=movement_event
            )
        if posture_event:
            await self._emit(
                EventType.POSTURE_CHANGED, ent, obs, metadata=posture_event
            )
        if interaction_event:
            await self._emit(
                EventType.INTERACTED_WITH, ent, obs, metadata=interaction_event
            )

        # §22.9 species-specific landmark dwell. Per-entity, per-landmark
        # consecutive-frame counter; once it reaches debounce, we emit a
        # specialized INTERACTED_WITH with metadata.interaction_kind so
        # the dashboard can filter (last_litterbox_visit, etc.). Resets
        # when the entity moves off the landmark.
        landmark_event = self._classify_landmark_dwell(ent, new_landmark)
        if landmark_event:
            await self._emit(
                EventType.INTERACTED_WITH, ent, obs, metadata=landmark_event
            )

        await self.store.upsert_entity(ent)

        # Hand off confident person matches to IdentityManager for
        # auto-enrollment quality consideration. Fire-and-forget so
        # the obs-batch handler stays fast.
        if (ent.entity_type == "person"
                and obs.person_id is not None
                and obs.person_match_confidence
                    >= self.cfg.get("enrollment_min_conf", 0.85)
                and obs.metadata.get("crop_path")
                and obs.metadata.get("face_embedding") is not None):
            self._bg_tasks.spawn(
                self._enroll_async(obs, attribution_conf),
                name=f"world.enroll:{obs.person_id}",
            )

    async def _enroll_async(
        self, obs: Observation, attribution_conf: float,
    ) -> None:
        """Hand off to IdentityManager — fire and forget. The IM
        `consider_new_sample_async` extension (§10) handles quality
        gating, diversity, and the bounded-capacity coreset. If the
        attribute is missing (e.g. an older IM build) we silently skip
        — the world model's job ends at the hand-off."""
        try:
            handler = getattr(
                self.identity_manager, "consider_new_sample_async", None
            )
            if handler is None:
                return
            # Compute face area so the IM quality gate can apply its
            # min_face_area_px check (§10). The bbox here is the PERSON
            # bbox; ObservationBuilder runs face detection on the
            # person crop, so the actual face fills a fraction of it —
            # we pass the person bbox area as a permissive proxy
            # (≥80×80 person bbox => face is plausibly ≥40×40, which
            # is the practical lower bound at typical Wyze distances).
            x1, y1, x2, y2 = obs.bbox
            person_area = max(0, int(x2) - int(x1)) * max(0, int(y2) - int(y1))
            await handler(
                person_id=obs.person_id,
                new_embedding=obs.metadata["face_embedding"],
                crop_path=obs.metadata["crop_path"],
                quality_metadata={
                    "yaw": obs.metadata.get("yaw"),
                    "pitch": obs.metadata.get("pitch"),
                    "blur_score": obs.metadata.get("blur_score"),
                    "face_area_px": person_area,
                    "association_confidence": attribution_conf,
                    "pose": obs.metadata.get("posture", "candid"),
                },
            )
        except Exception as e:
            logger.debug(f"[WorldModel] enrollment hand-off failed: {e}")

    # ── Delta classifiers ──────────────────────────────────────────────────

    def _classify_movement(
        self, ent: WorldEntity, obs: Observation
    ) -> Optional[dict]:
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

    def _classify_posture(
        self, ent: WorldEntity, obs: Observation
    ) -> Optional[dict]:
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

    def _classify_landmark_dwell(
        self, ent: WorldEntity, landmark: Optional[str],
    ) -> Optional[dict]:
        """§22.9 — fire INTERACTED_WITH once an entity has been over a
        registered landmark for `landmark_dwell_frames` consecutive
        observations. Returns the event metadata when the threshold is
        crossed; returns None on every other tick.

        State is kept in `ent.metadata['landmark_dwell']` so it survives
        upsert. The {landmark: count} map is reset to {} when the entity
        steps off (`landmark` becomes None or changes); we only debounce
        the *current* landmark to avoid double-firing during walks across
        multiple landmarks.

        Landmark-visit events for OBJECTS fire only for named / tagged
        objects (e.g. Cole's wallet at the leash hook) — never for
        unknown_* blobs: an unknown cell phone parked near the dog's
        water bowl is detector noise, not an interaction. People and pets
        always pass — a cat dwelling at the litterbox is the whole point.
        """
        etype = getattr(ent, "entity_type", None)
        if etype not in ("person", "cat", "dog"):
            name = getattr(ent, "display_name", None) or ""
            if not name or name.startswith("unknown_"):
                return None
        threshold = int(self.cfg.get("landmark_dwell_frames", 3))
        dwell: dict = ent.metadata.setdefault("landmark_dwell", {})
        if landmark is None:
            if dwell:
                ent.metadata["landmark_dwell"] = {}
            return None
        # Reset other landmark counters — the entity is now on `landmark`.
        if list(dwell.keys()) != [landmark]:
            dwell = {landmark: 0}
            ent.metadata["landmark_dwell"] = dwell
        dwell[landmark] = int(dwell.get(landmark, 0)) + 1
        if dwell[landmark] != threshold:
            # Already fired (count > threshold) or still ramping up.
            return None
        kind = self._LANDMARK_INTERACTION_KIND.get(landmark)
        if kind is None:
            return None
        return {
            "landmark": landmark,
            "interaction_kind": kind,
            "dwell_frames": int(dwell[landmark]),
        }

    def _classify_interaction(
        self, ent: WorldEntity, obs: Observation
    ) -> Optional[dict]:
        hand_bboxes = obs.metadata.get("hand_bboxes", [])
        if not hand_bboxes:
            return None
        for obj_ent in self.entities.values():
            if (obj_ent.entity_type != "object"
                    or obj_ent.last_seen_room != obs.room):
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

    # ────────────────────────────────────────────────────────────────────────
    # NEW OR RETURNING DETECTION
    # ────────────────────────────────────────────────────────────────────────

    async def _handle_unmatched_observation(
        self, obs: Observation, ts: datetime
    ) -> None:
        # Identity wins over spatial continuity for people: if obs has
        # a person_id that matches an existing entity, route directly —
        # BUT with a confidence guard for cross-camera moves. Without
        # this, a low-confidence face misidentification in room B can
        # steal Cole's entity away from room A where he actually is,
        # causing the "Cole is in two rooms at once" flickering.
        if obs.obj_class == "person" and obs.person_id is not None:
            existing = self._find_entity_by_person_id(obs.person_id)
            if existing:
                attribution_conf = obs.person_match_confidence or 0.0
                cross_camera = (
                    existing.last_seen_camera
                    and existing.last_seen_camera != obs.camera
                )
                if cross_camera:
                    # Time since the entity was last confirmed in its
                    # current room — fresh sightings carry weight.
                    last_seen = (
                        _as_utc(existing.last_seen_ts)
                        if existing.last_seen_ts else None
                    )
                    age_s = (
                        (ts - last_seen).total_seconds()
                        if last_seen else 999.0
                    )
                    freshness_s = float(self.cfg.get(
                        "identity_cross_camera_freshness_seconds", 5.0
                    ))
                    existing_conf = float(
                        existing.last_attribution_confidence or 0.0
                    )
                    margin = float(self.cfg.get(
                        "identity_cross_camera_margin", 0.15
                    ))
                    # Hold the entity in place when:
                    #   - it was just seen there with a real attribution
                    #   - AND the new claim is not decisively better
                    if (age_s < freshness_s
                            and existing_conf > 0.0
                            and attribution_conf < existing_conf + margin):
                        # Treat this as a misattribution — drop the
                        # person_id and let the spatial-continuity path
                        # below handle the obs as an unidentified person.
                        obs.person_id = None
                        obs.person_name = None
                        obs.person_match_confidence = 0.0
                    else:
                        existing.metadata["identity_overrode_continuity"] = True
                        await self._update_matched(
                            existing, obs, ts, attribution_conf
                        )
                        return
                else:
                    await self._update_matched(
                        existing, obs, ts, attribution_conf
                    )
                    return

        # Face attribution dropped out (no person_id) but a known named
        # person was just here. Prefer routing to them over spawning a
        # duplicate `unknown_person_*` that ping-pongs with the real
        # entity every frame the face match fails. Criteria:
        #   - same room, last seen within `person_continuity_seconds`
        #   - bbox center distance below 25% of frame diagonal
        # If both pass we treat this as the same person; the entity
        # keeps its name and we avoid the entity-split flicker that
        # produced the dashboard spam.
        if obs.obj_class == "person" and obs.person_id is None:
            continuity_s = float(self.cfg.get("person_continuity_seconds", 5.0))
            cam_cfg = self.cameras.get(obs.camera, {})
            fw = float(cam_cfg.get("frame_width", 640))
            fh = float(cam_cfg.get("frame_height", 480))
            diag = (fw * fw + fh * fh) ** 0.5
            max_dist = 0.25 * diag
            obs_center = bbox_center(obs.bbox)
            best_ent: Optional[WorldEntity] = None
            best_dist = max_dist + 1.0
            for ent in self.entities.values():
                if ent.entity_type != "person":
                    continue
                if ent.last_seen_room != obs.room:
                    continue
                if ent.last_seen_ts is None or ent.last_seen_bbox is None:
                    continue
                age = (ts - _as_utc(ent.last_seen_ts)).total_seconds()
                if age > continuity_s:
                    continue
                # Prefer named (resident) entities; fall back to merging
                # with a recent unknown to avoid a chain of duplicates.
                ent_center = bbox_center(ent.last_seen_bbox)
                dist = (
                    (ent_center[0] - obs_center[0]) ** 2
                    + (ent_center[1] - obs_center[1]) ** 2
                ) ** 0.5
                if dist > max_dist:
                    continue
                # Strong preference for the named person if both a named
                # and an unknown candidate exist — sort by (named first,
                # then closer).
                key = (0 if ent.display_name else 1, dist)
                best_key = (
                    0 if (best_ent is not None and best_ent.display_name) else 1,
                    best_dist,
                )
                if best_ent is None or key < best_key:
                    best_ent, best_dist = ent, dist
            if best_ent is not None:
                await self._update_matched(
                    best_ent, obs, ts,
                    attribution_conf=obs.person_match_confidence or 0.0,
                )
                return

        # Wider pool for cats/objects: same-type entity with strong
        # embedding match. People are caught by the person_id path above.
        # §23.8 — for objects we additionally lower the threshold when
        # an existing same-class same-room entity matches, so a phone
        # repeatedly detected on the same desk doesn't proliferate
        # into a new entity per frame.
        if obs.obj_class in ("cat", "object") and obs.visual_embedding is not None:
            best: Optional[WorldEntity] = None
            best_sim = 0.0
            obs_class = (
                obs.metadata.get("detected_class") or ""
                if obs.obj_class == "object" else None
            )
            for ent in self.entities.values():
                if ent.entity_type != obs.obj_class:
                    continue
                # Same-class hard filter for objects — different
                # detected_class can never match.
                if obs.obj_class == "object":
                    ent_class = ent.metadata.get("detected_class") or ""
                    if obs_class and ent_class and obs_class != ent_class:
                        continue
                emb = ent.metadata.get("_visual_embedding")
                if emb is None:
                    continue
                sim = float(
                    np.dot(obs.visual_embedding, emb)
                    / (np.linalg.norm(obs.visual_embedding)
                       * np.linalg.norm(emb) + 1e-9)
                )
                # §23.8 — lower threshold for same-class same-room.
                if obs.obj_class == "object" and ent.last_seen_room == obs.room:
                    threshold = self.cfg.get(
                        "cosine_match_strong_same_room", 0.45,
                    )
                else:
                    threshold = self.cfg.get("cosine_match_strong", 0.6)
                if sim > best_sim and sim >= threshold:
                    best, best_sim = ent, sim
            if best is not None:
                await self._update_matched(
                    best, obs, ts, attribution_conf=best_sim,
                )
                return

        # Genuinely new entity.
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
            last_seen_landmark=self._nearest_landmark(
                obs.camera, bbox_center(obs.bbox)
            ),
            last_state_change_ts=ts,
            confidence=obs.confidence,
            is_resident=(obs.person_id is not None),
        )
        self.entities[new_ent.id] = new_ent
        # Carry the YOLO class label onto the entity for objects so the
        # cost function can do hard class matching on subsequent ticks
        # (a cup obs must not link to a phone entity, etc.).
        if obs.obj_class == "object":
            detected = obs.metadata.get("detected_class")
            if detected:
                new_ent.metadata["detected_class"] = detected
            source = obs.metadata.get("source")
            if source:
                new_ent.metadata["source"] = source
        if obs.visual_embedding is not None:
            new_ent.metadata["_visual_embedding"] = obs.visual_embedding
        await self.store.upsert_entity(new_ent)
        if obs.visual_embedding is not None:
            await self.store.upsert_embedding(new_ent.id, obs.visual_embedding)
        await self._emit(EventType.FIRST_SEEN, new_ent, obs)

    def _find_entity_by_person_id(self, person_id: int) -> Optional[WorldEntity]:
        for ent in self.entities.values():
            if ent.entity_type == "person" and ent.person_id == person_id:
                return ent
        return None

    # ────────────────────────────────────────────────────────────────────────
    # ENTITY EXPECTED, NOT SEEN — bounded-house disappearance logic
    # ────────────────────────────────────────────────────────────────────────

    async def _handle_unmatched_entity(
        self, ent: WorldEntity, camera: str, ts: datetime
    ) -> None:
        # Only act on entities that were PRESENT in this camera.
        if ent.last_seen_camera != camera or ent.state != EntityState.PRESENT:
            return
        if (camera in self._unhealthy_cameras
                or ent.metadata.get("suspended_due_to_camera_health")):
            return

        # Hysteresis: detectors flicker frame-to-frame (especially YOLO on
        # cluttered scenes). We use a rolling seen/missed history per
        # entity. LOST_VISIBILITY only fires when ALL of:
        #   - last confirmed sighting was at least `visibility_grace_seconds`
        #     ago (wall-time floor),
        #   - we have at least `visibility_min_samples` recent observations
        #     in the smoothing window (otherwise we don't have enough data
        #     to trust the rate — hold),
        #   - the seen-fraction over the smoothing window is below
        #     `visibility_seen_fraction_floor` (truly mostly-missing,
        #     not just a flicker).
        # Exit-polygon hits skip the grace entirely — exits are
        # deliberate, low-flicker events worth firing fast on.
        exit_match = self._classify_exit(ent.last_seen_bbox, camera)
        if exit_match is None:
            _record_visibility(ent, ts, seen=False)
            grace_seconds = float(self.cfg.get("visibility_grace_seconds", 3.0))
            window_s = float(self.cfg.get("visibility_window_seconds", 6.0))
            min_samples = int(self.cfg.get("visibility_min_samples", 4))
            seen_floor = float(self.cfg.get(
                "visibility_seen_fraction_floor", 0.25
            ))
            last_seen = _as_utc(ent.last_seen_ts) if ent.last_seen_ts else None
            elapsed = (ts - last_seen).total_seconds() if last_seen else 0.0
            frac, samples = _seen_fraction(ent, ts, window_s)
            # Hold (return) when any condition says "not lost yet":
            #   - wall-time grace not elapsed
            #   - not enough samples to trust the smoothing
            #   - smoothing says we're still seeing the entity often
            if (elapsed < grace_seconds
                    or samples < min_samples
                    or frac > seen_floor):
                return

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
            ent.state = EntityState.DEPARTED
            ent.last_state_change_ts = ts
            ent.metadata["departed_via"] = exit_match.get("name")
            ent.metadata["departed_ts"] = ts.isoformat()
            await self._emit(
                EventType.DEPARTED, ent, obs=None,
                metadata={
                    "via_exit": exit_match.get("name"),
                    "last_landmark": ent.last_seen_landmark,
                },
            )

        await self.store.upsert_entity(ent)

    def _classify_exit(
        self, bbox: Optional[tuple], camera: str
    ) -> Optional[dict]:
        """Return the matching exit dict if bbox center is inside any
        exit polygon. None otherwise."""
        if not bbox or camera not in self.cameras:
            return None
        cx, cy = bbox_center(bbox)
        for exit_def in self.cameras[camera].get("exits", []):
            polygon = exit_def.get("polygon") or []
            if point_in_polygon(cx, cy, polygon):
                return exit_def
        return None

    # ────────────────────────────────────────────────────────────────────────
    # PERIODIC TIMERS
    # ────────────────────────────────────────────────────────────────────────

    async def _timer_loop(self) -> None:
        tick = float(self.cfg.get("timer_tick_seconds", 2.0))
        while not self._stopped:
            try:
                await asyncio.sleep(tick)
                now = _utcnow()
                async with self._lock:
                    for ent in list(self.entities.values()):
                        ent.last_state_change_ts = _as_utc(ent.last_state_change_ts)
                        elapsed = now - ent.last_state_change_ts

                        # UNKNOWN_AT_BOOT → resolve after boot grace.
                        boot_secs = self.cfg.get("boot_resolution_seconds", 30)
                        if (ent.state == EntityState.UNKNOWN_AT_BOOT
                                and elapsed > timedelta(seconds=boot_secs)):
                            if ent.is_resident:
                                ent.state = EntityState.IN_HOUSE_UNMONITORED
                                ent.metadata["entered_unmonitored_via"] = "boot"
                            ent.last_state_change_ts = now
                            await self.store.upsert_entity(ent)

                        # TRANSITIONING → resolve based on target kind.
                        if ent.state == EntityState.TRANSITIONING:
                            handoff = self.cfg.get("T_handoff_seconds", 8)
                            if elapsed > timedelta(seconds=handoff):
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
                                    # Handoff failed — neighbor cam never saw them.
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

                        # STATIONARY_LONG detection.
                        if ent.state == EntityState.PRESENT:
                            mins = self.cfg.get("stationary_long_minutes", 5)
                            if elapsed > timedelta(minutes=mins):
                                already_fired = ent.metadata.get("stationary_fired_at")
                                marker = ent.last_state_change_ts.isoformat()
                                if already_fired != marker:
                                    ent.metadata["stationary_fired_at"] = marker
                                    await self.store.upsert_entity(ent)
                                    await self._emit(
                                        EventType.STATIONARY_LONG, ent, obs=None
                                    )
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("[WorldModel] timer loop iteration failed")

    async def _snapshot_loop(self) -> None:
        interval = float(self.cfg.get("snapshot_interval_seconds", 30.0))
        while not self._stopped:
            try:
                await asyncio.sleep(interval)
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
                            "last_seen_ts": (
                                e.last_seen_ts.isoformat() if e.last_seen_ts else None
                            ),
                            "confidence": e.confidence,
                            "attribution_confidence": e.last_attribution_confidence,
                            "is_resident": e.is_resident,
                        }
                        for e in self.entities.values()
                    ]
                await self.bus.publish("world.state_snapshot", {"entities": snap})
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("[WorldModel] snapshot loop iteration failed")

    # ────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ────────────────────────────────────────────────────────────────────────

    # Event types that are pure noise for anonymous closed-vocab objects
    # (cups/bottles/chairs). The user-facing world-events feed shouldn't
    # be flooded with "cup blinked." Open-vocab tracked objects (wallet,
    # keys) DO have display_name, so they bypass this filter.
    _ANON_OBJECT_NOISE_EVENTS: frozenset = frozenset({
        EventType.LOST_VISIBILITY,
        EventType.REAPPEARED,
        EventType.MOVED_WITHIN_ROOM,
    })

    async def _emit(
        self,
        event_type: EventType,
        ent: WorldEntity,
        obs: Optional[Observation],
        metadata: Optional[dict] = None,
    ) -> None:
        # Anonymous closed-vocab object spam suppression — see
        # _ANON_OBJECT_NOISE_EVENTS above.
        if (ent.entity_type == "object"
                and not ent.display_name
                and event_type in self._ANON_OBJECT_NOISE_EVENTS):
            return
        ts = _as_utc(obs.ts) if obs else _utcnow()
        # For cat/dog events, blend the observation's visual descriptors
        # into event metadata so §22.5's AnimalClusterBuilder has signal
        # to work with at cold-start. Color histogram is excluded — too
        # large for the per-event log; cluster builder uses color_class
        # + room + size, which is sufficient for the household's
        # discriminating pairs (Spooky/Velcro by room, Sparta/Serval by
        # size).
        merged_metadata: dict = dict(metadata or {})
        if obs is not None and obs.obj_class in ("cat", "dog"):
            for k in ("color_class", "size_normalized",
                      "coat_texture", "breed_class"):
                v = obs.metadata.get(k)
                if v is not None and k not in merged_metadata:
                    merged_metadata[k] = v
        if ent.entity_type == "object":
            for k in ("detected_class", "source", "openvocab_query"):
                v = (obs.metadata.get(k) if obs is not None else None)
                if v is None:
                    v = ent.metadata.get(k)
                if v is not None and k not in merged_metadata:
                    merged_metadata[k] = v
        entity_name = ent.display_name
        if not entity_name and ent.entity_type == "object":
            detected = merged_metadata.get("detected_class")
            if detected:
                safe_class = str(detected).strip().lower().replace(" ", "_")
                entity_name = f"unknown_{safe_class}_{ent.id[:6]}"
        if not entity_name:
            entity_name = f"unknown_{ent.entity_type}_{ent.id[:6]}"
        payload = {
            "id": str(uuid.uuid4()),
            "ts": ts.isoformat(),
            "entity_id": ent.id,
            "entity_name": entity_name,
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
            "metadata": merged_metadata,
        }
        await self.store.append_event(payload)
        await self.bus.publish("world.entity_event", payload)

    def _candidate_entities_for_camera(self, camera: str) -> list[WorldEntity]:
        """Pool of entities a new observation on `camera` could plausibly
        match. Two paths union'd (deduped by id):

          1. Spatio-temporal: entities recently seen on this camera or a
             neighbor. Original behavior — favors continuity.
          2. Resident pets in their declared home_room (or for cyclic
             cats, any of `cyclic_home_rooms`; for dogs, `home_rooms`).
             Without this, freshly-bootstrapped pets — which start with
             last_seen_ts=None — never enter the candidate pool, so
             every animal observation creates a new anonymous entity
             instead of linking to Spooky/Velcro/Sparta/etc. on day 1.
             Per §22, attribution on a sparse-data day is *hedged*, not
             absent: the cost function should at least *consider* the
             declared resident.
        """
        lookback = int(self.cfg.get("candidate_lookback_minutes", 2))
        cutoff = _utcnow() - timedelta(minutes=lookback)
        cam_room = self.cameras.get(camera, {}).get("room")

        out: dict[str, WorldEntity] = {}
        for e in self.entities.values():
            if e.last_seen_ts is not None:
                e.last_seen_ts = _as_utc(e.last_seen_ts)
            recent_here = bool(
                e.last_seen_ts and e.last_seen_ts > cutoff and (
                    e.last_seen_camera == camera
                    or self._cameras_are_neighbors(e.last_seen_camera, camera)
                )
            )
            home_match = bool(
                e.is_resident
                and e.entity_type in ("cat", "dog")
                and e.archived_at is None
                and cam_room is not None
                and self._pet_home_room_matches(e, cam_room)
            )
            if recent_here or home_match:
                out[e.id] = e
        return list(out.values())

    @staticmethod
    def _pet_home_room_matches(ent: WorldEntity, room: str) -> bool:
        """Does this resident pet declare `room` as one of its home rooms?
        Reads the seed metadata that pets.bootstrap_pets_from_config wrote
        — handles cats (home_room / cyclic_home_rooms) and dogs (home_rooms)."""
        seed = ent.metadata.get("seed", {}) or {}
        # Cats: single home_room, or cyclic with a list.
        hr = seed.get("home_room")
        if hr and hr != "cyclic" and hr == room:
            return True
        cyclic = seed.get("cyclic_home_rooms") or []
        if isinstance(cyclic, list) and room in cyclic:
            return True
        # Dogs: list of home_rooms.
        home_rooms = seed.get("home_rooms") or []
        if isinstance(home_rooms, list) and room in home_rooms:
            return True
        return False

    def _cameras_are_neighbors(
        self, a: Optional[str], b: Optional[str]
    ) -> bool:
        if not a or not b:
            return False
        if a == b:
            return True
        a_neighbors = {
            ex["to"] for ex in self.cameras.get(a, {}).get("exits", [])
            if ex.get("kind") == "to_room"
        }
        b_room = self.cameras.get(b, {}).get("room")
        return b_room in a_neighbors

    def _nearest_landmark(
        self, camera: str, point: tuple
    ) -> Optional[str]:
        for lm in self.cameras.get(camera, {}).get("landmarks", []):
            polygon = lm.get("polygon") or []
            if point_in_polygon(point[0], point[1], polygon):
                return lm.get("name")
        return None

    # ────────────────────────────────────────────────────────────────────────
    # PUBLIC QUERY API
    # ────────────────────────────────────────────────────────────────────────

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

    async def build_snapshot_for_prompt(
        self,
        max_entities: int = 8,
        max_events: int = 3,
        recent_event_minutes: int = 10,
    ) -> Optional[str]:
        """
        Compact world-state blurb for injection into LLM prompts. Designed
        to stay under ~200 tokens (≤800 chars) so it can ride the
        per-turn `extras` slot without blowing the context window.

        Includes:
            - the named residents currently in any in-house state
            - any PRESENT non-resident (anonymous person, cat, object)
              up to the entity cap, residents-first
            - the last `max_events` state-change events from the last
              `recent_event_minutes` minutes

        Returns None when the world is empty (no entities, no events) so
        the orchestrator can skip the extras key entirely instead of
        injecting a meaningless header.
        """
        in_house_states = {
            EntityState.PRESENT, EntityState.IN_ROOM_UNSEEN,
            EntityState.TRANSITIONING, EntityState.IN_HOUSE_UNMONITORED,
        }
        # Residents first, then anyone else PRESENT, capped at max_entities.
        residents = [
            e for e in self.entities.values()
            if e.is_resident and e.display_name and e.state in in_house_states
        ]
        residents.sort(key=lambda e: e.display_name or "")
        non_residents_present = [
            e for e in self.entities.values()
            if not e.is_resident and e.state == EntityState.PRESENT
        ]
        ordered = (residents + non_residents_present)[:max_entities]

        lines: list[str] = []
        if ordered:
            lines.append("World state (presence is ground truth — prefer over guessing):")
            for e in ordered:
                bits = [f"{e.display_name or '?'} ({e.entity_type})"]
                state_str = e.state.value
                room = e.last_seen_room or "?"
                if e.state == EntityState.PRESENT:
                    bits.append(f"in {room}")
                elif e.state == EntityState.IN_ROOM_UNSEEN:
                    if e.last_seen_landmark:
                        bits.append(
                            f"in {room}, last near {e.last_seen_landmark}, "
                            "didn't leave through a door"
                        )
                    else:
                        bits.append(f"in {room}, didn't leave through a door")
                elif e.state == EntityState.IN_HOUSE_UNMONITORED:
                    via = e.metadata.get("entered_unmonitored_via") or "?"
                    bits.append(f"in {via} (no camera)")
                elif e.state == EntityState.TRANSITIONING:
                    target = e.metadata.get("transitioning_target") or "?"
                    bits.append(f"between rooms (heading toward {target})")
                else:
                    bits.append(state_str)
                lines.append("  - " + ", ".join(bits))

        # Recent events — pulled from the persistent log so the LLM gets
        # the full audit (e.g. "Cole departed via front_door 4m ago"
        # even when Cole isn't currently in any in-house state).
        if max_events > 0:
            try:
                cutoff = _utcnow() - timedelta(
                    minutes=recent_event_minutes
                )
                rows = await self.store.search_events(
                    since=cutoff, limit=max_events,
                )
            except Exception as e:
                logger.debug(
                    f"[WorldModel] snapshot event lookup failed: {e}"
                )
                rows = []
            if rows:
                lines.append("Recent changes:")
                for r in rows:
                    name = r.get("entity_name") or "?"
                    et = r.get("event_type") or "?"
                    rm = r.get("room") or "?"
                    lines.append(f"  - {name}: {et} ({rm})")

        if not lines:
            return None
        return "\n".join(lines)

    # ── Object cost function (§23.6) ──────────────────────────────────────

    def _typical_rooms_for(self, obj_class: Optional[str]) -> list[str]:
        """Lookup of typical_rooms[] for an object class. Built lazily
        from config.tracked_objects.open_vocabulary on first call so
        the WorldModel doesn't have to know about the open-vocab
        config at construct time."""
        if not hasattr(self, "_typical_rooms_cache"):
            self._typical_rooms_cache: dict[str, list[str]] = {}
            cfg_objects = (
                (self.cfg.get("tracked_objects") or {}).get(
                    "open_vocabulary",
                ) or []
            )
            for entry in cfg_objects:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name")
                if name:
                    self._typical_rooms_cache[name] = list(
                        entry.get("typical_rooms") or [],
                    )
        if not obj_class:
            return []
        return self._typical_rooms_cache.get(obj_class, [])

    # ── Animal cost function (§22.7) ───────────────────────────────────────

    # Per-species component weights. Cat weights match §22.7 exactly.
    # Dog weights deweight room prior (~30%) and bump size (~50%) per
    # the §22.7 closing note — Summer-vs-Dalila gets resolved on size +
    # breed_class alone, so location prior matters less.
    _COST_WEIGHTS: dict[str, dict[str, float]] = {
        "cat": {
            "emb": 0.20, "hist": 0.20, "size": 0.15,
            "location": 0.30, "time": 0.05,
        },
        "dog": {
            "emb": 0.15, "hist": 0.20, "size": 0.225,
            "location": 0.20, "time": 0.05,
        },
    }

    def _animal_pair_cost(
        self, obs: Observation, ent: WorldEntity, species: str,
    ) -> float:
        """
        Cats and dogs share structure: hard color filter, then a weighted
        sum of (visual emb / hist / size / location prior / time gap)
        plus continuity and co-occurrence bonuses. Weights differ per
        species (see `_COST_WEIGHTS`).
        """
        obs.ts = _as_utc(obs.ts)
        w = self._COST_WEIGHTS.get(species, self._COST_WEIGHTS["cat"])

        # Color class hard filter — solid-black observation can't be a
        # tabby resident, etc. unknown ↔ unknown is allowed (no signal).
        obs_color = obs.metadata.get("color_class", "unknown")
        ent_color = ent.metadata.get("seed", {}).get("color_class", "unknown")
        if (obs_color != "unknown" and ent_color != "unknown"
                and obs_color != ent_color):
            return self.cfg["cost_reject"] * 2

        # ── Visual: histogram (Bhattacharyya) + embedding (cosine).
        hist_cost = _hist_bhattacharyya(
            obs.metadata.get("color_histogram"),
            ent.metadata.get("color_histogram"),
        )

        ent_emb = ent.metadata.get("_visual_embedding")
        obs_emb = obs.visual_embedding
        if ent_emb is not None and obs_emb is not None:
            sim = float(
                np.dot(obs_emb, ent_emb)
                / (np.linalg.norm(obs_emb) * np.linalg.norm(ent_emb) + 1e-9)
            )
            emb_cost = 1.0 - sim
        else:
            emb_cost = 0.5

        # ── Size cost: prefer learned per-room stats, fall back to seed tier.
        profile = ent.metadata.get("behavioral_profile", {}) or {}
        obs_size = obs.metadata.get("size_normalized")
        per_room = profile.get("bbox_size_per_room", {}).get(obs.room, {}) or {}
        if per_room and obs_size is not None and per_room.get("n", 0) >= 5:
            mean_sz = per_room["mean"]
            std_sz = max(per_room["std"], 1e-3)
            z = abs(float(obs_size) - mean_sz) / std_sz
            size_cost = float(min(z / 3.0, 1.0))
        else:
            size_cost = _size_cost_from_seed(
                obs_size, ent.metadata.get("seed", {})
            )

        # ── Location prior — heavy hitter for Spooky/Velcro and Smudge/Onyx.
        hour = obs.ts.hour
        by_hour = profile.get("room_distribution_by_hour", {}) or {}
        # JSON keys come back as strings; accept both.
        hour_dist = by_hour.get(hour) or by_hour.get(str(hour)) or {}
        room_dist = profile.get("room_distribution", {}) or {}
        p_room = hour_dist.get(obs.room) or room_dist.get(obs.room) or 0.05
        location_cost = float(min(-np.log(p_room + 0.01), 2.0))

        # ── Continuity bonus: same camera in last few seconds is a strong
        # negative-cost (i.e. likely the same animal).
        if ent.last_seen_ts:
            ent.last_seen_ts = _as_utc(ent.last_seen_ts)
            seconds_gone = (obs.ts - ent.last_seen_ts).total_seconds()
        else:
            seconds_gone = 1e6
        if obs.camera == ent.last_seen_camera and seconds_gone < 5:
            continuity = -0.3
        elif obs.room == ent.last_seen_room and seconds_gone < 30:
            continuity = -0.15
        else:
            continuity = 0.0

        # ── Co-occurrence tie-breaker: if a known partner of this entity
        # was just seen in a *different* room, that's evidence this is the
        # entity (because partners typically aren't simultaneously alone
        # together — they cohabit). The §22.7 note specifically calls this
        # out for Smudge/Onyx; works for Spooky/Velcro the same way.
        co_partners = profile.get("co_occurrence_partners", {}) or {}
        other_seen = self._other_animal_seen_recently(
            ent, obs.ts, exclude_room=obs.room, lookback_seconds=30,
        )
        co_bonus = -0.2 if (other_seen and other_seen in co_partners) else 0.0

        cost = max(0.0, (
            w["emb"] * emb_cost
            + w["hist"] * hist_cost
            + w["size"] * size_cost
            + w["location"] * location_cost
            + w["time"] * (min(seconds_gone / 60.0, 1.0))
            + continuity
            + co_bonus
        ))
        return cost

    def _other_animal_seen_recently(
        self, ent: WorldEntity, ts: datetime, exclude_room: str,
        lookback_seconds: int = 30,
    ) -> Optional[str]:
        """Did any OTHER same-species entity get observed in a different
        room within `lookback_seconds`? Returns its display_name or None."""
        cutoff = ts - timedelta(seconds=lookback_seconds)
        for other in self.entities.values():
            if other.last_seen_ts is not None:
                other.last_seen_ts = _as_utc(other.last_seen_ts)
            if (other.entity_type == ent.entity_type
                    and other.id != ent.id
                    and other.last_seen_ts and other.last_seen_ts >= cutoff
                    and other.last_seen_room
                    and other.last_seen_room != exclude_room):
                return other.display_name
        return None

    # ── §23.8 stale-object pruning ────────────────────────────────────────

    async def prune_stale_objects(
        self, max_age_days: int = 30,
    ) -> int:
        """Soft-delete object entities that haven't been seen in
        `max_age_days` AND have no INTERACTED_WITH / PICKED_UP /
        PLACED_DOWN events in their history. Touched objects keep
        their row even when invisible — story value matters. Marked
        with metadata.pruned=True so the candidate-pool filter
        excludes them; the row itself stays for query history.

        Called nightly off the orchestrator's daily-task loop alongside
        BehavioralProfileBuilder. Returns the count of pruned entities."""
        now = _utcnow()
        cutoff = now - timedelta(days=max_age_days)
        pruned = 0
        async with self._lock:
            for ent in list(self.entities.values()):
                if ent.entity_type != "object":
                    continue
                if ent.metadata.get("pruned"):
                    continue
                if ent.last_seen_ts is not None:
                    ent.last_seen_ts = _as_utc(ent.last_seen_ts)
                if ent.last_seen_ts is not None and ent.last_seen_ts >= cutoff:
                    continue
                # Check for any interaction history before pruning.
                interactions = await self.store.search_events(
                    entity_id=ent.id,
                    event_types=[
                        "interacted_with", "picked_up", "placed_down",
                    ],
                    limit=1,
                )
                if interactions:
                    continue
                ent.metadata["pruned"] = True
                await self.store.upsert_entity(ent)
                pruned += 1
        if pruned:
            logger.info(
                f"[WorldModel] prune_stale_objects: marked {pruned} "
                f"object entit(y/ies) as pruned (≥{max_age_days}d, "
                "no interaction history)"
            )
        return pruned

    # ── Object cost (§23.6) ───────────────────────────────────────────────

    def _object_pair_cost(self, obs: Observation, ent: WorldEntity) -> float:
        """§23.6 — full object association cost. When CLIP embeddings
        are available on both sides, the visual signal dominates;
        without them the cost falls through to the closed-vocab
        spatial-temporal continuity logic that landed earlier (a
        same-class same-room recent match wins, otherwise reject).

        Cost components:
          • Class match (hard reject) — a wallet obs cannot match a
            phone entity even if their CLIP embeddings happen to
            cluster.
          • CLIP cosine similarity (when both sides have an embedding).
          • Room prior — same room is essentially free; a typical-
            rooms hit is a soft penalty; otherwise expensive (objects
            don't relocate themselves).
          • Time decay — entities not seen in days lose match priority
            so finding a similar object after a long gap doesn't
            silently re-attribute it.
        """
        obs.ts = _as_utc(obs.ts)
        obs_class = obs.metadata.get("detected_class")
        ent_class = ent.metadata.get("detected_class")
        if (obs_class and ent_class and obs_class != ent_class):
            return self.cfg["cost_reject"] * 2

        ent_emb = ent.metadata.get("_visual_embedding")
        obs_emb = obs.visual_embedding

        # Path A — both sides have CLIP embeddings: the spec's full
        # §23.6 cost function (emb 0.55 · room 0.30 · time 0.15).
        if ent_emb is not None and obs_emb is not None:
            sim = float(
                np.dot(obs_emb, ent_emb)
                / (np.linalg.norm(obs_emb)
                   * np.linalg.norm(ent_emb) + 1e-9)
            )
            emb_cost = 1.0 - sim
            if ent.last_seen_room == obs.room:
                room_cost = 0.0
            elif (obs_class and obs.room
                    and obs.room in self._typical_rooms_for(obs_class)):
                room_cost = 0.25
            else:
                room_cost = 0.5
            if ent.last_seen_ts:
                ent.last_seen_ts = _as_utc(ent.last_seen_ts)
                days_gone = (
                    (obs.ts - ent.last_seen_ts).total_seconds() / 86400.0
                )
                time_cost = float(min(days_gone / 14.0, 0.5))
            else:
                time_cost = 0.5
            return float(max(0.0, (
                0.55 * emb_cost
                + 0.30 * room_cost
                + 0.15 * time_cost
            )))

        # Path B — closed-vocab spatial-temporal fallback (the original
        # logic that landed before CLIP). Used when one or both sides
        # lack a visual embedding (Null encoder, embed compute failed).
        if not ent.last_seen_ts:
            return self.cfg["cost_reject"] * 2
        ent.last_seen_ts = _as_utc(ent.last_seen_ts)
        seconds_gone = (obs.ts - ent.last_seen_ts).total_seconds()
        if seconds_gone > 15 * 60:
            return self.cfg["cost_reject"] * 2

        if obs.room == ent.last_seen_room:
            room_cost_b = 0.0
        else:
            room_cost_b = 0.7

        spatial = self._spatial_distance(obs, ent)
        iou = (
            bbox_iou(obs.bbox, ent.last_seen_bbox)
            if ent.last_seen_bbox else 0.0
        )
        iou_term = 1.0 - float(iou)
        time_cost_b = min(seconds_gone / (15 * 60.0), 1.0)
        return float(max(0.0, (
            0.5 * room_cost_b
            + 0.25 * spatial
            + 0.15 * iou_term
            + 0.10 * time_cost_b
        )))


# ── Module-level helpers (§22.7) ────────────────────────────────────────────


def _hist_bhattacharyya(h1: Any, h2: Any) -> float:
    """Bhattacharyya-derived distance, normalized 0–1. Lower = more similar.
    Returns 0.5 (neutral) when either side is missing."""
    if h1 is None or h2 is None:
        return 0.5
    a = np.asarray(h1, dtype=np.float32)
    b = np.asarray(h2, dtype=np.float32)
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return 0.5
    bc = float(np.sum(np.sqrt(a * b)))
    return float(min(1.0, np.sqrt(max(0.0, 1.0 - bc))))


def _size_cost_from_seed(obs_size: Optional[float], seed: dict) -> float:
    """Size cost when learned per-room stats aren't available yet. Maps
    expected_size string → target normalized-area, returns log-distance."""
    if obs_size is None:
        return 0.5
    expected_size = (seed or {}).get("expected_size", "medium")
    targets = {
        "small": 0.02, "small-large": 0.03, "medium": 0.04,
        "medium-large": 0.06, "large": 0.07, "xl": 0.10,
    }
    target = targets.get(str(expected_size), 0.04)
    return float(min(abs(np.log(max(float(obs_size), 1e-4) / target)) / 2.0, 1.0))
