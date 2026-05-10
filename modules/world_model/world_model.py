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
from datetime import datetime, timedelta
from typing import Any, Optional

import numpy as np
from loguru import logger
from scipy.optimize import linear_sum_assignment

from modules.world_model.geometry import bbox_center, bbox_iou, point_in_polygon
from modules.world_model.store import WorldStore
from modules.world_model.types import (
    EntityState,
    EventType,
    Observation,
    WorldEntity,
)


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
            "movement_jitter_threshold": 0.08,
            "posture_debounce_frames": 3,
            "interaction_debounce_frames": 3,
            "T_handoff_seconds": 8,
            "stationary_long_minutes": 5,
            "cosine_match_strong": 0.6,
            "candidate_lookback_minutes": 2,
            "snapshot_interval_seconds": 30.0,
            "timer_tick_seconds": 2.0,
            "boot_resolution_seconds": 30,
            **(config or {}),
        }

        # Per-camera topology lookup from rooms config.
        self.cameras: dict[str, dict] = self._build_camera_topology(rooms_config)

        self.entities: dict[str, WorldEntity] = {}
        self._lock = asyncio.Lock()
        self._unhealthy_cameras: set[str] = set()
        # Background tasks the model owns; cancelled in stop().
        self._timer_task: Optional[asyncio.Task] = None
        self._snapshot_task: Optional[asyncio.Task] = None
        self._stopped = False

    # ── Topology ───────────────────────────────────────────────────────────

    @staticmethod
    def _build_camera_topology(rooms_config: list[dict]) -> dict:
        """
        Build a per-camera topology dict from the rooms[] config.
        Camera ID derives from room ID — one camera per room in the
        current config. Multi-cam-per-room is a future expansion.
        """
        topology: dict[str, dict] = {}
        for room in rooms_config:
            wm = room.get("world_model")
            if not wm or not wm.get("enabled", True):
                continue
            cam_id = room["id"]
            topology[cam_id] = {
                "room": room["id"],
                "frame_width": wm.get("frame_width", 640),
                "frame_height": wm.get("frame_height", 480),
                "exits": wm.get("exits", []),
                "landmarks": wm.get("landmarks", []),
            }
        return topology

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

    async def _load_from_store(self) -> None:
        """Hydrate entities from disk. Every PRESENT entity becomes
        UNKNOWN_AT_BOOT for the first 30s — observations resolve, the
        timer demotes survivors to IN_HOUSE_UNMONITORED."""
        for ent in await self.store.load_entities():
            self.entities[ent.id] = ent
        boot_ts = datetime.utcnow()
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
        async with self._lock:
            camera = payload["camera"]
            ts = payload["ts"] if isinstance(payload["ts"], datetime) \
                 else datetime.fromisoformat(payload["ts"])
            observations: list[Observation] = payload["observations"]

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
        # Identity wins if both sides have it.
        if obs.person_id is not None and ent.person_id is not None:
            if obs.person_id == ent.person_id:
                return 0.05 * self._spatial_distance(obs, ent)
            else:
                return self.cfg["cost_reject"] * 2  # different people, hard reject

        # Fallback to spatial-temporal continuity.
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
            asyncio.create_task(self._enroll_async(obs))

    async def _enroll_async(self, obs: Observation) -> None:
        """Hand off to IdentityManager — fire and forget. IM may or may
        not implement `consider_new_sample_async`; if not, we silently
        skip (the world model's job ends at the hand-off)."""
        try:
            handler = getattr(
                self.identity_manager, "consider_new_sample_async", None
            )
            if handler is None:
                return
            await handler(
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
        """
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
        # a person_id that matches an existing entity, route directly.
        if obs.obj_class == "person" and obs.person_id is not None:
            existing = self._find_entity_by_person_id(obs.person_id)
            if existing:
                attribution_conf = obs.person_match_confidence
                if (existing.last_seen_camera
                        and existing.last_seen_camera != obs.camera):
                    existing.metadata["identity_overrode_continuity"] = True
                await self._update_matched(existing, obs, ts, attribution_conf)
                return

        # Wider pool for cats/objects: same-type entity with strong
        # embedding match. People are caught by the person_id path above.
        if obs.obj_class in ("cat", "object") and obs.visual_embedding is not None:
            best: Optional[WorldEntity] = None
            best_sim = 0.0
            for ent in self.entities.values():
                if ent.entity_type != obs.obj_class:
                    continue
                emb = ent.metadata.get("_visual_embedding")
                if emb is None:
                    continue
                sim = float(
                    np.dot(obs.visual_embedding, emb)
                    / (np.linalg.norm(obs.visual_embedding)
                       * np.linalg.norm(emb) + 1e-9)
                )
                if sim > best_sim:
                    best, best_sim = ent, sim
            if best is not None and best_sim >= self.cfg.get("cosine_match_strong", 0.6):
                await self._update_matched(best, obs, ts, attribution_conf=best_sim)
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
                now = datetime.utcnow()
                async with self._lock:
                    for ent in list(self.entities.values()):
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

    async def _emit(
        self,
        event_type: EventType,
        ent: WorldEntity,
        obs: Optional[Observation],
        metadata: Optional[dict] = None,
    ) -> None:
        ts = obs.ts if obs else datetime.utcnow()
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
        payload = {
            "id": str(uuid.uuid4()),
            "ts": ts.isoformat(),
            "entity_id": ent.id,
            "entity_name": (
                ent.display_name or f"unknown_{ent.entity_type}_{ent.id[:6]}"
            ),
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
        cutoff = datetime.utcnow() - timedelta(minutes=lookback)
        cam_room = self.cameras.get(camera, {}).get("room")

        out: dict[str, WorldEntity] = {}
        for e in self.entities.values():
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
                cutoff = datetime.utcnow() - timedelta(
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
            if (other.entity_type == ent.entity_type
                    and other.id != ent.id
                    and other.last_seen_ts and other.last_seen_ts >= cutoff
                    and other.last_seen_room
                    and other.last_seen_room != exclude_room):
                return other.display_name
        return None

    # ── Object cost (§23 closed-vocab portion) ────────────────────────────

    def _object_pair_cost(self, obs: Observation, ent: WorldEntity) -> float:
        """Closed-vocab object association — for the YOLO-detected
        TRACKED_OBJECT_CLASSES (cell phone, cup, book, laptop, bottle,
        remote). Open-vocab + CLIP embedding matching is §23.4-§23.6
        and lands when the encoder is bootstrapped.

        Cost components:
          - Class match (hard): a phone observation can't link to a cup
            entity. Class is read from obs.metadata.detected_class /
            ent.metadata.detected_class because obs.obj_class is the
            generic "object" label.
          - Same-room continuity: an object is overwhelmingly likely
            to be the same instance if observed in the same room
            within the candidate-lookback window (default 2 minutes).
          - Bbox-center distance + IoU: tighter spatial match wins.
          - Time gap penalty.
        """
        obs_class = obs.metadata.get("detected_class")
        ent_class = ent.metadata.get("detected_class")
        if (obs_class and ent_class and obs_class != ent_class):
            return self.cfg["cost_reject"] * 2

        if not ent.last_seen_ts:
            # No history → fresh object, let the unmatched path pick it up.
            return self.cfg["cost_reject"] * 2
        seconds_gone = (obs.ts - ent.last_seen_ts).total_seconds()
        # Objects don't move themselves — broad lookback is fine, but
        # stale entities (>15 min idle) shouldn't capture new sightings.
        if seconds_gone > 15 * 60:
            return self.cfg["cost_reject"] * 2

        if obs.room == ent.last_seen_room:
            room_cost = 0.0
        else:
            # Different room is a strong negative — a coffee cup
            # appearing on a different camera is more likely a new
            # cup than the old one teleporting.
            room_cost = 0.7

        spatial = self._spatial_distance(obs, ent)
        iou = bbox_iou(obs.bbox, ent.last_seen_bbox) if ent.last_seen_bbox else 0.0
        # IoU is a positive bonus — higher overlap, lower cost.
        iou_term = 1.0 - float(iou)

        time_cost = min(seconds_gone / (15 * 60.0), 1.0)

        return float(max(0.0, (
            0.5 * room_cost
            + 0.25 * spatial
            + 0.15 * iou_term
            + 0.10 * time_cost
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
