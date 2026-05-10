"""
JARVIS — World Model
====================
Mission: Adapter from raw detector outputs to normalized Observation
         payloads consumed by WorldModel. Subscribes (via polling for
         this repo's CameraManager) to per-room frame streams, runs
         YOLO + face detection + identity match per frame, and emits
         `vision.observation` events on the bus.

         The doc's §18 reference uses an `iter_frames` async iterator
         on CameraManager — this repo uses `capture_frame_async(room)`
         pull-style instead. Same contract: ObservationBuilder is a
         consumer, owns no camera state, and adapts to whatever
         CameraManager exposes. One asyncio task per camera-equipped
         room, polling at the room's `fps_active` cadence.

         Phase 1.3 scope is people-only. Cat / object enrichers are
         stubbed per §22 / §23 — they emit minimal Observations so the
         pipeline shape is right, full enrichment lands in Phase 4.

Modules: modules/vision/observation_builder.py
Classes: ObservationBuilder, Detection
Spec:    new 2.md §12 (adapter) + §18 (full code).

#todo: Idle-mode FPS gate — when no person observed for N seconds,
       drop to fps_idle to save GPU. Today we run at fps_active
       continuously per room.
#todo: Per-room failure circuit-breaker — if `_build_for_frame` raises
       repeatedly (e.g. stale RTSP that never recovers), back off
       exponentially instead of hammering the CameraManager.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
from loguru import logger

from modules.world_model.types import Observation


@dataclass
class Detection:
    """Lightweight wrapper over the dict ObjectDetector returns. The §18
    spec uses `det.bbox / det.class_name / det.confidence` attributes;
    this gives us that ergonomic access without forcing ObjectDetector
    to change its return shape."""
    bbox: tuple[int, int, int, int]
    class_name: str
    confidence: float

    @classmethod
    def from_dict(cls, d: dict) -> "Detection":
        box = d.get("box") or d.get("bbox") or [0, 0, 0, 0]
        return cls(
            bbox=tuple(int(v) for v in box),  # type: ignore[arg-type]
            class_name=str(d.get("class") or d.get("class_name") or "unknown"),
            confidence=float(d.get("confidence", 0.0)),
        )


class ObservationBuilder:
    """
    Per-room observation pipeline. Reads frames from CameraManager, runs
    detections, enriches person detections with IdentityManager identity,
    and emits normalized Observation batches on `vision.observation`.

    One independent asyncio task per camera-equipped room — different
    rooms have different FPS budgets, no shared lock between them.
    """

    # Phase 4 starting set — cheap stuff, expand later. Phase 1.3 doesn't
    # USE these yet (cat/object enrichers are still stubs); listing here
    # so the dispatcher in `_build_for_frame` is structurally complete.
    TRACKED_OBJECT_CLASSES: set[str] = {
        "cell phone", "cup", "book", "laptop", "bottle", "remote",
    }

    def __init__(
        self,
        bus: Any,
        camera_manager: Any,
        object_detector: Any,
        face_recognizer: Any,
        identity_manager: Any,
        posture_analyzer: Optional[Any],
        rooms_config: list[dict],
        snapshot_dir: Optional[Path] = None,
    ) -> None:
        self.bus = bus
        self.cm = camera_manager
        self.detector = object_detector
        self.face = face_recognizer
        self.identity = identity_manager
        self.posture = posture_analyzer
        # Index rooms by id for fast `_loop_for_room` lookups; preserve
        # the original dicts so loop reads fps_active etc. directly.
        self.rooms: dict[str, dict] = {r["id"]: r for r in rooms_config}
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        if self.snapshot_dir is not None:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        # Track per-room loop tasks so we can cancel on shutdown without
        # leaking. Keyed by room id.
        self._tasks: dict[str, asyncio.Task] = {}
        self._stopped: bool = False

    # ── Lifecycle ───────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Spawn one polling loop per camera-equipped room. Idempotent —
        re-running on a started builder is a no-op."""
        if self._tasks:
            return
        # Honor the runtime CameraManager's view of which rooms actually
        # opened a stream (config can list a room whose driver failed to
        # connect — that room shouldn't get a polling loop).
        try:
            available = set(self.cm.get_available_rooms())
        except Exception as e:
            logger.warning(
                f"[ObservationBuilder] CameraManager.get_available_rooms() "
                f"failed: {e}; using config rooms"
            )
            available = set(self.rooms.keys())

        for room_id, room_cfg in self.rooms.items():
            wm = room_cfg.get("world_model") or {}
            if not wm.get("enabled", True):
                continue
            if room_id not in available:
                logger.info(
                    f"[ObservationBuilder] skipping '{room_id}' — no live camera"
                )
                continue
            self._tasks[room_id] = asyncio.create_task(
                self._loop_for_room(room_id),
                name=f"observation_builder:{room_id}",
            )
        logger.info(
            f"[ObservationBuilder] started on {len(self._tasks)} room(s): "
            + ", ".join(sorted(self._tasks.keys()))
        )

    async def stop(self) -> None:
        """Cancel all per-room loops. Safe to call before start()."""
        self._stopped = True
        for room_id, task in list(self._tasks.items()):
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks.clear()

    # ── Per-room frame pump ────────────────────────────────────────────────

    async def _loop_for_room(self, room_id: str) -> None:
        """One asyncio task per room. Polls CameraManager at fps_active,
        runs the pipeline, publishes `vision.observation`. Fails soft —
        any per-frame error logs + continues."""
        room_cfg = self.rooms.get(room_id, {})
        fps = max(1, int(room_cfg.get("fps_active", 5)))
        period_s = 1.0 / fps
        logger.debug(
            f"[ObservationBuilder] '{room_id}' polling at {fps} fps"
        )
        while not self._stopped:
            t0 = asyncio.get_event_loop().time()
            try:
                frame = await self.cm.capture_frame_async(room_id)
                if frame is not None:
                    ts = datetime.now(timezone.utc)
                    observations = await self._build_for_frame(
                        room_id, frame, ts
                    )
                    if observations:
                        await self.bus.publish("vision.observation", {
                            "camera": room_id,
                            "room": room_id,
                            "ts": ts,
                            "observations": observations,
                        })
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception(
                    f"[ObservationBuilder] frame error in '{room_id}': {e}"
                )
            # Pace the loop. capture_frame_async + detection together can
            # take longer than the period at high FPS — in that case we
            # just run flat-out, no sleep.
            elapsed = asyncio.get_event_loop().time() - t0
            slack = period_s - elapsed
            if slack > 0:
                try:
                    await asyncio.sleep(slack)
                except asyncio.CancelledError:
                    raise

    # ── Pipeline ───────────────────────────────────────────────────────────

    async def _build_for_frame(
        self, room: str, frame: np.ndarray, ts: datetime
    ) -> list[Observation]:
        observations: list[Observation] = []

        # 1. Object detection (YOLO).
        raw_dets = await self.detector.detect_async(frame)
        detections = [Detection.from_dict(d) for d in raw_dets]

        # 2. Optional: hand bboxes (Phase 5; for now empty list).
        hand_bboxes: list[tuple] = []

        # 3. Optional: room-wide posture (Phase 5; for now None).
        posture: Optional[str] = None

        frame_h, frame_w = frame.shape[:2]

        for det in detections:
            cls = det.class_name
            try:
                if cls == "person":
                    obs = await self._build_person_obs(
                        frame, det, room, ts, frame_w, frame_h,
                        hand_bboxes, posture,
                    )
                elif cls == "cat":
                    obs = self._build_cat_obs(
                        frame, det, room, ts, frame_w, frame_h
                    )
                elif cls in self.TRACKED_OBJECT_CLASSES:
                    obs = self._build_object_obs(
                        frame, det, room, ts, frame_w, frame_h
                    )
                else:
                    continue
                observations.append(obs)
            except Exception as e:
                logger.debug(
                    f"[ObservationBuilder] enricher failed for '{cls}' "
                    f"in '{room}': {e}"
                )

        return observations

    async def _build_person_obs(
        self,
        frame: np.ndarray,
        det: Detection,
        room: str,
        ts: datetime,
        fw: int,
        fh: int,
        hand_bboxes: list[tuple],
        posture: Optional[str],
    ) -> Observation:
        bbox = det.bbox
        x1, y1, x2, y2 = (int(v) for v in bbox)
        # Defensive clamp — YOLO occasionally emits negatives or out-of-frame.
        x1 = max(0, min(fw - 1, x1))
        y1 = max(0, min(fh - 1, y1))
        x2 = max(0, min(fw, x2))
        y2 = max(0, min(fh, y2))
        crop = frame[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else None

        # Save crop for enrollment / dashboard. Optional: gated on
        # snapshot_dir being set so tests / lightweight setups skip it.
        crop_path: Optional[Path] = None
        if self.snapshot_dir is not None and crop is not None and crop.size > 0:
            try:
                import cv2  # noqa: E402 — lazy so test envs without cv2 don't choke
                fname = (
                    f"person_{room}_{ts.strftime('%Y%m%dT%H%M%S')}_"
                    f"{uuid.uuid4().hex[:6]}.jpg"
                )
                crop_path = self.snapshot_dir / fname
                ok = cv2.imwrite(str(crop_path), crop)
                if not ok:
                    crop_path = None
            except Exception as e:
                logger.debug(
                    f"[ObservationBuilder] snapshot save failed in '{room}': {e}"
                )
                crop_path = None

        # Face detection + identity match. Run face detector on the
        # PERSON crop (not the whole frame) — much cheaper, and
        # avoids spurious matches on people standing behind the
        # primary subject. Falls back to no-face when crop is empty.
        face: Optional[dict] = None
        if crop is not None and crop.size > 0:
            try:
                face_results = await self.face.detect_and_embed(crop)
                if face_results:
                    face = face_results[0]
            except Exception as e:
                logger.debug(
                    f"[ObservationBuilder] face detect failed in '{room}': {e}"
                )

        person_id: Optional[int] = None
        person_name: Optional[str] = None
        identity_conf: float = 0.0
        face_metadata: dict[str, Any] = {}
        if face is not None:
            try:
                match = await self.identity.identify_from_embedding_async(
                    face["embedding"], modality="face"
                )
                if match is not None:
                    person_id = int(match.person_id)
                    person_name = match.name
                    identity_conf = float(match.similarity)
            except Exception as e:
                logger.debug(
                    f"[ObservationBuilder] identity match failed in '{room}': {e}"
                )
            face_metadata = {
                "face_embedding": face["embedding"],
                "yaw": float(face.get("yaw", 0.0)),
                "pitch": float(face.get("pitch", 0.0)),
                "roll": float(face.get("roll", 0.0)),
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

    def _build_cat_obs(
        self,
        frame: np.ndarray,
        det: Detection,
        room: str,
        ts: datetime,
        fw: int,
        fh: int,
    ) -> Observation:
        # Phase 4 — full enrichment in §22.3 (color/size/coat-texture).
        return Observation(
            camera=room, room=room, obj_class="cat",
            bbox=tuple(det.bbox), confidence=det.confidence, ts=ts,
            metadata={"frame_width": fw, "frame_height": fh},
        )

    def _build_object_obs(
        self,
        frame: np.ndarray,
        det: Detection,
        room: str,
        ts: datetime,
        fw: int,
        fh: int,
    ) -> Observation:
        # Phase 4 — full enrichment in §23.5 (CLIP embeddings, dedup).
        return Observation(
            camera=room, room=room, obj_class="object",
            bbox=tuple(det.bbox), confidence=det.confidence, ts=ts,
            metadata={
                "detected_class": det.class_name,
                "frame_width": fw, "frame_height": fh,
            },
        )


def _laplacian_var(image: Optional[np.ndarray]) -> float:
    """Blur metric — variance of the Laplacian. Lower = blurrier.
    Used by IdentityManager's quality gate on auto-enrolled samples
    so a motion-blurred face crop doesn't pollute the centroid bank.
    Returns 0.0 on empty input.
    """
    if image is None or image.size == 0:
        return 0.0
    try:
        import cv2  # noqa: E402 — lazy
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0
