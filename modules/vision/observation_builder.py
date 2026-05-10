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

         Phase 1.3 scope was people-only. Phase 4 (§22) lands the cat
         and dog enrichers — color/size/coat-texture for cats, color/
         breed-class for dogs. CLIP-based visual_embedding is still
         deferred to §23 (the encoder isn't bootstrapped here yet);
         until then, cost functions fall back to histogram + size +
         location prior, which is enough on day-30 disambiguation.

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

from modules.vision.hand_detector import (
    bbox_overlaps_or_within as _bbox_overlaps_or_within,
)
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

    # YOLO species → enricher method. New species register one row each.
    # Membership in this map gates `_build_animal_obs` from `_build_for_frame`,
    # so a class not listed here gets ignored at the entity layer (per §22.0
    # — non-whitelisted species ride the bus as raw observations only).
    ANIMAL_ENRICHERS: dict[str, str] = {
        "cat": "_build_cat_obs",
        "dog": "_build_dog_obs",
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
        hand_detector: Optional[Any] = None,
    ) -> None:
        self.bus = bus
        self.cm = camera_manager
        self.detector = object_detector
        self.face = face_recognizer
        self.identity = identity_manager
        self.posture = posture_analyzer
        # §24.1 — MediaPipe Hands. Optional; without it the hand_bboxes
        # field on person observations stays empty and INTERACTED_WITH /
        # PICKED_UP / PLACED_DOWN events never fire (state machine
        # safely degrades). NullHandDetector is the standard test stub.
        self.hand_detector = hand_detector
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
                    # Always publish — empty `observations` is exactly
                    # the signal WorldModel needs to fire LOST_VISIBILITY
                    # for entities that were PRESENT in this camera last
                    # tick but vanished this tick (under-desk, walked
                    # off-frame, etc.). Suppressing empty batches breaks
                    # every disappearance-driven state transition in
                    # §17, §22.9, §29.2. Camera capture failure (frame
                    # is None) is different — that's silence, not an
                    # empty observation; the WorldModel handles it via
                    # camera.health from CameraManager.
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

        # 2. Hand detection (§24.1) — once per frame, attached per-person
        # below. Cheap to skip when no detector is wired.
        all_hands: list[dict] = []
        if self.hand_detector is not None:
            try:
                all_hands = await self.hand_detector.detect_async(frame)
            except Exception as e:
                logger.debug(
                    f"[ObservationBuilder] hand detect failed in "
                    f"'{room}': {e}"
                )

        # 3. Optional: room-wide posture (Phase 5; for now None).
        posture: Optional[str] = None

        frame_h, frame_w = frame.shape[:2]

        for det in detections:
            cls = det.class_name
            try:
                if cls == "person":
                    # Attach hands whose bbox sits inside (or just at
                    # the edge of) this person's bbox. Multiple people
                    # in the same crop may share a hand attribution
                    # — the InteractionMonitor uses world-model identity
                    # to disambiguate at the event-correlation layer.
                    person_hand_details = [
                        h for h in all_hands
                        if _bbox_overlaps_or_within(
                            h.get("bbox", (0, 0, 0, 0)), det.bbox, slack=20,
                        )
                    ]
                    person_hand_bboxes = [
                        tuple(h["bbox"]) for h in person_hand_details
                    ]
                    obs = await self._build_person_obs(
                        frame, det, room, ts, frame_w, frame_h,
                        person_hand_bboxes, posture,
                        hand_details=person_hand_details,
                    )
                elif cls in self.ANIMAL_ENRICHERS:
                    obs = self._build_animal_obs(
                        cls, frame, det, room, ts, frame_w, frame_h
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
        hand_details: Optional[list[dict]] = None,
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
        identity_status: str = "no_face"
        face_metadata: dict[str, Any] = {}
        if face is not None:
            identity_status = "no_match"
            try:
                match = await self.identity.identify_from_embedding_async(
                    face["embedding"], modality="face"
                )
                # Treat ambiguous matches as "unknown for now". The
                # margin gate inside IdentityManager flagged this match
                # as too close to a runner-up — committing person_id
                # would let the WorldModel attribute observations to
                # the wrong resident, which corrupts every downstream
                # answer ("where's Cole?" returns Anna's office, etc.).
                # False unknowns are recoverable; false positives aren't.
                if match is not None and not match.is_ambiguous:
                    person_id = int(match.person_id)
                    person_name = match.name
                    identity_conf = float(match.similarity)
                    identity_status = "matched"
                elif match is not None and match.is_ambiguous:
                    identity_status = "ambiguous"
                    logger.debug(
                        f"[ObservationBuilder] ambiguous face match in '{room}' "
                        f"(top candidate: {match.name}, sim={match.similarity:.3f}); "
                        "treating as unknown"
                    )
            except Exception as e:
                logger.debug(
                    f"[ObservationBuilder] identity match failed in '{room}': {e}"
                )
                identity_status = "error"
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
                # Per-hand details (handedness label + wrist xy) for the
                # InteractionMonitor's "Cole picked it up with his right
                # hand" phrasing. Empty list when no hand detector is
                # wired or no hands attached to this person.
                "hand_details": list(hand_details or []),
                "posture": posture,
                # 'matched' | 'ambiguous' | 'no_match' | 'no_face' | 'error' —
                # consumers can branch on ambiguous (e.g. dashboard
                # highlights pending review) without re-deriving from
                # null person_id (which conflates "no face" with
                # "ambiguous match").
                "identity_status": identity_status,
                **face_metadata,
            },
        )

    # ── Animal enrichers (§22.3) ───────────────────────────────────────────

    def _build_animal_obs(
        self,
        species: str,
        frame: np.ndarray,
        det: Detection,
        room: str,
        ts: datetime,
        fw: int,
        fh: int,
    ) -> Observation:
        """Per-species dispatcher. Falls back to a generic enricher (size
        + obj_class only) when a tracked species has no specific method
        registered — useful as a stub for new species before its
        descriptors are implemented."""
        method_name = self.ANIMAL_ENRICHERS.get(species)
        if method_name is None:
            return self._build_generic_animal_obs(
                species, frame, det, room, ts, fw, fh
            )
        return getattr(self, method_name)(frame, det, room, ts, fw, fh)

    def _build_cat_obs(
        self,
        frame: np.ndarray,
        det: Detection,
        room: str,
        ts: datetime,
        fw: int,
        fh: int,
    ) -> Observation:
        bbox = det.bbox
        x1, y1, x2, y2 = (int(v) for v in bbox)
        x1 = max(0, min(fw - 1, x1))
        y1 = max(0, min(fh - 1, y1))
        x2 = max(0, min(fw, x2))
        y2 = max(0, min(fh, y2))
        crop = frame[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else None

        color_class = _classify_cat_color(crop)
        color_hist = _color_histogram(crop)
        coat_texture = _coat_texture_descriptor(crop)
        size_norm = ((x2 - x1) * (y2 - y1)) / max(fw * fh, 1)

        # Persist the crop alongside the metadata so the §22.5 cluster
        # builder dashboard can show an image grid for human labeling.
        # Mirrors the person-obs save path; gated on snapshot_dir +
        # non-empty crop. Failures are non-fatal — we still emit the
        # observation; the cluster UI just won't have an image for it.
        crop_path = self._save_animal_crop("cat", crop, room, ts)

        return Observation(
            camera=room, room=room, obj_class="cat",
            bbox=tuple(bbox), confidence=det.confidence, ts=ts,
            visual_embedding=None,  # CLIP encoder lands in §23
            metadata={
                "frame_width": fw, "frame_height": fh,
                "color_class": color_class,
                "color_histogram": color_hist,
                "coat_texture": coat_texture,
                "size_normalized": float(size_norm),
                "crop_path": str(crop_path) if crop_path else None,
            },
        )

    def _build_dog_obs(
        self,
        frame: np.ndarray,
        det: Detection,
        room: str,
        ts: datetime,
        fw: int,
        fh: int,
    ) -> Observation:
        bbox = det.bbox
        x1, y1, x2, y2 = (int(v) for v in bbox)
        x1 = max(0, min(fw - 1, x1))
        y1 = max(0, min(fh - 1, y1))
        x2 = max(0, min(fw, x2))
        y2 = max(0, min(fh, y2))
        # Persist the dog crop too — same reasoning as the cat path.
        # Computed below the bbox-clamp so we don't try to save an empty
        # slice; saved only if snapshot_dir is set + crop is non-empty.
        crop = frame[y1:y2, x1:x2] if y2 > y1 and x2 > x1 else None

        color_class = _classify_dog_color(crop)
        color_hist = _color_histogram(crop)
        # Breed-class is the dog analog of cat coat_texture — same role
        # in the cost function. Real impl is CLIP-zero-shot in §23; for
        # now use a coarse aspect-ratio + texture proxy so the field is
        # populated without a heavy classifier in the hot path.
        breed_class = _coarse_breed_class(crop, x2 - x1, y2 - y1)
        size_norm = ((x2 - x1) * (y2 - y1)) / max(fw * fh, 1)
        crop_path = self._save_animal_crop("dog", crop, room, ts)

        return Observation(
            camera=room, room=room, obj_class="dog",
            bbox=tuple(bbox), confidence=det.confidence, ts=ts,
            visual_embedding=None,
            metadata={
                "frame_width": fw, "frame_height": fh,
                "color_class": color_class,
                "color_histogram": color_hist,
                "breed_class": breed_class,
                "size_normalized": float(size_norm),
                "crop_path": str(crop_path) if crop_path else None,
            },
        )

    def _save_animal_crop(
        self,
        species: str,
        crop: Optional[np.ndarray],
        room: str,
        ts: datetime,
    ) -> Optional[Path]:
        """Save a cat/dog bbox crop next to the world snapshots so the
        §22.5 cluster builder dashboard can show the image grid. Same
        directory + naming convention as the person-obs save path —
        downstream just sees a JPEG path on the observation. Returns
        the full Path on success, None on no-op (no snapshot dir set,
        empty crop, cv2 unavailable, encode failure)."""
        if self.snapshot_dir is None or crop is None or crop.size == 0:
            return None
        try:
            import cv2  # noqa: E402 — lazy
            fname = (
                f"{species}_{room}_{ts.strftime('%Y%m%dT%H%M%S')}_"
                f"{uuid.uuid4().hex[:6]}.jpg"
            )
            path = self.snapshot_dir / fname
            ok = cv2.imwrite(str(path), crop)
            return path if ok else None
        except Exception as e:
            logger.debug(
                f"[ObservationBuilder] {species} crop save failed in "
                f"'{room}': {e}"
            )
            return None

    def _build_generic_animal_obs(
        self,
        species: str,
        frame: np.ndarray,  # noqa: ARG002 — symmetry with specific enrichers
        det: Detection,
        room: str,
        ts: datetime,
        fw: int,
        fh: int,
    ) -> Observation:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        size_norm = ((x2 - x1) * (y2 - y1)) / max(fw * fh, 1)
        return Observation(
            camera=room, room=room, obj_class=species,
            bbox=tuple(det.bbox), confidence=det.confidence, ts=ts,
            metadata={
                "frame_width": fw, "frame_height": fh,
                "size_normalized": float(size_norm),
            },
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


# ── Animal descriptor helpers (§22.3) ───────────────────────────────────────


def _classify_cat_color(crop: Optional[np.ndarray]) -> str:
    """Coarse color class — 'striped' | 'black' | 'unknown'.
    Stripe heuristic: high local-contrast variance (Laplacian).
    Black heuristic: low mean V + low mean S in HSV. Tunable thresholds;
    spend an afternoon labeling crops to retune for your own cameras."""
    if crop is None or crop.size == 0:
        return "unknown"
    try:
        import cv2  # noqa: E402
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    except Exception:
        return "unknown"
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_v = float(hsv[..., 2].mean())
    mean_s = float(hsv[..., 1].mean())
    if mean_v < 70 and mean_s < 40:
        return "black"
    if laplacian_var > 350:
        return "striped"
    return "unknown"


def _classify_dog_color(crop: Optional[np.ndarray]) -> str:
    """Coarse 8-class dog color in {tan, brown, black, white, tricolor,
    merle, brindle, cream, unknown}. The reliable household-discriminating
    classes for Cole's lineup are 'cream' (Summer) vs 'brindle' (Dalila),
    so the heuristic prioritizes those two then falls through to the rest.
    Real impl is CLIP-zero-shot in §23 — this gets us moving."""
    if crop is None or crop.size == 0:
        return "unknown"
    try:
        import cv2  # noqa: E402
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    except Exception:
        return "unknown"
    mean_v = float(hsv[..., 2].mean())
    mean_s = float(hsv[..., 1].mean())
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # Cream: high V + low S + low texture
    if mean_v > 160 and mean_s < 60 and laplacian_var < 400:
        return "cream"
    # White: very high V + very low S
    if mean_v > 200 and mean_s < 30:
        return "white"
    # Brindle: medium V, moderate S, high texture (the stripey look)
    if 60 < mean_v < 140 and laplacian_var > 500:
        return "brindle"
    # Black-base
    if mean_v < 70 and mean_s < 40:
        return "black"
    # Tan/brown — moderate V, warm hue (H in red-yellow range)
    mean_h = float(hsv[..., 0].mean())
    if 70 < mean_v < 170 and mean_s > 60 and (mean_h < 30 or mean_h > 150):
        return "tan" if mean_v > 120 else "brown"
    return "unknown"


def _color_histogram(crop: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Normalized 16x16x16 HSV histogram, flattened. Used by the
    Bhattacharyya-based hist_cost in animal cost functions."""
    if crop is None or crop.size == 0:
        return None
    try:
        import cv2  # noqa: E402
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], None, [16, 16, 16],
            [0, 180, 0, 256, 0, 256],
        )
        hist = hist.flatten().astype(np.float32)
        hist /= (hist.sum() + 1e-9)
        return hist
    except Exception:
        return None


def _coat_texture_descriptor(crop: Optional[np.ndarray]) -> Optional[list[float]]:
    """5-dim coat texture descriptor for the Spooky vs Velcro problem.
    Components:
      [0]   silhouette edge-density variance (fluffy = high)
      [1-4] mean Gabor-filter response at 4 orientations, single scale
    Reduced from §22.3's 24-dim Gabor stack to keep the hot path cheap;
    tune to per-species PCA later if accuracy stalls."""
    if crop is None or crop.size == 0:
        return None
    try:
        import cv2  # noqa: E402
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    except Exception:
        return None
    # Edge-density variance over a perimeter band (top/bottom/left/right
    # 10% of the bbox interior).
    h, w = gray.shape
    if h < 8 or w < 8:
        return None
    band = max(1, min(h, w) // 10)
    perim = np.concatenate([
        gray[:band, :].flatten(),
        gray[-band:, :].flatten(),
        gray[:, :band].flatten(),
        gray[:, -band:].flatten(),
    ])
    sob_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sob_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.hypot(sob_x, sob_y)
    edge_var = float(np.var(edge_mag) / (np.mean(edge_mag) ** 2 + 1e-9))
    edge_var = float(min(edge_var / 5.0, 1.0))  # squish to [0, 1]

    gabor_responses: list[float] = []
    for theta_deg in (0, 45, 90, 135):
        theta = np.deg2rad(theta_deg)
        kernel = cv2.getGaborKernel(
            (15, 15), sigma=3.0, theta=theta,
            lambd=8.0, gamma=0.5, psi=0,
        )
        resp = cv2.filter2D(gray, cv2.CV_32F, kernel)
        gabor_responses.append(float(np.mean(np.abs(resp))))
    # Discard `perim` after the variance read — it shaped where we
    # sampled, no further use.
    _ = perim
    return [edge_var, *gabor_responses]


def _coarse_breed_class(
    crop: Optional[np.ndarray], bbox_w: int, bbox_h: int,
) -> str:
    """Stub breed-class. Real impl is CLIP-zero-shot over the 7 classes
    in §22.3. For now: aspect-ratio (long vs short) plus a coat-texture
    proxy gets us small-vs-medium and shorthair-vs-longhair, which is
    enough for Summer (cream-longhair) vs Dalila (brindle-shorthair)."""
    if crop is None or crop.size == 0 or bbox_w <= 0 or bbox_h <= 0:
        return "unknown"
    try:
        import cv2  # noqa: E402
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    except Exception:
        return "unknown"
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    aspect = bbox_w / max(bbox_h, 1)
    coat = "longhair" if laplacian_var > 600 else "shorthair"
    if aspect < 0.7:
        size = "small"
    elif aspect < 1.4:
        size = "medium"
    else:
        size = "large"
    return f"{size}-{coat}"


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
