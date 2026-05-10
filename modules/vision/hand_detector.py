"""
JARVIS — World Model
====================
Mission: MediaPipe Hands wrapper. Returns hand bboxes per frame —
         each bbox is the axis-aligned bounding box around the 21
         keypoints MediaPipe produces per hand. Cheap (~3-5ms / frame
         on RTX 4070 Ti at 640×640), so safe to run at full FPS in
         the office and effectively free in Wyze rooms (5/1 fps).

         Wraps the sync mediapipe API in an async-friendly shim:
         detect_async runs the per-frame inference inside
         asyncio.to_thread so the per-room observation loop doesn't
         block. The detector instance is NOT thread-safe — share at
         most one instance per room polling task; the existing
         per-room observation_builder loop satisfies this naturally.

         Output dict shape (one entry per detected hand):
           {
             'bbox':       (x1, y1, x2, y2),
             'handedness': 'Left' | 'Right' | 'Unknown',
             'confidence': float,
             'wrist_xy':   (x, y)
           }

Modules: modules/vision/hand_detector.py
Classes: HandDetector, NullHandDetector
Spec:    new 2.md §24.1.

#todo: Per-handedness latency optimization. MediaPipe lets you ask
       for a single hand instead of multi-hand; if Cole's pipeline
       only ever needs one hand at a time, dropping to max=1 cuts
       latency ~30%. Premature for now; the office RTX has the
       budget.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

import numpy as np
from loguru import logger


class HandDetector:
    """MediaPipe Hands → bbox list. Construct once per ObservationBuilder
    (so per camera-equipped room), call detect / detect_async per frame.
    Thread-unsafe; the per-room polling loop guarantees serial calls."""

    def __init__(
        self,
        max_num_hands: int = 4,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        # Lazy-import mediapipe + cv2 so a test-only environment without
        # them doesn't pay the import cost just to instantiate the
        # NullHandDetector elsewhere in the codebase.
        import cv2 as _cv2          # noqa: F401 — used in detect()
        import mediapipe as mp_module
        self._cv2 = _cv2
        self._mp = mp_module
        # mediapipe.solutions.hands is dynamically registered at import
        # time; static type-checkers (Pylance) can't see it. The runtime
        # access works fine — narrow the lookup so the ignore is local.
        _hands_module = getattr(mp_module.solutions, "hands")
        self._hands = _hands_module.Hands(
            static_image_mode=False,
            max_num_hands=int(max_num_hands),
            min_detection_confidence=float(min_detection_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )
        self._max_hands = int(max_num_hands)
        self._detect_lock = asyncio.Lock()
        self._next_error_log_ts = 0.0
        logger.info(
            f"[HandDetector] MediaPipe Hands loaded "
            f"(max_num_hands={max_num_hands}, "
            f"min_detect={min_detection_confidence})"
        )

    def detect(self, image_bgr: Optional[np.ndarray]) -> list[dict]:
        """Sync detection — used for tests + as the worker behind
        detect_async. Returns [] for empty / None input."""
        if image_bgr is None or image_bgr.size == 0:
            return []
        try:
            rgb = self._cv2.cvtColor(image_bgr, self._cv2.COLOR_BGR2RGB)
        except Exception as e:
            logger.debug(f"[HandDetector] cvtColor failed: {e}")
            return []
        h, w = image_bgr.shape[:2]
        try:
            results = self._hands.process(rgb)
        except Exception as e:
            now = time.monotonic()
            if now >= self._next_error_log_ts:
                self._next_error_log_ts = now + 5.0
                logger.debug(f"[HandDetector] mediapipe process failed: {e}")
            return []
        if not getattr(results, "multi_hand_landmarks", None):
            return []
        out: list[dict] = []
        handedness_iter = list(results.multi_handedness or [])
        for i, landmarks in enumerate(results.multi_hand_landmarks):
            xs = [lm.x * w for lm in landmarks.landmark]
            ys = [lm.y * h for lm in landmarks.landmark]
            bbox = (
                int(max(0, min(w - 1, min(xs)))),
                int(max(0, min(h - 1, min(ys)))),
                int(max(0, min(w, max(xs)))),
                int(max(0, min(h, max(ys)))),
            )
            wrist_xy = (
                int(landmarks.landmark[0].x * w),
                int(landmarks.landmark[0].y * h),
            )
            handedness = handedness_iter[i] if i < len(handedness_iter) else None
            label, score = "Unknown", 0.0
            if handedness is not None and handedness.classification:
                label = handedness.classification[0].label
                score = float(handedness.classification[0].score)
            out.append({
                "bbox": bbox,
                "handedness": label,
                "confidence": score,
                "wrist_xy": wrist_xy,
            })
        return out

    async def detect_async(
        self, image_bgr: Optional[np.ndarray],
    ) -> list[dict]:
        """Async wrapper — runs sync detect() in the default executor
        so the per-room observation polling loop doesn't block on
        MediaPipe inference."""
        if image_bgr is None or image_bgr.size == 0:
            return []
        async with self._detect_lock:
            return await asyncio.to_thread(self.detect, image_bgr)

    def close(self) -> None:
        """Free MediaPipe resources. The graph's GPU buffers stick
        around otherwise on Windows; cheap to call on shutdown."""
        try:
            self._hands.close()
        except Exception:
            pass


class NullHandDetector:
    """No-op detector for tests / smoke runs without mediapipe.
    detect_async always returns []. Provides the same interface so
    ObservationBuilder can be constructed with either."""

    async def detect_async(self, image_bgr: Optional[np.ndarray]) -> list[dict]:
        return []

    def detect(self, image_bgr: Optional[np.ndarray]) -> list[dict]:
        return []

    def close(self) -> None:
        pass


# ── Helpers ────────────────────────────────────────────────────────────────


def bbox_overlaps_or_within(
    small: tuple, large: tuple, slack: int = 20,
) -> bool:
    """Returns True when `small` is mostly inside `large` (with slack
    pixels of tolerance). Used by ObservationBuilder to attach hand
    detections to the person whose bbox encloses them — the same hand
    can geometrically fit multiple person bboxes when people overlap,
    in which case the per-person enricher just sees that hand attached
    to whichever person it iterates first. Acceptable: the
    InteractionMonitor (§24.3) relies on world-model identity rather
    than per-frame attribution to disambiguate."""
    if not small or not large or len(small) != 4 or len(large) != 4:
        return False
    sx1, sy1, sx2, sy2 = small
    lx1, ly1, lx2, ly2 = large
    return (
        sx1 >= lx1 - slack
        and sy1 >= ly1 - slack
        and sx2 <= lx2 + slack
        and sy2 <= ly2 + slack
    )
