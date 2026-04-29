"""
JARVIS — Ambient Home AI
========================
Mission: Detect body posture (lying down, sitting, standing) from a camera
         frame using MediaPipe Pose. The posture signal feeds into SleepTracker
         and StateFusion as a high-confidence activity signal — if Cole is lying
         down for 15+ minutes, Jarvis knows to be quiet and check in gently.

Modules: modules/vision/posture_analyzer.py
Classes: PostureAnalyzer
Functions:
    PostureAnalyzer.__init__(config)        — Initialize mediapipe
    PostureAnalyzer.load()                  — Create the Pose solution instance
    PostureAnalyzer.analyze(frame)          — Sync: return posture label from frame
    PostureAnalyzer.analyze_async(frame)    — Async wrapper
    PostureAnalyzer._classify_pose(lms)     — Turn landmarks into posture label
    PostureAnalyzer._normalize_y(lms, idx) — Get normalized Y coordinate

Variables:
    PostureAnalyzer._pose         — mediapipe.solutions.pose.Pose instance
    PostureAnalyzer._last_posture — Last returned posture label (for caching)

Posture labels:
    "standing"  — Person is upright, head above shoulders by expected margin
    "sitting"   — Person upright but legs bent / body more compressed vertically
    "lying"     — Person horizontal, head at same Y level as hips/knees
    "unknown"   — No pose detected or confidence too low

#todo: Add confidence score per detection (use pose_world_landmarks visibility)
#todo: Add side-lying vs face-down classification
#todo: Add multiple-person detection for when Anna or Sophie are also in frame
#todo: Persist posture log to database for sleep pattern analysis
"""

import asyncio
from typing import Any, Optional, cast

import numpy as np
from loguru import logger

mp = None

try:
    import mediapipe as mp
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False
    logger.warning("[PostureAnalyzer] MediaPipe not available — posture analysis disabled")

cv2 = None

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


# MediaPipe Pose landmark indices (see mediapipe docs)
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

# Minimum visibility to trust a landmark
MIN_VISIBILITY = 0.5


class PostureAnalyzer:
    """
    Detects body posture using MediaPipe Pose estimation.

    Runs at low FPS (configurable, default 1 FPS) to keep CPU usage minimal.
    """

    def __init__(self, config: dict) -> None:
        context_cfg = config.get("context", {})
        self._fps: int = int(context_cfg.get("posture_analysis_fps", 1))
        self._pose: Optional[Any] = None
        self._last_posture: str = "unknown"

    def load(self) -> None:
        """Initialize the MediaPipe Pose model."""
        if not _MEDIAPIPE_AVAILABLE or mp is None:
            return
        try:
            mp_pose = cast(Any, mp.solutions).pose
            self._pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,          # 0 = fastest/smallest
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            logger.info("[PostureAnalyzer] MediaPipe Pose loaded")
        except Exception as e:
            logger.error(f"[PostureAnalyzer] Load failed: {e}")

    async def load_async(self) -> None:
        """Async wrapper for load() — runs in thread pool."""
        await asyncio.to_thread(self.load)

    def analyze(self, frame: Optional[np.ndarray]) -> str:
        """
        Run pose estimation on a single frame and return posture label.

        Args:
            frame: BGR numpy array from camera.

        Returns:
            One of: "standing", "sitting", "lying", "unknown"
        """
        if frame is None or self._pose is None:
            return self._last_posture

        try:
            if _CV2_AVAILABLE and cv2 is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb = frame[:, :, ::-1]  # Simple BGR→RGB flip

            results = self._pose.process(rgb)

            if not results.pose_landmarks:
                return "unknown"

            landmarks = results.pose_landmarks.landmark
            posture = self._classify_pose(landmarks)
            self._last_posture = posture
            return posture

        except Exception as e:
            logger.debug(f"[PostureAnalyzer] Analysis error: {e}")
            return self._last_posture

    async def analyze_async(self, frame: Optional[np.ndarray]) -> str:
        """Async wrapper — runs blocking mediapipe inference in thread pool."""
        return await asyncio.to_thread(self.analyze, frame)

    def _classify_pose(self, landmarks) -> str:
        """
        Classify posture from MediaPipe pose landmarks.

        Strategy:
        - Compute vertical positions of key points (normalized 0–1 in frame)
        - Standing: large vertical spread, head above hips by significant margin
        - Lying:    head Y ≈ hip Y (within ~0.15) — horizontal orientation
        - Sitting:  intermediate vertical spread
        """
        def y(idx: int) -> Optional[float]:
            lm = landmarks[idx]
            return lm.y if lm.visibility >= MIN_VISIBILITY else None

        nose_y     = y(NOSE)
        l_shoulder = y(LEFT_SHOULDER)
        r_shoulder = y(RIGHT_SHOULDER)
        l_hip      = y(LEFT_HIP)
        r_hip      = y(RIGHT_HIP)
        l_ankle    = y(LEFT_ANKLE)
        r_ankle    = y(RIGHT_ANKLE)

        # Need at least shoulders and hips
        shoulder_ys = [v for v in [l_shoulder, r_shoulder] if v is not None]
        hip_ys      = [v for v in [l_hip, r_hip] if v is not None]

        if not shoulder_ys or not hip_ys:
            return "unknown"

        shoulder_y = sum(shoulder_ys) / len(shoulder_ys)
        hip_y      = sum(hip_ys) / len(hip_ys)

        # Vertical difference between shoulders and hips (normalized 0–1)
        vert_spread = abs(shoulder_y - hip_y)

        # Lying: shoulders and hips at nearly the same height
        if vert_spread < 0.12:
            return "lying"

        # Standing vs sitting: use ankle position if available
        ankle_ys = [v for v in [l_ankle, r_ankle] if v is not None]
        if ankle_ys:
            ankle_y = sum(ankle_ys) / len(ankle_ys)
            total_spread = abs(shoulder_y - ankle_y)
            if total_spread > 0.5:
                return "standing"
            return "sitting"

        # Fallback: if vert_spread > 0.12 but no ankles visible
        if vert_spread > 0.25:
            return "standing"
        return "sitting"

    @property
    def is_loaded(self) -> bool:
        """True if MediaPipe is available and loaded."""
        return self._pose is not None
