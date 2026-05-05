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
        Run pose estimation on a single frame and return the posture label.

        Returns:
            One of: "standing", "sitting", "lying", "unknown".
            For richer per-person state (orientation, head tilt, arms,
            gesture hints) call analyze_full() instead.
        """
        full = self.analyze_full(frame)
        return full.get("posture", "unknown") if full else "unknown"

    async def analyze_async(self, frame: Optional[np.ndarray]) -> str:
        """Async wrapper — runs blocking mediapipe inference in thread pool."""
        return await asyncio.to_thread(self.analyze, frame)

    def analyze_full(self, frame: Optional[np.ndarray]) -> dict:
        """
        Rich per-person state from MediaPipe pose landmarks. Returns:
            {
              "posture":     "standing" | "sitting" | "lying" | "unknown",
              "orientation": "front" | "side_left" | "side_right" | "back" | None,
              "head_tilt":   "looking_up" | "looking_down" | "level" | None,
              "arms":        "raised" | "extended" | "crossed" | "down" | None,
              "lean":        "forward" | "backward" | None,
              "gesture":     "pointing" | "waving" | None,   # heuristic only
              "activity_hint": str | None,                   # rough guess from
                                                              # combined signals
              "confidence": 0.0-1.0,
              "landmarks_visible": int,
            }
        Each field is None when its underlying landmarks aren't visible
        enough — the vision LLM treats None as "no signal", not "absent."
        """
        empty = {
            "posture": "unknown", "orientation": None, "head_tilt": None,
            "arms": None, "lean": None, "gesture": None, "activity_hint": None,
            "confidence": 0.0, "landmarks_visible": 0,
        }
        if frame is None or self._pose is None:
            return empty
        try:
            if _CV2_AVAILABLE and cv2 is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb = frame[:, :, ::-1]
            results = self._pose.process(rgb)
            if not results.pose_landmarks:
                return empty
            landmarks = results.pose_landmarks.landmark
        except Exception as e:
            logger.debug(f"[PostureAnalyzer] analyze_full error: {e}")
            return empty

        out = dict(empty)
        out["posture"] = self._classify_pose(landmarks)
        self._last_posture = out["posture"]
        out["orientation"] = self._classify_orientation(landmarks)
        out["head_tilt"] = self._classify_head_tilt(landmarks)
        out["arms"] = self._classify_arms(landmarks)
        out["lean"] = self._classify_lean(landmarks)
        out["gesture"] = self._classify_gesture(landmarks)
        out["activity_hint"] = self._derive_activity_hint(out)
        out["landmarks_visible"] = sum(
            1 for lm in landmarks if lm.visibility >= MIN_VISIBILITY
        )
        # Confidence = average visibility of the core trunk landmarks
        core = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, NOSE]
        vis = [landmarks[i].visibility for i in core]
        out["confidence"] = float(sum(vis) / len(vis)) if vis else 0.0
        return out

    async def analyze_full_async(self, frame: Optional[np.ndarray]) -> dict:
        """Async wrapper for analyze_full."""
        return await asyncio.to_thread(self.analyze_full, frame)

    # ── Sub-classifiers ─────────────────────────────────────────────────────

    @staticmethod
    def _vis(lm, idx: int) -> bool:
        return lm[idx].visibility >= MIN_VISIBILITY

    @staticmethod
    def _xy(lm, idx: int) -> tuple[float, float]:
        return (lm[idx].x, lm[idx].y)

    def _classify_orientation(self, lm) -> Optional[str]:
        """front / side / back from shoulder horizontal spread.

        When facing the camera, shoulders are widely separated (~0.2-0.4 of
        frame width). When turned sideways, shoulder X coords collapse onto
        each other. 'Back' is approximated as front because we can't see the
        nose on a back-facing pose — handled separately below.
        """
        if not (self._vis(lm, LEFT_SHOULDER) and self._vis(lm, RIGHT_SHOULDER)):
            return None
        ls_x = lm[LEFT_SHOULDER].x
        rs_x = lm[RIGHT_SHOULDER].x
        spread = abs(ls_x - rs_x)
        nose_visible = self._vis(lm, NOSE)
        if spread < 0.08:
            # Shoulders overlap → torso turned sideways
            # Use which shoulder is in front (closer to side of frame) to pick L vs R
            return "side_left" if ls_x < rs_x else "side_right"
        if not nose_visible:
            return "back"
        return "front"

    def _classify_head_tilt(self, lm) -> Optional[str]:
        """Looking up / looking down / level, from nose vs shoulder Y."""
        if not (self._vis(lm, NOSE)
                and self._vis(lm, LEFT_SHOULDER)
                and self._vis(lm, RIGHT_SHOULDER)):
            return None
        nose_y = lm[NOSE].y
        sh_y = (lm[LEFT_SHOULDER].y + lm[RIGHT_SHOULDER].y) / 2.0
        # Negative delta = nose is ABOVE shoulders (looking up). Normal upright
        # head sits ~0.10-0.15 above the shoulder line.
        delta = sh_y - nose_y
        if delta > 0.20:
            return "looking_up"      # head well above shoulders → tilted up
        if delta < 0.05:
            return "looking_down"    # head pulled down toward shoulders
        return "level"

    def _classify_arms(self, lm) -> Optional[str]:
        """Arms position from wrist Y vs shoulder Y, plus elbow extension."""
        LEFT_WRIST, RIGHT_WRIST = 15, 16
        LEFT_ELBOW, RIGHT_ELBOW = 13, 14
        wrists = []
        if self._vis(lm, LEFT_WRIST):
            wrists.append(("L", lm[LEFT_WRIST].y, lm[LEFT_SHOULDER].y, lm[LEFT_ELBOW].x, lm[LEFT_WRIST].x))
        if self._vis(lm, RIGHT_WRIST):
            wrists.append(("R", lm[RIGHT_WRIST].y, lm[RIGHT_SHOULDER].y, lm[RIGHT_ELBOW].x, lm[RIGHT_WRIST].x))
        if not wrists:
            return None
        # Raised: wrist higher (smaller y) than shoulder
        if any(wy < sy - 0.05 for _, wy, sy, _, _ in wrists):
            return "raised"
        # Extended out from body: wrist x far from elbow x (horizontal stretch)
        if any(abs(wx - ex) > 0.18 for _, _, _, ex, wx in wrists):
            return "extended"
        return "down"

    def _classify_lean(self, lm) -> Optional[str]:
        """Forward / backward from shoulder x vs hip x offset."""
        if not (self._vis(lm, LEFT_SHOULDER) and self._vis(lm, RIGHT_SHOULDER)
                and self._vis(lm, LEFT_HIP) and self._vis(lm, RIGHT_HIP)):
            return None
        sh_x = (lm[LEFT_SHOULDER].x + lm[RIGHT_SHOULDER].x) / 2.0
        hip_x = (lm[LEFT_HIP].x + lm[RIGHT_HIP].x) / 2.0
        # Camera-relative: shoulders forward of hips → leaning forward (typical
        # 'leaning into the screen' pose). MediaPipe x grows left→right of
        # frame, so the absolute direction depends on orientation. Magnitude
        # matters more than sign.
        delta = sh_x - hip_x
        if abs(delta) < 0.04:
            return None
        return "forward" if delta > 0 else "backward"

    def _classify_gesture(self, lm) -> Optional[str]:
        """Single-frame gesture heuristic. Real waving / pointing detection
        needs temporal reasoning we don't have yet — this only catches
        'arm extended toward camera with index finger / open hand visible'."""
        LEFT_WRIST, RIGHT_WRIST = 15, 16
        LEFT_INDEX, RIGHT_INDEX = 19, 20
        # Pointing: index finger landmark visible AND extended further than wrist
        for wrist_idx, index_idx in ((LEFT_WRIST, LEFT_INDEX),
                                      (RIGHT_WRIST, RIGHT_INDEX)):
            if self._vis(lm, wrist_idx) and self._vis(lm, index_idx):
                if abs(lm[wrist_idx].x - lm[index_idx].x) > 0.04 \
                   or abs(lm[wrist_idx].y - lm[index_idx].y) > 0.04:
                    return "pointing"
        return None

    def _derive_activity_hint(self, state: dict) -> Optional[str]:
        """Combine signals into a rough activity guess. The vision LLM gets
        this as a hint, not a hard label — final description still wins."""
        posture = state.get("posture")
        head = state.get("head_tilt")
        lean = state.get("lean")
        arms = state.get("arms")
        if posture == "lying":
            return "resting or sleeping"
        if posture == "sitting" and head == "looking_down" and lean == "forward":
            return "focused at a desk / screen"
        if posture == "sitting" and head == "looking_up":
            return "looking up from work — perhaps watching something"
        if posture == "standing" and arms == "raised":
            return "reaching or stretching"
        if posture == "standing" and head == "looking_down":
            return "doing something with hands while standing"
        return None

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
