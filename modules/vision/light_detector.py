"""
JARVIS — Ambient Home AI
========================
Mission: Analyze a camera frame to determine whether the room lights are
         on or off. Uses simple brightness heuristics on the luminance channel
         rather than ML models — reliable, fast, and zero extra dependencies
         beyond OpenCV (already required for camera capture).

         Light state is an important contextual signal. Dark room during
         daytime = sleeping/napping. Dark room at night + lying down = sleep.
         Lights suddenly on at night = woke up.

Modules: modules/vision/light_detector.py
Classes: LightDetector
Functions:
    LightDetector.__init__(config)         — Initialize with brightness thresholds
    LightDetector.analyze(frame)           — Sync: return LightState from frame
    LightDetector.analyze_async(frame)     — Async wrapper
    LightDetector._compute_brightness(frame) — Get mean luminance of frame

Variables:
    LightDetector._on_threshold    — Mean brightness above this = lights on
    LightDetector._off_threshold   — Mean brightness below this = lights off
    LightDetector._last_state      — "on" | "off" | "unknown" for change detection

#todo: Add hysteresis — require N consecutive same readings before changing state
#todo: Add per-zone brightness (check windows separately for daylight vs lamp)
#todo: Add time-of-day calibration — adjust thresholds for sunrise/sunset
#todo: Learn per-room baseline brightness instead of global thresholds
"""

import asyncio
from typing import Optional

import numpy as np
from loguru import logger

cv2 = None

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


# Default brightness thresholds (0–255 mean luminance)
DEFAULT_ON_THRESHOLD: float = 60.0     # Mean Y above this → lights on
DEFAULT_OFF_THRESHOLD: float = 30.0    # Mean Y below this → lights off


class LightDetector:
    """
    Determines whether room lights are on by analyzing frame brightness.

    Uses the Y (luminance) channel from YCrCb colorspace for more stable
    brightness measurement than raw RGB average.
    """

    def __init__(self, config: dict) -> None:
        vision_cfg = config.get("vision", {})
        self._on_threshold: float = float(
            vision_cfg.get("light_on_threshold", DEFAULT_ON_THRESHOLD)
        )
        self._off_threshold: float = float(
            vision_cfg.get("light_off_threshold", DEFAULT_OFF_THRESHOLD)
        )
        self._last_state: Optional[bool] = None  # True=on, False=off

    def analyze(self, frame: Optional[np.ndarray]) -> Optional[bool]:
        """
        Analyze a frame to determine light state.

        Args:
            frame: BGR numpy array from camera capture, or None.

        Returns:
            True if lights are on, False if lights are off, None if unknown.
        """
        if frame is None:
            return self._last_state  # Return cached state if no new frame

        brightness = self._compute_brightness(frame)
        if brightness is None:
            return self._last_state

        if brightness >= self._on_threshold:
            new_state = True
        elif brightness <= self._off_threshold:
            new_state = False
        else:
            # Ambiguous zone — maintain last known state
            return self._last_state

        if new_state != self._last_state:
            logger.debug(
                f"[LightDetector] Light state changed: "
                f"{'ON' if new_state else 'OFF'} (brightness={brightness:.1f})"
            )

        self._last_state = new_state
        return new_state

    async def analyze_async(self, frame: Optional[np.ndarray]) -> Optional[bool]:
        """Async wrapper."""
        return await asyncio.to_thread(self.analyze, frame)

    def _compute_brightness(self, frame: np.ndarray) -> Optional[float]:
        """
        Convert frame to YCrCb and return mean luminance (Y channel).
        Falls back to grayscale mean if OpenCV is unavailable.
        """
        try:
            if _CV2_AVAILABLE and cv2 is not None:
                ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
                y_channel = ycrcb[:, :, 0]
            else:
                # Fallback: simple mean of all channels
                y_channel = frame.mean(axis=2) if frame.ndim == 3 else frame

            return float(np.mean(y_channel))
        except Exception as e:
            logger.debug(f"[LightDetector] Brightness computation failed: {e}")
            return None

    @property
    def lights_on(self) -> Optional[bool]:
        """Last known light state (True=on, False=off, None=unknown)."""
        return self._last_state
