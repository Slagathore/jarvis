"""
JARVIS — Ambient Home AI
========================
Mission: Track sleep patterns and detect when Cole is napping or sleeping
         so Jarvis can suppress speech appropriately. Combines posture signals
         (lying down), light detection (room dark), time-of-day heuristics,
         and inactivity duration to make a sleep/awake determination.

Modules: modules/context/sleep_tracker.py
Classes: SleepTracker
Functions:
    SleepTracker.__init__(config)           — Initialize with config
    SleepTracker.update(posture, lights_on, room) — Feed new signals
    SleepTracker.is_sleeping()              — True if likely sleeping
    SleepTracker.is_napping()               — True if likely napping (daytime)
    SleepTracker.get_sleep_signal()         — Return signal dict for StateFusion
    SleepTracker.record_wakeup()            — Explicitly mark as awake

Variables:
    SleepTracker._lying_since    — datetime when lying posture first detected
    SleepTracker._dark_since     — datetime when lights went off
    SleepTracker._confirmed_sleep — bool, set after sustained lying + dark
    SleepTracker._announced_sleep — bool, so Jarvis only says "nap?" once

#todo: Add microphone energy monitoring — silence for 10+ minutes reinforces sleep
#todo: Add time-series model for learning Cole's actual sleep schedule
#todo: Add smart alarm — detect light sleep phase via movement to wake gently
#todo: Integrate with OS idle time (no keyboard/mouse) as additional signal
"""

from datetime import datetime
from datetime import time as dtime
from typing import Optional

from loguru import logger

# How many minutes of continuous lying + dark before we confirm sleep
NAP_CONFIRM_MINUTES = 10
SLEEP_CONFIRM_MINUTES = 30
NIGHT_START_HOUR = 22   # 10 PM — after this, lying down = likely sleep not nap
NIGHT_END_HOUR = 8      # 8 AM


class SleepTracker:
    """
    Tracks whether Cole is sleeping or napping based on posture and lighting.

    Sleep confirmation requires:
      - Lying down posture (from MediaPipe)
      - Room lights off OR nighttime hours
      - Sustained for NAP_CONFIRM_MINUTES (nap) or SLEEP_CONFIRM_MINUTES (sleep)

    State is sticky — once sleep is confirmed, it stays until record_wakeup()
    is called or lying posture is no longer detected for 5+ minutes.
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._lying_since: Optional[datetime] = None
        self._dark_since: Optional[datetime] = None
        self._upright_since: Optional[datetime] = None  # tracking return to upright
        self._confirmed_sleep: bool = False
        self._confirmed_nap: bool = False
        self.announced_sleep: bool = False  # Used by curiosity engine

    def update(
        self,
        posture: Optional[str],
        lights_on: Optional[bool],
        room: str = "office",
    ) -> None:
        """
        Feed new sensor readings into the sleep tracker.

        Args:
            posture:   "lying", "sitting", "standing", or None if unknown.
            lights_on: True if room lights are on, False if off, None if unknown.
            room:      Room identifier (for future per-room tracking).
        """
        now = datetime.now()

        # Track lying down posture
        if posture == "lying":
            if self._lying_since is None:
                self._lying_since = now
                logger.debug(f"[Sleep] Lying posture started in '{room}'")
            self._upright_since = None

        elif posture in ("sitting", "standing"):
            if self._lying_since is not None:
                logger.debug(f"[Sleep] Lying posture ended in '{room}'")
            self._lying_since = None

            # Track how long they've been upright since lying
            if self._upright_since is None:
                self._upright_since = now

            # If upright for 5+ minutes, cancel confirmed sleep
            upright_minutes = (now - self._upright_since).total_seconds() / 60
            if upright_minutes >= 5 and (self._confirmed_sleep or self._confirmed_nap):
                logger.info("[Sleep] Woke up detected — cancelling sleep state")
                self.record_wakeup()

        # Track light state
        if lights_on is False:
            if self._dark_since is None:
                self._dark_since = now
        elif lights_on is True:
            if self._dark_since is not None:
                # Lights came on — if we were "sleeping" this may be a wake signal
                logger.debug("[Sleep] Lights came on")
            self._dark_since = None

        # Evaluate sleep/nap confirmation
        self._evaluate_sleep(now)

    def _evaluate_sleep(self, now: datetime) -> None:
        """Check whether we've been lying long enough to confirm sleep or nap."""
        if self._lying_since is None:
            return

        lying_minutes = (now - self._lying_since).total_seconds() / 60
        is_dark = self._dark_since is not None
        is_nighttime = self._is_nighttime(now)

        if (is_dark or is_nighttime) and lying_minutes >= SLEEP_CONFIRM_MINUTES:
            if not self._confirmed_sleep:
                self._confirmed_sleep = True
                logger.info(f"[Sleep] Sleep confirmed ({lying_minutes:.0f}min lying)")

        elif lying_minutes >= NAP_CONFIRM_MINUTES:
            if not self._confirmed_nap:
                self._confirmed_nap = True
                logger.info(f"[Sleep] Nap confirmed ({lying_minutes:.0f}min lying)")

    def is_sleeping(self) -> bool:
        """True if nighttime sleep is confirmed."""
        return self._confirmed_sleep

    def is_napping(self) -> bool:
        """True if daytime nap is confirmed (but not full nighttime sleep)."""
        return self._confirmed_nap and not self._confirmed_sleep

    def get_sleep_signal(self) -> Optional[dict]:
        """
        Return a signal dict for StateFusion if sleep/nap is confirmed.
        Returns None if sleep is not detected.
        """
        if self._confirmed_sleep:
            return {"activity": "sleeping", "confidence": 0.95, "context": {}}
        if self._confirmed_nap:
            return {"activity": "napping", "confidence": 0.85, "context": {}}
        return None

    def record_wakeup(self) -> None:
        """Explicitly reset all sleep state. Call when Cole is clearly awake."""
        self._confirmed_sleep = False
        self._confirmed_nap = False
        self.announced_sleep = False
        self._lying_since = None
        self._dark_since = None
        self._upright_since = None
        logger.debug("[Sleep] Wake-up recorded — sleep state cleared")

    @staticmethod
    def _is_nighttime(now: datetime) -> bool:
        """Return True during typical nighttime hours."""
        hour = now.hour
        return hour >= NIGHT_START_HOUR or hour < NIGHT_END_HOUR
