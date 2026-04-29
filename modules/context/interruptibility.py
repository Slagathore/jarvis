"""
JARVIS — Ambient Home AI
========================
Mission: Gate all Jarvis speech through an interruptibility check. Before Jarvis
         speaks proactively, this module decides whether it's actually appropriate.
         Factors: current activity score, quiet hours, and a cooldown timer that
         prevents Jarvis from being annoying by speaking too often.

         This is the social intelligence layer. Everything else detects what Cole
         is doing — this module decides whether Jarvis should act on it.

Modules: modules/context/interruptibility.py
Classes: InterruptibilityManager
Functions:
    InterruptibilityManager.__init__(config)  — Load thresholds from config
    InterruptibilityManager.can_interrupt(state, priority) — Main gate check
    InterruptibilityManager.record_interruption() — Start the cooldown timer
    InterruptibilityManager.get_score(activity)   — Look up activity score
    InterruptibilityManager.is_quiet_hours()       — True during sleep hours
    InterruptibilityManager.time_since_last()      — Seconds since last interruption

Variables:
    InterruptibilityManager._activity_scores — dict from config
    InterruptibilityManager._cooldown_minutes — min gap between interruptions
    InterruptibilityManager._quiet_start_hour — hour when quiet hours begin
    InterruptibilityManager._quiet_end_hour   — hour when quiet hours end
    InterruptibilityManager._last_interrupt   — datetime of last interruption

Priority levels (higher = can override stricter gates):
    "conversation"  — User explicitly asked (always allowed)
    "urgent"        — High-priority notification (dryer fire alarm, etc.)
    "notification"  — Normal notification (appliance done) — blocked during gaming
    "ambient"       — Proactive observation — blocked most of the time

#todo: Add escalating patience (Jarvis waits longer each time it's blocked)
#todo: Add per-person interruptibility profiles (Anna vs Sophie vs Cole)
#todo: Learn user override patterns to adjust scores automatically
#todo: Add "do not disturb" manual override via dashboard toggle
"""

import time
from datetime import datetime
from datetime import time as dtime
from typing import Optional

from loguru import logger

from modules.context.state import ActivityState


class InterruptibilityManager:
    """
    Decides whether Jarvis is allowed to speak at this moment.

    Priority hierarchy:
        "conversation" — Always allowed (user already initiated)
        "urgent"       — Allowed unless sleeping/napping
        "notification" — Blocked if score < 0.2 or during quiet hours
        "ambient"      — Blocked if score < configured threshold or cooldown active
    """

    # Minimum interruptibility scores required per priority level
    PRIORITY_THRESHOLDS: dict[str, float] = {
        "conversation": 0.0,   # Never blocked — user asked first
        "urgent":       0.05,  # Only blocked during sleep
        "notification": 0.20,  # Blocked during gaming, coding, video calls
        "ambient":      0.35,  # Proactive speech — only when genuinely free
    }

    def __init__(self, config: dict) -> None:
        interrupt_cfg = config.get("interruptibility", {})
        self._activity_scores: dict[str, float] = interrupt_cfg.get("activity_scores", {})
        self._cooldown_minutes: float = float(
            interrupt_cfg.get("interrupt_cooldown_minutes", 5)
        )

        # Parse quiet hours from "HH:MM" strings
        quiet_start_str = interrupt_cfg.get("quiet_hours_start", "22:00")
        quiet_end_str = interrupt_cfg.get("quiet_hours_end", "08:00")
        self._quiet_start = self._parse_time(quiet_start_str)
        self._quiet_end = self._parse_time(quiet_end_str)

        self._last_interrupt: Optional[float] = None  # monotonic timestamp

    def can_interrupt(
        self,
        state: Optional[ActivityState] = None,
        priority: str = "ambient",
    ) -> bool:
        """
        Main gate: return True if Jarvis is allowed to speak right now.

        Args:
            state:    Current ActivityState. If None, uses "unknown" score.
            priority: Speech priority level (see class docstring).

        Returns:
            True if the speech should proceed, False if Jarvis should stay quiet.
        """
        # Conversation priority is never blocked — user already spoke first
        if priority == "conversation":
            return True

        current_score = self._get_current_score(state)

        # Quiet hours suppress everything except urgent speech
        if self.is_quiet_hours() and priority != "urgent":
            logger.debug(
                f"[Interrupt] Blocked (quiet hours) — priority={priority}"
            )
            return False

        # Check minimum score for this priority
        threshold = self.PRIORITY_THRESHOLDS.get(priority, 0.5)
        if current_score < threshold:
            logger.debug(
                f"[Interrupt] Blocked — score={current_score:.2f} < threshold={threshold} "
                f"(activity={getattr(state, 'activity', 'unknown')}, priority={priority})"
            )
            return False

        # Check cooldown for ambient speech
        if priority == "ambient":
            since_last = self.time_since_last()
            cooldown_sec = self._cooldown_minutes * 60
            if since_last is not None and since_last < cooldown_sec:
                remaining = cooldown_sec - since_last
                logger.debug(
                    f"[Interrupt] Ambient blocked — cooldown {remaining:.0f}s remaining"
                )
                return False

        return True

    def record_interruption(self) -> None:
        """
        Call this immediately after Jarvis proactively speaks.
        Starts the cooldown timer for ambient-priority speech.
        Does NOT need to be called for conversation priority (user-initiated).
        """
        self._last_interrupt = time.monotonic()

    def get_score(self, activity: str) -> float:
        """Look up the interruptibility score for an activity label."""
        return float(self._activity_scores.get(activity, 0.5))

    def is_quiet_hours(self) -> bool:
        """
        Return True if the current time falls within the quiet hours window.
        Handles overnight ranges (e.g., 22:00 to 08:00 crosses midnight).
        """
        now = datetime.now().time()

        if self._quiet_start <= self._quiet_end:
            # Normal range: e.g., 02:00 to 06:00
            return self._quiet_start <= now < self._quiet_end
        else:
            # Overnight range: e.g., 22:00 to 08:00
            return now >= self._quiet_start or now < self._quiet_end

    def time_since_last(self) -> Optional[float]:
        """
        Return seconds since the last registered interruption, or None if
        Jarvis has never spoken proactively.
        """
        if self._last_interrupt is None:
            return None
        return time.monotonic() - self._last_interrupt

    def _get_current_score(self, state: Optional[ActivityState]) -> float:
        """Extract interruptibility from state, or fall back to activity lookup."""
        if state is not None:
            return state.interruptibility
        return self._activity_scores.get("unknown", 0.5)

    @staticmethod
    def _parse_time(time_str: str) -> dtime:
        """Parse "HH:MM" string to datetime.time. Defaults to midnight on failure."""
        try:
            h, m = time_str.split(":")
            return dtime(int(h), int(m))
        except (ValueError, AttributeError):
            logger.warning(f"[Interrupt] Invalid time format '{time_str}', using 00:00")
            return dtime(0, 0)
