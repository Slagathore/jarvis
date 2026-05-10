"""
JARVIS — Safety
===============
Mission: Alarm state-machine enums shared by every concrete alarm + the
         dispatcher. No I/O, no logic — just labels.

Modules: modules/safety/alarms/state.py
Spec:    new 2.md §29.1.
"""
from __future__ import annotations

from enum import Enum


class AlarmState(str, Enum):
    """Per-alarm-instance lifecycle.
    INACTIVE   — condition is false, alarm is dormant.
    FIRING_AUDIO — condition true, audio holding (subject to priority).
    MUTED      — condition still active but audio paused; rearm timer running.
    RESOLVED   — condition cleared; the most recent fire is closed out.
    SUPPRESSED — condition gated by a higher layer (global disarm,
                 per-cat suppression, etc.). Differs from MUTED in that
                 the alarm cannot fire at all while in this state.
    """
    INACTIVE = "inactive"
    FIRING_AUDIO = "firing_audio"
    MUTED = "muted"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class AlarmType:
    """String constants for the three v4 alarm types. Used as keys in
    config blocks and as the priority key in AlarmDispatcher's
    priority_order list. Centralizing prevents typo drift.
    """
    FIRE = "fire"
    CAT_ESCAPE = "cat_escape"
    DOOR_OPEN = "door_open"
