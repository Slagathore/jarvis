"""
JARVIS — Ambient Home AI
========================
Mission: Track sleep/nap state per identified person, per room, so Jarvis
         can defer to other sleepers (Anna napping in the bedroom) while
         remaining responsive to Cole at his desk. Combines posture
         (lying), light state, time of day, and inactivity duration.

         The tracker is signal-driven: any activity from a recognized
         person (PC active, wake word, voice, sustained upright posture)
         clears that person's sleep state. Other people's states are
         untouched. Brief disappearances from frame do NOT clear sleep —
         only an explicit awake signal does (the door-disappearance rule).

Modules: modules/context/sleep_tracker.py
Classes: SleepTracker, _PersonSleepState
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from loguru import logger

# Confirmation thresholds in minutes of continuous lying.
NAP_CONFIRM_MINUTES = 10
SLEEP_CONFIRM_MINUTES = 30
NIGHT_START_HOUR = 22
NIGHT_END_HOUR = 8

# Posture is sampled at ~1 fps. MediaPipe occasionally misclassifies a
# seated desk pose as 'lying' when hips are partially occluded. Require
# this many consecutive frames before flipping the lying/upright trackers
# so a single bad frame can't reset the upright timer or fake a nap.
POSTURE_HYSTERESIS_FRAMES = 3

# How long a per-person state survives without any new sighting before we
# consider it stale and stop using it for speech gating. The
# door-disappearance rule says a brief absence ≠ exit, so we keep the
# state alive a long time. 6h is long enough for a real nap, short enough
# that a forgotten state doesn't haunt us across days.
STALENESS_SECONDS = 6 * 3600


@dataclass
class _PersonSleepState:
    """Per-(room, person) sleep tracking. person_id is None for an
    unrecognized human seen in this room — we still want to know someone
    is napping there even if YOLO can't ID them.
    """
    room: str
    person_id: Optional[int]
    person_name: Optional[str]
    lying_since: Optional[datetime] = None
    upright_since: Optional[datetime] = None
    dark_since: Optional[datetime] = None
    confirmed_sleep: bool = False
    confirmed_nap: bool = False
    announced_sleep: bool = False
    last_seen: datetime = field(default_factory=datetime.now)
    # Hysteresis counters — the last few frames' classifications. We flip
    # the lying_since / upright_since trackers only after
    # POSTURE_HYSTERESIS_FRAMES same-class frames in a row.
    consecutive_lying: int = 0
    consecutive_upright: int = 0


class SleepTracker:
    """
    Tracks sleep state per identified person per room.

    Activity signals (PC active, voice, wake word, sustained upright
    posture) clear ONLY the matching person's state — not other sleepers
    in other rooms. Speech gating queries `is_anyone_sleeping_in(room)`
    before Jarvis speaks into a room, so a nap is respected even when the
    person requesting speech is awake elsewhere.
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        # Key: (room, person_id_or_None). One person can have an entry per
        # room; that's fine — sleep is location-bound (you don't carry the
        # bedroom nap with you when you walk to the kitchen).
        self._states: dict[tuple[str, Optional[int]], _PersonSleepState] = {}

    # ── Per-frame update from vision_loop ─────────────────────────────

    def update(
        self,
        room: str,
        posture: Optional[str],
        lights_on: Optional[bool],
        person_id: Optional[int] = None,
        person_name: Optional[str] = None,
    ) -> None:
        """Feed a frame's observation into the per-person tracker.

        Called from _vision_loop with the room's current posture +
        identity. posture may be 'unknown' or None when MediaPipe can't
        see the body — in that case we update last_seen but don't move
        the timers (preserves nap state across brief blind moments,
        per the door-disappearance rule).
        """
        now = datetime.now()
        key = (room, person_id)
        st = self._states.get(key)
        if st is None:
            st = _PersonSleepState(
                room=room, person_id=person_id, person_name=person_name
            )
            self._states[key] = st

        # Refresh identity if it just resolved
        if person_name and not st.person_name:
            st.person_name = person_name
        st.last_seen = now

        # Light-state tracking (room-wide, not per-person, but cheaper
        # to write here than to maintain a parallel structure).
        if lights_on is False:
            if st.dark_since is None:
                st.dark_since = now
        elif lights_on is True:
            st.dark_since = None

        # Posture transitions with hysteresis
        if posture == "lying":
            st.consecutive_lying += 1
            st.consecutive_upright = 0
            if (
                st.consecutive_lying >= POSTURE_HYSTERESIS_FRAMES
                and st.lying_since is None
            ):
                st.lying_since = now
                logger.debug(
                    f"[Sleep] lying confirmed in '{room}' "
                    f"({st.person_name or 'unknown'})"
                )
        elif posture in ("sitting", "standing"):
            st.consecutive_upright += 1
            st.consecutive_lying = 0
            if st.consecutive_upright >= POSTURE_HYSTERESIS_FRAMES:
                if st.lying_since is not None:
                    logger.debug(
                        f"[Sleep] upright in '{room}' "
                        f"({st.person_name or 'unknown'})"
                    )
                    st.lying_since = None
                if st.upright_since is None:
                    st.upright_since = now
                # 5 minutes of sustained upright = wake
                upright_min = (now - st.upright_since).total_seconds() / 60
                if upright_min >= 5 and (st.confirmed_sleep or st.confirmed_nap):
                    logger.info(
                        f"[Sleep] upright {upright_min:.0f}min — clearing "
                        f"sleep state for {st.person_name or 'unknown'} in {room}"
                    )
                    self._reset(st)
        else:
            # Unknown / no posture — leave timers alone. The
            # door-disappearance rule: brief absence isn't an exit and
            # isn't a wake-up. Just refresh last_seen above.
            pass

        self._evaluate(st, now)
        self._gc_stale(now)

    def _evaluate(self, st: _PersonSleepState, now: datetime) -> None:
        if st.lying_since is None:
            return
        lying_min = (now - st.lying_since).total_seconds() / 60
        is_dark = st.dark_since is not None
        is_night = self._is_nighttime(now)

        if (is_dark or is_night) and lying_min >= SLEEP_CONFIRM_MINUTES:
            if not st.confirmed_sleep:
                st.confirmed_sleep = True
                logger.info(
                    f"[Sleep] sleep confirmed for "
                    f"{st.person_name or 'unknown'} in '{st.room}' "
                    f"({lying_min:.0f}min)"
                )
        elif lying_min >= NAP_CONFIRM_MINUTES:
            if not st.confirmed_nap and not st.confirmed_sleep:
                st.confirmed_nap = True
                logger.info(
                    f"[Sleep] nap confirmed for "
                    f"{st.person_name or 'unknown'} in '{st.room}' "
                    f"({lying_min:.0f}min)"
                )

    # ── Activity-cancels-sleep API ────────────────────────────────────

    def record_activity(
        self,
        person_id: Optional[int] = None,
        person_name: Optional[str] = None,
        room: Optional[str] = None,
        signal: str = "activity",
    ) -> int:
        """Cancel sleep state for the matching person.

        Resolution order:
          1. If person_id is provided, clear all rooms for that pid.
          2. Else if person_name is provided, clear all rooms whose
             entry matches that name.
          3. Else if room is provided, clear ALL entries in that room
             (used when we have a presence signal but no identity —
             walking around in the kitchen wakes whoever was sleeping
             in the kitchen).

        Returns the count of states cleared, useful for logging.
        """
        cleared = 0
        for st in list(self._states.values()):
            if not (st.confirmed_sleep or st.confirmed_nap):
                continue
            match = False
            if person_id is not None and st.person_id == person_id:
                match = True
            elif person_name is not None and st.person_name == person_name:
                match = True
            elif room is not None and person_id is None and person_name is None:
                if st.room == room:
                    match = True
            if match:
                logger.info(
                    f"[Sleep] {signal} clears sleep for "
                    f"{st.person_name or 'unknown'} in '{st.room}'"
                )
                self._reset(st)
                cleared += 1
        return cleared

    def record_wakeup(
        self,
        person_id: Optional[int] = None,
        person_name: Optional[str] = None,
        room: Optional[str] = None,
    ) -> int:
        """Alias for record_activity with signal='explicit-wake'."""
        return self.record_activity(
            person_id=person_id,
            person_name=person_name,
            room=room,
            signal="explicit-wake",
        )

    # ── Queries ───────────────────────────────────────────────────────

    def is_anyone_sleeping_in(self, room: str) -> bool:
        """Speech gate: should Jarvis stay quiet in this room?"""
        return bool(self._fresh_sleepers_in(room))

    def get_sleepers_in(self, room: str) -> list[dict]:
        """List of {person_name, person_id, kind} for fresh sleepers in a
        room. kind = 'sleeping' or 'napping'.
        """
        out = []
        for st in self._fresh_sleepers_in(room):
            out.append({
                "person_id": st.person_id,
                "person_name": st.person_name,
                "kind": "sleeping" if st.confirmed_sleep else "napping",
                "since": st.lying_since.isoformat() if st.lying_since else None,
            })
        return out

    def get_room_sleep_signal(self, room: str) -> Optional[dict]:
        """State-fusion signal for the active-room context. Returns the
        strongest sleep state in the room (sleep beats nap), or None.
        """
        sleepers = self._fresh_sleepers_in(room)
        if not sleepers:
            return None
        names = [s.person_name for s in sleepers if s.person_name]
        if any(s.confirmed_sleep for s in sleepers):
            return {
                "activity": "sleeping",
                "confidence": 0.95,
                "context": {"sleepers": names},
            }
        return {
            "activity": "napping",
            "confidence": 0.85,
            "context": {"sleepers": names},
        }

    def snapshot(self) -> list[dict]:
        """Dashboard / scene snapshot of every tracked sleeper."""
        out = []
        for st in self._states.values():
            if not (st.confirmed_sleep or st.confirmed_nap):
                continue
            out.append({
                "room": st.room,
                "person_id": st.person_id,
                "person_name": st.person_name,
                "kind": "sleeping" if st.confirmed_sleep else "napping",
                "since": st.lying_since.isoformat() if st.lying_since else None,
                "last_seen": st.last_seen.isoformat(),
            })
        return out

    # ── Internal helpers ──────────────────────────────────────────────

    def _fresh_sleepers_in(self, room: str) -> list[_PersonSleepState]:
        now = datetime.now()
        return [
            st for st in self._states.values()
            if st.room == room
            and (st.confirmed_sleep or st.confirmed_nap)
            and (now - st.last_seen).total_seconds() < STALENESS_SECONDS
        ]

    @staticmethod
    def _reset(st: _PersonSleepState) -> None:
        st.confirmed_sleep = False
        st.confirmed_nap = False
        st.announced_sleep = False
        st.lying_since = None
        st.upright_since = None
        st.consecutive_lying = 0
        # Keep dark_since — light state is room-wide and unrelated to wake.

    def _gc_stale(self, now: datetime) -> None:
        """Drop entries that haven't been seen in a long time so the dict
        doesn't grow unbounded across guests / unknown faces.
        """
        stale = [
            key for key, st in self._states.items()
            if (now - st.last_seen).total_seconds() > STALENESS_SECONDS * 2
        ]
        for key in stale:
            self._states.pop(key, None)

    @staticmethod
    def _is_nighttime(now: datetime) -> bool:
        h = now.hour
        return h >= NIGHT_START_HOUR or h < NIGHT_END_HOUR
