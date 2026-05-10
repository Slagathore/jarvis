"""
JARVIS — Safety
===============
Mission: AlarmDispatcher — owns audio routing across multiple
         simultaneous alarms. Per §29.6, audio doesn't stack — the
         highest-priority FIRING alarm "owns" the speakers; lower-
         priority alarms remain ACTIVE in the subsystem (logged,
         dashboard banner, phone alert) but are AUDIO-SUPPRESSED. When
         the audio-owning alarm resolves or mutes, the next-highest
         alarm immediately takes the audio.

         Voice-silence semantics:
            "Jarvis, stop alarm"   → mute the current audio owner
            "Jarvis, silence all"  → mute every active alarm

Modules: modules/safety/alarms/dispatcher.py
Classes: AlarmDispatcher
Spec:    new 2.md §29.6, §29.7.

#todo: Fire-alarm signal-increase override per §29.4 — once a fire
       signal is rising, the fire alarm cannot be silenced for 30s.
       Implementation: `voice_silence` checks self.alarms['fire']
       for an `override_until_ts` stamp and refuses if set.
"""
from __future__ import annotations

from typing import Any, Optional

from loguru import logger

from modules.safety.alarms.alarm import Alarm
from modules.safety.alarms.state import AlarmState, AlarmType


class AlarmDispatcher:
    """Owns the priority-ordered map of registered alarms + the audio
    routing logic. Concrete alarms subclass `Alarm`, register here,
    and call `await self._notify_dispatcher` on every state change.
    """

    def __init__(self, audio: Any) -> None:
        self.alarms: dict[str, Alarm] = {}
        self.audio = audio
        # Priority: fire is highest, then cat, then door. Strings rather
        # than enum values so adding a new alarm type doesn't require
        # a code change here — just register it with a name and an
        # ALARM_PRIORITY value lower than the existing tail.
        self.priority_order: list[str] = [
            AlarmType.FIRE,
            AlarmType.CAT_ESCAPE,
            AlarmType.DOOR_OPEN,
            AlarmType.CLOWN,    # v4.1 — lowest; audio-suppressed by any other
        ]

    # ── Registration / lifecycle ────────────────────────────────────────────

    def register(self, alarm: Alarm) -> None:
        name = alarm.ALARM_TYPE
        self.alarms[name] = alarm
        alarm.attach(self)
        # Keep priority_order consistent with registered alarms — append
        # any new types after the canonical three.
        if name not in self.priority_order:
            self.priority_order.append(name)
        logger.info(f"[AlarmDispatcher] registered '{name}'")

    async def start(self) -> None:
        for a in self.alarms.values():
            try:
                await a.start()
            except Exception as e:
                logger.exception(
                    f"[AlarmDispatcher] start of '{a.ALARM_TYPE}' failed: {e}"
                )

    async def stop(self) -> None:
        for a in self.alarms.values():
            try:
                await a.stop()
            except Exception:
                pass
        try:
            await self.audio.stop()
        except Exception:
            pass

    # ── Read API used by the alarm-state-change callback ────────────────────

    def active_alarms(self) -> list[str]:
        """Alarms in either FIRING_AUDIO or MUTED — ordered by priority."""
        return [
            n for n in self.priority_order
            if n in self.alarms and self.alarms[n].state in (
                AlarmState.FIRING_AUDIO, AlarmState.MUTED
            )
        ]

    def audio_owner(self) -> Optional[str]:
        """Highest-priority alarm currently in FIRING_AUDIO."""
        for name in self.priority_order:
            a = self.alarms.get(name)
            if a is not None and a.state == AlarmState.FIRING_AUDIO:
                return name
        return None

    # ── State-change callback (called by Alarm._notify_dispatcher) ──────────

    async def on_alarm_state_change(
        self, name: str, old: AlarmState, new: AlarmState
    ) -> None:
        """Decide whether the audio owner changes, and ask AlarmAudio
        to play / stop accordingly. Called from inside Alarm methods
        AFTER the alarm's own state mutation has happened, so
        audio_owner() reflects the new world."""
        owner = self.audio_owner()
        suffix = self._suffix()
        if owner is None:
            await self.audio.stop()
            return

        owner_alarm = self.alarms[owner]
        announcement, _body = owner_alarm._announcement({})  # title only

        # Three transitions matter for audio:
        #  1. The alarm that just transitioned IS the new owner →
        #     start playing (covers INACTIVE→FIRING and a lower-priority
        #     alarm becoming owner because the previous owner muted).
        #  2. The alarm that just transitioned WAS the audio owner and
        #     dropped to MUTED/RESOLVED → audio_owner() now points at the
        #     next-highest alarm (or None). Play for new owner.
        #  3. Anything else → no audio change needed.
        was_owner = (old == AlarmState.FIRING_AUDIO and owner != name)
        is_owner_now = (owner == name and new == AlarmState.FIRING_AUDIO)
        if is_owner_now or was_owner:
            await self.audio.play_for(
                alarm_type=owner, announcement=announcement, suffix=suffix,
            )

    def _suffix(self) -> str:
        """Multi-alarm 'ALSO: …' tail for the TTS announcement.
        Empty string when only one alarm is active."""
        active = self.active_alarms()
        if len(active) <= 1:
            return ""
        owner = self.audio_owner()
        others = [
            a.upper().replace("_", " ") for a in active if a != owner
        ]
        return f" ALSO: {', '.join(others)} ALARM ACTIVE."

    # ── Voice commands ──────────────────────────────────────────────────────

    async def voice_silence_current(self) -> None:
        """'Jarvis, stop alarm' — silences whichever alarm currently
        holds the audio. The next-highest active alarm immediately
        takes the audio (handled by the on_alarm_state_change
        re-evaluation that fires when the alarm transitions to MUTED).
        """
        owner = self.audio_owner()
        if owner is None:
            return
        await self.alarms[owner].voice_silence()

    async def voice_silence_all(self) -> None:
        """'Jarvis, silence all alarms' — mute every currently-firing
        alarm with their rearm timers. Phone notifications + dashboard
        banners continue (per §29.6)."""
        for name in self.priority_order:
            a = self.alarms.get(name)
            if a is not None and a.state == AlarmState.FIRING_AUDIO:
                await a.voice_silence()
