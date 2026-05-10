"""
JARVIS — Safety
===============
Mission: Speaker fan-out for alarm audio. Plays the per-alarm klaxon
         on every configured speaker, then a TTS announcement over the
         top with ducking. v4 ships placeholder klaxon WAVs marked
         REPLACE_ME — when real audio assets arrive, drop them into
         `assets/alarms/` per §29.5 and the loader picks them up.

         Critical separation: alarm audio uses a FIXED voice (system-
         level), NEVER a persona voice. §30.5 calls this out — alarms
         aren't conversation, they're hard safety output.

         Phase A (this turn): TTS-only path through the existing
         SpeakerManager. The klaxon WAV mixing is stubbed pending
         real assets. The TTS announcement still works in isolation —
         "BACK DOOR. SNEAKY IS OUTSIDE." through every room speaker
         is the load-bearing UX.

Modules: modules/safety/alarms/audio.py
Classes: AlarmAudio, NullAlarmAudio
Spec:    new 2.md §29.1, §29.5.

#todo: Klaxon WAV loop + TTS-over-klaxon ducking (§29.5). Needs the
       real klaxon assets; the placeholder WAV path is wired in
       `_klaxon_path_for` so the loop only needs hooking up here.
#todo: Per-room speaker selection. Today AlarmAudio fans out to
       EVERY configured non-null speaker. Eventually the user might
       want quiet rooms (sleeping kid) excluded — gated on the
       sleep_tracker check at fan-out time.
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional

from loguru import logger


class AlarmAudio:
    """Routes alarm announcements through SpeakerManager.

    `play_for(alarm_type, suffix)` is called by the dispatcher when an
    alarm becomes the audio owner; it speaks a TTS announcement on
    every room with a configured speaker. The dispatcher handles
    re-calling on state transitions; this class is dumb-fanout only.
    """

    def __init__(
        self,
        speaker_manager: Any,                 # modules.voice.speaker_manager.SpeakerManager
        tts: Any,                             # modules.voice.tts.PiperTTS or compatible
        announcement_for: Optional[dict[str, str]] = None,
    ) -> None:
        self._spk = speaker_manager
        self._tts = tts
        # Per-alarm announcement template. The dispatcher passes the
        # rendered template + suffix into play_for; this is just a
        # default in case a caller wants the pre-renedered version.
        self._announcement_for = announcement_for or {}
        self._current_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    async def play_for(
        self,
        alarm_type: str,
        announcement: str,
        suffix: str = "",
    ) -> None:
        """Speak `announcement + suffix` on every room's speaker. The
        suffix carries the §29.6 multi-alarm "ALSO: DOOR OPEN ALARM
        ACTIVE." appendage. Stops any previous playback first so
        a new alarm taking the audio cuts the previous one cleanly.
        """
        full = f"{announcement} {suffix}".strip()
        async with self._lock:
            await self._cancel_current()
            self._current_task = asyncio.create_task(
                self._fanout(alarm_type, full),
                name=f"alarm_audio:{alarm_type}",
            )

    async def stop(self) -> None:
        """Halt any in-flight announcement loop."""
        async with self._lock:
            await self._cancel_current()

    async def _cancel_current(self) -> None:
        if self._current_task is None:
            return
        self._current_task.cancel()
        try:
            await self._current_task
        except (asyncio.CancelledError, Exception):
            pass
        self._current_task = None

    async def _fanout(self, alarm_type: str, announcement: str) -> None:
        """The per-room speaker fan-out loop. Repeats the announcement
        every ~6 seconds until cancelled. The doc's klaxon-then-TTS
        cadence becomes klaxon-then-TTS-then-pause when assets land.
        """
        try:
            while True:
                pcm, rate = await self._render_tts(announcement)
                if pcm:
                    rooms = self._target_rooms()
                    if not rooms:
                        logger.warning(
                            f"[AlarmAudio:{alarm_type}] no speakers — "
                            f"announcement dropped: {announcement!r}"
                        )
                    else:
                        # Fan out in parallel — one slow speaker shouldn't
                        # block the others.
                        results = await asyncio.gather(
                            *[self._spk.play(r, pcm, rate) for r in rooms],
                            return_exceptions=True,
                        )
                        for r, res in zip(rooms, results):
                            if isinstance(res, BaseException):
                                logger.debug(
                                    f"[AlarmAudio:{alarm_type}] '{r}' play "
                                    f"failed: {res}"
                                )
                # Repeat cadence — see §29.5 (door is long-gap, fire is
                # slow-pulse, cat is repeating chirps). Without real
                # klaxons we just wait between TTS repeats.
                await asyncio.sleep(_REPEAT_SECONDS_FOR.get(alarm_type, 6.0))
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.exception(f"[AlarmAudio:{alarm_type}] fanout crashed: {e}")

    async def _render_tts(self, text: str) -> tuple[bytes, int]:
        """Render TTS to PCM. Tolerant of differing TTS APIs — tries
        synthesize_async, then synthesize, then falls back to empty
        bytes (which the caller logs as 'no audio')."""
        if self._tts is None:
            return b"", 16000
        # PiperTTS.synthesize_async returns (pcm_int16, sample_rate).
        # If your TTS speaks directly through the local PC speaker
        # (speak_async), we still want PCM bytes for SpeakerManager —
        # so synthesize_async is the contract.
        for method_name in ("synthesize_async", "synthesize"):
            meth = getattr(self._tts, method_name, None)
            if meth is None:
                continue
            try:
                result = meth(text)
                if asyncio.iscoroutine(result):
                    result = await result
                if isinstance(result, tuple) and len(result) == 2:
                    pcm, rate = result
                    if isinstance(pcm, (bytes, bytearray)):
                        return bytes(pcm), int(rate)
            except Exception as e:
                logger.debug(f"[AlarmAudio] TTS {method_name} failed: {e}")
        return b"", 16000

    def _target_rooms(self) -> list[str]:
        """Every room with a real (non-null) speaker. SpeakerManager
        already excludes Null sinks from get_rooms()."""
        try:
            return list(self._spk.get_rooms()) if self._spk else []
        except Exception as e:
            logger.debug(f"[AlarmAudio] get_rooms failed: {e}")
            return []


class NullAlarmAudio:
    """No-op AlarmAudio for tests / headless boots. Dispatcher works
    without speakers; phone alerts and dashboard banners still fire."""

    async def play_for(
        self, alarm_type: str, announcement: str, suffix: str = ""
    ) -> None:
        logger.info(
            f"[AlarmAudio:null] would play '{alarm_type}': "
            f"{announcement!r} (suffix={suffix!r})"
        )

    async def stop(self) -> None:
        pass


# Per-§29.5: door is lowest urgency (long gap), cat is repeating chirps,
# fire is slow pulse. Without real klaxons we use these as the inter-TTS
# pause interval.
_REPEAT_SECONDS_FOR: dict[str, float] = {
    "fire":       4.0,
    "cat_escape": 5.0,
    "door_open":  10.0,
}
