"""
JARVIS — Safety
===============
Mission: Speaker fan-out for alarm audio. Each repeat cycle plays the
         per-alarm klaxon on every configured speaker, then the TTS
         announcement on top. v4 sources klaxon files from
         `modules/safety/alarms/*.{mp3,wav}` via KlaxonLibrary; absent
         klaxon → degrades to TTS-only for that alarm type without
         crashing.

         Critical separation: alarm audio uses a FIXED voice (system-
         level), NEVER a persona voice. §30.5 calls this out — alarms
         aren't conversation, they're hard safety output.

         Per-alarm cadence (matches §29.5 escalation tiers):
            fire        — 4s gap (slow pulse)
            cat_escape  — 5s gap (repeating chirps)
            door_open   — 10s gap (lowest urgency, long gap)

         The klaxon-then-TTS pattern is sequential, not mixed — running
         two streams concurrently through SpeakerManager would require
         a per-room mixer that doesn't exist yet. Sequential is the
         safer default and matches how typical fire alarm panels
         interleave horn + voice.

Modules: modules/safety/alarms/audio.py
Classes: AlarmAudio, NullAlarmAudio
Spec:    new 2.md §29.1, §29.5.

#todo: Per-room speaker selection. Today AlarmAudio fans out to
       EVERY configured non-null speaker. Eventually the user might
       want quiet rooms (sleeping kid) excluded — gated on the
       sleep_tracker check at fan-out time.
#todo: True ducking — play klaxon at 0.4× while TTS speaks, 1.0×
       between announcements. Needs concurrent audio streams + a
       mixer; for now we sequence (klaxon → 100ms gap → TTS → wait).
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
        klaxons: Optional[Any] = None,        # KlaxonLibrary (already loaded)
    ) -> None:
        self._spk = speaker_manager
        self._tts = tts
        # Per-alarm announcement template. The dispatcher passes the
        # rendered template + suffix into play_for; this is just a
        # default in case a caller wants the pre-renedered version.
        self._announcement_for = announcement_for or {}
        # Optional KlaxonLibrary. None / empty → TTS-only for every
        # alarm type. Per-alarm-type lookup at play time so loading
        # a new klaxon at runtime takes effect on the next loop.
        self._klaxons = klaxons
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
        """Per-cycle: klaxon → 100ms gap → TTS → sleep. Cancelable at
        any point so a higher-priority alarm preempting the audio cuts
        the current cycle cleanly.

        Klaxon and TTS each fan out in parallel across rooms — one slow
        speaker can't block the others. Klaxon is skipped silently when
        no library entry exists for this alarm type (TTS-only mode)."""
        try:
            klaxon = self._klaxon_for(alarm_type)
            while True:
                rooms = self._target_rooms()
                if not rooms:
                    logger.warning(
                        f"[AlarmAudio:{alarm_type}] no speakers — "
                        f"announcement dropped: {announcement!r}"
                    )
                else:
                    # 1. Klaxon (skip if absent for this type).
                    if klaxon is not None:
                        kpcm, krate = klaxon
                        await self._fanout_play(
                            alarm_type, "klaxon", rooms, kpcm, krate,
                        )
                        # Tiny gap so the klaxon tail doesn't bleed into
                        # the announcement. 100ms is below human-audible
                        # cadence break but enough for the speaker
                        # buffers to flush on the slowest sink.
                        await asyncio.sleep(0.1)

                    # 2. TTS announcement.
                    pcm, rate = await self._render_tts(announcement)
                    if pcm:
                        await self._fanout_play(
                            alarm_type, "tts", rooms, pcm, rate,
                        )

                # Repeat cadence — see §29.5 (door is long-gap, fire is
                # slow-pulse, cat is repeating chirps).
                await asyncio.sleep(_REPEAT_SECONDS_FOR.get(alarm_type, 6.0))
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.exception(f"[AlarmAudio:{alarm_type}] fanout crashed: {e}")

    async def _fanout_play(
        self,
        alarm_type: str,
        kind: str,
        rooms: list[str],
        pcm: bytes,
        rate: int,
    ) -> None:
        """Parallel speaker fan-out for one PCM buffer."""
        results = await asyncio.gather(
            *[self._spk.play(r, pcm, rate) for r in rooms],
            return_exceptions=True,
        )
        for r, res in zip(rooms, results):
            if isinstance(res, BaseException):
                logger.debug(
                    f"[AlarmAudio:{alarm_type}] '{r}' {kind} play "
                    f"failed: {res}"
                )

    def _klaxon_for(
        self, alarm_type: str,
    ) -> Optional[tuple[bytes, int]]:
        """KlaxonLibrary lookup with safe-fallback. None → TTS-only."""
        if self._klaxons is None:
            return None
        try:
            return self._klaxons.get(alarm_type)
        except Exception as e:
            logger.debug(
                f"[AlarmAudio:{alarm_type}] klaxon lookup failed: {e}"
            )
            return None

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
