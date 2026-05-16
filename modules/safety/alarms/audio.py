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

import numpy as np
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

    async def play_clown_sequence(
        self,
        announcement: str,
        on_complete: Optional[Any] = None,
        horn_loop_count: int = 3,
    ) -> None:
        """v4.1 §29.8 — fire-once-then-resolve clown audio.
        Sequence: clown horns × N → TTS announcement → calliope full
        song → invoke `on_complete` callback so the alarm can
        transition itself to RESOLVED. Distinct from `play_for` which
        loops forever; the clown alarm's natural-end semantics are
        bounded by the calliope file's duration (the joke ends when
        the music does).
        """
        async with self._lock:
            await self._cancel_current()
            self._current_task = asyncio.create_task(
                self._clown_fanout(announcement, on_complete, horn_loop_count),
                name="alarm_audio:clown",
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

    async def _clown_fanout(
        self,
        announcement: str,
        on_complete: Optional[Any],
        horn_loop_count: int,
    ) -> None:
        """v4.1 §29.8.2 single-pass clown sequence:
            1. clown horns × horn_loop_count
            2. TTS announcement (rendered via the existing TTS path)
            3. calliope full track
            4. invoke `on_complete()` so the alarm self-resolves

        No looping — when the calliope finishes the alarm is done. The
        horn × N gives the LLM-generated improv response time to
        complete in parallel (per §29.8.3, generation is kicked off
        when `clown.detected` fires; this method is invoked only after
        generation has resolved or fallen back).

        A higher-priority alarm preempting the audio cancels this task
        cleanly via _cancel_current — the on_complete callback is NOT
        invoked in that case (the dispatcher's normal state-change
        flow takes over)."""
        try:
            horns = self._klaxon_for("clown")
            calliope = self._klaxon_for("calliope")

            rooms = self._target_rooms()
            if not rooms:
                logger.warning(
                    "[AlarmAudio:clown] no speakers — "
                    f"announcement dropped: {announcement!r}"
                )
            else:
                # 1. Clown horns × N. Each loop fans out in parallel.
                if horns is not None:
                    hpcm, hrate = horns
                    for i in range(max(1, int(horn_loop_count))):
                        await self._fanout_play(
                            "clown", f"horns[{i+1}/{horn_loop_count}]",
                            rooms, hpcm, hrate,
                        )
                        # 100ms gap so successive loops don't slur into
                        # one long honk.
                        await asyncio.sleep(0.1)

                # 2. TTS announcement.
                tpcm, trate = await self._render_tts(announcement)
                if tpcm:
                    await asyncio.sleep(0.15)
                    await self._fanout_play(
                        "clown", "tts", rooms, tpcm, trate,
                    )

                # 3. Calliope full song. Skipping if the file isn't
                # loaded means the clown alarm simply ends after TTS,
                # which is a degraded but coherent experience.
                if calliope is not None:
                    cpcm, crate = calliope
                    await asyncio.sleep(0.3)
                    await self._fanout_play(
                        "clown", "calliope", rooms, cpcm, crate,
                    )

            # 4. Self-resolve via callback so ClownAlarm's state
            # transitions correctly (RESOLVED, dispatcher releases
            # audio, state row + alarm_fires resolution stamped).
            if on_complete is not None:
                try:
                    result = on_complete()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.debug(
                        f"[AlarmAudio:clown] on_complete callback "
                        f"raised: {e}"
                    )
        except asyncio.CancelledError:
            # Cancellation is the preempt path — DO NOT call
            # on_complete; the alarm stays in FIRING_AUDIO and the
            # dispatcher's normal state machine routes audio to the
            # higher-priority alarm. ClownAlarm's resume path picks
            # up from the start of the current segment when audio
            # comes back.
            return
        except Exception as e:
            logger.exception(f"[AlarmAudio:clown] sequence crashed: {e}")

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
        """Render TTS to PCM. Tolerant of differing TTS APIs.

        PiperTTS.synthesize[_async]() returns a float32 np.ndarray in
        [-1, 1] — NOT a (bytes, rate) tuple. Earlier code only handled
        the tuple shape, so every spoken alarm announcement silently
        rendered to empty audio. Both shapes are handled now; other TTS
        backends that return (bytes, rate) keep working unchanged."""
        if self._tts is None:
            return b"", 16000
        for method_name in ("synthesize_async", "synthesize"):
            meth = getattr(self._tts, method_name, None)
            if meth is None:
                continue
            try:
                result = meth(text)
                if asyncio.iscoroutine(result):
                    result = await result
                # Backend A: (pcm_bytes, sample_rate)
                if isinstance(result, tuple) and len(result) == 2:
                    pcm, rate = result
                    if isinstance(pcm, (bytes, bytearray)):
                        return bytes(pcm), int(rate)
                # Backend B: PiperTTS — float32 ndarray in [-1, 1]
                if isinstance(result, np.ndarray) and result.size:
                    pcm_i16 = (
                        np.clip(result, -1.0, 1.0) * 32767.0
                    ).astype(np.int16)
                    rate = int(getattr(self._tts, "_sample_rate", 22050))
                    return pcm_i16.tobytes(), rate
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
