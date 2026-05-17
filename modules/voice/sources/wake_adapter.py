"""
JARVIS — Ambient Home AI
========================
Mission: Bridge MicSource (callback-based, room I/O surface) and WakeSource
         (async-iterator-based, openWakeWord input). The two existed because
         they grew separately — MicSource for "I want to capture audio from
         a room into the orchestrator", WakeSource for "feed openWakeWord
         from somewhere that isn't the PC mic." This adapter lets one
         driver back both protocols so a Wyze RTSP cam (or any future
         MicSource) gets per-room wake-word detection without a second
         RTSP connection.

         Bridging shape:
           MicSource emits (pcm_int16_bytes, sample_rate) via callback at
           ~80ms/1280-sample chunks (matched to OWW's frame size).
           WakeSource yields np.int16 arrays of shape (1280,) via async
           iterator. The adapter starts the MicSource lazily on first
           stream() consumption, pipes chunks through an asyncio.Queue,
           and yields them as ndarrays.

Modules: modules/voice/sources/wake_adapter.py
Classes: MicSourceWakeAdapter

#todo: If MicSource.start() can fail (RTSP unreachable), surface that
       upstream so the WakeSourceManager can drop the room rather than
       silently leaving an empty queue forever. For now an unreachable
       cam just produces no wake events for that room — same as before.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from typing import AsyncIterator, Awaitable, Callable, Optional

import numpy as np
from loguru import logger

from modules.voice.sources.base import MicCallback, MicSource


# Audio-level tap signature: (room, rms_db, peak_db, sample_rate).
# rms_db and peak_db are dBFS (negative numbers; 0 = clipping). Used by
# the dashboard to render per-room mic bars for wake-word diagnosis.
AudioLevelTap = Callable[[str, float, float, int], Awaitable[None]]


def _pcm_levels(pcm: bytes) -> tuple[float, float]:
    """Return (rms_db, peak_db) for int16 PCM bytes. dBFS scale; -90
    floor when input is silence."""
    if not pcm:
        return -90.0, -90.0
    samples = np.frombuffer(pcm, dtype=np.int16)
    if samples.size == 0:
        return -90.0, -90.0
    # Avoid float overflow on the square — promote to int64 first.
    sq = samples.astype(np.int64)
    sq = sq * sq
    mean = float(sq.mean())
    rms = math.sqrt(mean) if mean > 0 else 0.0
    peak = float(np.max(np.abs(samples)))
    rms_db = 20.0 * math.log10(rms / 32768.0) if rms > 0 else -90.0
    peak_db = 20.0 * math.log10(peak / 32768.0) if peak > 0 else -90.0
    return max(rms_db, -90.0), max(peak_db, -90.0)

# OWW's required frame size; chunks shorter than this get padded by the
# WakeRunner with a logged warning. Chunks longer get split.
_OWW_CHUNK_SIZE = 1280


class MicSourceWakeAdapter:
    """Wraps a MicSource so it satisfies the WakeSource Protocol.

    Lifecycle:
      - First call to stream() schedules MicSource.start(callback) and
        returns the async iterator.
      - Each subsequent callback invocation pushes one chunk to the queue.
      - The iterator yields chunks as int16 ndarrays of shape (1280,).
      - stop() (via WakeSourceManager.stop) calls MicSource.stop().

    Bounded queue: if the wake-word predictor falls behind for any reason
    (rare — predict is ~5ms on CPU), drop oldest. Stale audio is worse
    than missing audio for wake detection.
    """

    def __init__(self, mic_source: MicSource) -> None:
        self._src = mic_source
        # Queue holds raw chunk bytes; we convert to ndarray inside the
        # generator so the predict thread doesn't pay for an extra copy on
        # every chunk we drop.
        self._queue: "asyncio.Queue[bytes]" = asyncio.Queue(maxsize=64)
        self._started = False
        self._start_lock = asyncio.Lock()
        self._stopped = False
        # Buffer for chunks that don't already match OWW_CHUNK_SIZE — Wyze
        # mic emits 1280-sample chunks today, but USB mic blocksize can
        # drift slightly under load.
        self._partial = bytearray()
        # Rolling buffer of recent OWW-sized chunks (int16, 16 kHz), ~5 s
        # deep — 80 chunks × 1280 samples. Lets the orchestrator's ambient
        # AudioClassifier read a recent snapshot via get_recent_audio()
        # without opening a second stream. This is the role the legacy
        # WakeWordDetector's buffer played for the office PC mic; routing
        # the office mic through the cascade moves that duty here.
        # _mic_callback and get_recent_audio both run on the event loop
        # (UsbMicSource bridges its audio thread → loop), so no lock.
        self._recent_audio: deque = deque(maxlen=80)
        # Optional recording tap: when set, every raw mic chunk is also
        # forwarded here. Used by the orchestrator's _on_wake_detected /
        # _listen_followup paths so they can capture audio from the SAME
        # room the wake fired in, instead of always recording from the
        # office PC mic. The tap receives raw `(pcm_bytes, sample_rate)`
        # straight from the MicSource — no resampling, no chunk-size
        # alignment. The recording path applies its own RMS / silence
        # detection on top.
        self._recording_tap: Optional[MicCallback] = None
        self._recording_sample_rate: int = 16000
        # Audio-level tap: see AudioLevelTap. Computed once per chunk and
        # forwarded for the dashboard's per-room mic bars. Throttled to
        # one event per `_audio_level_min_interval_s` per room so a fast
        # mic at 80ms chunks doesn't drown the WebSocket.
        self._audio_level_tap: Optional[AudioLevelTap] = None
        self._audio_level_last_emit: float = 0.0
        self._audio_level_min_interval_s: float = 0.1  # ~10Hz

    @property
    def room(self) -> str:
        return self._src.room

    def stream(self) -> AsyncIterator[np.ndarray]:
        """Async-iterate int16 ndarrays of shape (1280,). Lazily starts the
        underlying MicSource on first consumption — this method itself
        returns the iterator immediately so callers can use the standard
        `async for chunk in source.stream():` pattern without an await.
        """
        return self._iter()

    async def _ensure_started(self) -> None:
        async with self._start_lock:
            if self._started:
                return
            self._started = True
            try:
                await self._src.start(self._mic_callback)
                logger.info(
                    f"[WakeAdapter:{self.room}] MicSource started for wake-word"
                )
            except Exception as e:
                # Don't raise — a Wyze cam being offline at boot shouldn't
                # crash the whole wake_sources manager. The empty queue
                # just means no wake events come from this room.
                logger.warning(
                    f"[WakeAdapter:{self.room}] MicSource start failed: {e}"
                )

    def attach_recording_tap(self, callback: MicCallback) -> None:
        """Install a recording tap. Every chunk this adapter receives from
        its MicSource is also forwarded to `callback`. Used by the
        orchestrator's wake-recording path to capture audio from the room
        the wake actually fired in. Replaces any existing tap silently."""
        self._recording_tap = callback

    def detach_recording_tap(self) -> None:
        """Remove the recording tap. No-op if none was attached."""
        self._recording_tap = None

    def attach_audio_level_tap(self, callback: AudioLevelTap) -> None:
        """Install an audio-level tap. RMS + peak (dBFS) are computed
        per chunk and forwarded to `callback` at ~10Hz. Used by the
        dashboard to render per-room mic bars for wake-word debugging."""
        self._audio_level_tap = callback

    def detach_audio_level_tap(self) -> None:
        """Remove the audio-level tap. No-op if none was attached."""
        self._audio_level_tap = None

    @property
    def recording_sample_rate(self) -> int:
        """Best-known sample rate for chunks delivered to the recording tap.
        Defaults to 16000 until the first chunk is observed; updated on
        every callback so consumers can read it after a few chunks have
        flowed."""
        return self._recording_sample_rate

    def get_recent_audio(self, seconds: float) -> Optional[np.ndarray]:
        """Last `seconds` of buffered mic audio as a float32 [-1, 1] mono
        waveform at 16 kHz. None when nothing has been buffered yet.

        Mirrors WakeWordDetector.get_recent_audio so the orchestrator's
        ambient AudioClassifier (YAMNet) can read from a cascade-driven
        mic exactly the way it read from the legacy PC wake detector —
        the OWW chunks are 16 kHz (MicSources resample to match OWW)."""
        if not self._recent_audio:
            return None
        n_chunks = max(1, int(seconds * 16000 / _OWW_CHUNK_SIZE))
        chunks = list(self._recent_audio)[-n_chunks:]
        if not chunks:
            return None
        pcm_int16 = np.concatenate(chunks)
        return pcm_int16.astype(np.float32) / 32768.0

    async def _mic_callback(self, pcm: bytes, sample_rate: int) -> None:
        """Called by MicSource per chunk. Slice into OWW-sized chunks and
        push to the queue. sample_rate is captured for logging only — OWW
        is hard-wired to 16kHz and the underlying MicSources resample to
        match, so there's no rate negotiation here.
        """
        if not pcm:
            return
        # Forward to the recording tap FIRST. The tap gets raw chunks at
        # the source's native rate (typically 16kHz from WyzeRtspMicSource);
        # the recording layer handles its own framing. We do this before
        # the OWW slicing so the tap sees whole chunks even if OWW's queue
        # is full and would have dropped them.
        self._recording_sample_rate = sample_rate
        tap = self._recording_tap
        if tap is not None:
            try:
                await tap(pcm, sample_rate)
            except Exception as e:
                # Tap failures must NOT break wake detection. Log and move on.
                logger.warning(
                    f"[WakeAdapter:{self.room}] recording tap raised: {e}"
                )
        # Audio-level tap — throttled. Compute once per chunk; emit only
        # if enough wall time has passed since the last emit. Wake-debug
        # bars don't need every 80ms sample.
        level_tap = self._audio_level_tap
        if level_tap is not None:
            now = time.monotonic()
            if now - self._audio_level_last_emit >= self._audio_level_min_interval_s:
                self._audio_level_last_emit = now
                try:
                    rms_db, peak_db = _pcm_levels(pcm)
                    await level_tap(self.room, rms_db, peak_db, sample_rate)
                except Exception as e:
                    logger.debug(
                        f"[WakeAdapter:{self.room}] audio_level tap raised: {e}"
                    )
        target_bytes = _OWW_CHUNK_SIZE * 2  # int16 = 2 bytes per sample
        self._partial.extend(pcm)
        while len(self._partial) >= target_bytes:
            chunk = bytes(self._partial[:target_bytes])
            del self._partial[:target_bytes]
            # Record into the ambient buffer BEFORE the OWW queue — the
            # AudioClassifier wants continuous audio even on the chunks
            # the (bounded) wake queue would drop under backpressure.
            self._recent_audio.append(np.frombuffer(chunk, dtype=np.int16))
            try:
                self._queue.put_nowait(chunk)
            except asyncio.QueueFull:
                # Drop oldest, take newest. Stale wake audio is useless.
                try:
                    _ = self._queue.get_nowait()
                    self._queue.put_nowait(chunk)
                except Exception:
                    pass

    async def _iter(self) -> AsyncIterator[np.ndarray]:
        # Start the source on first iteration. Doing it here (rather than
        # in stream()) means a caller that creates the iterator and never
        # iterates it doesn't accidentally open an RTSP connection.
        await self._ensure_started()
        try:
            while not self._stopped:
                try:
                    pcm = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Periodically wake so a stop() call doesn't have to wait
                    # for the next mic chunk to take effect. Keeps shutdown snappy.
                    continue
                arr = np.frombuffer(pcm, dtype=np.int16)
                if arr.shape != (_OWW_CHUNK_SIZE,):
                    # Defensive — _mic_callback enforces this, but a bad
                    # MicSource implementation could slip through.
                    continue
                yield arr
        except asyncio.CancelledError:
            raise
        finally:
            # The WakeSourceManager's runner will see the iterator end and
            # tear down. Don't stop the MicSource here — there could be
            # other consumers (e.g. STT) bound to the same source via a
            # multiplexer in the future. For now, MicManager.close()
            # handles the actual stop on shutdown.
            return
