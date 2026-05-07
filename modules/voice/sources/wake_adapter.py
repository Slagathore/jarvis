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
from typing import AsyncIterator, Optional

import numpy as np
from loguru import logger

from modules.voice.sources.base import MicSource

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

    async def _mic_callback(self, pcm: bytes, sample_rate: int) -> None:
        """Called by MicSource per chunk. Slice into OWW-sized chunks and
        push to the queue. sample_rate is captured for logging only — OWW
        is hard-wired to 16kHz and the underlying MicSources resample to
        match, so there's no rate negotiation here.
        """
        if not pcm:
            return
        target_bytes = _OWW_CHUNK_SIZE * 2  # int16 = 2 bytes per sample
        self._partial.extend(pcm)
        while len(self._partial) >= target_bytes:
            chunk = bytes(self._partial[:target_bytes])
            del self._partial[:target_bytes]
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
