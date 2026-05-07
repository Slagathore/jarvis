"""
JARVIS — Ambient Home AI
========================
Mission: Demux the audio track from a Wyze V2 RTSP stream and emit
         16kHz mono int16 PCM chunks to the manager's callback. Uses PyAV
         (which bundles FFmpeg, so no system FFmpeg needed) to handle the
         G.711/AAC variation that wz_mini_hacks emits depending on firmware
         build — PyAV's AudioResampler decodes both transparently.

         The demux loop is blocking (PyAV's container.demux() is a sync
         generator), so it runs inside asyncio.to_thread for its entire
         lifetime. The callback is awaited via run_coroutine_threadsafe
         from the worker thread back into the asyncio loop. This is more
         efficient than calling to_thread per packet, which would burn a
         thread context-switch ~30 times per second.

         Each emitted chunk targets ~80ms (1280 samples at 16kHz) so it
         lines up with openWakeWord's frame size — but PyAV decodes whatever
         frame size the RTP packets bring, so we accumulate into a slice
         buffer and emit fixed-size chunks.

Modules: modules/voice/sources/wyze_rtsp_mic.py
Classes: WyzeRtspMicSource

#todo: Add a packet-loss watchdog — if PyAV emits N consecutive zero-byte
       frames, force a reconnect rather than silently dropping audio
#todo: Expose the codec PyAV detected (PCMU vs AAC) on the dashboard so
       we can spot wz_mini_hacks build drift between cams
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Optional

import numpy as np
from loguru import logger

from modules.voice.sources.base import MicCallback, MicSource

# Target chunk: 80ms at 16kHz mono int16 → 1280 samples → 2560 bytes.
TARGET_CHUNK_SAMPLES = 1280


class WyzeRtspMicSource(MicSource):
    """PyAV-based RTSP audio demux. One thread holds the open container; the
    asyncio side just waits for the worker to call back.

    On RTSP failure (cam reboot, WiFi blip), the worker exits and start()
    can be called again to reconnect. We don't auto-reconnect inside the
    driver because the manager already polls source health and can decide
    whether to restart based on global policy.
    """

    def __init__(
        self,
        room: str,
        url: str,
        transport: str = "tcp",
        sample_rate_hz: int = 16000,
        channels: int = 1,
        max_consecutive_errors: int = 5,
    ) -> None:
        self._room = room
        self._url = url
        self._transport = transport
        self._sample_rate = sample_rate_hz
        self._channels = channels
        self._max_consecutive_errors = max_consecutive_errors
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._callback: Optional[MicCallback] = None

    @property
    def room(self) -> str:
        return self._room

    async def start(self, callback: MicCallback) -> None:
        if self._worker is not None and self._worker.is_alive():
            logger.warning(
                f"[WyzeMic:{self._room}] start() called while worker alive — ignoring"
            )
            return
        self._loop = asyncio.get_running_loop()
        self._callback = callback
        self._stop_event.clear()
        self._worker = threading.Thread(
            target=self._run_blocking,
            name=f"wyze-rtsp-mic-{self._room}",
            daemon=True,
        )
        self._worker.start()
        logger.info(
            f"[WyzeMic:{self._room}] Demuxer started for {self._url} "
            f"({self._transport}, target {self._sample_rate}Hz/{self._channels}ch)"
        )

    async def stop(self) -> None:
        self._stop_event.set()
        worker = self._worker
        self._worker = None
        if worker is not None and worker.is_alive():
            # Block briefly so the thread can release the AV container before
            # the next caller (or test cleanup) tries to reopen the URL.
            # PyAV's container.close() can take a moment on a half-broken
            # stream, hence the generous timeout.
            await asyncio.to_thread(worker.join, 5.0)
        logger.info(f"[WyzeMic:{self._room}] Stopped")

    # ── Worker thread ────────────────────────────────────────────────────────

    def _run_blocking(self) -> None:
        """Runs in a thread for the source's full lifetime. Opens the AV
        container, iterates demux'd audio frames, accumulates into the
        target chunk size, and posts each chunk back to the asyncio loop.
        """
        try:
            import av
        except ImportError:
            logger.error(
                f"[WyzeMic:{self._room}] PyAV not installed — pip install av"
            )
            return

        container: Any = None
        try:
            try:
                container = av.open(
                    self._url,
                    options={
                        "rtsp_transport": self._transport,
                        # 5s socket timeout. Without this, PyAV inherits
                        # FFmpeg's default 'forever' and a cam that drops mid-
                        # stream wedges the worker thread until we kill the
                        # whole process.
                        "stimeout": "5000000",
                    },
                )
            except Exception as e:
                logger.warning(
                    f"[WyzeMic:{self._room}] av.open failed: {e}"
                )
                return

            audio_stream = next(
                (s for s in container.streams if s.type == "audio"), None
            )
            if audio_stream is None:
                logger.warning(
                    f"[WyzeMic:{self._room}] No audio stream in {self._url} — "
                    "is wz_mini_hacks audio enabled?"
                )
                return

            # Resampler converts whatever the cam emits (G.711 µ-law @ 8kHz
            # or AAC @ 16kHz, depending on build) into the int16 mono target.
            resampler = av.AudioResampler(
                format="s16",
                layout="mono" if self._channels == 1 else "stereo",
                rate=self._sample_rate,
            )

            buffer = bytearray()
            target_bytes = TARGET_CHUNK_SAMPLES * 2 * self._channels  # int16 = 2 bytes
            consecutive_errors = 0

            for packet in container.demux(audio_stream):
                if self._stop_event.is_set():
                    break
                try:
                    for frame in packet.decode():
                        for resampled in self._iter_resampled(resampler, frame):
                            arr = resampled.to_ndarray()
                            buffer.extend(arr.astype(np.int16).tobytes())
                            while len(buffer) >= target_bytes:
                                chunk = bytes(buffer[:target_bytes])
                                del buffer[:target_bytes]
                                self._post_chunk(chunk)
                    consecutive_errors = 0
                except Exception as e:
                    consecutive_errors += 1
                    logger.debug(
                        f"[WyzeMic:{self._room}] decode err "
                        f"({consecutive_errors}/{self._max_consecutive_errors}): {e}"
                    )
                    if consecutive_errors >= self._max_consecutive_errors:
                        logger.warning(
                            f"[WyzeMic:{self._room}] {consecutive_errors} consecutive "
                            "decode errors — bailing out, manager will restart"
                        )
                        break
        finally:
            if container is not None:
                try:
                    container.close()
                except Exception:
                    pass

    @staticmethod
    def _iter_resampled(resampler: Any, frame: Any) -> Any:
        """PyAV's AudioResampler.resample() returns either a single frame
        or a list of frames, version-dependent. Normalize to an iterable.
        """
        out = resampler.resample(frame)
        if out is None:
            return ()
        if isinstance(out, list):
            return out
        return (out,)

    def _post_chunk(self, chunk: bytes) -> None:
        """Hand the chunk back to the asyncio loop. We schedule the callback
        as a task — fire-and-forget — so a slow callback doesn't backpressure
        the demux thread (which would let RTP packets pile up in PyAV's
        internal buffer and re-introduce the latency we just fought to avoid).
        """
        loop = self._loop
        cb = self._callback
        if loop is None or cb is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._schedule_cb, cb, chunk)
        except RuntimeError:
            pass

    def _schedule_cb(self, cb: MicCallback, chunk: bytes) -> None:
        try:
            asyncio.create_task(self._safe_invoke(cb, chunk))
        except RuntimeError:
            pass  # loop closing

    async def _safe_invoke(self, cb: MicCallback, chunk: bytes) -> None:
        try:
            await cb(chunk, self._sample_rate)
        except Exception as e:
            logger.warning(f"[WyzeMic:{self._room}] callback raised: {e}")
