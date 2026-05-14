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

    # Class-level lock around av.open(). libav's RTSP+codec init isn't
    # thread-safe across containers on Windows — three concurrent opens
    # (one per Wyze room) would race inside libav and segfault with a
    # 0xC0000005 access violation. Serializing only the open is enough;
    # demux/decode after open is fully concurrent.
    _open_lock = threading.Lock()

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
        # Live PyAV container reference. Owned by _run_blocking but also
        # reachable from stop() so we can force-close it from the asyncio
        # side; demux() blocks in native code on socket recv and won't
        # notice _stop_event for many seconds otherwise. Closing the
        # container forces av_read_frame to return immediately, which lets
        # the worker thread fall out of its for-loop and exit cleanly.
        # _close_lock arbitrates ownership: whichever of {stop(), worker
        # finally} grabs the lock first takes the local reference, nulls
        # the shared one, and is the sole closer. The other side gets None
        # and skips. Without this, both threads attempted close() on the
        # same libav context and segfaulted on freed memory (observed
        # 2026-05-09 during shutdown).
        self._container: Optional[Any] = None
        self._close_lock = threading.Lock()
        # Cap callback tasks queued onto the event loop. If the loop stalls
        # or a callback blocks, fresh RTSP packets should be dropped rather
        # than accumulating one asyncio Task per 80 ms audio chunk forever.
        self._callback_tasks_inflight: int = 0
        self._max_callback_tasks_inflight: int = 16

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
        # Atomically take ownership of the container under the close lock,
        # then call close() OUTSIDE the lock (close can take a moment on a
        # half-broken stream and we don't want to hold the lock that long
        # — the worker's finally is also trying to acquire it). If we
        # win the race, container is non-None and we close it; the worker's
        # finally will see None and skip. If the worker beat us here, we
        # see None and skip — the worker handles cleanup.
        container_to_close: Optional[Any] = None
        with self._close_lock:
            container_to_close = self._container
            self._container = None
        if container_to_close is not None:
            try:
                await asyncio.to_thread(container_to_close.close)
            except Exception:
                pass
        worker = self._worker
        self._worker = None
        if worker is not None and worker.is_alive():
            # Short timeout: the close above should make demux exit in
            # milliseconds. If we hit 2s here, something is genuinely stuck
            # and a full-process kill is the right escalation.
            await asyncio.to_thread(worker.join, 2.0)
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

        # Silence PyAV's libav stderr (PPS/decode_slice_header noise during
        # cold starts and reconnects). FATAL keeps real failures, drops
        # everything else. av.logging is a global config; calling once per
        # worker is idempotent. Accessed via getattr because PyAV's type
        # stubs don't export the logging submodule even though it exists
        # at runtime.
        try:
            av_logging = getattr(av, "logging", None)
            if av_logging is not None:
                av_logging.set_level(av_logging.FATAL)
        except Exception:
            pass

        container: Any = None
        try:
            try:
                with WyzeRtspMicSource._open_lock:
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
                self._container = container
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

            try:
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
            except OSError as e:
                # Expected when stop() force-closes the container from
                # outside — libav's read returns WSAENOTSOCK / EBADF and
                # demux raises. Don't let it bubble up to the thread's
                # default unhandled-exception printer (the noisy
                # multi-line traceback we'd otherwise see for every
                # shutdown). Any OSError outside a stop is genuinely
                # unexpected and worth a warning.
                if self._stop_event.is_set():
                    logger.debug(
                        f"[WyzeMic:{self._room}] demux exited on stop: {e}"
                    )
                else:
                    logger.warning(
                        f"[WyzeMic:{self._room}] demux raised OSError "
                        f"unexpectedly (no stop signaled): {e}"
                    )
        finally:
            # Atomically take ownership of close() under the same lock
            # stop() uses, then close OUTSIDE the lock. If stop() already
            # took the container, our local read returns None and we skip.
            # Without this gate (or with a naive double-close), both
            # threads would call libav's avformat_close_input on the same
            # context and segfault on freed memory — observed 2026-05-09.
            with self._close_lock:
                local_container = self._container
                self._container = None
            if local_container is not None:
                try:
                    local_container.close()
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
        if self._callback_tasks_inflight >= self._max_callback_tasks_inflight:
            return
        self._callback_tasks_inflight += 1
        try:
            task = asyncio.create_task(self._safe_invoke(cb, chunk))
            task.add_done_callback(lambda _t: self._callback_task_done())
        except RuntimeError:
            self._callback_task_done()
            pass  # loop closing

    def _callback_task_done(self) -> None:
        self._callback_tasks_inflight = max(0, self._callback_tasks_inflight - 1)

    async def _safe_invoke(self, cb: MicCallback, chunk: bytes) -> None:
        try:
            await cb(chunk, self._sample_rate)
        except Exception as e:
            logger.warning(f"[WyzeMic:{self._room}] callback raised: {e}")
