"""
JARVIS — Ambient Home AI
========================
Mission: USB / built-in microphone driver. Wraps sounddevice.RawInputStream
         in a long-lived background thread that pushes int16 PCM chunks to
         the manager's async callback via a thread-safe bridge.

         Each chunk is sized for openWakeWord (1280 samples = 80ms at 16kHz)
         so the mic-side and the wake-word side speak the same frame size
         without intermediate re-buffering.

         device_name does substring matching against sd.query_devices() —
         "default", "USB Microphone", "Realtek" all work. device_index is
         exact (use it when two devices share a name).

Modules: modules/voice/sources/usb_mic.py
Classes: UsbMicSource

#todo: Add per-device gain control once we have noisy-room calibration data
#todo: Surface stream overflow events to the dashboard so a misbehaving USB
       hub becomes visible without grepping logs
"""

from __future__ import annotations

import asyncio
import threading
from typing import Optional

import numpy as np
import sounddevice as sd
from loguru import logger

from modules.voice.sources.base import MicCallback, MicSource

OWW_CHUNK_SIZE = 1280  # 80ms at 16kHz — matches openWakeWord's frame size


class UsbMicSource(MicSource):
    """sounddevice RawInputStream → async callback bridge.

    The audio thread runs sounddevice's blocking RawInputStream and pushes
    each chunk to a thread-safe queue. An async pump task drains the queue
    and awaits the user's callback. Splitting it like this lets the audio
    thread keep pace with hardware regardless of how slow the callback is —
    if the callback falls behind, we drop chunks and log instead of letting
    the kernel buffer overrun.
    """

    def __init__(
        self,
        room: str,
        device_name: Optional[str] = None,
        device_index: Optional[int] = None,
        sample_rate_hz: int = 16000,
        channels: int = 1,
    ) -> None:
        self._room = room
        self._device = self._resolve_device(device_name, device_index)
        self._sample_rate = sample_rate_hz
        self._channels = channels
        self._stream: Optional[sd.RawInputStream] = None
        self._stop_event = threading.Event()
        # Bounded queue — if the callback can't keep up, drop oldest chunks
        # rather than growing memory unboundedly. Wake-word is far cheaper
        # than the chunks-per-second rate, so this only fires under bug or
        # GIL-contention.
        self._queue: "asyncio.Queue[bytes]" = asyncio.Queue(maxsize=64)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._pump_task: Optional[asyncio.Task] = None

    @staticmethod
    def _resolve_device(name: Optional[str], index: Optional[int]) -> Optional[int]:
        """Resolve device_name (substring) or device_index to a sounddevice
        device ID. Returns None for the system default — sounddevice treats
        None as "use the host's default input device".
        """
        if index is not None:
            return int(index)
        if not name or name.lower() == "default":
            return None
        try:
            devices = sd.query_devices()
        except Exception as e:
            logger.warning(f"[UsbMic] query_devices failed: {e}")
            return None
        needle = name.lower()
        for i, info in enumerate(devices):
            # sd.query_devices() with no args returns a DeviceList of dicts,
            # but its typeshed union also covers single-device lookups that
            # return a str — narrow before .get() so Pylance is happy.
            if not isinstance(info, dict):
                continue
            try:
                # Only consider devices with input channels — sounddevice
                # also enumerates outputs in the same list.
                if int(info.get("max_input_channels", 0)) <= 0:
                    continue
                if needle in str(info.get("name", "")).lower():
                    return i
            except Exception:
                continue
        logger.warning(
            f"[UsbMic] No input device matched '{name}' — falling back to system default"
        )
        return None

    @property
    def room(self) -> str:
        return self._room

    async def start(self, callback: MicCallback) -> None:
        """Open the input stream and start the async pump task."""
        if self._stream is not None:
            logger.warning(f"[UsbMic:{self._room}] start() called twice — ignoring second call")
            return
        self._loop = asyncio.get_running_loop()
        self._stop_event.clear()

        def _audio_cb(indata, frames, time_info, status):  # type: ignore[no-untyped-def]
            if status:
                logger.debug(f"[UsbMic:{self._room}] sounddevice status: {status}")
            # indata is a CFFI buffer; convert to bytes once.
            chunk = bytes(indata)
            try:
                # call_soon_threadsafe so we don't touch the asyncio loop
                # from the audio thread directly.
                if self._loop is not None and not self._loop.is_closed():
                    self._loop.call_soon_threadsafe(self._enqueue_nowait, chunk)
            except RuntimeError:
                pass  # loop closed mid-shutdown

        try:
            self._stream = sd.RawInputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                dtype="int16",
                blocksize=OWW_CHUNK_SIZE,
                device=self._device,
                callback=_audio_cb,
            )
            self._stream.start()
        except Exception as e:
            logger.warning(
                f"[UsbMic:{self._room}] Could not open device {self._device}: {e}"
            )
            self._stream = None
            return

        self._pump_task = asyncio.create_task(self._pump(callback))
        logger.info(
            f"[UsbMic:{self._room}] Capturing from device {self._device} "
            f"@ {self._sample_rate}Hz / {self._channels}ch"
        )

    def _enqueue_nowait(self, chunk: bytes) -> None:
        """Try to enqueue without blocking. If the queue is full, drop the
        oldest chunk and put the new one in — fresh audio matters more than
        backlog.
        """
        try:
            self._queue.put_nowait(chunk)
        except asyncio.QueueFull:
            try:
                _ = self._queue.get_nowait()
                self._queue.put_nowait(chunk)
            except Exception:
                pass

    async def _pump(self, callback: MicCallback) -> None:
        """Drain the queue and await the callback for each chunk. Cancels
        cleanly when stop() sets the event.
        """
        try:
            while not self._stop_event.is_set():
                try:
                    chunk = await asyncio.wait_for(self._queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                try:
                    await callback(chunk, self._sample_rate)
                except Exception as e:
                    logger.warning(f"[UsbMic:{self._room}] callback raised: {e}")
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        self._stop_event.set()
        if self._pump_task is not None:
            self._pump_task.cancel()
            try:
                await self._pump_task
            except asyncio.CancelledError:
                pass
            self._pump_task = None
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.debug(f"[UsbMic:{self._room}] stream close error: {e}")
            self._stream = None
        # Drain any leftover queued chunks so they're GC'd promptly
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        logger.info(f"[UsbMic:{self._room}] Stopped")
