"""
JARVIS — Ambient Home AI
========================
Mission: USB / built-in speaker driver. Wraps sounddevice playback with the
         existing audio_focus duck so PC apps quiet down while Jarvis talks.

         The actual playback already lives in modules/voice/audio_utils —
         this is a thin SpeakerSink shim around it so per-room dispatch can
         pick "play to PC speaker" the same way it picks "play to Wyze SSH"
         or "play to ESP32 MQTT".

Modules: modules/voice/sources/usb_speaker.py
Classes: UsbSpeakerSink
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import sounddevice as sd
from loguru import logger

from modules.voice.audio_utils import play_audio_array_async
from modules.voice.sources.base import SpeakerSink


class UsbSpeakerSink(SpeakerSink):
    """Plays PCM through sounddevice on the host. Routes through
    play_audio_array_async so the existing audio-focus / ducking integration
    (pycaw on Windows) still applies.
    """

    def __init__(
        self,
        room: str,
        device_name: Optional[str] = None,
        device_index: Optional[int] = None,
        sample_rate_hz: int = 22050,
        channels: int = 1,
    ) -> None:
        self._room = room
        self._device = self._resolve_device(device_name, device_index)
        self._sample_rate = sample_rate_hz
        self._channels = channels

    @staticmethod
    def _resolve_device(name: Optional[str], index: Optional[int]) -> Optional[int]:
        if index is not None:
            return int(index)
        if not name or name.lower() == "default":
            return None
        try:
            devices = sd.query_devices()
        except Exception as e:
            logger.warning(f"[UsbSpeaker] query_devices failed: {e}")
            return None
        needle = name.lower()
        for i, info in enumerate(devices):
            # See usb_mic.py — sounddevice's query_devices() return type is a
            # union that includes str, even though the no-arg call always
            # gives a DeviceList of dicts.
            if not isinstance(info, dict):
                continue
            try:
                if int(info.get("max_output_channels", 0)) <= 0:
                    continue
                if needle in str(info.get("name", "")).lower():
                    return i
            except Exception:
                continue
        logger.warning(
            f"[UsbSpeaker] No output device matched '{name}' — using system default"
        )
        return None

    @property
    def room(self) -> str:
        return self._room

    async def play(self, pcm: bytes, sample_rate: int) -> None:
        if not pcm:
            return
        # Decode int16 PCM bytes to a numpy array. play_audio_array_async
        # handles ducking + the actual sd.play() blocking call in a thread.
        try:
            arr = np.frombuffer(pcm, dtype=np.int16)
        except Exception as e:
            logger.warning(f"[UsbSpeaker:{self._room}] PCM decode failed: {e}")
            return
        # Reshape if multi-channel was packed interleaved.
        if self._channels > 1 and arr.size % self._channels == 0:
            arr = arr.reshape(-1, self._channels)
        await play_audio_array_async(
            arr,
            sample_rate=sample_rate,
            device=self._device,
        )

    async def close(self) -> None:
        # play_audio_array_async opens/closes per call — no persistent
        # resource to release here.
        return
