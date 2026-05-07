"""
JARVIS — Ambient Home AI
========================
Mission: Null implementations of MicSource and SpeakerSink. Used when a
         room declares `mic.type: none` or `speaker.type: none`. Lets the
         dispatcher always return an instance — callers never check for None.

Modules: modules/voice/sources/null_audio.py
Classes: NullMicSource, NullSpeakerSink
"""

from __future__ import annotations

from loguru import logger

from modules.voice.sources.base import MicCallback, MicSource, SpeakerSink


class NullMicSource(MicSource):
    """No-op mic. start() returns immediately and never invokes the callback."""

    def __init__(self, room: str) -> None:
        self._room = room

    async def start(self, callback: MicCallback) -> None:
        logger.debug(f"[NullMic:{self._room}] Mic disabled — start() is a no-op")

    async def stop(self) -> None:
        return

    @property
    def room(self) -> str:
        return self._room


class NullSpeakerSink(SpeakerSink):
    """No-op speaker. play() drops the buffer on the floor."""

    def __init__(self, room: str) -> None:
        self._room = room

    async def play(self, pcm: bytes, sample_rate: int) -> None:
        logger.debug(
            f"[NullSpeaker:{self._room}] Drop {len(pcm)} bytes @ {sample_rate}Hz "
            "(speaker disabled in config)"
        )

    async def close(self) -> None:
        return

    @property
    def room(self) -> str:
        return self._room
