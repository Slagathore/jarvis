"""
JARVIS — Ambient Home AI
========================
Mission: Define the abstract base classes for per-room audio I/O. Two
         independent surfaces — MicSource produces PCM bytes; SpeakerSink
         consumes PCM bytes. Each room's config.yaml picks one of each via
         a `type:` toggle, and the corresponding manager (MicManager /
         SpeakerManager) instantiates the right concrete subclass.

         The contract is intentionally small. Mic drivers run forever and
         push chunks to a callback; speaker drivers play one buffer per call
         and return when playback completes. Anything more complex (volume
         ramps, format negotiation, parallel mixing) belongs in higher layers.

Modules: modules/voice/sources/base.py
Classes: MicSource, SpeakerSink
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Awaitable, Callable

# An async callback the mic driver invokes per chunk: (pcm_int16_bytes, sample_rate).
# Receivers (wake-word detector, STT) can immediately push the bytes onto a
# queue and return; the mic loop will keep draining without blocking on
# downstream processing.
MicCallback = Callable[[bytes, int], Awaitable[None]]


class MicSource(ABC):
    """One mic driver per room. Lifecycle: start() → callback*N → stop()."""

    @abstractmethod
    async def start(self, callback: MicCallback) -> None:
        """Begin streaming. The driver is expected to keep producing chunks
        until stop() is called. Each chunk should be int16 PCM, ideally
        ~80ms (1280 samples at 16 kHz) so it slots straight into
        openWakeWord's frame size. Larger chunks are accepted but waste
        latency.
        """
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop streaming and release resources. Idempotent — a second call
        after the source is already stopped is a no-op.
        """
        ...

    @property
    @abstractmethod
    def room(self) -> str:
        """The room ID this source publishes for — used by the manager to tag
        events on the event bus."""
        ...


class SpeakerSink(ABC):
    """One speaker driver per room. play() is a single-buffer call that must
    block until playback completes — barge-in handling happens at a higher
    layer (the orchestrator already coordinates "stop talking when wake
    fires") and shouldn't be re-implemented per driver.
    """

    @abstractmethod
    async def play(self, pcm: bytes, sample_rate: int) -> None:
        """Play the given int16 PCM buffer at the given rate. Must complete
        before returning — caller awaits and assumes playback is done when
        the await resolves.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release any persistent resources (SSH session, audio device handle).
        Idempotent — called once at orchestrator shutdown.
        """
        ...

    @property
    @abstractmethod
    def room(self) -> str:
        ...
