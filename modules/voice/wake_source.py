"""
JARVIS — Ambient Home AI
========================
Mission: Multi-room wake-word source abstraction.

         The default WakeWordDetector owns the PC mic via sounddevice. To add
         per-room wake (so wake-word can fire from any room with a microphone),
         we don't want to refactor the existing detector — instead we expose a
         WakeSource protocol that any audio producer can implement, plus a
         MicWakeRunner that runs openWakeWord against the producer's stream
         and publishes room-tagged 'voice.wake_detected' events.

         Cole's Wyze v2 cam adapter (in progress) and the laundry-room ESP
         node (when its mic firmware ships) both implement WakeSource by
         iterating their incoming audio stream as 1280-sample int16 numpy
         arrays. The orchestrator registers each source with WakeSourceManager
         on startup; each source gets its own OWW Model instance + listener
         task, with cooldowns scoped per-room so a wake in 'office' doesn't
         block a follow-up wake in 'kitchen'.

         Today this module ships with NO sources registered — it's the seam
         that the Wyze adapter and ESP firmware will plug into. The PC mic
         continues to flow through the original WakeWordDetector unchanged.

Modules: modules/voice/wake_source.py
Classes: WakeSource (Protocol), MicWakeRunner, WakeSourceManager
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator, Optional, Protocol, runtime_checkable

import numpy as np
from loguru import logger

from core.event_bus import EventBus

OWW_CHUNK_SIZE = 1280       # 80ms at 16kHz — required by openWakeWord
OWW_SAMPLE_RATE = 16000


@runtime_checkable
class WakeSource(Protocol):
    """A producer of 16kHz mono int16 audio chunks for one room.

    Implementations: LocalMicSource, WyzeMicSource (Cole's WIP), ESPNodeSource.
    """

    @property
    def room(self) -> str:
        """The room this source represents (e.g. 'living_room')."""
        ...

    def stream(self) -> AsyncIterator[np.ndarray]:
        """Yield audio chunks of shape (OWW_CHUNK_SIZE,) dtype int16, ~80ms each.

        Declared non-async on the Protocol because async-generator functions
        return an AsyncIterator immediately when called — callers use
        `async for chunk in source.stream():` without an extra await.
        Implementations should make this cancellation-safe — when the
        consumer task is cancelled, the source must release its underlying
        capture (close socket, stop ffmpeg, etc.) and exit cleanly.
        """
        ...


class MicWakeRunner:
    """Runs openWakeWord against a single WakeSource and publishes detections.

    One runner per room. Each owns its own OWW Model instance because the
    library isn't documented as thread-safe across simultaneous predicts;
    independent models also let us tune sensitivity per room later.
    """

    def __init__(
        self,
        config: dict,
        bus: EventBus,
        source: WakeSource,
    ) -> None:
        self._cfg = config["voice"]["wake_word"]
        self._bus = bus
        self._source = source
        self._model: Optional[Any] = None
        self._task: Optional[asyncio.Task] = None
        self._last_detection: float = 0.0
        self._last_score_emit: float = 0.0
        self._running: bool = False

    @property
    def room(self) -> str:
        return self._source.room

    def load(self) -> None:
        """Load a per-room OWW model. Blocking — call from a thread or during startup."""
        try:
            from openwakeword.model import Model
        except ImportError as e:
            raise RuntimeError(
                "openwakeword not installed — multi-room wake disabled"
            ) from e
        model_name = self._cfg.get("model", "hey_jarvis")
        self._model = Model(wakeword_models=[model_name], inference_framework="onnx")
        logger.info(
            f"[WakeSource] Loaded OWW model '{model_name}' for room '{self.room}'"
        )

    async def run(self) -> None:
        """Consume the source's stream, predict per chunk, publish wake events."""
        if self._model is None:
            await asyncio.to_thread(self.load)
        sensitivity = float(self._cfg.get("sensitivity", 0.5))
        cooldown = float(self._cfg.get("cooldown_seconds", 2))
        self._running = True
        try:
            async for chunk in self._source.stream():
                if not self._running:
                    break
                if chunk is None:
                    continue
                if chunk.dtype != np.int16:
                    chunk = chunk.astype(np.int16)
                if chunk.shape != (OWW_CHUNK_SIZE,):
                    # Source supplied wrong shape — skip rather than corrupt the model.
                    logger.debug(
                        f"[WakeSource:{self.room}] discarding chunk shape {chunk.shape}"
                    )
                    continue
                model = self._model
                if model is None:
                    break
                try:
                    predictions = await asyncio.to_thread(model.predict, chunk)
                except Exception as e:
                    logger.debug(f"[WakeSource:{self.room}] predict failed: {e}")
                    continue
                if isinstance(predictions, tuple):
                    predictions = predictions[0]
                if not isinstance(predictions, dict):
                    continue
                for model_name, score in predictions.items():
                    score_v = float(score)
                    now = time.monotonic()
                    if now - self._last_score_emit >= 0.5:
                        self._last_score_emit = now
                        await self._bus.publish(
                            "voice.wake_score",
                            {
                                "room": self.room,
                                "model": model_name,
                                "score": score_v,
                                "sensitivity": sensitivity,
                            },
                        )
                    if score_v < sensitivity:
                        continue
                    if (now - self._last_detection) < cooldown:
                        continue
                    self._last_detection = now
                    logger.info(
                        f"[WakeSource] Detected '{model_name}' (score={score_v:.3f}) "
                        f"in room '{self.room}'"
                    )
                    await self._bus.publish(
                        "voice.wake_detected",
                        {"room": self.room, "confidence": score_v, "model": model_name},
                    )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"[WakeSource:{self.room}] run loop error: {e}")
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()


class WakeSourceManager:
    """Registry of additional WakeSources beyond the default PC mic.

    Add sources before or after `start()`; sources added afterwards begin
    running immediately. `stop()` cancels all runners on shutdown.
    """

    def __init__(self, config: dict, bus: EventBus) -> None:
        self._config = config
        self._bus = bus
        self._runners: dict[str, MicWakeRunner] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._started: bool = False

    def register(self, source: WakeSource) -> None:
        """Add a source. Idempotent per room — second registration replaces."""
        room = source.room
        if room in self._runners:
            logger.info(f"[WakeSource] Replacing source for room '{room}'")
            self._runners[room].stop()
            old_task = self._tasks.pop(room, None)
            if old_task is not None and not old_task.done():
                old_task.cancel()
        runner = MicWakeRunner(self._config, self._bus, source)
        self._runners[room] = runner
        if self._started:
            self._tasks[room] = asyncio.create_task(runner.run())
            logger.info(f"[WakeSource] Started runner for room '{room}'")

    def start(self) -> None:
        """Spawn run-tasks for every registered source. Call once during init."""
        if self._started:
            return
        self._started = True
        for room, runner in self._runners.items():
            self._tasks[room] = asyncio.create_task(runner.run())
            logger.info(f"[WakeSource] Started runner for room '{room}'")

    async def stop(self) -> None:
        for runner in self._runners.values():
            runner.stop()
        for task in self._tasks.values():
            if not task.done():
                task.cancel()
        for task in self._tasks.values():
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        self._tasks.clear()
        self._started = False

    def registered_rooms(self) -> list[str]:
        return list(self._runners.keys())

    def get_source(self, room: str) -> Optional[Any]:
        """Return the WakeSource registered for a room, or None.

        Used by the orchestrator's wake-recording path to reach the adapter
        and install a recording tap when wake fires in a non-office room.
        Returns the underlying source object (typically a
        MicSourceWakeAdapter), so the caller can `isinstance`-check or
        getattr the tap-management methods on it.
        """
        runner = self._runners.get(room)
        if runner is None:
            return None
        return runner._source
