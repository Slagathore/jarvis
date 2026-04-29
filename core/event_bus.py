"""
JARVIS — Ambient Home AI
========================
Mission: Provide an async publish/subscribe event bus that decouples all modules
         from each other. Modules communicate exclusively through events — no
         module imports another module directly (they only import from core/).

         This makes it trivial to add, remove, or mock any subsystem without
         touching the rest of the codebase.

Architecture:
    - Producers call: await bus.publish("topic.name", {"key": "value"})
    - Consumers call: bus.subscribe("topic.name", async_handler_func)
    - The run() coroutine processes events sequentially from an internal queue,
      dispatching each to all subscribers concurrently.

Event topics used in this project:
    voice.wake_detected       — Wake word fired: {room, confidence}
    voice.transcript          — Audio transcribed: {room, text, duration_s}
    voice.speech_start        — TTS started: {room, text}
    voice.speech_end          — TTS finished: {room}
    context.state_changed     — Activity state updated: {state: ActivityState}
    context.interruptibility  — Score changed: {score, activity, reason}
    appliance.state_changed   — Appliance event: {appliance, status, runtime_minutes}
    vision.frame_processed    — Camera analysis done: {room, lights_on, person_present, description}
    vision.posture            — Posture detected: {room, posture}
    node.status               — ESP32 node online/offline: {room, online, ip}
    node.audio_received       — Raw audio from node: {room, audio_bytes}

Modules: core/event_bus.py
Classes: EventBus
Variables: EventHandler (type alias)

#todo: Add wildcard subscriptions (e.g., subscribe to "voice.*")
#todo: Add priority queue so critical events (wake word) jump the queue
#todo: Add event replay / history for dashboard debugging
#todo: Add per-topic message rate limiting to prevent runaway producers
#todo: Add dead-letter queue for events with no subscribers
"""

import asyncio
from collections import defaultdict
from typing import Any, Callable, Coroutine

from loguru import logger

# Type alias: any async function that accepts a dict payload
EventHandler = Callable[[dict], Coroutine[Any, Any, None]]


class EventBus:
    """
    Central async publish/subscribe event bus.

    Thread-safety: All operations must run in the same event loop.
    The queue is unbounded — producers will never block.
    Errors in individual handlers are logged but do not crash the bus.

    Usage:
        bus = EventBus()
        bus.subscribe("voice.wake_detected", my_handler)
        asyncio.create_task(bus.run())          # Start the dispatch loop
        await bus.publish("voice.wake_detected", {"room": "office"})
    """

    def __init__(self) -> None:
        # topic string → list of registered async handlers
        self._subscribers: dict[str, list[EventHandler]] = defaultdict(list)
        # Unbounded FIFO queue of (topic, payload) tuples
        self._queue: asyncio.Queue[tuple[str, dict]] = asyncio.Queue()
        self._running: bool = False

    # ── Public API ───────────────────────────────────────────────────────────

    def subscribe(self, topic: str, handler: EventHandler) -> None:
        """
        Register an async handler for the given topic.
        Multiple handlers per topic are all called concurrently.
        Calling this after run() has started is safe — changes take effect immediately.
        """
        self._subscribers[topic].append(handler)
        logger.debug(f"[EventBus] '{topic}' ← {handler.__qualname__}")

    async def publish(self, topic: str, payload: dict) -> None:
        """
        Enqueue an event for dispatch. Returns immediately — does not wait
        for handlers to execute. Safe to call from any coroutine.
        """
        await self._queue.put((topic, payload))

    async def run(self) -> None:
        """
        Main dispatch loop. Processes events one at a time from the queue.
        Each event's handlers are all dispatched concurrently (gathered).
        This coroutine runs forever — wrap in an asyncio.Task.
        """
        self._running = True
        logger.debug("[EventBus] Dispatch loop started")

        while self._running:
            try:
                # Use timeout so we can respond to _running=False promptly
                topic, payload = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            handlers = self._subscribers.get(topic, [])
            if not handlers:
                logger.debug(f"[EventBus] No subscribers for '{topic}' — event dropped")
                self._queue.task_done()
                continue

            # Dispatch all handlers for this event concurrently
            tasks = [
                asyncio.create_task(
                    self._safe_call(h, topic, payload),
                    name=f"event:{topic}:{h.__qualname__}",
                )
                for h in handlers
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
            self._queue.task_done()

        logger.debug("[EventBus] Dispatch loop stopped")

    async def stop(self) -> None:
        """Signal the run loop to exit cleanly after draining remaining events."""
        # Drain the queue first so no events are lost
        await self._queue.join()
        self._running = False

    # ── Internal ─────────────────────────────────────────────────────────────

    async def _safe_call(self, handler: EventHandler, topic: str, payload: dict) -> None:
        """
        Call a single handler, catching and logging any exception.
        One handler crashing does not affect other handlers for the same event.
        """
        try:
            await handler(payload)
        except asyncio.CancelledError:
            raise  # Propagate cancellation — don't swallow it
        except Exception:
            logger.exception(
                f"[EventBus] Unhandled exception in handler '{handler.__qualname__}' "
                f"for topic '{topic}'"
            )

    @property
    def queue_depth(self) -> int:
        """Current number of unprocessed events. Useful for health monitoring."""
        return self._queue.qsize()

    @property
    def subscriber_count(self) -> int:
        """Total number of registered handler functions across all topics."""
        return sum(len(v) for v in self._subscribers.values())
