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
import heapq
import itertools
import time
from collections import defaultdict
from typing import Any, Callable, Coroutine, Optional

from loguru import logger

# Type alias: any async function that accepts a dict payload
EventHandler = Callable[[dict], Coroutine[Any, Any, None]]


class EventPriority:
    """Semantic event priorities. Lower numbers dispatch first."""

    CRITICAL = 0
    VOICE = 10
    CONTROL = 20
    WORLD = 50
    TELEMETRY = 80
    DEBUG = 90


_DEFAULT_TOPIC_PRIORITIES: dict[str, int] = {
    "safety.": EventPriority.CRITICAL,
    "alarm.": EventPriority.CRITICAL,
    "fire.": EventPriority.CRITICAL,
    "cat_escape.": EventPriority.CRITICAL,
    "voice.wake_detected": EventPriority.VOICE,
    "voice.": EventPriority.VOICE,
    "reminder.due": EventPriority.CONTROL,
    "node.status": EventPriority.CONTROL,
    "camera.health": EventPriority.CONTROL,
    "world.": EventPriority.WORLD,
    "vision.observation": EventPriority.WORLD,
    "audio.level": EventPriority.TELEMETRY,
    "telemetry.": EventPriority.TELEMETRY,
    "debug.": EventPriority.DEBUG,
}

_DEFAULT_TOPIC_RATE_LIMITS: dict[str, tuple[float, int]] = {
    # High-frequency dashboard meters. Fresh samples replace stale samples
    # visually, so dropping excess is better than making wake/safety wait.
    "audio.level": (12.0, 24),
    "telemetry.": (20.0, 40),
    "debug.": (5.0, 10),
    # World snapshots can be large; entity-level events still flow normally.
    "world.state_snapshot": (0.5, 2),
}


class _SubscriptionHandle:
    """Handle returned by subscribe(). Two shapes for two callers:

    1. Old call sites that ignore the return value still work — the
       handler is already registered before the handle is returned.
    2. `await bus.subscribe(...)` still works — the handle is an
       already-completed awaitable.
    3. New: callers that need to unregister (plugins, selfedit reloads,
       per-request handlers) can hold the handle and call `.unsubscribe()`
       later.
    """

    def __init__(
        self,
        bus: Optional["EventBus"] = None,
        topic: Optional[str] = None,
        handler: Optional["EventHandler"] = None,
    ) -> None:
        self._bus = bus
        self._topic = topic
        self._handler = handler
        self._active = bus is not None

    def __await__(self):
        if False:
            yield None
        return None

    def unsubscribe(self) -> bool:
        """Idempotent removal. Returns True iff the handler was still
        registered when the call arrived."""
        if not self._active or self._bus is None or self._topic is None \
                or self._handler is None:
            return False
        self._active = False
        return self._bus._unsubscribe(self._topic, self._handler)


class EventBus:
    """
    Central async publish/subscribe event bus.

    Thread-safety: All operations must run in the same event loop.
    The queue is bounded — sustained overload backpressures producers
    instead of growing memory without limit.
    Errors in individual handlers are logged but do not crash the bus.

    Usage:
        bus = EventBus()
        bus.subscribe("voice.wake_detected", my_handler)
        asyncio.create_task(bus.run())          # Start the dispatch loop
        await bus.publish("voice.wake_detected", {"room": "office"})
    """

    def __init__(
        self,
        max_queue_size: int = 1000,
        *,
        topic_priorities: Optional[dict[str, int]] = None,
        topic_rate_limits: Optional[dict[str, tuple[float, int]]] = None,
    ) -> None:
        # topic string → list of registered async handlers
        self._subscribers: dict[str, list[EventHandler]] = defaultdict(list)
        # Bounded semantic priority queue of (priority, sequence, topic,
        # payload) tuples. Critical wake/safety events dispatch before
        # telemetry bursts while preserving FIFO order inside each priority.
        self._queue: asyncio.PriorityQueue[tuple[int, int, str, dict]] = asyncio.PriorityQueue(
            maxsize=max(1, int(max_queue_size))
        )
        self._sequence = itertools.count()
        self._topic_priorities = dict(_DEFAULT_TOPIC_PRIORITIES)
        if topic_priorities:
            self._topic_priorities.update(topic_priorities)
        self._topic_rate_limits = dict(_DEFAULT_TOPIC_RATE_LIMITS)
        if topic_rate_limits:
            self._topic_rate_limits.update(topic_rate_limits)
        self._rate_buckets: dict[str, dict[str, float]] = {}
        self._dropped_by_topic: dict[str, int] = defaultdict(int)
        self._running: bool = False

    # ── Public API ───────────────────────────────────────────────────────────

    def subscribe(self, topic: str, handler: EventHandler) -> _SubscriptionHandle:
        """
        Register an async handler for the given topic.
        Multiple handlers per topic are all called concurrently.
        Calling this after run() has started is safe — changes take effect immediately.

        Returns a SubscriptionHandle. Callers that need to remove the
        handler later (plugins, selfedit reloads, dashboard hot-reload)
        can hold the handle and call `.unsubscribe()` — older callers
        that ignore the return value or `await` it still work unchanged.
        """
        self._subscribers[topic].append(handler)
        logger.debug(f"[EventBus] '{topic}' ← {handler.__qualname__}")
        return _SubscriptionHandle(bus=self, topic=topic, handler=handler)

    def unsubscribe(self, topic: str, handler: EventHandler) -> bool:
        """Remove a previously-registered handler. Idempotent — returns
        True if the handler was registered and got removed, False if
        either the topic or the handler had no matching entry."""
        return self._unsubscribe(topic, handler)

    def _unsubscribe(self, topic: str, handler: EventHandler) -> bool:
        handlers = self._subscribers.get(topic)
        if not handlers:
            return False
        try:
            handlers.remove(handler)
        except ValueError:
            return False
        if not handlers:
            # Drop the empty list so subscriber_count / dispatch lookup
            # don't see a phantom topic.
            del self._subscribers[topic]
        logger.debug(f"[EventBus] '{topic}' ✗ {handler.__qualname__}")
        return True

    async def publish(
        self,
        topic: str,
        payload: dict,
        *,
        priority: Optional[int] = None,
    ) -> None:
        """
        Enqueue an event for dispatch. Returns immediately — does not wait
        for handlers to execute. Safe to call from any coroutine.
        """
        if not self._allow_by_rate_limit(topic):
            self._dropped_by_topic[topic] += 1
            logger.debug(f"[EventBus] Rate-limited '{topic}' — event dropped")
            return
        event_priority = int(priority) if priority is not None else self._priority_for(topic)
        await self._put_event(event_priority, topic, payload)

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
                _, _, topic, payload = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
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

    def _priority_for(self, topic: str) -> int:
        best_len = -1
        best_priority = EventPriority.WORLD
        for pattern, priority in self._topic_priorities.items():
            if topic == pattern or topic.startswith(pattern):
                if len(pattern) > best_len:
                    best_len = len(pattern)
                    best_priority = int(priority)
        return best_priority

    def _rate_limit_for(self, topic: str) -> Optional[tuple[str, float, int]]:
        best_pattern = ""
        best_limit: Optional[tuple[float, int]] = None
        for pattern, limit in self._topic_rate_limits.items():
            if topic == pattern or topic.startswith(pattern):
                if len(pattern) > len(best_pattern):
                    best_pattern = pattern
                    best_limit = limit
        if best_limit is None:
            return None
        rate, burst = best_limit
        return best_pattern, max(0.0, float(rate)), max(1, int(burst))

    def _allow_by_rate_limit(self, topic: str) -> bool:
        limit = self._rate_limit_for(topic)
        if limit is None:
            return True
        bucket_key, rate, burst = limit
        if rate <= 0.0:
            return False
        now = time.monotonic()
        bucket = self._rate_buckets.get(bucket_key)
        if bucket is None:
            self._rate_buckets[bucket_key] = {"tokens": float(burst - 1), "ts": now}
            return True
        elapsed = max(0.0, now - bucket["ts"])
        bucket["tokens"] = min(float(burst), bucket["tokens"] + elapsed * rate)
        bucket["ts"] = now
        if bucket["tokens"] >= 1.0:
            bucket["tokens"] -= 1.0
            return True
        return False

    async def _put_event(self, priority: int, topic: str, payload: dict) -> None:
        item = (priority, next(self._sequence), topic, payload)
        try:
            self._queue.put_nowait(item)
            return
        except asyncio.QueueFull:
            pass

        # If telemetry filled the bounded queue, a critical/voice/control
        # event should evict the worst queued low-priority event instead of
        # waiting behind it. This uses asyncio.PriorityQueue internals, but
        # keeps the public API stable and preserves bounded memory.
        if priority <= EventPriority.CONTROL:
            raw_queue = self._queue._queue  # type: ignore[attr-defined]
            if raw_queue:
                worst_idx, worst_item = max(
                    enumerate(raw_queue),
                    key=lambda pair: (pair[1][0], pair[1][1]),
                )
                if worst_item[0] > priority:
                    dropped = raw_queue[worst_idx]
                    del raw_queue[worst_idx]
                    heapq.heapify(raw_queue)
                    self._queue.task_done()
                    self._dropped_by_topic[dropped[2]] += 1
                    self._queue.put_nowait(item)
                    logger.warning(
                        f"[EventBus] Queue full; dropped lower-priority "
                        f"'{dropped[2]}' for '{topic}'"
                    )
                    return

        await self._queue.put(item)

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

    @property
    def dropped_counts(self) -> dict[str, int]:
        """Events dropped by rate limit or priority eviction, keyed by topic."""
        return dict(self._dropped_by_topic)
