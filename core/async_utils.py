"""
JARVIS — Ambient Home AI
========================
Mission: Small async helpers shared across modules. The first occupant is
         TrackedTaskSet — a thin wrapper that solves the three real problems
         with `asyncio.create_task(...)` without storing the return value:
           1. asyncio only holds a weak reference to a running task, so an
              un-stored task can be GC'd before it finishes.
           2. Exceptions in fire-and-forget tasks land in "Task exception
              was never retrieved" warnings instead of the project logger,
              which makes them invisible.
           3. On shutdown the orchestrator has no way to cancel/await them,
              so background work bleeds past the lifetime of the modules
              it depends on.

         The pattern was already in use in modules/world_model/interactions.py
         (`self._tasks.append(task); task.add_done_callback(self._tasks.remove)`).
         This file makes it available everywhere with one import.

Usage:
    self._tracked = TrackedTaskSet(label="Orchestrator")
    self._tracked.spawn(some_coro(), name="memory.extract")
    # on shutdown:
    await self._tracked.shutdown()
"""

import asyncio
from typing import Coroutine, Optional

from loguru import logger


class TrackedTaskSet:
    """Track fire-and-forget asyncio tasks so they survive GC, log their
    exceptions, and can be cancelled on shutdown.

    Not thread-safe — every operation must happen on the same event loop.
    """

    def __init__(self, label: str = "tracked") -> None:
        self._tasks: set[asyncio.Task] = set()
        self._label = label

    def spawn(
        self,
        coro: Coroutine,
        *,
        name: Optional[str] = None,
    ) -> asyncio.Task:
        """Schedule `coro` and track the resulting task.

        Exceptions are logged via the project logger; cancellation is
        treated as expected and not logged.
        """
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._on_done)
        return task

    def _on_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.opt(exception=exc).warning(
                f"[{self._label}] background task "
                f"{task.get_name()!r} raised"
            )

    async def shutdown(self, timeout: float = 5.0) -> None:
        """Cancel every outstanding task and wait for them to settle.

        Bounded by `timeout` — if a task ignores cancellation we give up
        and log; we don't want shutdown to hang forever on one stuck
        background job.
        """
        if not self._tasks:
            return
        tasks = list(self._tasks)
        for t in tasks:
            t.cancel()
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            stuck = sum(1 for t in tasks if not t.done())
            logger.warning(
                f"[{self._label}] {stuck} task(s) did not finish within "
                f"{timeout:.1f}s shutdown timeout"
            )

    def __len__(self) -> int:
        return len(self._tasks)
