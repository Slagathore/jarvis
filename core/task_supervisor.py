"""
JARVIS — Ambient Home AI
========================
Mission: Supervised fire-and-forget task management. TaskSupervisor is the
         policy-bearing successor to core.async_utils.TrackedTaskSet — it
         keeps everything TrackedTaskSet did (strong refs so tasks survive
         GC, exception logging, bounded shutdown) and adds a TaskPolicy so
         callers can express:

           - singleton_key   — at most one live task per key
           - cancel_previous — ...or the newest spawn cancels the prior one
           - timeout_s       — auto-cancel a task that runs too long

         This replaces the ad-hoc per-room singleton dicts the orchestrator
         grew (`_scene_tasks`, `_followup_tasks`) with one declarative path.

         TrackedTaskSet is intentionally left in place — appliance_tracker,
         world_model, and the dashboard still use it unchanged. Only callers
         that need policy migrate to TaskSupervisor.

Usage:
    self._bg = TaskSupervisor(label="Orchestrator")
    # plain fire-and-forget (TrackedTaskSet-compatible):
    self._bg.spawn(some_coro(), name="memory.extract")
    # at most one scene job per room, newest cycle skipped if one runs:
    self._bg.spawn(scene_coro(), name=f"scene:{room}",
                   policy=TaskPolicy(singleton_key=f"scene:{room}",
                                     timeout_s=20.0))
    # newest follow-up listener cancels the prior one for that room:
    self._bg.spawn(listen_coro(), name=f"followup:{room}",
                   policy=TaskPolicy(singleton_key=f"followup:{room}",
                                     cancel_previous=True))
    await self._bg.shutdown()
"""

import asyncio
from dataclasses import dataclass
from typing import Coroutine, Optional

from loguru import logger


@dataclass(frozen=True)
class TaskPolicy:
    """Declarative spawn policy. The default instance reproduces plain
    TrackedTaskSet behavior (untracked-key, no timeout)."""

    # When set, at most one live task may carry this key.
    singleton_key: Optional[str] = None
    # With a singleton_key: if a live task already holds the key, cancel
    # it and spawn the new one. When False, the new coroutine is dropped.
    cancel_previous: bool = False
    # Auto-cancel the task after this many seconds (asyncio.wait_for).
    timeout_s: Optional[float] = None


_DEFAULT_POLICY = TaskPolicy()


class TaskSupervisor:
    """Track + supervise fire-and-forget asyncio tasks.

    Not thread-safe — every operation must happen on the same event loop.
    """

    def __init__(self, label: str = "supervisor") -> None:
        self._label = label
        self._tasks: set[asyncio.Task] = set()
        # singleton_key → the live task currently holding it.
        self._by_key: dict[str, asyncio.Task] = {}
        # task → its singleton_key (for cleanup in the done-callback).
        self._task_keys: dict[asyncio.Task, str] = {}

    def spawn(
        self,
        coro: Coroutine,
        *,
        name: Optional[str] = None,
        policy: TaskPolicy = _DEFAULT_POLICY,
    ) -> Optional[asyncio.Task]:
        """Schedule `coro` under `policy`. Returns the task, or None when a
        singleton policy gated the spawn (a live task already holds the key
        and cancel_previous is False). A gated coroutine is closed so it
        does not leak an "un-awaited coroutine" warning."""
        key = policy.singleton_key
        if key is not None:
            existing = self._by_key.get(key)
            if existing is not None and not existing.done():
                if policy.cancel_previous:
                    existing.cancel()
                else:
                    coro.close()
                    logger.debug(
                        f"[{self._label}] singleton '{key}' busy — spawn skipped"
                    )
                    return None

        run: Coroutine = coro
        if policy.timeout_s is not None:
            run = asyncio.wait_for(coro, timeout=policy.timeout_s)

        task = asyncio.create_task(run, name=name)
        self._tasks.add(task)
        if key is not None:
            self._by_key[key] = task
            self._task_keys[task] = key
        task.add_done_callback(self._on_done)
        return task

    def _on_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        key = self._task_keys.pop(task, None)
        if key is not None and self._by_key.get(key) is task:
            del self._by_key[key]
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None or isinstance(exc, asyncio.TimeoutError):
            # A timeout is a deliberate policy outcome, not a crash.
            return
        logger.opt(exception=exc).warning(
            f"[{self._label}] background task {task.get_name()!r} raised"
        )

    async def shutdown(self, timeout: float = 5.0) -> None:
        """Cancel every outstanding task and wait, bounded by `timeout`."""
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

    def status(self) -> list[dict]:
        """Snapshot of live tasks — for a dashboard observability panel."""
        return [
            {
                "name": t.get_name(),
                "key": self._task_keys.get(t),
                "done": t.done(),
            }
            for t in self._tasks
        ]

    def __len__(self) -> int:
        return len(self._tasks)
