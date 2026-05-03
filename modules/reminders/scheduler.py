"""
JARVIS — Ambient Home AI
========================
Mission: Background async loop that fires due reminders. Polls the store every
         N seconds, publishes a `reminder.due` event for each one that's ready,
         and marks them as fired so they don't repeat. The orchestrator
         subscribes to `reminder.due` and pipes the message through _speak().

Modules: modules/reminders/scheduler.py
Classes: ReminderScheduler
Functions:
    ReminderScheduler.__init__(config, store, event_bus) — Wire deps
    ReminderScheduler.run()                              — Async polling loop
"""

import asyncio
from datetime import datetime

from loguru import logger

from modules.reminders.store import RemindersStore


class ReminderScheduler:
    """
    Polls the reminders store and emits `reminder.due` events for fired ones.

    Config keys (from config["reminders"]):
        poll_interval_seconds: How often to check the DB. Default 30.
    """

    def __init__(self, config: dict, store: RemindersStore, event_bus) -> None:
        cfg = config.get("reminders", {}) if isinstance(config.get("reminders"), dict) else {}
        self._store = store
        self._bus = event_bus
        # 30s default trades latency for DB load. For a single-user system either
        # extreme is fine; this is mid-range.
        self._poll_seconds: float = float(cfg.get("poll_interval_seconds", 30))

    async def run(self) -> None:
        """Run forever — poll, emit, mark, sleep. Exits on cancellation."""
        logger.info(
            f"[Reminders] Scheduler started (poll every {self._poll_seconds:.0f}s)"
        )
        while True:
            try:
                due = await self._store.list_due(datetime.now())
                for reminder in due:
                    await self._bus.publish(
                        "reminder.due",
                        {
                            "id":           reminder["id"],
                            "message":      reminder["message"],
                            "trigger_time": reminder["trigger_time"],
                        },
                    )
                    # Mark fired before next poll so a slow speak path can't
                    # cause the reminder to be re-emitted.
                    await self._store.mark_fired(reminder["id"])
                    logger.info(
                        f"[Reminders] Fired #{reminder['id']}: {reminder['message']!r}"
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Reminders] Scheduler loop error: {e}")
            await asyncio.sleep(self._poll_seconds)
