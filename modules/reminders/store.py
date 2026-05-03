"""
JARVIS — Ambient Home AI
========================
Mission: Async CRUD over the existing `reminders` SQLite table. Owns no business
         logic — just persistence. The scheduler polls list_due(), the dashboard
         calls list_pending(), the voice parser calls add().

Modules: modules/reminders/store.py
Classes: RemindersStore
Functions:
    RemindersStore.__init__(db)               — Wrap a DatabaseManager
    RemindersStore.add(message, trigger_time) — Insert pending reminder, return id
    RemindersStore.list_pending()             — All not-yet-fired reminders
    RemindersStore.list_due(now)              — Reminders whose trigger_time <= now
    RemindersStore.mark_fired(reminder_id)    — Set last_triggered = now
    RemindersStore.delete(reminder_id)        — Remove a reminder
"""

from datetime import datetime, timedelta
from typing import Any, Optional

from loguru import logger

from modules.memory.database import DatabaseManager


class RemindersStore:
    """Thin async CRUD layer over the `reminders` table."""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    async def add(
        self,
        message: str,
        trigger_time: datetime,
        recurrence_seconds: Optional[int] = None,
    ) -> int:
        """
        Insert a pending reminder. Returns the new row's id.

        Args:
            message:            What to remind about.
            trigger_time:       First fire time.
            recurrence_seconds: If set, the reminder re-arms `recurrence_seconds`
                                after each fire instead of being marked done.
                                E.g., 86400 = daily, 3600 = hourly.
        """
        rid = await self._db.execute(
            "INSERT INTO reminders (message, trigger_time, recurring, recurrence_seconds) "
            "VALUES (?, ?, ?, ?)",
            (
                message,
                trigger_time.isoformat(),
                1 if recurrence_seconds else 0,
                recurrence_seconds,
            ),
        )
        recurring_str = (
            f" (recurring every {recurrence_seconds}s)" if recurrence_seconds else ""
        )
        logger.info(
            f"[Reminders] Added #{rid}: {message!r} for {trigger_time.isoformat()}{recurring_str}"
        )
        return rid

    async def list_pending(self) -> list[dict[str, Any]]:
        """
        All reminders that have not yet fired (one-shots) plus all recurring
        reminders. Recurring ones stay in the list forever — last_triggered
        gets updated each fire but the row remains.
        """
        rows = await self._db.fetchall(
            "SELECT id, message, trigger_time, recurring, recurrence_seconds, last_triggered "
            "FROM reminders "
            "WHERE last_triggered IS NULL OR recurrence_seconds IS NOT NULL "
            "ORDER BY trigger_time ASC"
        )
        return [dict(r) for r in rows]

    async def list_due(self, now: datetime) -> list[dict[str, Any]]:
        """
        Reminders whose trigger_time <= now and that are still active.
        For one-shots: last_triggered must be NULL.
        For recurring: trigger_time itself advances after each fire (in
        mark_fired) so we can use the same comparison — we don't gate on
        last_triggered for recurring reminders.
        """
        rows = await self._db.fetchall(
            "SELECT id, message, trigger_time, recurring, recurrence_seconds, last_triggered "
            "FROM reminders "
            "WHERE trigger_time <= ? "
            "  AND (last_triggered IS NULL OR recurrence_seconds IS NOT NULL) "
            "ORDER BY trigger_time ASC",
            (now.isoformat(),),
        )
        return [dict(r) for r in rows]

    async def mark_fired(self, reminder_id: int) -> None:
        """
        Record a fire. For one-shots, set last_triggered so they don't re-fire.
        For recurring reminders, also advance trigger_time forward by
        recurrence_seconds so the next due-check picks them up at the right time.
        """
        row = await self._db.fetchone(
            "SELECT trigger_time, recurrence_seconds FROM reminders WHERE id = ?",
            (reminder_id,),
        )
        now_iso = datetime.now().isoformat()
        if not row or not row["recurrence_seconds"]:
            await self._db.execute(
                "UPDATE reminders SET last_triggered = ? WHERE id = ?",
                (now_iso, reminder_id),
            )
            return

        # Recurring: advance trigger_time. If we missed multiple intervals
        # (system was off, etc.), jump to the next future tick rather than
        # firing N times catching up.
        try:
            current = datetime.fromisoformat(row["trigger_time"])
        except (TypeError, ValueError):
            current = datetime.now()
        seconds = int(row["recurrence_seconds"])
        next_fire = current + timedelta(seconds=seconds)
        now = datetime.now()
        while next_fire <= now:
            next_fire += timedelta(seconds=seconds)
        await self._db.execute(
            "UPDATE reminders SET trigger_time = ?, last_triggered = ? WHERE id = ?",
            (next_fire.isoformat(), now_iso, reminder_id),
        )

    async def delete(self, reminder_id: int) -> None:
        """Hard-delete a reminder (used by dashboard dismiss)."""
        await self._db.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
        logger.info(f"[Reminders] Deleted #{reminder_id}")
