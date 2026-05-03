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

from datetime import datetime
from typing import Any

from loguru import logger

from modules.memory.database import DatabaseManager


class RemindersStore:
    """Thin async CRUD layer over the `reminders` table."""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    async def add(self, message: str, trigger_time: datetime) -> int:
        """Insert a pending reminder. Returns the new row's id."""
        rid = await self._db.execute(
            "INSERT INTO reminders (message, trigger_time, recurring) VALUES (?, ?, 0)",
            (message, trigger_time.isoformat()),
        )
        logger.info(
            f"[Reminders] Added #{rid}: {message!r} for {trigger_time.isoformat()}"
        )
        return rid

    async def list_pending(self) -> list[dict[str, Any]]:
        """All reminders that have not yet fired (last_triggered IS NULL)."""
        rows = await self._db.fetchall(
            "SELECT id, message, trigger_time, recurring, last_triggered "
            "FROM reminders WHERE last_triggered IS NULL "
            "ORDER BY trigger_time ASC"
        )
        return [dict(r) for r in rows]

    async def list_due(self, now: datetime) -> list[dict[str, Any]]:
        """Reminders that should have fired by `now` and haven't yet."""
        rows = await self._db.fetchall(
            "SELECT id, message, trigger_time, recurring, last_triggered "
            "FROM reminders WHERE last_triggered IS NULL AND trigger_time <= ? "
            "ORDER BY trigger_time ASC",
            (now.isoformat(),),
        )
        return [dict(r) for r in rows]

    async def mark_fired(self, reminder_id: int) -> None:
        """Mark a reminder as fired so it doesn't fire again next poll."""
        await self._db.execute(
            "UPDATE reminders SET last_triggered = ? WHERE id = ?",
            (datetime.now().isoformat(), reminder_id),
        )

    async def delete(self, reminder_id: int) -> None:
        """Hard-delete a reminder (used by dashboard dismiss)."""
        await self._db.execute("DELETE FROM reminders WHERE id = ?", (reminder_id,))
        logger.info(f"[Reminders] Deleted #{reminder_id}")
