"""
JARVIS — Ambient Home AI
========================
Mission: Log all significant events and conversations to the database for
         persistent memory across restarts. Provides retrieval helpers for
         recent events and conversations so the orchestrator and curiosity
         engine can reference what happened earlier.

Modules: modules/memory/event_log.py
Classes: EventLogger
Functions:
    EventLogger.__init__(db)                   — Initialize with DatabaseManager
    EventLogger.log_event(room, type, content) — Write an event record (async)
    EventLogger.log_conversation(room, role, content) — Write a conversation turn
    EventLogger.get_recent_events(room, n)     — Fetch N most recent events
    EventLogger.get_recent_conversation(room, n) — Fetch N recent turns
    EventLogger.acknowledge_event(event_id)    — Mark event as handled

Variables:
    EventLogger._db — DatabaseManager reference

Event types (used as the `type` column):
    "wake_detected"   — Wake word fired
    "speech"          — Jarvis spoke
    "appliance"       — Appliance state changed
    "vision"          — Camera observation
    "posture"         — Posture detected
    "activity"        — Activity state changed
    "reminder"        — Reminder triggered
    "system"          — System start/stop/error

#todo: Add full-text search over conversation_log for memory recall
#todo: Add event deduplication (don't log identical consecutive events)
#todo: Add summarization job that compresses old events to save space
#todo: Expose event timeline to dashboard for playback/review
"""

from datetime import datetime
from typing import Optional

from loguru import logger

from modules.memory.database import DatabaseManager


class EventLogger:
    """
    High-level event logging interface over DatabaseManager.

    All methods are async and safe to call from any coroutine.
    The underlying database must already be initialized (db.init() called).
    """

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    async def log_event(
        self,
        room: str,
        event_type: str,
        content: str,
    ) -> int:
        """
        Write a general event to the events table.

        Args:
            room:       Room identifier (e.g., "office", "kitchen").
            event_type: Category string (see Event types above).
            content:    Human-readable description of what happened.

        Returns:
            The new row's ID.
        """
        now = datetime.now().isoformat()
        row_id = await self._db.execute(
            "INSERT INTO events (timestamp, room, type, content) VALUES (?,?,?,?)",
            (now, room, event_type, content),
        )
        logger.debug(f"[EventLog] [{room}] {event_type}: {content[:80]}")
        return row_id

    async def log_conversation(
        self,
        room: str,
        role: str,
        content: str,
    ) -> int:
        """
        Write a conversation turn to the conversation_log table.

        Args:
            room:    Room where the exchange happened.
            role:    "user" or "assistant".
            content: The message text.

        Returns:
            The new row's ID.
        """
        now = datetime.now().isoformat()
        row_id = await self._db.execute(
            "INSERT INTO conversation_log (timestamp, room, role, content) VALUES (?,?,?,?)",
            (now, room, role, content),
        )
        return row_id

    async def get_recent_events(
        self,
        room: Optional[str] = None,
        n: int = 20,
        event_type: Optional[str] = None,
    ) -> list[dict]:
        """
        Fetch the N most recent events, optionally filtered by room and/or type.

        Returns:
            List of dicts with keys: id, timestamp, room, type, content, acknowledged.
        """
        params: list = []
        where_clauses: list[str] = []

        if room:
            where_clauses.append("room = ?")
            params.append(room)
        if event_type:
            where_clauses.append("type = ?")
            params.append(event_type)

        where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        params.append(n)

        rows = await self._db.fetchall(
            f"SELECT * FROM events {where} ORDER BY timestamp DESC LIMIT ?",
            tuple(params),
        )
        return [dict(row) for row in rows]

    async def get_recent_conversation(
        self,
        room: str,
        n: int = 10,
    ) -> list[dict]:
        """
        Fetch the N most recent conversation turns for a room.

        Returns:
            List of dicts: id, timestamp, room, role, content — oldest first.
        """
        rows = await self._db.fetchall(
            """
            SELECT * FROM (
                SELECT * FROM conversation_log
                WHERE room = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ) sub ORDER BY timestamp ASC
            """,
            (room, n),
        )
        return [dict(row) for row in rows]

    async def acknowledge_event(self, event_id: int) -> None:
        """Mark an event as acknowledged (handled by Jarvis or the user)."""
        await self._db.execute(
            "UPDATE events SET acknowledged=1 WHERE id=?",
            (event_id,),
        )

    async def get_unacknowledged_events(self, room: Optional[str] = None) -> list[dict]:
        """Return all events that haven't been acknowledged yet."""
        if room:
            rows = await self._db.fetchall(
                "SELECT * FROM events WHERE acknowledged=0 AND room=? ORDER BY timestamp",
                (room,),
            )
        else:
            rows = await self._db.fetchall(
                "SELECT * FROM events WHERE acknowledged=0 ORDER BY timestamp",
            )
        return [dict(row) for row in rows]
