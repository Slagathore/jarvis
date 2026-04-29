"""
JARVIS — Ambient Home AI
========================
Mission: Provide an async SQLite database layer for all persistent storage.
         The schema covers events, conversation logs, room baselines, user routines,
         and reminders. A single DatabaseManager instance is shared across all
         modules. All queries are async via aiosqlite to avoid blocking the event loop.

Modules: modules/memory/database.py
Classes: DatabaseManager
Functions:
    DatabaseManager.__init__(config)    — Store config, DB path
    DatabaseManager.init()              — Create tables if not exist (async)
    DatabaseManager.execute(sql, params)— Run a write query (INSERT/UPDATE/DELETE)
    DatabaseManager.fetchall(sql, params)— Run a SELECT, return list of Row objects
    DatabaseManager.fetchone(sql, params)— Run a SELECT, return first row or None
    DatabaseManager.close()             — Close the connection pool

Variables:
    DatabaseManager._db_path — Path to the .db file
    DatabaseManager._conn    — aiosqlite.Connection, opened on first init()

Schema tables:
    events            — General event log (room activity, observations)
    conversation_log  — Full conversation history (user + assistant turns)
    room_baselines    — Per-room visual baseline descriptions
    user_routines     — Learned patterns (day/hour/activity frequencies)
    reminders         — User-set reminders with trigger times

#todo: Add full-text search index on conversation_log.content
#todo: Add database migration system for schema evolution
#todo: Add automatic backup on startup (copy .db before opening)
#todo: Expose query stats (row counts, DB size) to the health dashboard
#todo: Add vacuum() helper for periodic DB compaction
"""

import asyncio
from pathlib import Path
from typing import Optional

import aiosqlite
from loguru import logger

from core.exceptions import DatabaseError

# DDL for all tables — idempotent (CREATE IF NOT EXISTS)
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    room            TEXT    NOT NULL,
    type            TEXT    NOT NULL,
    content         TEXT    NOT NULL,
    acknowledged    INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS conversation_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    room        TEXT    NOT NULL,
    role        TEXT    NOT NULL,    -- "user" | "assistant"
    content     TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS room_baselines (
    room            TEXT PRIMARY KEY,
    baseline_desc   TEXT,
    updated_at      TEXT
);

CREATE TABLE IF NOT EXISTS user_routines (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    day_of_week TEXT,
    hour        INTEGER,
    activity    TEXT,
    frequency   INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS reminders (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    message         TEXT    NOT NULL,
    trigger_time    TEXT,
    recurring       INTEGER DEFAULT 0,
    last_triggered  TEXT
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_events_room_ts ON events (room, timestamp);
CREATE INDEX IF NOT EXISTS idx_conv_room_ts   ON conversation_log (room, timestamp);
CREATE INDEX IF NOT EXISTS idx_reminders_time ON reminders (trigger_time);
"""


class DatabaseManager:
    """
    Async SQLite database manager. All methods are coroutines.

    The connection is opened once on init() and kept open for the lifetime
    of the process. WAL mode is enabled for concurrent reads during writes.

    Usage:
        db = DatabaseManager(config)
        await db.init()
        await db.execute("INSERT INTO events ...", ("office", "vision", "..."))
        rows = await db.fetchall("SELECT * FROM events WHERE room=?", ("office",))
    """

    def __init__(self, config: dict) -> None:
        self._db_path = Path(config["memory"]["db_path"])
        self._conn: Optional[aiosqlite.Connection] = None

    async def init(self) -> None:
        """
        Open the database connection and create all tables.
        Safe to call multiple times — all DDL is idempotent.

        Raises:
            DatabaseError: If the file cannot be created or the schema fails.
        """
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = await aiosqlite.connect(str(self._db_path))
            # Row factory for dict-like access: row["column_name"]
            conn.row_factory = aiosqlite.Row

            # WAL mode allows concurrent reads while writing
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA foreign_keys=ON")

            # Apply schema
            await conn.executescript(SCHEMA_SQL)
            await conn.commit()
            self._conn = conn

            logger.info(f"[DB] Ready: {self._db_path}")

        except Exception as e:
            raise DatabaseError(f"Database init failed: {e}") from e

    async def execute(
        self,
        sql: str,
        params: tuple = (),
    ) -> int:
        """
        Execute a write query (INSERT, UPDATE, DELETE).
        Returns the last inserted row ID (useful for INSERT).

        Raises:
            DatabaseError: On SQL error or if init() was not called.
        """
        conn = self._get_connection()
        try:
            cursor = await conn.execute(sql, params)
            await conn.commit()
            return int(cursor.lastrowid or 0)
        except aiosqlite.Error as e:
            raise DatabaseError(f"Execute failed [{sql[:80]}]: {e}") from e

    async def fetchall(
        self,
        sql: str,
        params: tuple = (),
    ) -> list[aiosqlite.Row]:
        """
        Execute a SELECT and return all rows.
        Rows support dict-like access: row["column_name"].

        Raises:
            DatabaseError: On SQL error.
        """
        conn = self._get_connection()
        try:
            cursor = await conn.execute(sql, params)
            return list(await cursor.fetchall())
        except aiosqlite.Error as e:
            raise DatabaseError(f"Fetchall failed [{sql[:80]}]: {e}") from e

    async def fetchone(
        self,
        sql: str,
        params: tuple = (),
    ) -> Optional[aiosqlite.Row]:
        """
        Execute a SELECT and return the first row, or None if no results.

        Raises:
            DatabaseError: On SQL error.
        """
        conn = self._get_connection()
        try:
            cursor = await conn.execute(sql, params)
            return await cursor.fetchone()
        except aiosqlite.Error as e:
            raise DatabaseError(f"Fetchone failed [{sql[:80]}]: {e}") from e

    async def close(self) -> None:
        """Close the database connection gracefully."""
        conn = self._conn
        if conn is not None:
            await conn.close()
            self._conn = None
            logger.debug("[DB] Connection closed")

    def _get_connection(self) -> aiosqlite.Connection:
        """Return the live connection or raise if init() was not called."""
        conn = self._conn
        if conn is None:
            raise DatabaseError("Database not initialized — call await db.init() first")
        return conn
