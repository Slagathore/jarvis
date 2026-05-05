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
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    message            TEXT    NOT NULL,
    trigger_time       TEXT,
    recurring          INTEGER DEFAULT 0,        -- legacy bool flag
    recurrence_seconds INTEGER,                  -- if set, repeat every N seconds after firing
    last_triggered     TEXT
);

-- Legacy single-centroid identity tables. Identity v2 (persons + face_samples
-- + voice_samples) supersedes these; rows here are migrated forward on first
-- boot of v2 and the tables are kept read-only for one release cycle.
CREATE TABLE IF NOT EXISTS speakers (
    name           TEXT PRIMARY KEY,
    embedding      BLOB NOT NULL,    -- 256-dim float32 centroid from resemblyzer
    sample_count   INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS faces (
    name           TEXT PRIMARY KEY,
    embedding      BLOB NOT NULL,    -- 128-dim float32 centroid from deepface (Facenet)
    sample_count   INTEGER DEFAULT 1
);

-- ── Identity v2 ──────────────────────────────────────────────────────────────
-- A `person` is a single human identity. Each person can have many face_samples
-- and many voice_samples (never deleted). Match across modalities is by
-- person_id, so voice ID and face ID are anchored to the same person regardless
-- of modality drift (cold, haircut, glasses, etc.).
CREATE TABLE IF NOT EXISTS persons (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    created_at  TEXT    NOT NULL,
    notes       TEXT
);

CREATE TABLE IF NOT EXISTS face_samples (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id    INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    embedding    BLOB    NOT NULL,    -- 128-dim float32 (Facenet)
    pose         TEXT,                -- center | left | right | up | down | candid
    captured_at  TEXT    NOT NULL,
    source       TEXT    NOT NULL,    -- enroll | drift_capture | live_question | migration
    image_jpeg   BLOB                 -- preview thumbnail for the dashboard; nullable for legacy/migrated rows
);

CREATE TABLE IF NOT EXISTS voice_samples (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id    INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    embedding    BLOB    NOT NULL,    -- 256-dim float32 (Resemblyzer)
    prompt_id    TEXT,                -- 'sentence_1' | 'sentence_2' | 'sentence_3' | 'wake'
    captured_at  TEXT    NOT NULL,
    source       TEXT    NOT NULL     -- enroll | drift_capture | live_question | migration
);

-- Pending captures that the dashboard surfaces for human review.
-- kind: 'face_drift' or 'voice_drift' = anchor said this is person X but the
--       captured sample didn't match X loosely. Reviewer confirms or rejects.
-- kind: 'pending_cluster' = sample matched no enrolled person; clusters with
--       similar pending samples form a candidate persona that the reviewer
--       can name.
CREATE TABLE IF NOT EXISTS identity_pending (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    kind            TEXT    NOT NULL,    -- face_drift | voice_drift | pending_cluster_face | pending_cluster_voice
    person_id       INTEGER REFERENCES persons(id) ON DELETE SET NULL,
    cluster_id      INTEGER,             -- groups pending samples of the same modality
    embedding       BLOB    NOT NULL,
    image_jpeg      BLOB,                -- preview JPEG for face captures
    audio_pcm16     BLOB,                -- preview PCM (16-bit, 16kHz) for voice captures (~3s)
    similarity      REAL,                -- best cosine score against enrolled persons
    anchored_via    TEXT,                -- 'voice' | 'face' | NULL for cold pending clusters
    captured_at     TEXT    NOT NULL,
    resolved        INTEGER DEFAULT 0    -- 0 = needs review, 1 = applied, 2 = rejected
);

CREATE INDEX IF NOT EXISTS idx_face_samples_person ON face_samples (person_id);
CREATE INDEX IF NOT EXISTS idx_voice_samples_person ON voice_samples (person_id);
CREATE INDEX IF NOT EXISTS idx_pending_resolved ON identity_pending (resolved);
CREATE INDEX IF NOT EXISTS idx_pending_cluster ON identity_pending (cluster_id);

-- ── Notifications ────────────────────────────────────────────────────────────
-- A persistent, dashboard-surfaced inbox for things that need Cole's attention.
-- Sources include identity drift/conflict captures, system errors that
-- couldn't be resolved automatically, and any future module that wants to
-- raise a flag without spamming proactive speech. The dashboard renders a
-- bell icon with an unread count and a dropdown list; clicking a notification
-- navigates to the relevant card (pending review, person modal, etc.).
CREATE TABLE IF NOT EXISTS notifications (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    kind         TEXT    NOT NULL,    -- e.g. 'identity.drift', 'identity.cluster', 'system.error'
    title        TEXT    NOT NULL,
    message      TEXT,
    target_type  TEXT,                 -- 'person' | 'pending' | 'room' | NULL
    target_id    INTEGER,              -- id within target_type
    action       TEXT,                 -- frontend nav hint: 'open_pending', 'open_person', etc.
    severity     TEXT    DEFAULT 'info', -- 'info' | 'warning' | 'error'
    created_at   TEXT    NOT NULL,
    read         INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_notifications_unread ON notifications (read, created_at);

-- Per-activity transition log used for predicted-duration + routine learning.
-- A row is open (ended_at NULL) while the activity is current; closed when the
-- activity changes.
CREATE TABLE IF NOT EXISTS activity_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    activity         TEXT NOT NULL,
    started_at       TEXT NOT NULL,
    ended_at         TEXT,
    duration_seconds INTEGER,
    room             TEXT
);

CREATE INDEX IF NOT EXISTS idx_activity_log_activity ON activity_log (activity);
CREATE INDEX IF NOT EXISTS idx_activity_log_started  ON activity_log (started_at);

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

            # Idempotent column additions for older databases. SQLite has no
            # native ADD COLUMN IF NOT EXISTS so we just try and swallow the
            # "duplicate column" error.
            for migration_sql in (
                "ALTER TABLE reminders ADD COLUMN recurrence_seconds INTEGER",
                "ALTER TABLE face_samples ADD COLUMN image_jpeg BLOB",
            ):
                try:
                    await conn.execute(migration_sql)
                except aiosqlite.OperationalError:
                    pass  # column already exists

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
