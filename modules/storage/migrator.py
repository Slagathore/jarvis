"""
JARVIS — Ambient Home AI
========================
Mission: Forward-only schema migration runner.

         Before this, schema ownership was scattered: DatabaseManager.SCHEMA_SQL
         (core tables), ad-hoc `ALTER TABLE` calls in DatabaseManager.init(),
         WorldStore.ensure_schema() (world/alarm/door tables), and
         NotificationDispatcher._ensure_delivery_table() (lazy create). Adding
         a new table meant picking one of four inconsistent places.

         SchemaMigrator gives one ordered, idempotent path. Each migration is
         a numbered `.sql` file under migrations/. On boot the runner applies
         every file whose id is not yet recorded in `schema_migrations`, inside
         a transaction, and records it. Re-running is a no-op.

         The pre-existing schema (SCHEMA_SQL / ensure_schema / lazy creates)
         is intentionally left in place — those tables already exist in every
         live DB. The migrator owns NEW schema from here forward (the
         BeliefResolver tables are the first occupant). A later cleanup pass
         can fold the legacy DDL into baseline migration files.

Modules: modules/storage/migrator.py
Classes: SchemaMigrator

Migration file convention:
    modules/storage/migrations/NNN_short_name.sql
    - NNN is a zero-padded ordering prefix (001, 002, ...).
    - The full filename (minus .sql) is the migration id stored in
      schema_migrations — so files must never be renamed once shipped.
    - SQL must be idempotent-friendly (CREATE TABLE IF NOT EXISTS etc.);
      a partially-applied migration that failed mid-way can then be
      retried safely on the next boot.
"""

from pathlib import Path
from typing import Any

from loguru import logger

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"

_SCHEMA_MIGRATIONS_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    id          TEXT PRIMARY KEY,
    applied_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


class SchemaMigrator:
    """Applies pending `.sql` migrations in filename order.

    Usage (from DatabaseManager.init, after the legacy schema is in place):
        await SchemaMigrator(db).run()
    """

    def __init__(self, db: Any, migrations_dir: Path = _MIGRATIONS_DIR) -> None:
        self._db = db
        self._dir = migrations_dir

    async def run(self) -> list[str]:
        """Apply every migration not yet recorded. Returns the ids applied
        this run (empty when the DB is already current)."""
        await self._db.executescript(_SCHEMA_MIGRATIONS_DDL)

        applied: set[str] = set()
        for row in await self._db.fetchall("SELECT id FROM schema_migrations"):
            applied.add(row["id"])

        if not self._dir.is_dir():
            logger.debug(f"[Migrator] no migrations dir at {self._dir}")
            return []

        pending = sorted(
            p for p in self._dir.glob("*.sql") if p.stem not in applied
        )
        if not pending:
            logger.debug(
                f"[Migrator] schema current ({len(applied)} migration(s) applied)"
            )
            return []

        done: list[str] = []
        for path in pending:
            mid = path.stem
            sql = path.read_text(encoding="utf-8")
            try:
                # executescript wraps the file in its own transaction and
                # commits; the recording INSERT commits separately. A crash
                # between them just means the migration re-runs next boot —
                # which is why migration SQL must be idempotent-friendly.
                await self._db.executescript(sql)
                await self._db.execute(
                    "INSERT INTO schema_migrations (id) VALUES (?)", (mid,)
                )
            except Exception as e:
                logger.error(f"[Migrator] migration '{mid}' failed: {e}")
                raise
            logger.info(f"[Migrator] applied migration '{mid}'")
            done.append(mid)

        logger.info(f"[Migrator] {len(done)} migration(s) applied")
        return done
