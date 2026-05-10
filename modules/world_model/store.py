"""
JARVIS — World Model
====================
Mission: Async storage layer for the World Model. Uses the existing
         aiosqlite-based DatabaseManager — does NOT open its own
         connection, does NOT hold its own lock, does NOT import
         sqlite3 directly. Mixing sync and async DB access in the
         same process is a known source of deadlocks under load.

         Schema mirrors §8 of new 2.md. Three tables:
            world_entities             — current state per entity
            world_entity_events        — append-only event log
            world_entity_embeddings    — visual fingerprints (cats/objects)

         Phase 4 (§22.0a, §32) adds:
            world_entities.household_owner_id     INTEGER → persons.id
            world_entities.unmonitored_home_room  TEXT
            world_entities.archived_at            TEXT
            pet_affinities                         table — per-pet, per-person
                                                   strength + contexts

         The ALTER TABLE / CREATE TABLE for these is idempotent and runs
         via the same `ensure_schema` path the Phase 1 tables use, so
         existing DBs migrate forward in place without a separate
         migrations runner.

Modules: modules/world_model/store.py
Spec:    new 2.md §8 (Storage Layer) and §16 (Full Code: WorldStore).
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

import numpy as np

from modules.world_model.types import (
    EntityState,
    WorldEntity,
)


class WorldStore:
    """
    Thin async persistence layer over the existing DatabaseManager.
    Every method is async-safe; concurrent calls are fine — DatabaseManager
    serializes writes per its existing discipline (one aiosqlite connection,
    auto-commit per `execute`).
    """

    def __init__(self, db_manager) -> None:
        # db_manager is modules/memory/database.py:DatabaseManager — exposes
        # async execute/fetchall/fetchone. Each `execute` already commits;
        # we don't call a separate commit().
        self.db = db_manager

    async def ensure_schema(self) -> None:
        """
        Idempotent CREATE TABLE IF NOT EXISTS for world_* tables.
        Called once at startup. Schema mirrors §8 + Phase 4 §32 additions.
        """
        statements = [
            """CREATE TABLE IF NOT EXISTS world_entities (
                id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                person_id INTEGER REFERENCES persons(id),
                display_name TEXT,
                state TEXT NOT NULL,
                last_seen_ts TEXT,
                last_seen_room TEXT,
                last_seen_camera TEXT,
                last_seen_bbox TEXT,
                last_seen_landmark TEXT,
                last_state_change_ts TEXT,
                confidence REAL,
                last_attribution_confidence REAL,
                is_resident INTEGER DEFAULT 0,
                metadata TEXT,
                household_owner_id INTEGER REFERENCES persons(id),
                unmonitored_home_room TEXT,
                archived_at TEXT
            )""",
            "CREATE INDEX IF NOT EXISTS idx_world_entities_person ON world_entities(person_id)",
            "CREATE INDEX IF NOT EXISTS idx_world_entities_state ON world_entities(state)",
            "CREATE INDEX IF NOT EXISTS idx_world_entities_room ON world_entities(last_seen_room)",
            "CREATE INDEX IF NOT EXISTS idx_world_entities_owner ON world_entities(household_owner_id)",
            """CREATE TABLE IF NOT EXISTS world_entity_events (
                id TEXT PRIMARY KEY,
                ts TEXT NOT NULL,
                entity_id TEXT NOT NULL REFERENCES world_entities(id),
                person_id INTEGER REFERENCES persons(id),
                entity_name TEXT,
                entity_type TEXT NOT NULL,
                event_type TEXT NOT NULL,
                room TEXT,
                camera TEXT,
                bbox TEXT,
                landmark TEXT,
                state TEXT,
                confidence REAL,
                snapshot_path TEXT,
                related_entity_id TEXT,
                metadata TEXT
            )""",
            "CREATE INDEX IF NOT EXISTS idx_world_events_entity_ts ON world_entity_events(entity_id, ts DESC)",
            "CREATE INDEX IF NOT EXISTS idx_world_events_room_ts ON world_entity_events(room, ts DESC)",
            "CREATE INDEX IF NOT EXISTS idx_world_events_type_ts ON world_entity_events(event_type, ts DESC)",
            "CREATE INDEX IF NOT EXISTS idx_world_events_person_ts ON world_entity_events(person_id, ts DESC)",
            """CREATE TABLE IF NOT EXISTS world_entity_embeddings (
                entity_id TEXT PRIMARY KEY REFERENCES world_entities(id),
                embedding BLOB NOT NULL,
                dimension INTEGER NOT NULL,
                updated_ts TEXT NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS pet_affinities (
                pet_entity_id TEXT NOT NULL REFERENCES world_entities(id) ON DELETE CASCADE,
                person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
                strength TEXT NOT NULL CHECK (strength IN ('low','medium','high')),
                contexts TEXT NOT NULL,
                PRIMARY KEY (pet_entity_id, person_id)
            )""",
            "CREATE INDEX IF NOT EXISTS idx_pet_affinities_pet ON pet_affinities(pet_entity_id)",
            # ── §32 v4 schema — alarm + door tables ─────────────────────────
            # Created here for now because WorldStore is the only module with
            # a deterministic ensure_schema entry-point. When the §29.3 door
            # alarm and §29's alarm_fires/state persistence land, ownership
            # moves to those modules; the CREATEs are idempotent so a
            # double-create is a no-op. notification_deliveries is owned by
            # NotificationDispatcher (lazy-created on first send).
            """CREATE TABLE IF NOT EXISTS alarm_state (
                alarm_type   TEXT PRIMARY KEY,
                state        TEXT NOT NULL,
                state_since  TEXT NOT NULL,
                metadata     TEXT
            )""",
            """CREATE TABLE IF NOT EXISTS alarm_fires (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                alarm_type   TEXT NOT NULL,
                fired_at     TEXT NOT NULL,
                resolved_at  TEXT,
                resolution   TEXT,
                metadata     TEXT
            )""",
            "CREATE INDEX IF NOT EXISTS idx_alarm_fires_type_time ON alarm_fires(alarm_type, fired_at DESC)",
            """CREATE TABLE IF NOT EXISTS door_state (
                door_id      TEXT PRIMARY KEY,
                state        TEXT NOT NULL CHECK (state IN ('open','closed','unknown')),
                state_since  TEXT NOT NULL,
                source       TEXT NOT NULL CHECK (source IN ('vision','reed','manual'))
            )""",
        ]
        for stmt in statements:
            await self.db.execute(stmt)
        # Forward-migrate any DB created before §22.0a — SQLite doesn't
        # support `ALTER TABLE ADD COLUMN IF NOT EXISTS`, so we read
        # PRAGMA table_info and add only the missing ones.
        await self._ensure_columns(
            "world_entities",
            [
                ("household_owner_id", "INTEGER REFERENCES persons(id)"),
                ("unmonitored_home_room", "TEXT"),
                ("archived_at", "TEXT"),
            ],
        )

    async def _ensure_columns(
        self, table: str, columns: list[tuple[str, str]]
    ) -> None:
        """Add any missing columns to `table`. SQLite has no `IF NOT
        EXISTS` for column additions, so we PRAGMA-introspect first."""
        rows = await self.db.fetchall(f"PRAGMA table_info({table})")
        existing = {r["name"] for r in rows}
        for col_name, col_decl in columns:
            if col_name not in existing:
                await self.db.execute(
                    f"ALTER TABLE {table} ADD COLUMN {col_name} {col_decl}"
                )

    # ── Entities ─────────────────────────────────────────────────────────────

    async def upsert_entity(self, ent: WorldEntity) -> None:
        await self.db.execute(
            """
            INSERT INTO world_entities (
                id, entity_type, person_id, display_name, state,
                last_seen_ts, last_seen_room, last_seen_camera, last_seen_bbox,
                last_seen_landmark, last_state_change_ts, confidence,
                last_attribution_confidence, is_resident, metadata,
                household_owner_id, unmonitored_home_room, archived_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                entity_type=excluded.entity_type,
                person_id=excluded.person_id,
                display_name=excluded.display_name,
                state=excluded.state,
                last_seen_ts=excluded.last_seen_ts,
                last_seen_room=excluded.last_seen_room,
                last_seen_camera=excluded.last_seen_camera,
                last_seen_bbox=excluded.last_seen_bbox,
                last_seen_landmark=excluded.last_seen_landmark,
                last_state_change_ts=excluded.last_state_change_ts,
                confidence=excluded.confidence,
                last_attribution_confidence=excluded.last_attribution_confidence,
                is_resident=excluded.is_resident,
                metadata=excluded.metadata,
                household_owner_id=excluded.household_owner_id,
                unmonitored_home_room=excluded.unmonitored_home_room,
                archived_at=excluded.archived_at
            """,
            (
                ent.id, ent.entity_type, ent.person_id, ent.display_name,
                ent.state.value,
                _iso(ent.last_seen_ts), ent.last_seen_room, ent.last_seen_camera,
                json.dumps(ent.last_seen_bbox) if ent.last_seen_bbox else None,
                ent.last_seen_landmark,
                _iso(ent.last_state_change_ts), ent.confidence,
                ent.last_attribution_confidence,
                int(ent.is_resident), json.dumps(_clean_metadata(ent.metadata)),
                ent.household_owner_id, ent.unmonitored_home_room,
                _iso(ent.archived_at),
            ),
        )

    async def upsert_embedding(self, entity_id: str, embedding: np.ndarray) -> None:
        await self.db.execute(
            """
            INSERT INTO world_entity_embeddings (entity_id, embedding, dimension, updated_ts)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(entity_id) DO UPDATE SET
                embedding=excluded.embedding,
                dimension=excluded.dimension,
                updated_ts=excluded.updated_ts
            """,
            (
                entity_id,
                embedding.astype(np.float32).tobytes(),
                int(embedding.shape[0]),
                datetime.utcnow().isoformat(),
            ),
        )

    async def load_entities(self) -> list[WorldEntity]:
        rows = await self.db.fetchall("SELECT * FROM world_entities")
        emb_rows = await self.db.fetchall(
            "SELECT entity_id, embedding FROM world_entity_embeddings"
        )
        emb_map = {
            r["entity_id"]: np.frombuffer(r["embedding"], dtype=np.float32)
            for r in emb_rows
        }

        # Affinity rows joined per pet so behavioral cost-functions and
        # the persona snapshot can read them without an extra round-trip.
        aff_rows = await self.db.fetchall(
            "SELECT pet_entity_id, person_id, strength, contexts FROM pet_affinities"
        )
        aff_map: dict[str, list[dict]] = {}
        for r in aff_rows:
            aff_map.setdefault(r["pet_entity_id"], []).append({
                "person_id": r["person_id"],
                "strength": r["strength"],
                "contexts": [c for c in (r["contexts"] or "").split(",") if c],
            })

        entities: list[WorldEntity] = []
        row_keys = set(rows[0].keys()) if rows else set()
        has_phase4_cols = "household_owner_id" in row_keys
        for row in rows:
            ent = WorldEntity(
                id=row["id"],
                entity_type=row["entity_type"],
                person_id=row["person_id"],
                display_name=row["display_name"],
                state=EntityState(row["state"]),
                last_seen_ts=_parse_iso(row["last_seen_ts"]),
                last_seen_room=row["last_seen_room"],
                last_seen_camera=row["last_seen_camera"],
                last_seen_bbox=tuple(json.loads(row["last_seen_bbox"]))
                    if row["last_seen_bbox"] else None,
                last_seen_landmark=row["last_seen_landmark"],
                last_state_change_ts=_parse_iso(row["last_state_change_ts"])
                    or datetime.utcnow(),
                confidence=row["confidence"] or 0.0,
                last_attribution_confidence=row["last_attribution_confidence"] or 0.0,
                is_resident=bool(row["is_resident"]),
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                household_owner_id=(row["household_owner_id"]
                                    if has_phase4_cols else None),
                unmonitored_home_room=(row["unmonitored_home_room"]
                                       if has_phase4_cols else None),
                archived_at=(_parse_iso(row["archived_at"])
                             if has_phase4_cols else None),
            )
            # Stash visual embedding (cats/objects only) on the entity for fast access
            if ent.id in emb_map:
                ent.metadata["_visual_embedding"] = emb_map[ent.id]
            if ent.id in aff_map:
                ent.metadata["affinities"] = aff_map[ent.id]
            entities.append(ent)
        return entities

    # ── Affinities ───────────────────────────────────────────────────────────

    async def replace_affinities(
        self, pet_entity_id: str, affinities: list[dict]
    ) -> None:
        """Idempotent replace of all affinity rows for one pet.
        `affinities` is a list of {person_id, strength, contexts: list[str]}."""
        await self.db.execute(
            "DELETE FROM pet_affinities WHERE pet_entity_id = ?",
            (pet_entity_id,),
        )
        for aff in affinities:
            await self.db.execute(
                """
                INSERT INTO pet_affinities (
                    pet_entity_id, person_id, strength, contexts
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    pet_entity_id,
                    int(aff["person_id"]),
                    str(aff["strength"]),
                    ",".join(aff.get("contexts", [])),
                ),
            )

    # ── Events ───────────────────────────────────────────────────────────────

    async def append_event(self, payload: dict) -> None:
        await self.db.execute(
            """
            INSERT INTO world_entity_events (
                id, ts, entity_id, person_id, entity_name, entity_type,
                event_type, room, camera, bbox, landmark, state, confidence,
                snapshot_path, related_entity_id, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["id"], payload["ts"], payload["entity_id"],
                payload.get("person_id"),
                payload.get("entity_name"), payload.get("entity_type"),
                payload["event_type"], payload.get("room"), payload.get("camera"),
                json.dumps(payload.get("bbox")) if payload.get("bbox") else None,
                payload.get("landmark"),
                payload.get("state"), payload.get("confidence", 0.0),
                payload.get("snapshot_path"), payload.get("related_entity_id"),
                json.dumps(payload.get("metadata", {})),
            ),
        )

    async def search_events(
        self,
        entity_id: Optional[str] = None,
        person_id: Optional[int] = None,
        room: Optional[str] = None,
        event_types: Optional[list[str]] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 50,
    ) -> list[dict]:
        q = "SELECT * FROM world_entity_events WHERE 1=1"
        params: list[Any] = []
        if entity_id:
            q += " AND entity_id = ?"
            params.append(entity_id)
        if person_id is not None:
            q += " AND person_id = ?"
            params.append(person_id)
        if room:
            q += " AND room = ?"
            params.append(room)
        if event_types:
            placeholders = ",".join("?" for _ in event_types)
            q += f" AND event_type IN ({placeholders})"
            params.extend(event_types)
        if since:
            q += " AND ts >= ?"
            params.append(since.isoformat())
        if until:
            q += " AND ts <= ?"
            params.append(until.isoformat())
        q += " ORDER BY ts DESC LIMIT ?"
        params.append(limit)
        rows = await self.db.fetchall(q, tuple(params))
        return [dict(r) for r in rows]


# ── helpers ─────────────────────────────────────────────────────────────────


def _iso(ts: Optional[datetime]) -> Optional[str]:
    return ts.isoformat() if ts else None


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    return datetime.fromisoformat(s) if s else None


def _clean_metadata(metadata: dict) -> dict:
    """Strip non-JSON-serializable values (numpy arrays, etc.) before persisting."""
    return {
        k: v for k, v in metadata.items()
        if not (k.startswith("_") or isinstance(v, np.ndarray))
    }
