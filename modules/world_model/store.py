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

Modules: modules/world_model/store.py
Spec:    new 2.md §8 (Storage Layer) and §16 (Full Code: WorldStore).

#todo: Phase 4 schema additions from §32 (household_owner_id,
       unmonitored_home_room, archived_at, pet_affinities) get added
       when we implement Phase 4 — defer until then to keep Phase 1
       schema lean.
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
        Called once at startup. Schema mirrors §8.
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
                metadata TEXT
            )""",
            "CREATE INDEX IF NOT EXISTS idx_world_entities_person ON world_entities(person_id)",
            "CREATE INDEX IF NOT EXISTS idx_world_entities_state ON world_entities(state)",
            "CREATE INDEX IF NOT EXISTS idx_world_entities_room ON world_entities(last_seen_room)",
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
        ]
        for stmt in statements:
            await self.db.execute(stmt)

    # ── Entities ─────────────────────────────────────────────────────────────

    async def upsert_entity(self, ent: WorldEntity) -> None:
        await self.db.execute(
            """
            INSERT INTO world_entities (
                id, entity_type, person_id, display_name, state,
                last_seen_ts, last_seen_room, last_seen_camera, last_seen_bbox,
                last_seen_landmark, last_state_change_ts, confidence,
                last_attribution_confidence, is_resident, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                metadata=excluded.metadata
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

        entities: list[WorldEntity] = []
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
            )
            # Stash visual embedding (cats/objects only) on the entity for fast access
            if ent.id in emb_map:
                ent.metadata["_visual_embedding"] = emb_map[ent.id]
            entities.append(ent)
        return entities

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
