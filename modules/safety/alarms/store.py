"""
JARVIS — Safety
===============
Mission: Persistence wrapper for the alarm subsystem. Writes one
         row per fire to `alarm_fires` and tracks current
         per-alarm-type state in `alarm_state`. Both tables live in
         data/jarvis.db (created via WorldStore.ensure_schema, §32).

         Why a thin wrapper instead of inlining INSERTs in `Alarm`:
         the alarm subclasses already own a lot (state machine,
         condition watching, voice silence, rearm timers); routing
         their persistence through one helper keeps the migration
         path clean if we later move alarm tables to their own
         schema owner.

Modules: modules/safety/alarms/store.py
Classes: AlarmStore, NullAlarmStore
Spec:    new 2.md §32.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from loguru import logger


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AlarmStore:
    """Async wrapper over alarm_fires + alarm_state. One method per
    state-machine transition the alarm wants logged."""

    def __init__(self, db: Any) -> None:
        self._db = db

    async def record_fire(
        self,
        fire_id: str,
        alarm_type: str,
        fired_at: datetime,
        metadata: Optional[dict] = None,
    ) -> None:
        """Append-only audit row. Same alarm refiring (rearm) reuses
        fire_id so the audit trail can group rearms with their
        original fire."""
        try:
            existing = await self._db.fetchone(
                "SELECT id FROM alarm_fires WHERE metadata LIKE ?",
                (f'%"fire_id": "{fire_id}"%',),
            )
            if existing is not None:
                # Already logged this fire_id — don't duplicate; rearms
                # within the same fire are tracked via state transitions.
                return
            await self._db.execute(
                "INSERT INTO alarm_fires "
                "(alarm_type, fired_at, metadata) VALUES (?, ?, ?)",
                (
                    alarm_type,
                    fired_at.isoformat(),
                    json.dumps({"fire_id": fire_id, **(metadata or {})}),
                ),
            )
        except Exception as e:
            logger.debug(f"[AlarmStore] record_fire failed: {e}")

    async def record_resolution(
        self,
        fire_id: str,
        alarm_type: str,
        resolution: str,
        resolved_at: Optional[datetime] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Stamp the resolved_at + resolution on the alarm_fires row
        for this fire_id. resolution ∈ {condition_clear, voice_silence,
        visual_confirm, manual, mute_expired_clear}."""
        # Compute ts once outside the try so the except-branch fallback
        # path has a definite value (Pylance flagged the original as
        # possibly-unbound).
        ts = (resolved_at or datetime.now(timezone.utc)).isoformat()
        try:
            await self._db.execute(
                "UPDATE alarm_fires SET resolved_at = ?, resolution = ?, "
                "metadata = json_patch(COALESCE(metadata, '{}'), ?) "
                "WHERE alarm_type = ? AND metadata LIKE ?",
                (
                    ts,
                    resolution,
                    json.dumps({"resolution_metadata": (metadata or {})}),
                    alarm_type,
                    f'%"fire_id": "{fire_id}"%',
                ),
            )
        except Exception as e:
            # SQLite without json_patch is a real possibility; degrade
            # to a simple resolution stamp.
            try:
                await self._db.execute(
                    "UPDATE alarm_fires SET resolved_at = ?, "
                    "resolution = ? "
                    "WHERE alarm_type = ? AND metadata LIKE ?",
                    (
                        ts, resolution, alarm_type,
                        f'%"fire_id": "{fire_id}"%',
                    ),
                )
            except Exception as e2:
                logger.debug(
                    f"[AlarmStore] record_resolution failed: {e2}"
                )
            else:
                _ = e  # silence linter

    async def record_state(
        self,
        alarm_type: str,
        state: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """Upsert the current state for an alarm_type into alarm_state.
        One row per type — the dashboard reads this to render the
        per-alarm banner without hitting the audit log."""
        try:
            await self._db.execute(
                "INSERT INTO alarm_state (alarm_type, state, state_since, metadata) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(alarm_type) DO UPDATE SET "
                "state = excluded.state, "
                "state_since = excluded.state_since, "
                "metadata = excluded.metadata",
                (
                    alarm_type, state, _now_iso(),
                    json.dumps(metadata or {}),
                ),
            )
        except Exception as e:
            logger.debug(f"[AlarmStore] record_state failed: {e}")

    async def recent_fires(
        self, alarm_type: Optional[str] = None, limit: int = 50,
    ) -> list[dict]:
        """Read the last N alarm_fires rows. Used by the dashboard's
        alarm-history panel."""
        try:
            if alarm_type:
                rows = await self._db.fetchall(
                    "SELECT id, alarm_type, fired_at, resolved_at, "
                    "resolution, metadata FROM alarm_fires "
                    "WHERE alarm_type = ? ORDER BY fired_at DESC LIMIT ?",
                    (alarm_type, limit),
                )
            else:
                rows = await self._db.fetchall(
                    "SELECT id, alarm_type, fired_at, resolved_at, "
                    "resolution, metadata FROM alarm_fires "
                    "ORDER BY fired_at DESC LIMIT ?",
                    (limit,),
                )
            return [dict(r) for r in rows]
        except Exception as e:
            logger.debug(f"[AlarmStore] recent_fires failed: {e}")
            return []


class NullAlarmStore:
    """No-op store for tests / smoke runs without a DB."""

    async def record_fire(self, *a, **kw) -> None: pass
    async def record_resolution(self, *a, **kw) -> None: pass
    async def record_state(self, *a, **kw) -> None: pass
    async def recent_fires(self, *a, **kw) -> list[dict]: return []
