"""
JARVIS — Ambient Home AI
========================
Mission: Track and retrieve per-room visual baselines — descriptions of what
         each room normally looks like. When the vision system describes a room,
         it compares against the baseline to detect anomalies (mess, missing items,
         rearrangement) worth noting. Baselines are updated periodically and stored
         in SQLite.

Modules: modules/memory/room_baselines.py
Classes: RoomBaselines
Functions:
    RoomBaselines.__init__(db, config) — Initialize with DB and config
    RoomBaselines.update(room, desc)   — Store a new baseline description
    RoomBaselines.get(room)            — Get baseline description for a room
    RoomBaselines.get_all()            — Get all room baselines as a dict
    RoomBaselines.needs_update(room)   — True if baseline is stale/missing

Variables:
    RoomBaselines._db                   — DatabaseManager reference
    RoomBaselines._update_interval_hours — From config memory.room_baseline_update_hours

#todo: Add anomaly scoring (compare current description to baseline via LLM similarity)
#todo: Add visual diff highlights for the dashboard (what changed vs baseline)
#todo: Support multiple baselines per room (day/night, clean/in-use variants)
#todo: Auto-generate baseline on first camera capture of a room
"""

from datetime import datetime, timedelta
from typing import Optional

from loguru import logger

from modules.memory.database import DatabaseManager


class RoomBaselines:
    """
    Stores and retrieves room visual baseline descriptions.

    A baseline is a short natural-language description of the room's
    normal appearance (captured by the vision/scene analyzer). Used to
    give Jarvis a reference point for anomaly detection.
    """

    def __init__(self, db: DatabaseManager, config: dict) -> None:
        self._db = db
        self._update_interval_hours: float = config["memory"].get(
            "room_baseline_update_hours", 24
        )

    async def update(self, room: str, description: str) -> None:
        """
        Store or overwrite the baseline description for a room.

        Args:
            room:        Room identifier string.
            description: Natural-language description from the vision system.
        """
        now = datetime.now().isoformat()
        await self._db.execute(
            """
            INSERT INTO room_baselines (room, baseline_desc, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(room) DO UPDATE SET
                baseline_desc = excluded.baseline_desc,
                updated_at    = excluded.updated_at
            """,
            (room, description.strip(), now),
        )
        logger.info(f"[Baselines] Updated baseline for '{room}'")

    async def get(self, room: str) -> Optional[str]:
        """
        Retrieve the baseline description for a room.
        Returns None if no baseline exists yet.
        """
        row = await self._db.fetchone(
            "SELECT baseline_desc FROM room_baselines WHERE room=?",
            (room,),
        )
        return row["baseline_desc"] if row else None

    async def get_all(self) -> dict[str, str]:
        """
        Return all room baselines as {room_id: description} dict.
        Rooms with no baseline are not included.
        """
        rows = await self._db.fetchall("SELECT room, baseline_desc FROM room_baselines")
        return {row["room"]: row["baseline_desc"] for row in rows if row["baseline_desc"]}

    async def needs_update(self, room: str) -> bool:
        """
        Return True if the baseline for this room is missing or older than
        the configured update interval.
        """
        row = await self._db.fetchone(
            "SELECT updated_at FROM room_baselines WHERE room=?",
            (room,),
        )
        if not row or not row["updated_at"]:
            return True

        try:
            last_update = datetime.fromisoformat(row["updated_at"])
            age_hours = (datetime.now() - last_update).total_seconds() / 3600
            return age_hours >= self._update_interval_hours
        except (ValueError, TypeError):
            return True
