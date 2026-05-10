"""
JARVIS — Ambient Home AI
========================
Mission: Persist a per-room door graph so Jarvis can reason about how rooms
         connect and where the user is heading when they walk out of frame.
         The user teaches it interactively — they stand near a door, point
         at it, and say "this door goes to the kitchen". The orchestrator
         captures the pointing wrist's normalized frame coordinates and
         calls DoorMap.add_door(); subsequent teach calls just append more
         doors per room.

         Schema is intentionally tiny: rooms map to a list of door dicts,
         each with a label, the neighbor room id (resolved against known
         rooms in config.yaml), and a normalized (fx, fy) frame coordinate
         the vision loop can later watch for transit events.

         The graph is bidirectional in spirit but stored unidirectionally.
         Saying "this door in the office goes to the kitchen" creates a
         door under office only — the reverse "kitchen → office" is taught
         separately (different camera, different frame coords). A future
         transit-inference pass can pair them up by neighbor_room match.

Modules: modules/layout/door_map.py
Classes: DoorEntry, DoorMap

#todo: Reverse-link auto-suggestion — when the office side is taught and
       the kitchen camera also has a person in frame near the same wall
       direction, prompt "should this be the office door from the kitchen
       side?" instead of waiting for a separate teach.
#todo: Confidence + source field per door. Today every door is treated
       equal; if vision auto-detects a likely doorway later we'll want to
       distinguish "user-taught" from "auto-detected" so user-taught wins
       merges.
"""
from __future__ import annotations

import asyncio
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from loguru import logger


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DoorEntry(dict):
    """Typed dict-style wrapper. Stays a plain dict for trivial JSON
    round-tripping; the subclass exists for IDE autocomplete only.

    Keys:
        id            — short hex id, stable across saves so the dashboard
                        can render selection state
        label         — the natural-language name the user gave it
                        ("kitchen door", "the door to the bedroom")
        neighbor_room — resolved room id (matches an entry in config.yaml's
                        rooms[]) or None if the user named a room we don't
                        know about yet
        fx, fy        — normalized frame coordinates (0-1) of the door's
                        centroid in the room's primary camera. Default 0.5,
                        0.5 (center) when no pointing gesture was detected.
        created_at    — ISO 8601 UTC
        updated_at    — ISO 8601 UTC
    """


class DoorMap:
    """In-memory + JSON-persisted house door graph.

    Reads are sync and lock-free. Writes go through an asyncio lock so
    rapid-fire teach calls don't half-write the file. The on-disk schema
    matches the in-memory structure — no migrations needed yet.
    """

    SCHEMA_VERSION = 1

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        # Top-level: {"version": int, "rooms": {room_id: {"doors": [...]}}}
        self._data: dict[str, Any] = {"version": self.SCHEMA_VERSION, "rooms": {}}
        self._lock = asyncio.Lock()
        self._load()

    # ── Read API ─────────────────────────────────────────────────────────────

    def get_doors(self, room: str) -> list[DoorEntry]:
        """All doors in `room`, oldest first. Returns a copy so callers
        can't mutate the in-memory store."""
        room_block = self._data.get("rooms", {}).get(room, {})
        doors = room_block.get("doors", [])
        return [DoorEntry(d) for d in doors]

    def find_by_label(self, room: str, label_substr: str) -> Optional[DoorEntry]:
        """First door in `room` whose label contains `label_substr` (case
        insensitive). Used by the "forget the kitchen door" voice path."""
        needle = label_substr.lower().strip()
        if not needle:
            return None
        for door in self.get_doors(room):
            if needle in str(door.get("label", "")).lower():
                return door
            neigh = str(door.get("neighbor_room") or "").lower()
            if neigh and neigh in needle:
                return door
        return None

    def all_rooms(self) -> list[str]:
        """Every room id with at least one taught door."""
        return list(self._data.get("rooms", {}).keys())

    def snapshot(self) -> dict[str, Any]:
        """Full graph snapshot — for the dashboard / debug dumps. Deep-ish
        copy via json round-trip is overkill; nested lists/dicts are tiny
        and readers should treat the result as read-only."""
        return json.loads(json.dumps(self._data))

    # ── Write API ────────────────────────────────────────────────────────────

    async def add_door(
        self,
        room: str,
        label: str,
        neighbor_room: Optional[str] = None,
        fx: float = 0.5,
        fy: float = 0.5,
    ) -> DoorEntry:
        """Append a new door to `room`. Returns the new entry. Coordinates
        are clamped to [0, 1] — the wrist landmark can drift slightly
        outside the frame on edge poses."""
        entry = DoorEntry(
            id=secrets.token_hex(4),
            label=label.strip(),
            neighbor_room=neighbor_room,
            fx=max(0.0, min(1.0, float(fx))),
            fy=max(0.0, min(1.0, float(fy))),
            created_at=_now_iso(),
            updated_at=_now_iso(),
        )
        async with self._lock:
            rooms = self._data.setdefault("rooms", {})
            block = rooms.setdefault(room, {"doors": []})
            block.setdefault("doors", []).append(dict(entry))
            await asyncio.to_thread(self._save)
        logger.info(
            f"[DoorMap] Added door in '{room}': '{entry['label']}' "
            f"→ {neighbor_room or '?'} @ ({entry['fx']:.2f}, {entry['fy']:.2f})"
        )
        return entry

    async def remove_door(self, room: str, door_id: str) -> bool:
        """Delete by id. Returns True if a row was removed."""
        async with self._lock:
            block = self._data.get("rooms", {}).get(room)
            if not block:
                return False
            doors = block.get("doors", [])
            new_doors = [d for d in doors if d.get("id") != door_id]
            if len(new_doors) == len(doors):
                return False
            block["doors"] = new_doors
            await asyncio.to_thread(self._save)
        logger.info(f"[DoorMap] Removed door {door_id} from '{room}'")
        return True

    async def clear_room(self, room: str) -> int:
        """Forget every door in `room`. Returns the number removed."""
        async with self._lock:
            block = self._data.get("rooms", {}).get(room)
            if not block:
                return 0
            n = len(block.get("doors", []))
            block["doors"] = []
            await asyncio.to_thread(self._save)
        logger.info(f"[DoorMap] Cleared {n} door(s) from '{room}'")
        return n

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(
                f"[DoorMap] {self._path} unreadable ({e}); starting empty"
            )
            return
        if not isinstance(data, dict):
            return
        # Ignore unknown future schema versions silently rather than
        # crashing — a v1 build seeing v2 should still come up; the
        # dashboard will surface the mismatch separately if we ever
        # add a v2.
        rooms = data.get("rooms")
        if isinstance(rooms, dict):
            self._data = {
                "version": int(data.get("version", self.SCHEMA_VERSION)),
                "rooms": {
                    rid: {"doors": list(b.get("doors", []))}
                    for rid, b in rooms.items()
                    if isinstance(b, dict)
                },
            }
            n = sum(len(b["doors"]) for b in self._data["rooms"].values())
            logger.info(
                f"[DoorMap] Loaded {n} door(s) across "
                f"{len(self._data['rooms'])} room(s) from {self._path}"
            )

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(self._path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(self._data, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            tmp.replace(self._path)
        except Exception as e:
            logger.warning(f"[DoorMap] Save to {self._path} failed: {e}")
