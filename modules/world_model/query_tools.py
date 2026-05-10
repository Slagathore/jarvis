"""
JARVIS — World Model
====================
Mission: LLM-facing tool functions for World Model queries. Registered
         into the orchestrator's tool registry alongside the existing
         calendar / vision / memory tools; the brain LLM picks them
         naturally based on the user's question.

         Four entry points, all async, all returning JSON-serializable
         dicts/lists:
            get_entity_status(name)         — "where is Cole right now?"
            list_entities_in_room(room)     — "who's in the kitchen?"
            who_is_home()                   — "is anyone home?"
            search_recent_events(...)       — "what happened in the office today?"

         These read from the in-memory WorldModel registry (cheap) and,
         for `search_recent_events`, the persistent event log via
         WorldStore. They never mutate state.

Modules: modules/world_model/query_tools.py
Classes: WorldQueryTools
Spec:    new 2.md §20 (Query Layer).

#todo: Add `get_recent_movements(name, hours_ago)` — a higher-level
       narrative reducer over MOVED_TO + LOST_VISIBILITY + REAPPEARED
       events that summarises the path "Cole was in office → kitchen →
       living_room over the last 2 hours". Useful for the persona's
       "what's been going on" question pattern.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from modules.world_model.world_model import WorldModel


class WorldQueryTools:
    """Read-only query surface over WorldModel. Constructed once, lives
    as long as the orchestrator does. Stateless — every call hits the
    underlying WorldModel."""

    def __init__(self, world: WorldModel) -> None:
        self.world = world

    async def get_entity_status(self, name: str) -> dict:
        """
        Where is X right now? Use for 'where is Cole', 'is Anna home',
        'is Spooky in the bedroom', etc.

        Returns a dict with `found: bool` so the LLM can branch cleanly.
        State-specific extras (`departed_via`, `entered_unmonitored_via`)
        are included unconditionally — they're None when not applicable
        and let the LLM produce natural phrasing without follow-up tool
        calls.
        """
        ent = self.world.find_entity_by_name(name)
        if not ent:
            return {
                "found": False,
                "message": f"No entity named {name} in registry.",
            }
        elapsed = datetime.utcnow() - ent.last_state_change_ts
        return {
            "found": True,
            "name": ent.display_name,
            "type": ent.entity_type,
            "state": ent.state.value,
            "last_seen_room": ent.last_seen_room,
            "last_seen_camera": ent.last_seen_camera,
            "last_seen_landmark": ent.last_seen_landmark,
            "last_seen_ts": (
                ent.last_seen_ts.isoformat() if ent.last_seen_ts else None
            ),
            "duration_in_state_seconds": int(elapsed.total_seconds()),
            "confidence": ent.confidence,
            "attribution_confidence": ent.last_attribution_confidence,
            "is_resident": ent.is_resident,
            "departed_via": ent.metadata.get("departed_via"),
            "departed_ts": ent.metadata.get("departed_ts"),
            "entered_unmonitored_via": ent.metadata.get("entered_unmonitored_via"),
            "last_event": await self.world.most_recent_event(ent.id),
        }

    async def list_entities_in_room(self, room: str) -> list[dict]:
        """Roster of who/what's in a room right now (PRESENT only).
        Cats and objects show up alongside people — the LLM filters by
        `type` if it cares ('who's' = person filter, 'what's' = wider).
        """
        return [
            {
                "name": e.display_name or f"unknown_{e.entity_type}",
                "type": e.entity_type,
                "state": e.state.value,
                "confidence": e.confidence,
            }
            for e in self.world.entities.values()
            if e.last_seen_room == room and e.state.value == "present"
        ]

    async def who_is_home(self) -> list[dict]:
        """List residents currently considered 'home' — any in-house
        state (PRESENT, IN_ROOM_UNSEEN, TRANSITIONING, IN_HOUSE_UNMONITORED).
        Excludes DEPARTED and UNKNOWN_AT_BOOT. Visitors / non-residents
        and unnamed entities are excluded.
        """
        in_house_states = {
            "present", "in_room_unseen", "transitioning", "in_house_unmonitored",
        }
        return [
            {
                "name": e.display_name,
                "state": e.state.value,
                "last_room": e.last_seen_room,
            }
            for e in self.world.entities.values()
            if e.is_resident and e.display_name
            and e.state.value in in_house_states
        ]

    async def search_recent_events(
        self,
        entity_name: Optional[str] = None,
        room: Optional[str] = None,
        event_types: Optional[list[str]] = None,
        hours_ago: int = 24,
        limit: int = 20,
    ) -> list[dict]:
        """Search the entity event log. All filters are optional and
        AND-combined. Returns rows in DESC ts order.
        """
        entity_id: Optional[str] = None
        if entity_name:
            ent = self.world.find_entity_by_name(entity_name)
            if not ent:
                return []
            entity_id = ent.id
        since = datetime.utcnow() - timedelta(hours=hours_ago)
        return await self.world.store.search_events(
            entity_id=entity_id,
            room=room,
            event_types=event_types,
            since=since,
            limit=limit,
        )
