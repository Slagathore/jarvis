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

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from modules.world_model.world_model import WorldModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


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
        elapsed = _utcnow() - _as_utc(ent.last_state_change_ts)
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
        since = _utcnow() - timedelta(hours=hours_ago)
        return await self.world.store.search_events(
            entity_id=entity_id,
            room=room,
            event_types=event_types,
            since=since,
            limit=limit,
        )

    # ── Pet-aware queries (§22) ────────────────────────────────────────────

    async def list_pets(
        self,
        species: Optional[str] = None,
        include_archived: bool = False,
    ) -> list[dict]:
        """List resident pets — every cat and/or dog the world model
        knows about. `species` filters to "cat" or "dog"; default is
        both. `include_archived` defaults False so rehomed/passed pets
        don't pollute current-state queries.
        """
        out: list[dict] = []
        for e in self.world.entities.values():
            if e.entity_type not in ("cat", "dog"):
                continue
            if species and e.entity_type != species:
                continue
            if e.archived_at is not None and not include_archived:
                continue
            if not e.is_resident:
                continue
            out.append({
                "name": e.display_name,
                "species": e.entity_type,
                "owner_person_id": e.household_owner_id,
                "state": e.state.value,
                "last_seen_room": e.last_seen_room,
                "unmonitored_home": e.unmonitored_home_room,
                "is_archived": e.archived_at is not None,
            })
        return sorted(out, key=lambda d: (d["species"], d["name"] or ""))

    async def where_is_pet(self, name: str) -> dict:
        """Pet-flavored variant of get_entity_status. Includes
        unmonitored_home so 'where's Velcro?' resolves cleanly to
        'jeff_room (no camera)' even when no recent observation exists.
        """
        ent = self.world.find_entity_by_name(name)
        if not ent or ent.entity_type not in ("cat", "dog"):
            return {"found": False,
                    "message": f"No pet named {name} in registry."}
        elapsed = _utcnow() - _as_utc(ent.last_state_change_ts)
        unmon = ent.unmonitored_home_room
        # If we haven't seen the pet AND it has an unmonitored home,
        # surface that as the most likely location.
        likely_room = ent.last_seen_room
        is_likely_inferred = False
        if (ent.state.value in ("in_house_unmonitored", "in_room_unseen")
                and unmon and not likely_room):
            likely_room = unmon
            is_likely_inferred = True
        return {
            "found": True,
            "name": ent.display_name,
            "species": ent.entity_type,
            "state": ent.state.value,
            "last_seen_room": ent.last_seen_room,
            "last_seen_landmark": ent.last_seen_landmark,
            "last_seen_ts": (
                ent.last_seen_ts.isoformat() if ent.last_seen_ts else None
            ),
            "duration_in_state_seconds": int(elapsed.total_seconds()),
            "likely_room": likely_room,
            "likely_room_inferred": is_likely_inferred,
            "unmonitored_home": unmon,
            "owner_person_id": ent.household_owner_id,
            "is_archived": ent.archived_at is not None,
        }

    # ── §23.7 find_object — CLIP text-query over tracked objects ──────────

    async def find_object(
        self,
        description: str,
        k: int = 3,
    ) -> dict:
        """Embed `description` with CLIP and return the top-k most-
        similar tracked objects. Hedges when the top similarity is
        borderline so the LLM phrases the answer as a guess instead
        of an assertion. Returns {found: false} when no encoder is
        wired or no entities have visual embeddings — the LLM should
        then say something like 'I don't have a tracked object that
        looks like X'."""
        encoder = getattr(self.world, "clip_encoder", None)
        if encoder is None:
            return {
                "found": False,
                "message": (
                    "Object visual search isn't available "
                    "(CLIP encoder not loaded)."
                ),
            }
        try:
            text_emb = encoder.encode_text(description)
        except Exception as e:
            return {
                "found": False,
                "message": f"CLIP text encode failed: {e}",
            }
        if text_emb is None:
            return {
                "found": False,
                "message": (
                    "Object visual search isn't available "
                    "(CLIP encoder is a no-op stub)."
                ),
            }

        import numpy as _np
        candidates: list[tuple[float, Any]] = []
        for e in self.world.entities.values():
            if e.entity_type != "object":
                continue
            if e.metadata.get("pruned"):
                continue
            emb = e.metadata.get("_visual_embedding")
            if emb is None:
                continue
            sim = float(
                _np.dot(text_emb, emb)
                / (_np.linalg.norm(text_emb)
                   * _np.linalg.norm(emb) + 1e-9)
            )
            candidates.append((sim, e))
        candidates.sort(key=lambda x: x[0], reverse=True)

        threshold = float(
            (self.world.cfg or {}).get("clip_match_threshold", 0.25)
        )
        if not candidates or candidates[0][0] < threshold:
            return {
                "found": False,
                "message": (
                    f"I don't have a tracked object that looks like "
                    f"'{description}'."
                ),
                "checked_entities": len(candidates),
            }
        top = candidates[: max(1, int(k))]
        primary_sim, primary = top[0]
        return {
            "found": True,
            "name": (
                primary.display_name
                or primary.metadata.get("detected_class")
                or "object"
            ),
            "last_seen_room": primary.last_seen_room,
            "last_seen_landmark": primary.last_seen_landmark,
            "last_seen_ts": (
                primary.last_seen_ts.isoformat()
                if primary.last_seen_ts else None
            ),
            "match_similarity": primary_sim,
            "alternatives": [
                {
                    "name": (
                        e.display_name
                        or e.metadata.get("detected_class")
                        or "object"
                    ),
                    "room": e.last_seen_room,
                    "similarity": s,
                }
                for s, e in top[1:]
            ],
            # If primary similarity is borderline, the LLM should
            # phrase the answer with explicit hedge.
            "hedge": primary_sim < float(
                (self.world.cfg or {}).get("clip_hedge_threshold", 0.32)
            ),
        }

    # ── §24.5 Interaction queries ──────────────────────────────────────────

    async def what_did_someone_do_with(
        self,
        person_name: str,
        object_name: Optional[str] = None,
        hours_ago: int = 24,
    ) -> list[dict]:
        """'What did Cole do with the wallet?' — chronological list of
        INTERACTED_WITH / PICKED_UP / PLACED_DOWN events involving the
        named person, optionally filtered to one object. Oldest first
        so the LLM can phrase it as a narrative."""
        person_ent = self.world.find_entity_by_name(person_name)
        if not person_ent:
            return []
        since = _utcnow() - timedelta(hours=hours_ago)
        events = await self.world.store.search_events(
            person_id=person_ent.person_id,
            event_types=["interacted_with", "picked_up", "placed_down"],
            since=since,
            limit=200,
        )
        if object_name:
            obj_ent = self.world.find_entity_by_name(object_name)
            obj_id = obj_ent.id if obj_ent else None
            if obj_id is not None:
                # Metadata is stored as a JSON string on the event row;
                # decode lazily and filter by object_id. Match against
                # entity_id too for events emitted by InteractionMonitor
                # where the object is the primary entity.
                import json as _json
                filtered: list[dict] = []
                for e in events:
                    if e.get("entity_id") == obj_id:
                        filtered.append(e); continue
                    raw = e.get("metadata")
                    meta = (
                        raw if isinstance(raw, dict)
                        else (
                            _json.loads(raw)
                            if isinstance(raw, str) and raw else {}
                        )
                    )
                    if meta.get("object_id") == obj_id:
                        filtered.append(e)
                events = filtered
            else:
                # Object name not in registry — fall back to a name-only
                # text match against metadata.object_name.
                import json as _json
                obj_lc = object_name.lower()
                filtered = []
                for e in events:
                    raw = e.get("metadata")
                    meta = (
                        raw if isinstance(raw, dict)
                        else (
                            _json.loads(raw)
                            if isinstance(raw, str) and raw else {}
                        )
                    )
                    name = (meta.get("object_name") or "").lower()
                    if name and obj_lc in name:
                        filtered.append(e)
                events = filtered
        return list(reversed(events))

    async def who_last_touched(self, object_name: str) -> dict:
        """'Who last touched my wallet?' — most recent PICKED_UP /
        PLACED_DOWN / INTERACTED_WITH event for this object. Returns
        {found, object_name, event_type, ts, person_name, room}."""
        obj_ent = self.world.find_entity_by_name(object_name)
        if not obj_ent:
            return {"found": False,
                    "message": f"No object named {object_name} in registry."}
        events = await self.world.store.search_events(
            entity_id=obj_ent.id,
            event_types=["picked_up", "placed_down", "interacted_with"],
            limit=1,
        )
        if not events:
            return {
                "found": False,
                "message": f"No interaction events for {object_name}.",
            }
        e = events[0]
        raw = e.get("metadata")
        if isinstance(raw, dict):
            meta = raw
        elif isinstance(raw, str) and raw:
            import json as _json
            try:
                meta = _json.loads(raw)
            except Exception:
                meta = {}
        else:
            meta = {}
        return {
            "found": True,
            "object_name": object_name,
            "event_type": e["event_type"],
            "ts": e["ts"],
            "person_name": meta.get("person_name") or e.get("entity_name"),
            "person_id": meta.get("person_id") or e.get("person_id"),
            "room": e.get("room"),
        }

    async def pet_care_summary(
        self, name: str, hours_ago: int = 24,
    ) -> dict:
        """§22.9 — answer 'has Spooky used the litterbox today?',
        'when did Summer last eat?'. Reads INTERACTED_WITH events for
        the named pet within `hours_ago` and groups by interaction_kind.

        Returns a dict {interaction_kind: {count, last_ts, last_room}}
        plus the pet's current state and a `found` flag.
        """
        ent = self.world.find_entity_by_name(name)
        if not ent or ent.entity_type not in ("cat", "dog"):
            return {"found": False,
                    "message": f"No pet named {name} in registry."}
        since = _utcnow() - timedelta(hours=hours_ago)
        events = await self.world.store.search_events(
            entity_id=ent.id,
            event_types=["interacted_with"],
            since=since,
            limit=200,
        )
        # Aggregate by interaction_kind. Events stored metadata as JSON
        # strings — decode and pick out the kind.
        import json
        by_kind: dict[str, dict] = {}
        for ev in events:
            raw_meta = ev.get("metadata")
            if isinstance(raw_meta, str) and raw_meta:
                try:
                    meta = json.loads(raw_meta)
                except Exception:
                    meta = {}
            elif isinstance(raw_meta, dict):
                meta = raw_meta
            else:
                meta = {}
            kind = meta.get("interaction_kind")
            if not kind:
                continue
            slot = by_kind.setdefault(kind, {
                "count": 0, "last_ts": None, "last_room": None,
                "last_landmark": meta.get("landmark"),
            })
            slot["count"] += 1
            ts = ev.get("ts")
            # Events are returned DESC, so the first hit per kind is
            # automatically the most recent — only stamp once.
            if slot["last_ts"] is None and ts:
                slot["last_ts"] = ts
                slot["last_room"] = ev.get("room")
                slot["last_landmark"] = meta.get("landmark")
        return {
            "found": True,
            "name": ent.display_name,
            "species": ent.entity_type,
            "state": ent.state.value,
            "window_hours": hours_ago,
            "by_kind": by_kind,
        }
