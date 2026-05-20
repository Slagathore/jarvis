"""
JARVIS — World Model / Patterns
===============================
Mission: PatternMiner — the nightly job that builds a behavioral profile
         for each resident from the rolling 30-day world-event log
         (audit roadmap §25). The profile is a set of HISTOGRAMS — not
         single numbers — because AnomalyScorer needs the full shape of
         "when/where does this person usually do things", not just a mean.

         Profiles are stored on `entity.metadata['pattern_profile']`.
         People only — pets get their own behavioral profile elsewhere.

         "Anna usually gets home by 6 PM, it's 11 PM and she hasn't" is
         meaningfully more useful than "Anna is not home." The first is
         concern; the second is just data. This module produces the
         distributions that make the first sentence possible.

Modules: modules/world_model/patterns.py
Classes: PatternMiner
"""

from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

from loguru import logger

from modules.world_model.event_windows import RoomTimeIndex

_PRESENT_EVENTS = ("reappeared", "moved_to", "first_seen")
# Safety cap for the one-shot "all present events in the window" query
# used by _co_presence. Covers ~30 days of household event volume with
# headroom; if it's ever exceeded we lose oldest-first via the DESC LIMIT
# in store.search_events -- acceptable degradation for a 30-day profile.
_PRESENT_QUERY_LIMIT = 250000


def _parse_ts(value: Any) -> datetime:
    """Parse a stored ISO timestamp to an aware UTC datetime."""
    try:
        dt = datetime.fromisoformat(str(value))
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


class PatternMiner:
    """Nightly behavioral-profile builder for residents."""

    def __init__(self, world: Any, days_back: int = 30) -> None:
        self.world = world
        self.days_back = int(days_back)

    async def run_nightly(self) -> int:
        """Rebuild the pattern profile once per distinct resident.
        Returns the number of residents whose profile was rebuilt."""
        try:
            async with self.world._lock:
                residents = [
                    e for e in self.world.entities.values()
                    if getattr(e, "is_resident", False)
                    and getattr(e, "entity_type", None) == "person"
                    and getattr(e, "person_id", None)
                    and getattr(e, "archived_at", None) is None
                ]
        except Exception as e:
            logger.warning(f"[PatternMiner] resident scan failed: {e}")
            return 0

        # The world model accumulates many entities per resident — one per
        # tracking session. A profile is derived purely from person_id, so
        # build it ONCE per distinct resident and fan the result out to
        # every entity sharing that id. Iterating raw entities instead
        # meant an N-session household ran N full 100k-event queries —
        # minutes of DB hammering that starved the rest of the system.
        by_person: dict[Any, list] = {}
        for ent in residents:
            by_person.setdefault(ent.person_id, []).append(ent)

        rebuilt = 0
        for _person_id, ents in by_person.items():
            try:
                profile = await self._build_profile_for(ents[0])
                ts = datetime.now(timezone.utc).isoformat()
                for ent in ents:
                    ent.metadata["pattern_profile"] = profile
                    ent.metadata["pattern_profile_updated_ts"] = ts
                    await self.world.store.upsert_entity(ent)
                rebuilt += 1
                logger.info(
                    f"[PatternMiner] rebuilt profile for "
                    f"{ents[0].display_name}: "
                    f"{profile.get('n_events', 0)} events analyzed "
                    f"({len(ents)} entit{'y' if len(ents) == 1 else 'ies'})"
                )
            except Exception:
                logger.exception(
                    f"[PatternMiner] profile build failed for "
                    f"{getattr(ents[0], 'display_name', '?')}"
                )
        return rebuilt

    async def _build_profile_for(self, ent: Any) -> dict[str, Any]:
        since = datetime.now(timezone.utc) - timedelta(days=self.days_back)
        events = await self.world.store.search_events(
            person_id=ent.person_id, since=since, limit=100000,
        )
        if not events:
            return {"n_events": 0, "window_start": since.isoformat()}
        # _co_presence does its own (one) windowed DB query; everything
        # else is pure CPU and gets offloaded so we never block the
        # event loop for more than the DB round-trip itself, even with
        # 100k events.
        co_presence = await self._co_presence(ent, events)
        return await asyncio.to_thread(
            self._build_histograms, events, since, co_presence,
        )

    def _build_histograms(
        self,
        events: list[dict],
        since: datetime,
        co_presence: dict[str, float],
    ) -> dict[str, Any]:
        """Pure-CPU profile assembly. Called via asyncio.to_thread so the
        per-pass histogram work never touches the event loop."""
        return {
            "n_events": len(events),
            "window_start": since.isoformat(),
            "window_end": datetime.now(timezone.utc).isoformat(),
            "arrival_by_weekday": self._arrival_distribution(events),
            "departure_by_weekday": self._departure_distribution(events),
            "room_by_weekday_hour": self._room_by_weekday_hour(events),
            "co_presence": co_presence,
            "morning_routine": self._morning_routine(events),
            "long_stays_by_room": self._long_stays_by_room(events),
            "weekly_active_hours": self._weekly_active_hours(events),
        }

    # ── arrivals / departures ────────────────────────────────────────────────

    def _arrival_distribution(self, events: list[dict]) -> dict[int, dict[int, int]]:
        """{weekday: {hour: count}} for REAPPEARED-after-DEPARTED — i.e.
        when this person comes home, by weekday."""
        out: dict[int, dict[int, int]] = {wd: defaultdict(int) for wd in range(7)}
        prev_state = None
        for e in sorted(events, key=lambda x: x["ts"]):
            if prev_state == "departed" and e.get("event_type") == "reappeared":
                t = _parse_ts(e["ts"])
                out[t.weekday()][t.hour] += 1
            prev_state = e.get("state")
        return {wd: dict(d) for wd, d in out.items()}

    def _departure_distribution(self, events: list[dict]) -> dict[int, dict[int, int]]:
        out: dict[int, dict[int, int]] = {wd: defaultdict(int) for wd in range(7)}
        for e in events:
            if e.get("event_type") == "departed":
                t = _parse_ts(e["ts"])
                out[t.weekday()][t.hour] += 1
        return {wd: dict(d) for wd, d in out.items()}

    # ── room-by-time ─────────────────────────────────────────────────────────

    def _room_by_weekday_hour(self, events: list[dict]) -> dict:
        """{weekday: {hour: {room: probability}}}, normalized per bucket."""
        bucket: dict[int, dict[int, dict[str, int]]] = {
            wd: {h: defaultdict(int) for h in range(24)} for wd in range(7)
        }
        for e in events:
            r = e.get("room")
            if not r:
                continue
            t = _parse_ts(e["ts"])
            bucket[t.weekday()][t.hour][r] += 1
        out: dict = {}
        for wd, by_hour in bucket.items():
            out[wd] = {}
            for h, counts in by_hour.items():
                total = sum(counts.values()) or 1
                out[wd][h] = {r: c / total for r, c in counts.items()}
        return out

    # ── co-presence ──────────────────────────────────────────────────────────

    async def _co_presence(self, ent: Any, my_events: list[dict]) -> dict[str, float]:
        """{other_resident_name: avg matching events per my present-window}.

        Same semantic as the original: for each of my present-events
        in (room, ts), count how many OTHER people's present-events
        fall within +/-2 min in the same room, then divide by the
        count of my windows. (Can exceed 1.0 -- it's a count ratio,
        not a probability.)

        Mechanics: one windowed query for every PRESENT-event in the
        analysis window, then an in-memory bisect via RoomTimeIndex.
        The previous shape was one DB query per of-my-events and was
        the dominant share of the boot-time DB freeze -- a resident
        with 30k present-events fired 30k sequential round-trips
        through the single shared aiosqlite connection.
        """
        my_present = [
            (_parse_ts(e["ts"]), e.get("room"))
            for e in my_events
            if e.get("event_type") in _PRESENT_EVENTS and e.get("room")
        ]
        if not my_present:
            return {}
        since = datetime.now(timezone.utc) - timedelta(days=self.days_back)
        all_present = await self.world.store.search_events(
            event_types=list(_PRESENT_EVENTS),
            since=since,
            limit=_PRESENT_QUERY_LIMIT,
        )
        index = RoomTimeIndex(all_present)
        overlaps: dict[str, int] = defaultdict(int)
        for ts, room in my_present:
            for w in index.window(room, ts, 120.0):
                if (w.get("entity_type") == "person"
                        and w.get("entity_id") != ent.id
                        and w.get("entity_name")):
                    overlaps[w["entity_name"]] += 1
        n = len(my_present)
        return {name: c / n for name, c in overlaps.items()}

    # ── morning routine ──────────────────────────────────────────────────────

    def _morning_routine(self, events: list[dict]) -> dict[str, Any]:
        """The typical ordered room sequence in the first ~3 h of each day."""
        by_day: dict[str, list[tuple[datetime, str]]] = defaultdict(list)
        for e in events:
            if e.get("event_type") in _PRESENT_EVENTS and e.get("room"):
                t = _parse_ts(e["ts"])
                by_day[t.strftime("%Y-%m-%d")].append((t, e["room"]))
        sequences: list[tuple[str, ...]] = []
        for entries in by_day.values():
            entries.sort(key=lambda x: x[0])
            if not entries:
                continue
            cutoff = entries[0][0] + timedelta(hours=3)
            seq: list[str] = []
            seen: set[str] = set()
            for t, r in entries:
                if t > cutoff:
                    break
                if r not in seen:
                    seq.append(r)
                    seen.add(r)
            if len(seq) >= 2:
                sequences.append(tuple(seq))
        if not sequences:
            return {}
        return {
            "most_common_sequences": [
                {"sequence": list(s), "count": c}
                for s, c in Counter(sequences).most_common(3)
            ],
            "n_days_analyzed": len(by_day),
        }

    def _long_stays_by_room(self, events: list[dict]) -> dict[str, dict[str, float]]:
        per_room: dict[str, int] = defaultdict(int)
        for e in events:
            if e.get("event_type") == "stationary_long" and e.get("room"):
                per_room[e["room"]] += 1
        return {
            r: {"n": n, "rate_per_day": n / max(self.days_back, 1)}
            for r, n in per_room.items()
        }

    def _weekly_active_hours(self, events: list[dict]) -> dict[int, list[int]]:
        """Sorted hours per weekday in which this person was observed at all."""
        out: dict[int, set] = {wd: set() for wd in range(7)}
        for e in events:
            t = _parse_ts(e["ts"])
            out[t.weekday()].add(t.hour)
        return {wd: sorted(s) for wd, s in out.items()}
