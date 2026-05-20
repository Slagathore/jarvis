"""
JARVIS -- World Model / event-window co-presence index
=======================================================
Mission: a shared in-memory co-presence helper for the nightly
         profile builders.

PatternMiner (people) and BehavioralProfileBuilder (pets) both answer
the same question many times per nightly pass: "for one of my
PRESENT-events at time T in room R, which OTHER entities were in R
within +/- W seconds of T?" The naive shape is one
`search_events(room=R, since=T-W, until=T+W, ...)` query per event of
mine.

That shape is what froze Jarvis at boot. A resident with 30k
present-events in the 30-day window fired 30k sequential round-trips
through the single shared aiosqlite connection. SQLite is fast, but
30k async hops on a serialized connection take minutes -- and every
other DB consumer in the process (dashboard handlers, vision event
log, world-model writes) is queued behind every one of those hops.
The result was a multi-minute DB stall ~60s after every boot.

This module replaces the per-event query pattern with:
   1. ONE windowed query that pulls every PRESENT-event in the
      analysis window for everyone.
   2. A room-bucketed timestamp-sorted index built once in memory.
   3. O(log N) bisect lookups per of-my-events question.

One DB round-trip instead of tens of thousands. The CPU cost of the
index (sort + bisect) is small enough that it stays on the event
loop, but callers are free to wrap it in asyncio.to_thread if they
want extra insurance.

Modules: modules/world_model/event_windows.py
Classes: RoomTimeIndex
Funcs:   parse_ts
"""
from __future__ import annotations

import bisect
from datetime import datetime, timezone
from typing import Any, Optional


def parse_ts(value: Any) -> Optional[datetime]:
    """Parse a stored ISO timestamp to an aware-UTC datetime, or None.

    Naive datetimes are assumed UTC (matches the rest of the world model);
    unparseable values return None so the caller can skip them. This is
    the single source of truth for ts parsing across event_windows.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(
            tzinfo=timezone.utc,
        )
    try:
        dt = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


class RoomTimeIndex:
    """Room-bucketed, timestamp-sorted index over a flat event list.

    Build once from every PRESENT-event in the analysis window, then
    call `window()` to answer co-presence questions without touching
    the database again.

    Events without a `room` or with an unparseable `ts` are silently
    dropped -- they can't contribute to any room-and-time co-presence
    question, so they're not worth indexing.
    """

    __slots__ = ("_by_room",)

    def __init__(self, events: list[dict]) -> None:
        # room -> (sorted parallel lists: epoch_seconds, original event dicts).
        # Parallel lists let bisect find the slice boundaries in O(log N)
        # over the float list, and we hand back the original event dicts.
        staged: dict[str, list[tuple[float, dict]]] = {}
        for e in events:
            room = e.get("room")
            ts = parse_ts(e.get("ts"))
            if not room or ts is None:
                continue
            staged.setdefault(room, []).append((ts.timestamp(), e))
        self._by_room: dict[str, tuple[list[float], list[dict]]] = {}
        for room, pairs in staged.items():
            pairs.sort(key=lambda p: p[0])
            self._by_room[room] = (
                [p[0] for p in pairs],
                [p[1] for p in pairs],
            )

    def window(
        self,
        room: Optional[str],
        center: datetime,
        half_width_s: float,
    ) -> list[dict]:
        """Events in `room` whose ts is within +/- half_width_s of `center`.

        Returns a (possibly empty) list of the original event dicts, in
        ascending-ts order. `room=None` or an unknown room returns [].
        """
        if not room:
            return []
        bucket = self._by_room.get(room)
        if bucket is None:
            return []
        keys, evs = bucket
        c = center.timestamp()
        lo = bisect.bisect_left(keys, c - half_width_s)
        hi = bisect.bisect_right(keys, c + half_width_s)
        return evs[lo:hi]
