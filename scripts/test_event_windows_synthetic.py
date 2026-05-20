"""
JARVIS -- synthetic test: event_windows.RoomTimeIndex
======================================================
Verifies the in-memory co-presence index that replaced the per-event
DB query storm in PatternMiner / BehavioralProfileBuilder. The index
backs the nightly profile builders' "for each of my events, who else
was in this room at roughly the same time?" question -- and gets that
answer with one DB query + bisect lookups instead of tens of
thousands of sequential round-trips.

The boot-freeze post-mortem identified the per-event query pattern as
the dominant cost (a 30k-event resident fired 30k DB hops). This test
locks in the contract the rewrite relies on.

Run:  .venv\\Scripts\\python.exe scripts\\test_event_windows_synthetic.py
ASCII-only print output (Windows cp1252 console).
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.world_model.event_windows import RoomTimeIndex, parse_ts


# -- helpers ---------------------------------------------------------------

def _ev(room, ts, entity_id="x", entity_type="person",
        entity_name=None) -> dict:
    """Minimal event dict shaped like search_events() output."""
    return {
        "room": room,
        "ts": ts.isoformat() if isinstance(ts, datetime) else ts,
        "entity_id": entity_id,
        "entity_type": entity_type,
        "entity_name": entity_name or entity_id,
        "event_type": "reappeared",
    }


def _utc(year=2026, month=5, day=18, hour=12, minute=0, second=0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


# -- parse_ts --------------------------------------------------------------

def test_parse_ts_iso_naive_becomes_utc() -> None:
    dt = parse_ts("2026-05-18T12:00:00")
    assert dt is not None and dt.tzinfo is not None, dt
    assert dt == _utc(), dt


def test_parse_ts_iso_aware_preserves_tz() -> None:
    dt = parse_ts("2026-05-18T12:00:00+00:00")
    assert dt is not None and dt == _utc(), dt


def test_parse_ts_garbage_returns_none() -> None:
    assert parse_ts("not a timestamp") is None
    assert parse_ts(None) is None
    assert parse_ts("") is None


def test_parse_ts_datetime_passthrough() -> None:
    naive = datetime(2026, 5, 18, 12, 0, 0)
    dt = parse_ts(naive)
    assert dt is not None and dt.tzinfo is not None, dt
    assert dt == _utc(), dt


# -- RoomTimeIndex basics --------------------------------------------------

def test_empty_index_returns_empty_window() -> None:
    idx = RoomTimeIndex([])
    assert idx.window("office", _utc(), 60.0) == []


def test_single_event_in_window() -> None:
    e = _ev("office", _utc(), entity_id="anna")
    idx = RoomTimeIndex([e])
    assert idx.window("office", _utc(), 60.0) == [e]


def test_event_outside_window_excluded() -> None:
    e = _ev("office", _utc(hour=12, minute=5), entity_id="anna")
    idx = RoomTimeIndex([e])
    # Center at 12:00, half-width 60s -> window is [11:59, 12:01].
    # Event at 12:05 is well outside.
    assert idx.window("office", _utc(), 60.0) == []


def test_window_is_inclusive_at_both_ends() -> None:
    """bisect_left + bisect_right means events at exactly +/- W are kept.
    This matches the previous semantics: the SQL `since`/`until` were
    inclusive (>=, <=)."""
    lo = _ev("office", _utc(hour=11, minute=59), entity_id="anna")
    hi = _ev("office", _utc(hour=12, minute=1), entity_id="jeff")
    idx = RoomTimeIndex([lo, hi])
    got = idx.window("office", _utc(), 60.0)
    assert lo in got and hi in got, got


def test_rooms_are_isolated() -> None:
    same = _ev("office", _utc(), entity_id="anna")
    other = _ev("kitchen", _utc(), entity_id="jeff")
    idx = RoomTimeIndex([same, other])
    assert idx.window("office", _utc(), 60.0) == [same]
    assert idx.window("kitchen", _utc(), 60.0) == [other]


def test_unknown_room_returns_empty() -> None:
    e = _ev("office", _utc(), entity_id="anna")
    idx = RoomTimeIndex([e])
    assert idx.window("bathroom", _utc(), 60.0) == []
    assert idx.window(None, _utc(), 60.0) == []


def test_missing_room_or_ts_silently_dropped() -> None:
    """Events without a room or with a bad ts can't contribute to any
    room-and-time question; the index drops them rather than crashing."""
    no_room = {"room": None, "ts": _utc().isoformat(),
               "entity_id": "x", "entity_type": "person",
               "entity_name": "x", "event_type": "reappeared"}
    bad_ts = {"room": "office", "ts": "not a timestamp",
              "entity_id": "y", "entity_type": "person",
              "entity_name": "y", "event_type": "reappeared"}
    good = _ev("office", _utc(), entity_id="z")
    idx = RoomTimeIndex([no_room, bad_ts, good])
    got = idx.window("office", _utc(), 60.0)
    assert got == [good], got


def test_window_returns_ascending_order() -> None:
    """The room bucket is sorted by ts internally so consumers can rely
    on a stable order. This matters when the orchestrator displays
    co-presence events back in the dashboard."""
    a = _ev("office", _utc(hour=11, minute=59, second=30), entity_id="a")
    b = _ev("office", _utc(hour=12, minute=0, second=15), entity_id="b")
    c = _ev("office", _utc(hour=12, minute=0, second=45), entity_id="c")
    idx = RoomTimeIndex([c, a, b])  # deliberately out-of-order input
    got = idx.window("office", _utc(), 60.0)
    assert [e["entity_id"] for e in got] == ["a", "b", "c"], got


def test_large_room_uses_bisect_not_full_scan() -> None:
    """Smoke test: 5000 events in one room, lookup centered on the
    middle. The result slice should be small; the operation should be
    sub-second. This is what the new design buys us."""
    base = _utc()
    events = [
        _ev("office", base + timedelta(seconds=i), entity_id=f"e{i}")
        for i in range(5000)
    ]
    idx = RoomTimeIndex(events)
    got = idx.window("office", base + timedelta(seconds=2500), 30.0)
    # Half-width 30s, one event per second -> ~61 events in the window
    # (the center second + 30 each side, inclusive endpoints).
    assert 55 <= len(got) <= 65, len(got)


# -- runner ---------------------------------------------------------------

def main() -> int:
    tests = [
        test_parse_ts_iso_naive_becomes_utc,
        test_parse_ts_iso_aware_preserves_tz,
        test_parse_ts_garbage_returns_none,
        test_parse_ts_datetime_passthrough,
        test_empty_index_returns_empty_window,
        test_single_event_in_window,
        test_event_outside_window_excluded,
        test_window_is_inclusive_at_both_ends,
        test_rooms_are_isolated,
        test_unknown_room_returns_empty,
        test_missing_room_or_ts_silently_dropped,
        test_window_returns_ascending_order,
        test_large_room_uses_bisect_not_full_scan,
    ]
    passed = 0
    for test in tests:
        try:
            test()
            print(f"[OK]   {test.__name__}")
            passed += 1
        except AssertionError as exc:
            print(f"[FAIL] {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"[ERR]  {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
