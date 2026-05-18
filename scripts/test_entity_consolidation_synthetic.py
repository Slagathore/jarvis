"""
JARVIS — synthetic test: world-model entity consolidation
=========================================================
Verifies WorldModel.consolidate_entities() — the fix for the unbounded
entity accumulation (live DB had 1761 rows: 101 Cole, 783 dogs, 609
unknown-person, and nothing ever archived).

  - duplicate person entities collapse to ONE canonical per person_id
    (the newest by last_seen);
  - duplicate named-pet entities collapse to one per (type, name);
  - a named pet's lone canonical is kept even when very stale;
  - non-PRESENT unknown entities older than the cutoff are archived;
  - recent unknowns and PRESENT entities are kept;
  - archiving is a soft-delete (archived_at set) and drops the entity
    from the live in-memory set.

consolidate_entities only touches self.cfg / self.entities / self.store,
so it runs against a tiny fake `self`.

Run:  .venv\\Scripts\\python.exe scripts\\test_entity_consolidation_synthetic.py
ASCII-only print output (Windows cp1252 console).
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.world_model.world_model import WorldModel
from modules.world_model.types import EntityState, WorldEntity


class _StubStore:
    def __init__(self) -> None:
        self.upserts: list[str] = []

    async def upsert_entity(self, ent: WorldEntity) -> None:
        self.upserts.append(ent.id)


class _FakeWM:
    """Just enough of WorldModel for consolidate_entities."""

    def __init__(self, entities: list[WorldEntity]) -> None:
        self.cfg = {"entity_stale_archive_days": 5.0}
        self.entities = {e.id: e for e in entities}
        self.store = _StubStore()


def _ent(eid, etype, *, pid=None, name=None, days_ago=0.0,
         state=EntityState.IN_ROOM_UNSEEN) -> WorldEntity:
    return WorldEntity(
        id=eid, entity_type=etype, person_id=pid, display_name=name,
        state=state,
        last_seen_ts=datetime.now(timezone.utc) - timedelta(days=days_ago),
    )


async def _consolidate(entities: list[WorldEntity]):
    wm = _FakeWM(entities)
    result = await WorldModel.consolidate_entities(wm)
    return wm, result


# ── tests ────────────────────────────────────────────────────────────────

async def test_person_dedup_keeps_newest() -> None:
    wm, res = await _consolidate([
        _ent("c1", "person", pid=1, name="Cole", days_ago=10),
        _ent("c2", "person", pid=1, name="Cole", days_ago=1),    # newest
        _ent("c3", "person", pid=1, name="Cole", days_ago=20),
        _ent("a1", "person", pid=2, name="Anna", days_ago=2),
    ])
    assert set(wm.entities) == {"c2", "a1"}, set(wm.entities)
    assert res["duplicates"] == 2, res
    assert res["kept"] == 2, res
    assert wm.entities["c2"].archived_at is None


async def test_stale_unknowns_archived_recent_kept() -> None:
    wm, res = await _consolidate([
        _ent("u1", "person", days_ago=10),     # unknown, stale -> archive
        _ent("u2", "person", days_ago=1),      # unknown, recent -> keep
        _ent("d1", "dog", days_ago=30),        # unknown dog, stale -> archive
        _ent("d2", "dog", days_ago=0.5),       # unknown dog, recent -> keep
    ])
    assert set(wm.entities) == {"u2", "d2"}, set(wm.entities)
    assert res["stale"] == 2, res
    assert res["duplicates"] == 0, res


async def test_named_pet_deduped() -> None:
    wm, res = await _consolidate([
        _ent("s1", "cat", name="Summer", days_ago=30),
        _ent("s2", "cat", name="Summer", days_ago=40),
        _ent("s3", "cat", name="Summer", days_ago=2),     # newest
    ])
    assert set(wm.entities) == {"s3"}, set(wm.entities)
    assert res["duplicates"] == 2, res


async def test_named_pet_canonical_survives_staleness() -> None:
    """A named pet's single entity is the identity anchor — kept even
    if it has not been seen in months."""
    wm, res = await _consolidate([_ent("s1", "cat", name="Summer", days_ago=99)])
    assert set(wm.entities) == {"s1"}, set(wm.entities)
    assert res["duplicates"] == 0 and res["stale"] == 0, res


async def test_present_entity_never_archived() -> None:
    wm, res = await _consolidate([
        _ent("p1", "person", days_ago=99, state=EntityState.PRESENT),
    ])
    assert set(wm.entities) == {"p1"}, set(wm.entities)
    assert res["stale"] == 0, res


async def test_archived_is_soft_delete() -> None:
    wm, res = await _consolidate([
        _ent("c1", "person", pid=1, name="Cole", days_ago=10),
        _ent("c2", "person", pid=1, name="Cole", days_ago=1),
    ])
    # c1 archived: gone from the live set, archived_at stamped, persisted.
    assert "c1" not in wm.entities
    assert "c1" in wm.store.upserts, wm.store.upserts
    assert res["live"] == 1, res


# ── runner ───────────────────────────────────────────────────────────────

async def _run() -> int:
    tests = [
        test_person_dedup_keeps_newest,
        test_stale_unknowns_archived_recent_kept,
        test_named_pet_deduped,
        test_named_pet_canonical_survives_staleness,
        test_present_entity_never_archived,
        test_archived_is_soft_delete,
    ]
    passed = 0
    for test in tests:
        try:
            await test()
            print(f"[OK]   {test.__name__}")
            passed += 1
        except AssertionError as exc:
            print(f"[FAIL] {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"[ERR]  {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_run()))
