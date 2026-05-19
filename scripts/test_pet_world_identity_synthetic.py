"""
JARVIS — synthetic test: per-pet identity in the world model
============================================================
Verifies WorldModel._resolve_pet_entity() — the cat/dog analogue of the
people-have-person_id routing. A cat/dog observation that didn't match
by continuity is matched against the confirmed pet-sample bank and
routed to that pet's canonical world entity, instead of spawning a
fresh entity every re-acquisition (the runaway 783-dog count).

  - a dog observation whose descriptor matches "Velcro"'s tagged
    samples resolves to the live Velcro entity;
  - no tagged samples -> (None, 0.0), so untagged pets fall through
    to the existing path (no regression);
  - a matched pet name with no live entity -> (None, 0.0).

_resolve_pet_entity only touches self.store.db / self.entities, so it
runs against a tiny fake `self`.

Run:  .venv\\Scripts\\python.exe scripts\\test_pet_world_identity_synthetic.py
ASCII-only print output (Windows cp1252 console).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.world_model.world_model import WorldModel
from modules.world_model.types import Observation, WorldEntity


class _StubDB:
    def __init__(self, pet_rows: list[dict] | None = None) -> None:
        self._pet_rows = pet_rows or []

    async def fetchall(self, sql: str, params=()):
        if "pet_visual_samples" in sql:
            return list(self._pet_rows)
        return []


class _FakeWM:
    """Just enough of WorldModel for _resolve_pet_entity."""

    def __init__(self, pet_rows, entities) -> None:
        self.store = SimpleNamespace(db=_StubDB(pet_rows))
        self.entities = {e.id: e for e in entities}

    # _resolve_pet_entity calls self.find_entity_by_name — bind the real one.
    find_entity_by_name = WorldModel.find_entity_by_name


_VELCRO_DESC = {
    "species": "dog", "room": "jeff_room", "bbox": [0, 0, 90, 70],
    "color_class": "black", "color_histogram": [0.3, 0.3, 0.2, 0.2],
    "size_normalized": 0.06, "breed_class": "medium",
}


def _pet_rows(name: str, entity_id: str, desc: dict, n: int = 4) -> list[dict]:
    return [
        {"pet_entity_id": entity_id, "pet_name": name, "species": "dog",
         "room": desc["room"], "bbox": "[0,0,90,70]",
         "descriptor_json": json.dumps(desc)}
        for _ in range(n)
    ]


def _dog_obs(desc: dict) -> Observation:
    from datetime import datetime, timezone
    return Observation(
        camera="jeff_room", room=desc["room"], obj_class="dog",
        bbox=tuple(desc["bbox"]), confidence=0.85,
        ts=datetime.now(timezone.utc),
        metadata={
            "color_class": desc["color_class"],
            "color_histogram": desc["color_histogram"],
            "size_normalized": desc["size_normalized"],
            "breed_class": desc["breed_class"],
            "frame_width": 1280, "frame_height": 720,
        },
    )


def _velcro_entity() -> WorldEntity:
    return WorldEntity(id="velcro-1", entity_type="dog", display_name="Velcro")


# ── tests ────────────────────────────────────────────────────────────────

async def test_matched_pet_routes_to_its_entity() -> None:
    velcro = _velcro_entity()
    wm = _FakeWM(_pet_rows("Velcro", "velcro-1", _VELCRO_DESC), [velcro])
    ent, score = await WorldModel._resolve_pet_entity(wm, _dog_obs(_VELCRO_DESC))
    assert ent is velcro, ent
    assert score > 0.0, score


async def test_no_tagged_samples_falls_through() -> None:
    """No pet_visual_samples -> (None, 0.0): untagged pets behave exactly
    as before (no regression)."""
    wm = _FakeWM([], [_velcro_entity()])
    ent, score = await WorldModel._resolve_pet_entity(wm, _dog_obs(_VELCRO_DESC))
    assert ent is None and score == 0.0, (ent, score)


async def test_match_without_live_entity_returns_none() -> None:
    """Descriptor matches 'Velcro' samples but no live Velcro entity
    exists -> (None, 0.0), no crash."""
    wm = _FakeWM(_pet_rows("Velcro", "velcro-1", _VELCRO_DESC), [])
    ent, score = await WorldModel._resolve_pet_entity(wm, _dog_obs(_VELCRO_DESC))
    assert ent is None and score == 0.0, (ent, score)


async def test_archived_pet_entity_not_returned() -> None:
    velcro = _velcro_entity()
    from datetime import datetime, timezone
    velcro.archived_at = datetime.now(timezone.utc)
    wm = _FakeWM(_pet_rows("Velcro", "velcro-1", _VELCRO_DESC), [velcro])
    ent, score = await WorldModel._resolve_pet_entity(wm, _dog_obs(_VELCRO_DESC))
    assert ent is None, "an archived pet entity must not be a match target"


# ── runner ───────────────────────────────────────────────────────────────

async def _run() -> int:
    tests = [
        test_matched_pet_routes_to_its_entity,
        test_no_tagged_samples_falls_through,
        test_match_without_live_entity_returns_none,
        test_archived_pet_entity_not_returned,
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
