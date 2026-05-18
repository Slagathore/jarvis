"""
JARVIS — synthetic test: belief naming + pet individuation
==========================================================
Verifies the multi-resident-naming change to the belief resolver:

  - Two distinct cats in ONE room produce TWO separate hypotheses
    (keyed pet:<id>), instead of collapsing into one "a cat is here".
  - Unidentified pets still fall back to the coarse species:room key
    (graceful degradation — no regression when nothing is tagged).
  - People and pets carry a display_name through snapshot() and the
    world.belief_changed event.
  - match_pet_from_descriptor ranks an already-built descriptor against
    the confirmed-sample bank (the path the resolver uses — no frame).

Run:  .venv\\Scripts\\python.exe scripts\\test_belief_naming_synthetic.py
ASCII-only print output (Windows cp1252 console).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.world_model.belief.resolver import BeliefResolver
from modules.world_model.pet_identity import match_pet_from_descriptor


# ── stubs ────────────────────────────────────────────────────────────────

class StubBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, dict]] = []

    async def publish(self, topic: str, payload: dict) -> None:
        self.published.append((topic, payload))

    def subscribe(self, topic, cb):
        return SimpleNamespace(unsubscribe=lambda: None)


class StubDB:
    """Minimal async DB stub. Returns pet rows for pet_visual_samples
    queries; everything else is empty / a no-op."""

    def __init__(self, pet_rows: list[dict] | None = None) -> None:
        self._pet_rows = pet_rows or []
        self.executed: list[tuple] = []

    async def execute(self, sql: str, params=()):
        self.executed.append((sql, params))
        return 1

    async def fetchone(self, sql: str, params=()):
        return None

    async def fetchall(self, sql: str, params=()):
        if "pet_visual_samples" in sql:
            return list(self._pet_rows)
        return []


def _cat_obs(color: str, bbox=(10, 10, 110, 110)) -> SimpleNamespace:
    return SimpleNamespace(
        person_id=None, obj_class="cat", confidence=0.85,
        person_match_confidence=0.0, person_name=None, bbox=bbox,
        metadata={
            "color_class": color, "size_normalized": 0.05,
            "color_histogram": [0.4, 0.3, 0.2, 0.1],
            "coat_texture": "short", "frame_width": 1280,
            "frame_height": 720,
        },
    )


def _person_obs(pid: int, name: str | None, bbox=(20, 20, 120, 220)) -> SimpleNamespace:
    return SimpleNamespace(
        person_id=pid, obj_class="person", confidence=0.9,
        person_match_confidence=0.95, person_name=name, bbox=bbox,
        metadata={},
    )


# ── tests ────────────────────────────────────────────────────────────────

async def test_two_cats_individuated() -> None:
    """Two cats, same room, different identity -> two hypotheses."""
    resolver = BeliefResolver(StubBus(), StubDB(), {"shadow": True})

    async def fake_resolve(species, room, obs, omd):
        return {
            "orange": ("summer", "Summer"),
            "black": ("atlas", "Atlas"),
        }.get(omd.get("color_class"), (None, None))

    resolver._resolve_pet = fake_resolve  # type: ignore[method-assign]
    await resolver._on_observation({
        "room": "kitchen", "camera": "kitchen",
        "observations": [_cat_obs("orange"), _cat_obs("black")],
    })
    keys = set(resolver._entities.keys())
    assert keys == {"pet:summer", "pet:atlas"}, keys
    assert resolver._entities["pet:summer"][0].display_name == "Summer"
    assert resolver._entities["pet:atlas"][0].display_name == "Atlas"


async def test_unidentified_pets_collapse() -> None:
    """No identity -> coarse species:room key (no regression)."""
    resolver = BeliefResolver(StubBus(), StubDB(), {"shadow": True})

    async def fake_none(species, room, obs, omd):
        return (None, None)

    resolver._resolve_pet = fake_none  # type: ignore[method-assign]
    await resolver._on_observation({
        "room": "kitchen", "camera": "kitchen",
        "observations": [_cat_obs("orange"), _cat_obs("black")],
    })
    keys = set(resolver._entities.keys())
    assert keys == {"cat:kitchen"}, keys


async def test_person_name_in_snapshot() -> None:
    """A recognized person carries display_name into snapshot()."""
    resolver = BeliefResolver(StubBus(), StubDB(), {"shadow": True})
    await resolver._on_observation({
        "room": "office", "camera": "office",
        "observations": [_person_obs(5, "Anna")],
    })
    assert resolver._entities["person:5"][0].display_name == "Anna"
    snap = resolver.snapshot()
    named = [e for e in snap if e.get("display_name") == "Anna"]
    assert named, f"no Anna in snapshot: {snap}"
    assert named[0]["primary"]["display_name"] == "Anna"


async def test_belief_changed_carries_name() -> None:
    """A live (non-shadow) transition publishes entity_name."""
    bus = StubBus()
    resolver = BeliefResolver(bus, StubDB(), {"shadow": False})
    await resolver._on_observation({
        "room": "office", "camera": "office",
        "observations": [_person_obs(7, "Cole")],
    })
    events = [p for (t, p) in bus.published if t == "world.belief_changed"]
    assert events, "no belief_changed event published"
    assert events[0]["entity_name"] == "Cole", events[0]


async def test_unnamed_person_falls_back() -> None:
    """An unrecognized person (person_name=None) does not crash and
    falls back to the entity_type as display_name."""
    resolver = BeliefResolver(StubBus(), StubDB(), {"shadow": True})
    await resolver._on_observation({
        "room": "office", "camera": "office",
        "observations": [_person_obs(9, None)],
    })
    assert resolver._entities["person:9"][0].display_name == "person"


async def test_match_pet_from_descriptor() -> None:
    """The descriptor-based matcher ranks the right pet and accepts a
    clean, well-separated match."""
    desc_a = {
        "species": "cat", "room": "kitchen", "bbox": [0, 0, 100, 100],
        "color_class": "orange", "color_histogram": [0.4, 0.3, 0.2, 0.1],
        "size_normalized": 0.05, "coat_texture": "short",
    }
    desc_b = {
        "species": "cat", "room": "bedroom", "bbox": [0, 0, 50, 50],
        "color_class": "black", "color_histogram": [0.1, 0.1, 0.1, 0.7],
        "size_normalized": 0.02, "coat_texture": "long",
    }
    rows = []
    for _ in range(3):
        rows.append({
            "pet_entity_id": "e_summer", "pet_name": "Summer",
            "species": "cat", "room": "kitchen", "bbox": "[0,0,100,100]",
            "descriptor_json": json.dumps(desc_a),
        })
    for _ in range(3):
        rows.append({
            "pet_entity_id": "e_atlas", "pet_name": "Atlas",
            "species": "cat", "room": "bedroom", "bbox": "[0,0,50,50]",
            "descriptor_json": json.dumps(desc_b),
        })
    db = StubDB(pet_rows=rows)
    result = await match_pet_from_descriptor(
        db=db, species="cat", room="kitchen", query=dict(desc_a),
    )
    assert result is not None, "expected a match"
    assert result["pet_name"] == "Summer", result
    assert result["entity_id"] == "e_summer", result
    assert result["accepted"] is True, result

    # No samples -> None, no crash.
    empty = await match_pet_from_descriptor(
        db=StubDB(), species="cat", room="kitchen", query=dict(desc_a),
    )
    assert empty is None


# ── runner ───────────────────────────────────────────────────────────────

async def _run() -> int:
    tests = [
        test_two_cats_individuated,
        test_unidentified_pets_collapse,
        test_person_name_in_snapshot,
        test_belief_changed_carries_name,
        test_unnamed_person_falls_back,
        test_match_pet_from_descriptor,
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
