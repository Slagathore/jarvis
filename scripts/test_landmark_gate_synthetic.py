"""
JARVIS — synthetic test: named-object landmark gate
===================================================
Verifies that landmark-visit interaction events fire only for the right
entities:

  - a NAMED object (Cole's wallet at the leash hook) -> fires;
  - an UNKNOWN object (unknown_cell_phone_* near the dog water bowl) ->
    never fires -- the noise Cole flagged;
  - an un-named object -> never fires;
  - people and pets ALWAYS pass the gate, regardless of name (a cat at
    the litterbox is the whole point of the feature).

_classify_landmark_dwell only touches self.cfg and the class-level
_LANDMARK_INTERACTION_KIND map, so it is exercised with a tiny fake
`self` rather than a full WorldModel.

Run:  .venv\\Scripts\\python.exe scripts\\test_landmark_gate_synthetic.py
ASCII-only print output (Windows cp1252 console).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.world_model.world_model import WorldModel


class _FakeWM:
    """Just enough of WorldModel for _classify_landmark_dwell."""
    cfg = {"landmark_dwell_frames": 3}
    _LANDMARK_INTERACTION_KIND = WorldModel._LANDMARK_INTERACTION_KIND


def _ent(entity_type: str, display_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        entity_type=entity_type, display_name=display_name, metadata={},
    )


def _dwell(fake, ent, landmark: str, n: int) -> list:
    """Call _classify_landmark_dwell n times; return each result."""
    return [
        WorldModel._classify_landmark_dwell(fake, ent, landmark)
        for _ in range(n)
    ]


# ── tests ────────────────────────────────────────────────────────────────

def test_named_object_fires() -> None:
    results = _dwell(_FakeWM(), _ent("object", "wallet"), "leash_hook", 3)
    assert results[0] is None and results[1] is None, results
    assert results[2] is not None, "named object should fire at threshold"
    assert results[2]["interaction_kind"] == "leash_interaction", results[2]


def test_unknown_object_never_fires() -> None:
    """THE fix — an unknown_* blob over a landmark must stay silent."""
    results = _dwell(
        _FakeWM(), _ent("object", "unknown_cell_phone_3"),
        "dog_water_bowl", 6,
    )
    assert all(r is None for r in results), results


def test_unnamed_object_never_fires() -> None:
    results = _dwell(_FakeWM(), _ent("object", ""), "leash_hook", 5)
    assert all(r is None for r in results), results


def test_pet_fires_at_litterbox() -> None:
    results = _dwell(_FakeWM(), _ent("cat", "Summer"), "litterbox", 3)
    assert results[2] is not None, results
    assert results[2]["interaction_kind"] == "litterbox_visit", results[2]


def test_pet_with_unknown_name_still_fires() -> None:
    """Pets always pass — the gate restricts objects only."""
    results = _dwell(
        _FakeWM(), _ent("cat", "unknown_cat_1"), "litterbox", 3
    )
    assert results[2] is not None, results


def test_person_fires() -> None:
    results = _dwell(_FakeWM(), _ent("person", "Cole"), "leash_hook", 3)
    assert results[2] is not None, results


# ── runner ───────────────────────────────────────────────────────────────

def main() -> int:
    tests = [
        test_named_object_fires,
        test_unknown_object_never_fires,
        test_unnamed_object_never_fires,
        test_pet_fires_at_litterbox,
        test_pet_with_unknown_name_still_fires,
        test_person_fires,
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
