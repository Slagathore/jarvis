"""
JARVIS — synthetic test: object-vocab spatial persistence + Review feed
=======================================================================
Verifies the persistence Cole asked for on unknown objects:

  - note_unknown accumulates a rolling bbox trail per unknown;
  - _location_summary collapses that trail into a center + a 0..1
    stability score — high when an object keeps appearing in the SAME
    spot ("it's really there"), low when sightings are scattered;
  - review_items() / pending_question() expose crop + location so the
    dashboard Review tab can show picture evidence;
  - a crop_path sticks (a later None sighting does not wipe it);
  - answer / dismiss still work and take the item out of review.

Run:  .venv\\Scripts\\python.exe scripts\\test_object_vocab_review_synthetic.py
ASCII-only print output (Windows cp1252 console).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.vision.object_vocab import ObjectVocabLearner


def _learner() -> ObjectVocabLearner:
    """A learner backed by a fresh temp store (never touches the real
    data/learned_objects.json)."""
    tmp = Path(tempfile.mkdtemp()) / "learned.json"
    return ObjectVocabLearner({"enabled": True}, store_path=str(tmp))


def _note(learner, key, bbox, *, room="office", cls="bottle", crop=None):
    learner.note_unknown(
        key=key, room=room,
        descriptor={"yolo_class": cls, "bbox": bbox, "confidence": 0.7},
        crop_path=crop,
    )


# ── tests ────────────────────────────────────────────────────────────────

def test_bbox_trail_accumulates() -> None:
    learner = _learner()
    for _ in range(5):
        _note(learner, "office:bottle", [100, 100, 140, 180])
    items = learner.review_items()
    assert len(items) == 1, items
    item = items[0]
    assert item["count"] == 5, item
    assert item["yolo_class"] == "bottle", item
    assert item["location"]["n"] == 5, item


def test_same_spot_is_stable() -> None:
    """All sightings in one place -> stability ~1.0."""
    learner = _learner()
    for _ in range(6):
        _note(learner, "office:bottle", [100, 100, 140, 180])
    loc = learner.review_items()[0]["location"]
    assert loc["stability"] > 0.9, loc
    assert loc["center"] == [120.0, 140.0], loc


def test_scattered_is_unstable() -> None:
    """Sightings all over the frame -> low stability (likely noise)."""
    learner = _learner()
    for b in ([0, 0, 40, 40], [300, 300, 340, 340],
              [600, 100, 640, 140], [50, 500, 90, 540]):
        _note(learner, "office:thing", b, cls="thing")
    loc = learner.review_items()[0]["location"]
    assert loc["stability"] < 0.5, loc


def test_crop_path_is_sticky() -> None:
    """A new crop updates the evidence; a None sighting must NOT wipe it."""
    learner = _learner()
    _note(learner, "k", [0, 0, 10, 10], crop="/snaps/1.jpg")
    _note(learner, "k", [0, 0, 10, 10], crop=None)
    assert learner.review_items()[0]["crop_path"] == "/snaps/1.jpg"
    _note(learner, "k", [0, 0, 10, 10], crop="/snaps/2.jpg")
    assert learner.review_items()[0]["crop_path"] == "/snaps/2.jpg"


def test_pending_question_carries_evidence() -> None:
    """An unknown past the ask threshold exposes crop + location."""
    learner = _learner()  # ask_after defaults to 3
    for _ in range(3):
        _note(learner, "office:mug", [10, 10, 50, 50], cls="mug",
              crop="/snaps/mug.jpg")
    question = learner.pending_question()
    assert question is not None, "should be ready to ask"
    assert question["crop_path"] == "/snaps/mug.jpg", question
    assert question["location"]["n"] == 3, question
    assert "stability" in question["location"], question


def test_answer_and_dismiss_clear_review() -> None:
    learner = _learner()
    for _ in range(3):
        _note(learner, "office:mug", [1, 1, 9, 9], cls="mug")
    entry = learner.record_answer("office:mug", "coffee mug")
    assert entry is not None and entry["name"] == "coffee mug", entry
    assert learner.review_items() == [], "answered item must leave review"

    for _ in range(3):
        _note(learner, "office:x", [1, 1, 9, 9], cls="x")
    learner.dismiss("office:x")
    assert learner.review_items() == [], "dismissed item must leave review"
    # A dismissed key must not creep back on the next sighting.
    _note(learner, "office:x", [1, 1, 9, 9], cls="x")
    assert learner.review_items() == [], "dismissed key re-added"


# ── runner ───────────────────────────────────────────────────────────────

def main() -> int:
    tests = [
        test_bbox_trail_accumulates,
        test_same_spot_is_stable,
        test_scattered_is_unstable,
        test_crop_path_is_sticky,
        test_pending_question_carries_evidence,
        test_answer_and_dismiss_clear_review,
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
