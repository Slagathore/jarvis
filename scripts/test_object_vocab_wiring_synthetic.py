"""
JARVIS — §23 Object-Vocab Wiring
================================
Synthetic test for wiring ObjectVocabLearner into the vision loop. The
learner module already existed; this proves the three integration
points it was wired into:

  1. UNKNOWN SIGNAL — ObservationBuilder._build_for_frame's else-branch
     feeds un-tracked YOLO objects to learner.note_unknown (gated by
     should_note: ignore-list + confidence floor).
  2. LEARNED QUERIES — _open_vocab_loop_for_room merges
     learner.learned_query_names() into the OWLv2 query set each tick.
  3. ASK/ANSWER — recurrence → pending_question, record_answer persists,
     dismiss is permanent; plus the object-name extraction heuristic.

No YOLO / OWLv2 / cameras — stub detectors + frames.

Run: python scripts/test_object_vocab_wiring_synthetic.py
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import modules.vision.observation_builder as ob_module
from modules.vision.object_vocab import ObjectVocabLearner
from modules.vision.observation_builder import ObservationBuilder
from core.orchestrator_loops import LoopsMixin


# ── Stubs ───────────────────────────────────────────────────────────────────


class StubBus:
    def __init__(self) -> None:
        self.published: list = []

    async def publish(self, topic: str, payload: dict) -> None:
        self.published.append((topic, payload))


class StubDetector:
    """YOLO stand-in — returns a fixed detection list."""

    def __init__(self, dets: list[dict]) -> None:
        self._dets = dets

    async def detect_async(self, frame) -> list[dict]:
        return list(self._dets)


class StubOpenVocab:
    """OWLv2 stand-in — records the query set it was handed each call."""

    score_threshold = 0.2

    def __init__(self) -> None:
        self.query_sets: list[list[str]] = []

    def detect(self, frame, queries: list[str]) -> list[dict]:
        self.query_sets.append(list(queries))
        return []


class StubCamera:
    async def capture_frame_async(self, room: str):
        return np.zeros((120, 160, 3), dtype=np.uint8)

    def get_available_rooms(self) -> list[str]:
        return ["office"]


def _learner() -> ObjectVocabLearner:
    """A learner backed by a throwaway temp store."""
    p = Path(tempfile.gettempdir()) / "jarvis_objvocab_test.json"
    if p.exists():
        p.unlink()
    return ObjectVocabLearner(
        {"enabled": True, "ask_after_sightings": 2, "min_confidence": 0.4},
        store_path=str(p),
    )


# ── Tests: learner policy + ask/answer cycle ────────────────────────────────


def test_should_note_policy() -> None:
    learner = _learner()
    assert learner.should_note("suitcase", 0.7) is True
    assert learner.should_note("chair", 0.99) is False, "furniture is ignored"
    assert learner.should_note("suitcase", 0.20) is False, "below conf floor"
    assert learner.should_note("", 0.9) is False, "blank class rejected"
    disabled = ObjectVocabLearner({"enabled": False})
    assert disabled.should_note("suitcase", 0.9) is False, "disabled → no note"
    print("PASS: should_note gates on enabled / ignore-list / confidence")


def test_recurrence_then_ask_then_learn() -> None:
    learner = _learner()  # ask_after_sightings = 2
    learner.note_unknown("office:suitcase", "office",
                         descriptor={"yolo_class": "suitcase"})
    assert learner.pending_question() is None, "1 sighting < threshold"
    learner.note_unknown("office:suitcase", "office",
                         descriptor={"yolo_class": "suitcase"})
    q = learner.pending_question()
    assert q is not None and q["key"] == "office:suitcase", q
    assert q["count"] == 2 and q["descriptor"]["yolo_class"] == "suitcase", q
    learner.mark_asked(q["key"])
    assert learner.pending_question() is None, "asked → not re-surfaced"
    learner.record_answer(q["key"], "guitar case")
    assert "guitar case" in learner.learned_queries(), learner.learned_queries()
    assert learner.learned_query_names() == {"guitar case": "guitar case"}
    print("PASS: recurrence crosses threshold -> ask -> record_answer learns it")


def test_dismiss_is_permanent() -> None:
    learner = _learner()
    learner.note_unknown("office:vase2", "office")
    learner.dismiss("office:vase2")
    # Re-sighting a dismissed key must NOT re-add it to pending.
    for _ in range(5):
        learner.note_unknown("office:vase2", "office")
    assert learner.pending_question() is None, "dismissed key must not re-ask"
    # And the dismissal survives a reload from disk.
    reloaded = ObjectVocabLearner(
        {"enabled": True, "ask_after_sightings": 1},
        store_path=learner._store_path.as_posix(),
    )
    reloaded.note_unknown("office:vase2", "office")
    assert reloaded.pending_question() is None, "dismissal must persist"
    print("PASS: dismiss() is permanent + persists across reload")


# ── Test: integration point 1 — ObservationBuilder unknown signal ───────────


async def test_observation_builder_notes_unknown() -> None:
    # Make filter_detections a no-op so the test doesn't depend on the
    # machine's ignore-zone config file.
    orig_filter = ob_module.filter_detections
    ob_module.filter_detections = lambda dets, room: dets
    try:
        learner = _learner()
        builder = ObservationBuilder(
            bus=StubBus(), camera_manager=StubCamera(),
            object_detector=StubDetector([
                {"class": "suitcase", "box": [10, 10, 60, 80],
                 "confidence": 0.72},   # unknown → should be noted
                {"class": "chair", "box": [0, 0, 40, 40],
                 "confidence": 0.95},   # furniture → ignored by should_note
                {"class": "person", "box": [5, 5, 30, 90],
                 "confidence": 0.30},   # below person floor → dropped anyway
            ]),
            face_recognizer=None, identity_manager=None,
            posture_analyzer=None, rooms_config=[],
            object_vocab=learner,
        )
        ts = datetime.now(timezone.utc)
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        await builder._build_for_frame("office", frame, ts)
        # The suitcase was noted; the chair was not.
        learner.note_unknown("office:suitcase", "office")  # 2nd → cross thr.
        q = learner.pending_question()
        assert q is not None and q["key"] == "office:suitcase", q
        print("PASS: ObservationBuilder else-branch notes the unknown object")
    finally:
        ob_module.filter_detections = orig_filter


# ── Test: integration point 2 — learned queries reach the OWLv2 loop ────────


async def test_open_vocab_loop_merges_learned_queries() -> None:
    learner = _learner()
    learner.record_answer("office:suitcase", "guitar case")
    openvocab = StubOpenVocab()
    builder = ObservationBuilder(
        bus=StubBus(), camera_manager=StubCamera(),
        object_detector=StubDetector([]),
        face_recognizer=None, identity_manager=None, posture_analyzer=None,
        rooms_config=[{"id": "office"}],
        openvocab_detector=openvocab,
        tracked_objects_open_vocab=[
            {"name": "wallet", "description": "a small leather wallet"},
        ],
        openvocab_interval_seconds=0.02,
        object_vocab=learner,
    )
    task = asyncio.create_task(builder._open_vocab_loop_for_room("office"))
    # Let the loop tick a few times, then stop it.
    for _ in range(50):
        await asyncio.sleep(0.01)
        if openvocab.query_sets:
            break
    builder._stopped = True
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert openvocab.query_sets, "open-vocab loop never ran a detection"
    queries = openvocab.query_sets[-1]
    assert "a small leather wallet" in queries, queries
    assert "guitar case" in queries, (
        f"learned query missing from OWLv2 query set: {queries}"
    )
    print("PASS: open-vocab loop merges learned queries into the OWLv2 set")


# ── Test: integration point 3 — object-name extraction heuristic ────────────


async def test_object_name_extraction_heuristic() -> None:
    # llm=None forces the regex fallback path.
    class _FakeSelf:
        llm = None

    extract = LoopsMixin._extract_object_name_from_reply
    assert await extract(_FakeSelf(), "it's a guitar case") == "guitar case"
    assert await extract(_FakeSelf(), "that's my air purifier") == "air purifier"
    assert await extract(_FakeSelf(), "the humidifier") == "humidifier"
    assert await extract(_FakeSelf(), "") is None
    print("PASS: object-name extraction heuristic strips filler / articles")


async def main() -> None:
    test_should_note_policy()
    test_recurrence_then_ask_then_learn()
    test_dismiss_is_permanent()
    await test_observation_builder_notes_unknown()
    await test_open_vocab_loop_merges_learned_queries()
    await test_object_name_extraction_heuristic()
    print("\nAll §23 object-vocab wiring tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
