"""
JARVIS — synthetic test: SoundVocabLearner
==========================================
Verifies the unknown-sound review store that backs the dashboard's
"Unknown sounds" Review section:

  - note_unknown logs a sound; review_items returns newest-first;
  - the recency log is bounded by max_unknown;
  - record_answer learns the sound, removes it from review, persists;
  - dismiss removes a one-off;
  - clip_path_for resolves the saved audio clip;
  - a blank name is rejected.

Run:  .venv\\Scripts\\python.exe scripts\\test_sound_vocab_synthetic.py
ASCII-only print output (Windows cp1252 console).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.voice.sound_vocab import SoundVocabLearner


def _learner(max_unknown: int = 5) -> SoundVocabLearner:
    tmp = Path(tempfile.mkdtemp()) / "learned_sounds.json"
    return SoundVocabLearner(
        {"enabled": True, "max_unknown": max_unknown}, store_path=str(tmp)
    )


# ── tests ────────────────────────────────────────────────────────────────

def test_note_and_review_newest_first() -> None:
    sv = _learner()
    first = sv.note_unknown("kitchen", clip_path="/c/1.wav", duration_s=1.5)
    second = sv.note_unknown("office", clip_path="/c/2.wav", duration_s=2.0)
    items = sv.review_items()
    assert len(items) == 2, items
    assert items[0]["id"] == second["id"], "newest must be first"
    assert items[1]["id"] == first["id"], items
    assert items[0]["room"] == "office", items[0]


def test_recency_log_is_bounded() -> None:
    sv = _learner(max_unknown=5)
    for i in range(8):
        sv.note_unknown("kitchen", clip_path=f"/c/{i}.wav")
    items = sv.review_items()
    assert len(items) == 5, f"expected bound of 5, got {len(items)}"


def test_record_answer_learns_and_removes() -> None:
    sv = _learner()
    item = sv.note_unknown("kitchen", clip_path="/c/x.wav")
    entry = sv.record_answer(item["id"], "dishwasher beep")
    assert entry is not None and entry["name"] == "dishwasher beep", entry
    assert sv.review_items() == [], "answered sound must leave review"
    learned = sv.snapshot()["learned"]
    assert any(e["name"] == "dishwasher beep" for e in learned), learned


def test_record_answer_persists() -> None:
    tmp = Path(tempfile.mkdtemp()) / "ls.json"
    sv1 = SoundVocabLearner({"enabled": True}, store_path=str(tmp))
    item = sv1.note_unknown("kitchen", clip_path="/c/x.wav")
    sv1.record_answer(item["id"], "kettle whistle")
    sv2 = SoundVocabLearner({"enabled": True}, store_path=str(tmp))
    learned = sv2.snapshot()["learned"]
    assert any(e["name"] == "kettle whistle" for e in learned), learned


def test_dismiss_removes() -> None:
    sv = _learner()
    item = sv.note_unknown("kitchen", clip_path="/c/x.wav")
    sv.dismiss(item["id"])
    assert sv.review_items() == [], "dismissed sound must leave review"


def test_clip_path_for() -> None:
    sv = _learner()
    item = sv.note_unknown("kitchen", clip_path="/c/abc.wav")
    assert sv.clip_path_for(item["id"]) == "/c/abc.wav"
    assert sv.clip_path_for(999999) is None


def test_blank_answer_rejected() -> None:
    sv = _learner()
    item = sv.note_unknown("kitchen")
    assert sv.record_answer(item["id"], "   ") is None
    assert len(sv.review_items()) == 1, "rejected answer must not drop it"


# ── runner ───────────────────────────────────────────────────────────────

def main() -> int:
    tests = [
        test_note_and_review_newest_first,
        test_recency_log_is_bounded,
        test_record_answer_learns_and_removes,
        test_record_answer_persists,
        test_dismiss_removes,
        test_clip_path_for,
        test_blank_answer_rejected,
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
