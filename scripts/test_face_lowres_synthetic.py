"""
JARVIS — synthetic test: low-res face flag
==========================================
Verifies FaceRecognizer flags a detected face whose shorter side is
below ArcFace's 112px aligned input — those produce noisy embeddings
that poison the face bank, and the flag (+ a debug log) makes them
visible to quality gates and diagnostics.

Run:  .venv\\Scripts\\python.exe scripts\\test_face_lowres_synthetic.py
ASCII-only print output (Windows cp1252 console).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.vision.face_recognizer import _ARCFACE_INPUT_PX, _face_to_dict


def _fake_face() -> SimpleNamespace:
    """A minimal stand-in for an InsightFace Face object."""
    return SimpleNamespace(
        normed_embedding=np.zeros(512, dtype=np.float32),
        det_score=0.9,
        pose=None,
        age=30,
        gender=1,
    )


# ── tests ────────────────────────────────────────────────────────────────

def test_small_face_flagged_low_res() -> None:
    d = _face_to_dict(_fake_face(), (0, 0, 80, 80))  # 80x80, below 112
    assert d["low_res"] is True, d


def test_full_size_face_not_flagged() -> None:
    d = _face_to_dict(_fake_face(), (0, 0, 200, 160))  # both sides >= 112
    assert d["low_res"] is False, d


def test_uses_shorter_side() -> None:
    """A wide-but-short crop is still low-res — ArcFace upscales it."""
    d = _face_to_dict(_fake_face(), (0, 0, 320, 90))  # 90 tall < 112
    assert d["low_res"] is True, d


def test_exactly_threshold_is_full_res() -> None:
    d = _face_to_dict(_fake_face(), (0, 0, _ARCFACE_INPUT_PX,
                                     _ARCFACE_INPUT_PX))
    assert d["low_res"] is False, d  # 112 is not < 112


def test_dict_shape_intact() -> None:
    """The new field is purely additive — existing keys unchanged."""
    d = _face_to_dict(_fake_face(), (0, 0, 150, 150))
    for key in ("bbox", "embedding", "det_score", "yaw", "pitch",
                "roll", "age", "gender", "low_res"):
        assert key in d, f"missing key {key}"


# ── runner ───────────────────────────────────────────────────────────────

def main() -> int:
    tests = [
        test_small_face_flagged_low_res,
        test_full_size_face_not_flagged,
        test_uses_shorter_side,
        test_exactly_threshold_is_full_res,
        test_dict_shape_intact,
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
