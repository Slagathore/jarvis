"""
JARVIS — synthetic test: recognition passive-capture pipeline
=============================================================
Verifies the two new recognition-quality mechanisms in IdentityManager:

  _passes_coherence_gate — a passively-captured face sample is kept only
    if it AGREES with the person's existing bank (cosine to the bank mean
    >= _COHERENCE_FLOOR). Deliberate enroll and cold-start bypass it.

  prune_bank_incoherent — the nightly gardener: quarantines the lowest-
    centrality face samples, conservatively (min-keep floor, max-evict
    cap, absolute outlier floor). No-op on a healthy bank.

Both run against a tiny fake `self`.

Run:  .venv\\Scripts\\python.exe scripts\\test_recognition_pipeline_synthetic.py
ASCII-only print output (Windows cp1252 console).
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.identity.identity_manager import IdentityManager

_DIM = 512


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return (v / n if n else v).astype(np.float32)


def _rand_unit(seed: int) -> np.ndarray:
    return _unit(np.random.default_rng(seed).normal(0.0, 1.0, _DIM))


def _near(base: np.ndarray, seed: int, jitter: float = 0.35) -> np.ndarray:
    """A vector close to `base` — same person, different capture. Noise
    is unit-scaled first so `jitter` is the real perturbation size
    relative to the unit base (jitter 0.35 -> ~0.94 cosine to base)."""
    noise = np.random.default_rng(seed).normal(0.0, 1.0, _DIM)
    noise = noise / (float(np.linalg.norm(noise)) or 1.0)
    return _unit(base + jitter * noise)


_BASE = _rand_unit(7)


class _StubDB:
    def __init__(self, n_rows: int) -> None:
        self._n = n_rows
        self.updates: list = []

    async def fetchall(self, sql: str, params=()):
        if "face_samples" in sql:
            return [{"id": i} for i in range(self._n)]
        return []

    async def execute(self, sql: str, params=()):
        self.updates.append(params)
        return 1


class _FakeIM:
    def __init__(self, face_samples: dict, db=None) -> None:
        self._face_samples = face_samples
        self._db = db


# ── coherence gate ───────────────────────────────────────────────────────

def test_enroll_bypasses_gate() -> None:
    im = _FakeIM({1: []})
    assert IdentityManager._passes_coherence_gate(
        im, 1, _rand_unit(99), "enroll"
    ) is True


def test_cold_start_bypasses_gate() -> None:
    """A person with < _COHERENCE_MIN_CORE samples has no core yet."""
    im = _FakeIM({1: [_near(_BASE, i) for i in range(3)]})
    assert IdentityManager._passes_coherence_gate(
        im, 1, _rand_unit(99), "drift_capture"
    ) is True


def test_coherent_sample_passes() -> None:
    core = [_near(_BASE, i) for i in range(10)]
    im = _FakeIM({1: core})
    fresh = _near(_BASE, 555)  # same person, new capture
    assert IdentityManager._passes_coherence_gate(
        im, 1, fresh, "drift_capture"
    ) is True


def test_incoherent_sample_rejected() -> None:
    core = [_near(_BASE, i) for i in range(10)]
    im = _FakeIM({1: core})
    impostor = _rand_unit(999)  # nothing like the person's bank
    assert IdentityManager._passes_coherence_gate(
        im, 1, impostor, "live_question"
    ) is False


# ── nightly gardener ─────────────────────────────────────────────────────

async def test_gardener_quarantines_outliers() -> None:
    """25 coherent + 5 random outliers -> the 5 outliers quarantined."""
    embs = [_near(_BASE, i) for i in range(25)] + [
        _rand_unit(200 + i) for i in range(5)
    ]
    im = _FakeIM({1: embs}, db=_StubDB(n_rows=30))
    res = await IdentityManager.prune_bank_incoherent(im)
    assert res["quarantined"] == 5, res
    assert len(im._face_samples[1]) == 25, len(im._face_samples[1])


async def test_gardener_noop_on_healthy_bank() -> None:
    embs = [_near(_BASE, i) for i in range(30)]
    im = _FakeIM({1: embs}, db=_StubDB(n_rows=30))
    res = await IdentityManager.prune_bank_incoherent(im)
    assert res["quarantined"] == 0, res
    assert len(im._face_samples[1]) == 30


async def test_gardener_respects_min_keep() -> None:
    """22 all-incoherent samples -> can only evict down to the 20 floor."""
    embs = [_rand_unit(300 + i) for i in range(22)]
    im = _FakeIM({1: embs}, db=_StubDB(n_rows=22))
    res = await IdentityManager.prune_bank_incoherent(im)
    assert res["quarantined"] == 2, res  # 22 - 20 min-keep
    assert len(im._face_samples[1]) == 20


# ── runner ───────────────────────────────────────────────────────────────

async def _run() -> int:
    sync_tests = [
        test_enroll_bypasses_gate,
        test_cold_start_bypasses_gate,
        test_coherent_sample_passes,
        test_incoherent_sample_rejected,
    ]
    async_tests = [
        test_gardener_quarantines_outliers,
        test_gardener_noop_on_healthy_bank,
        test_gardener_respects_min_keep,
    ]
    passed = 0
    total = len(sync_tests) + len(async_tests)
    for test in sync_tests:
        try:
            test()
            print(f"[OK]   {test.__name__}")
            passed += 1
        except AssertionError as exc:
            print(f"[FAIL] {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"[ERR]  {test.__name__}: {type(exc).__name__}: {exc}")
    for test in async_tests:
        try:
            await test()
            print(f"[OK]   {test.__name__}")
            passed += 1
        except AssertionError as exc:
            print(f"[FAIL] {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"[ERR]  {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{passed}/{total} passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_run()))
