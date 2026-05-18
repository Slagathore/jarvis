"""
JARVIS — synthetic test: per-room adaptive silence threshold
============================================================
Exercises MicSourceWakeAdapter.get_noise_floor_db() — the per-room
adaptive floor that replaces the static -45 dBFS in the room-tap
recording path. Every room (office + each Wyze room) now calibrates
its silence threshold from its OWN rolling ambient buffer.

The office regression this guards against: a static -45 dBFS sits
below office ambient (~-37 dBFS), so the recorder treats ambient as
unending speech, burns the full 60 s max_duration, and Whisper returns
an empty transcript.

Run:  .venv\\Scripts\\python.exe scripts\\test_per_room_threshold_synthetic.py
ASCII-only print output (Windows cp1252 console).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.voice.sources.wake_adapter import MicSourceWakeAdapter

# One seeded RNG for the whole run -> reproducible chunks, slight
# realistic per-chunk variation. p25 over 40 chunks is stable well
# inside the +/-2 dB tolerances below.
_RNG = np.random.default_rng(20260517)
_CHUNK = 1280  # OWW frame size, the unit _recent_audio stores


class _StubMic:
    """Minimal MicSource stand-in. get_noise_floor_db only touches
    _recent_audio, so the adapter never actually starts this."""

    room = "test_room"

    async def start(self, callback) -> None:  # pragma: no cover - unused
        pass

    async def stop(self) -> None:  # pragma: no cover - unused
        pass


def _chunk_at_db(db_fs: float, n: int = _CHUNK) -> np.ndarray:
    """White-noise int16 chunk whose RMS lands at ~db_fs dBFS."""
    amp = 32768.0 * (10.0 ** (db_fs / 20.0))  # target RMS in int16 units
    sig = _RNG.normal(0.0, amp, n)
    return np.clip(sig, -32768, 32767).astype(np.int16)


def _adapter_with(levels_db: list[float]) -> MicSourceWakeAdapter:
    adapter = MicSourceWakeAdapter(_StubMic())
    for db in levels_db:
        adapter._recent_audio.append(_chunk_at_db(db))
    return adapter


def _measured_db(chunk: np.ndarray) -> float:
    sq = chunk.astype(np.int64)
    sq = sq * sq
    mean = float(sq.mean())
    rms = math.sqrt(mean) if mean > 0 else 0.0
    return 20.0 * math.log10(rms / 32768.0) if rms > 0 else -90.0


# ── tests ────────────────────────────────────────────────────────────────

def test_fallback_when_empty() -> None:
    """No buffered audio yet -> the static fallback, unchanged."""
    adapter = _adapter_with([])
    assert adapter.get_noise_floor_db(fallback_db=-45.0) == -45.0


def test_fallback_when_too_few_chunks() -> None:
    """Under 5 chunks is too little to trust -> fallback."""
    adapter = _adapter_with([-50.0] * 3)
    assert adapter.get_noise_floor_db(fallback_db=-45.0) == -45.0


def test_quiet_room_floor() -> None:
    """A genuinely quiet room: floor ~= ambient + margin, no cap hit."""
    adapter = _adapter_with([-55.0] * 40)
    threshold = adapter.get_noise_floor_db(margin_db=8.0, fallback_db=-45.0)
    # p25(~-55) + 8 ~= -47
    assert -49.0 <= threshold <= -45.0, threshold


def test_noisy_office_floor_rises_above_static() -> None:
    """THE regression test. Office ambient ~-37 dBFS must produce a
    threshold ABOVE the old static -45, so ambient stops registering as
    speech."""
    adapter = _adapter_with([-37.0] * 40)
    threshold = adapter.get_noise_floor_db(margin_db=8.0, fallback_db=-45.0)
    # p25(~-37) + 8 ~= -29
    assert -31.0 <= threshold <= -27.0, threshold
    assert threshold > -45.0, "office floor must rise above the static -45"
    # ...and still leave headroom for real speech (~-17 dBFS at the mic).
    assert threshold < -17.0, "floor must stay below normal speech level"


def test_loud_room_is_capped() -> None:
    """A very loud room must not push the bar above a normal voice —
    hard cap at -25 dBFS."""
    adapter = _adapter_with([-10.0] * 40)
    threshold = adapter.get_noise_floor_db(margin_db=8.0, fallback_db=-45.0)
    assert threshold == -25.0, threshold


def test_percentile_ignores_speech_in_buffer() -> None:
    """When this is called right after a wake, the buffer holds the wake
    word + start of speech. p25 must stay in the ambient cluster and not
    be dragged up by those loud samples."""
    adapter = _adapter_with([-50.0] * 30 + [-15.0] * 10)
    threshold = adapter.get_noise_floor_db(margin_db=8.0, fallback_db=-45.0)
    # p25 sits inside the -50 cluster -> ~-42, NOT pulled toward -15.
    assert -44.0 <= threshold <= -40.0, threshold


def test_chunk_generator_is_accurate() -> None:
    """Sanity check the test's own dBFS synthesis before trusting it."""
    for target in (-55.0, -37.0, -25.0):
        measured = _measured_db(_chunk_at_db(target))
        assert abs(measured - target) <= 1.5, (target, measured)


# ── runner ───────────────────────────────────────────────────────────────

def main() -> int:
    tests = [
        test_chunk_generator_is_accurate,
        test_fallback_when_empty,
        test_fallback_when_too_few_chunks,
        test_quiet_room_floor,
        test_noisy_office_floor_rises_above_static,
        test_loud_room_is_capped,
        test_percentile_ignores_speech_in_buffer,
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
