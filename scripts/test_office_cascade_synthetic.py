"""
JARVIS — Office Mic → Cascade
=============================
Synthetic test for the office-mic-on-cascade wiring. Routing the office
PC mic through the cascade reuses the existing WakeSource path, so the
two pieces of genuinely new logic are exercised here:

  1. MicSourceWakeAdapter.get_recent_audio() — the rolling ambient buffer
     that replaces WakeWordDetector's buffer for the office mic, so the
     orchestrator's AudioClassifier (YAMNet) keeps getting audio.

  2. CascadeWakeRunner echo suppression — _on_segment must drop any
     segment that closes while Jarvis is speaking in the room (+ tail),
     so the office cascade never transcribes + triages Jarvis's own
     voice off the local PC speaker.

No audio hardware, models, or event loop wiring needed — stub MicSource /
bus / cascade, drive the callbacks directly.

Run: python scripts/test_office_cascade_synthetic.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.voice.cascade import CascadeAction, CascadeDecision
from modules.voice.cascade_runner import CascadeWakeRunner
from modules.voice.sources.base import MicSource
from modules.voice.sources.wake_adapter import MicSourceWakeAdapter

_OWW_CHUNK_SIZE = 1280  # samples — 80 ms at 16 kHz


# ── Stubs ───────────────────────────────────────────────────────────────────


class StubMicSource(MicSource):
    """Minimal MicSource — never started in these tests; only .room is read."""

    def __init__(self, room: str = "office") -> None:
        self._room = room

    async def start(self, callback) -> None:  # pragma: no cover - unused
        pass

    async def stop(self) -> None:  # pragma: no cover - unused
        pass

    @property
    def room(self) -> str:
        return self._room


class StubBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, dict]] = []

    async def publish(self, topic: str, payload: dict) -> None:
        self.published.append((topic, payload))

    def subscribe(self, topic, handler):  # CascadeWakeRunner.run() calls this
        class _Sub:
            def unsubscribe(self_inner) -> None:
                pass
        return _Sub()


class StubCascade:
    """Records evaluate_segment calls so the test can prove echo
    suppression skipped (or allowed) the segment evaluation."""

    def __init__(self) -> None:
        self.evaluate_calls = 0

    async def evaluate_segment(self, audio, *, room: str = "") -> CascadeDecision:
        self.evaluate_calls += 1
        return CascadeDecision(CascadeAction.DROP, stage="stub")


def _pcm(n_chunks: int, value: int = 1000) -> bytes:
    """n_chunks worth of OWW-sized int16 PCM as raw bytes."""
    samples = np.full(n_chunks * _OWW_CHUNK_SIZE, value, dtype=np.int16)
    return samples.tobytes()


def _runner() -> tuple[CascadeWakeRunner, StubCascade]:
    config = {
        "voice": {
            "wake_word": {"model": "hey_jarvis", "sensitivity": 0.5},
            "cascade": {"speech_echo_tail_s": 1.5, "wake_suppress_s": 30},
            "vad": {},
            "barge_in": {},
        }
    }
    runner = CascadeWakeRunner(config, StubBus(), StubMicSource("office"))
    stub_cascade = StubCascade()
    runner._cascade = stub_cascade  # bypass load()
    return runner, stub_cascade


# ── Tests: MicSourceWakeAdapter ambient buffer ──────────────────────────────


async def test_adapter_buffer_empty_returns_none() -> None:
    adapter = MicSourceWakeAdapter(StubMicSource("office"))
    assert adapter.get_recent_audio(1.0) is None, "empty buffer must be None"
    print("PASS: adapter get_recent_audio() returns None before any audio")


async def test_adapter_buffer_accumulates_and_returns_float32() -> None:
    adapter = MicSourceWakeAdapter(StubMicSource("office"))
    # Feed 5 OWW-sized chunks (5 × 80 ms = 400 ms of audio).
    await adapter._mic_callback(_pcm(5), 16000)
    wave = adapter.get_recent_audio(1.0)
    assert wave is not None, "buffer should hold audio after a callback"
    assert wave.dtype == np.float32, wave.dtype
    assert wave.size == 5 * _OWW_CHUNK_SIZE, wave.size
    # int16 1000 → float32 1000/32768 ≈ 0.0305
    assert abs(float(wave[0]) - 1000 / 32768.0) < 1e-4, float(wave[0])
    assert -1.0 <= float(wave.min()) and float(wave.max()) <= 1.0
    print("PASS: adapter buffer accumulates chunks, returns 16 kHz float32")


async def test_adapter_buffer_windows_to_requested_seconds() -> None:
    adapter = MicSourceWakeAdapter(StubMicSource("office"))
    await adapter._mic_callback(_pcm(40), 16000)  # 40 chunks = 3.2 s
    # 0.5 s window → 0.5 * 16000 / 1280 = 6 chunks.
    wave = adapter.get_recent_audio(0.5)
    assert wave is not None and wave.size == 6 * _OWW_CHUNK_SIZE, (
        None if wave is None else wave.size
    )
    print("PASS: adapter get_recent_audio(seconds) windows correctly")


async def test_adapter_buffer_is_bounded() -> None:
    adapter = MicSourceWakeAdapter(StubMicSource("office"))
    # Feed 200 chunks — far past the 80-chunk (~5 s) deque cap.
    await adapter._mic_callback(_pcm(200), 16000)
    wave = adapter.get_recent_audio(60.0)  # ask for more than the cap holds
    assert wave is not None, "buffer should hold audio"
    assert wave.size == 80 * _OWW_CHUNK_SIZE, (
        f"buffer must cap at 80 chunks, got {wave.size // _OWW_CHUNK_SIZE}"
    )
    print("PASS: adapter ambient buffer is bounded (~5 s, drops oldest)")


# ── Tests: CascadeWakeRunner echo suppression ───────────────────────────────


async def test_segment_evaluated_when_jarvis_silent() -> None:
    runner, cascade = _runner()
    await runner._on_segment(np.zeros(8000, dtype=np.float32))
    assert cascade.evaluate_calls == 1, "a normal segment must be evaluated"
    print("PASS: segment is evaluated when Jarvis is not speaking")


async def test_segment_dropped_while_jarvis_speaking() -> None:
    runner, cascade = _runner()
    # voice.speech_start for this room → _jarvis_speaking True.
    await runner._on_speech_start({"room": "office"})
    await runner._on_segment(np.zeros(8000, dtype=np.float32))
    assert cascade.evaluate_calls == 0, (
        "segment closing during Jarvis speech must be dropped (echo)"
    )
    print("PASS: segment dropped while Jarvis is speaking (echo suppression)")


async def test_segment_dropped_within_echo_tail() -> None:
    runner, cascade = _runner()
    await runner._on_speech_start({"room": "office"})
    await runner._on_speech_end({"room": "office"})  # arms _speaking_until
    # Immediately after speech_end we are inside the 1.5 s echo tail.
    await runner._on_segment(np.zeros(8000, dtype=np.float32))
    assert cascade.evaluate_calls == 0, "echo tail must still suppress"
    # Fast-forward past the tail → evaluation resumes.
    runner._speaking_until = time.monotonic() - 0.01
    await runner._on_segment(np.zeros(8000, dtype=np.float32))
    assert cascade.evaluate_calls == 1, "evaluation resumes after the tail"
    print("PASS: echo tail suppresses post-speech, then evaluation resumes")


async def test_echo_suppression_is_room_scoped() -> None:
    runner, cascade = _runner()  # runner.room == "office"
    # Speech in a DIFFERENT room must not gate the office runner.
    await runner._on_speech_start({"room": "kitchen"})
    await runner._on_segment(np.zeros(8000, dtype=np.float32))
    assert cascade.evaluate_calls == 1, (
        "another room's speech must not suppress this room's cascade"
    )
    print("PASS: echo suppression is scoped to the runner's own room")


async def main() -> None:
    await test_adapter_buffer_empty_returns_none()
    await test_adapter_buffer_accumulates_and_returns_float32()
    await test_adapter_buffer_windows_to_requested_seconds()
    await test_adapter_buffer_is_bounded()
    await test_segment_evaluated_when_jarvis_silent()
    await test_segment_dropped_while_jarvis_speaking()
    await test_segment_dropped_within_echo_tail()
    await test_echo_suppression_is_room_scoped()
    print("\nAll office-mic-cascade tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
