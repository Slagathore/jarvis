"""
JARVIS — TTS Router
===================
Synthetic test for modules/voice/tts_router.py — the dispatcher that
fronts the Piper and Kokoro TTS backends so a single voice menu can
offer both and route synthesis to whichever backend owns the voice.

No real Piper binary or Kokoro model needed: the router is constructed
normally (PiperTTS/KokoroTTS __init__ are cheap, no model load), then
its two backend handles are swapped for recording stubs so the routing
logic itself is exercised.

Covers:
  • available_voices() merges Piper voices + Kokoro "KOK <id>" voices
  • Piper-voice selection keeps the Piper backend active
  • KOK-voice selection lazily loads Kokoro and flips the active backend
  • _sample_rate / _active_voice track the live backend
  • synchronous set_voice() refuses a not-yet-loaded Kokoro voice
  • speak/synthesize forward to the active backend
  • kokoro.enabled=false hides KOK voices and rejects KOK selection

Run: python scripts/test_tts_router_synthetic.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.voice.tts_router import TTSRouter


# ── Stubs ───────────────────────────────────────────────────────────────────


class StubPiper:
    """Mimics the PiperTTS surface the router touches."""

    def __init__(self) -> None:
        self.loaded = True
        self._active_voice = "glados"
        self._sample_rate = 22050
        self._use_sapi = False
        self.spoken: list[str] = []
        self.load_calls = 0

    def load(self) -> None:
        self.load_calls += 1

    def available_voices(self) -> list[str]:
        return ["glados", "wheatley"]

    def set_voice(self, name: str) -> bool:
        if name in ("glados", "wheatley"):
            self._active_voice = name
            return True
        return False

    def speak(self, text: str) -> None:
        self.spoken.append(text)

    async def speak_async(self, text: str) -> None:
        self.spoken.append(text)

    def synthesize(self, text: str) -> np.ndarray:
        return np.ones(8, dtype=np.float32)

    async def synthesize_async(self, text: str) -> np.ndarray:
        return np.ones(8, dtype=np.float32)


class StubKokoro:
    """Mimics the KokoroTTS surface. loaded flips True only on load()."""

    def __init__(self) -> None:
        self.loaded = False
        self._active_voice = "af_nicole"
        self._sample_rate = 24000
        self.spoken: list[str] = []
        self.load_calls = 0

    def load(self) -> None:
        self.load_calls += 1
        self.loaded = True

    def available_voices(self) -> list[str]:
        return ["KOK af_nicole", "KOK am_adam"]

    def set_voice(self, name: str) -> bool:
        if not self.loaded:
            return False
        raw = name[4:] if name.startswith("KOK ") else name
        if raw in ("af_nicole", "am_adam"):
            self._active_voice = raw
            return True
        return False

    def speak(self, text: str) -> None:
        self.spoken.append(text)

    async def speak_async(self, text: str) -> None:
        self.spoken.append(text)

    def synthesize(self, text: str) -> np.ndarray:
        return np.full(16, 2.0, dtype=np.float32)

    async def synthesize_async(self, text: str) -> np.ndarray:
        return np.full(16, 2.0, dtype=np.float32)


def _make_router(kokoro_enabled: bool = True) -> tuple[TTSRouter, StubPiper, StubKokoro]:
    """Build a router, then swap in recording stubs for both backends."""
    cfg = {
        "voice": {
            "tts": {
                "engine": "piper",
                "active_voice": "glados",
                "voices": {"glados": "x.onnx"},
                "kokoro": {"enabled": kokoro_enabled, "voice": "af_nicole"},
            }
        }
    }
    router = TTSRouter(cfg)
    piper, kokoro = StubPiper(), StubKokoro()
    router._piper = piper
    router._active = piper
    if kokoro_enabled:
        router._kokoro = kokoro
    return router, piper, kokoro


# ── Tests ───────────────────────────────────────────────────────────────────


async def test_available_voices_merges_both() -> None:
    router, _, _ = _make_router()
    voices = router.available_voices()
    assert "glados" in voices and "wheatley" in voices, voices
    assert "KOK af_nicole" in voices and "KOK am_adam" in voices, voices
    # Piper voices come first so the default backend leads the menu.
    assert voices.index("glados") < voices.index("KOK af_nicole"), voices
    print("PASS: available_voices() merges Piper + Kokoro voices")


async def test_piper_voice_keeps_piper_active() -> None:
    router, piper, kokoro = _make_router()
    ok = await router.set_voice_async("wheatley")
    assert ok is True
    assert router._active is piper, "Piper voice must keep Piper active"
    assert router._sample_rate == 22050, router._sample_rate
    assert router._active_voice == "wheatley", router._active_voice
    assert kokoro.load_calls == 0, "Kokoro must not load for a Piper voice"
    print("PASS: Piper-voice selection keeps Piper active, Kokoro untouched")


async def test_kok_voice_lazy_loads_and_switches() -> None:
    router, _, kokoro = _make_router()
    assert kokoro.loaded is False, "Kokoro starts unloaded"
    ok = await router.set_voice_async("KOK am_adam")
    assert ok is True
    assert kokoro.load_calls == 1, "first KOK pick must lazily load Kokoro"
    assert router._active is kokoro, "KOK voice must flip the active backend"
    # Backend-tracking properties follow the switch.
    assert router._sample_rate == 24000, router._sample_rate
    assert router._active_voice == "KOK am_adam", router._active_voice
    # A second KOK pick must NOT reload the model.
    ok2 = await router.set_voice_async("KOK af_nicole")
    assert ok2 is True and kokoro.load_calls == 1, "Kokoro loaded at most once"
    print("PASS: KOK-voice selection lazy-loads Kokoro once + switches backend")


async def test_sync_set_voice_refuses_unloaded_kokoro() -> None:
    router, piper, kokoro = _make_router()
    # Kokoro not loaded yet — the synchronous path can't download, so refuse.
    ok = router.set_voice("KOK af_nicole")
    assert ok is False, "sync set_voice must refuse an unloaded Kokoro voice"
    assert router._active is piper, "failed switch must not change the backend"
    # A Piper voice still works synchronously.
    assert router.set_voice("glados") is True
    print("PASS: sync set_voice() refuses unloaded Kokoro, still serves Piper")


async def test_speak_synthesize_follow_active_backend() -> None:
    router, piper, kokoro = _make_router()
    await router.speak_async("hello from piper")
    assert piper.spoken == ["hello from piper"], piper.spoken
    out = await router.synthesize_async("p")
    assert len(out) == 8, "Piper stub returns 8 samples"

    await router.set_voice_async("KOK af_nicole")
    await router.speak_async("hello from kokoro")
    assert kokoro.spoken == ["hello from kokoro"], kokoro.spoken
    assert piper.spoken == ["hello from piper"], "Piper must not see Kokoro speech"
    out2 = await router.synthesize_async("k")
    assert len(out2) == 16, "Kokoro stub returns 16 samples"
    print("PASS: speak/synthesize forward to the active backend")


async def test_kokoro_disabled_hides_and_rejects() -> None:
    router, _, _ = _make_router(kokoro_enabled=False)
    assert router._kokoro is None, "disabled Kokoro must not be constructed"
    voices = router.available_voices()
    assert not any(v.startswith("KOK ") for v in voices), voices
    ok = await router.set_voice_async("KOK af_nicole")
    assert ok is False, "KOK selection must fail when Kokoro disabled"
    print("PASS: kokoro.enabled=false hides KOK voices + rejects KOK selection")


async def main() -> None:
    await test_available_voices_merges_both()
    await test_piper_voice_keeps_piper_active()
    await test_kok_voice_lazy_loads_and_switches()
    await test_sync_set_voice_refuses_unloaded_kokoro()
    await test_speak_synthesize_follow_active_backend()
    await test_kokoro_disabled_hides_and_rejects()
    print("\nAll TTS router tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
