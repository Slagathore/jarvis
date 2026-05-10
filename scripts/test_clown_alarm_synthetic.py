"""
JARVIS - v4.1 Clown Alarm
=========================
Synthetic test for the §29.8 ClownAlarm - drives state machine
transitions + the patch-patch's 3-layer improv fallback without
needing a real LLM, audio system, or speakers.

Spec: v4_1_clown_alarm_micropatch.md + clown_alarm_patch_patch.md
Runs in <2s.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.safety.alarms.clown import (
    ClownAlarm, ClownResponse, MIN_EXAMPLES_FOR_GENERATION,
    parse_cooldown_phrase,
)
from modules.safety.alarms.state import AlarmState


# ── Stubs ───────────────────────────────────────────────────────────────────


class StubBus:
    def __init__(self) -> None:
        self.handlers: dict[str, list] = {}
        self.published: list[tuple[str, dict]] = []

    async def subscribe(self, topic: str, handler: Any) -> None:
        self.handlers.setdefault(topic, []).append(handler)

    async def publish(self, topic: str, payload: dict) -> None:
        self.published.append((topic, payload))
        for h in self.handlers.get(topic, []):
            await h(payload)


class StubAudio:
    """Records what the alarm asked us to play, but doesn't actually
    play anything. Lets the test assert audio sequence requests
    without owning real speakers."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def play_clown_sequence(
        self,
        announcement: str,
        on_complete: Optional[Any] = None,
        horn_loop_count: int = 3,
    ) -> None:
        self.calls.append(("play_clown_sequence", {
            "announcement": announcement,
            "horn_loop_count": horn_loop_count,
            "has_on_complete": on_complete is not None,
        }))
        # Simulate the natural-end path so condition_cleared fires.
        if on_complete is not None:
            res = on_complete()
            if asyncio.iscoroutine(res):
                await res

    async def play_for(self, *a, **kw) -> None:
        self.calls.append(("play_for", {"args": a, "kw": kw}))

    async def stop(self) -> None:
        self.calls.append(("stop", {}))


class StubDispatcher:
    """Pretends the clown alarm IS the audio owner so play_clown_sequence
    runs in the test. Also accepts the state-change callback the alarm
    base class fires on every transition."""

    def __init__(self, owner: Optional[str] = "clown") -> None:
        self._owner = owner
        self.state_changes: list[tuple[str, Any, Any]] = []

    def audio_owner(self) -> Optional[str]:
        return self._owner

    async def on_alarm_state_change(
        self, name: str, old: Any, new: Any,
    ) -> None:
        self.state_changes.append((name, old, new))


class StubLLM:
    """Configurable LLM stub. `responses` is a queue of strings to
    return; `errors` is a queue of exceptions to raise."""

    def __init__(
        self,
        responses: Optional[list[str]] = None,
        errors: Optional[list[BaseException]] = None,
        delay: float = 0.0,
    ) -> None:
        self.responses = list(responses or [])
        self.errors = list(errors or [])
        self.delay = float(delay)
        self.calls: list[dict] = []

    async def complete(
        self,
        prompt: str,
        max_tokens: int = 80,
        temperature: float = 0.9,
    ) -> str:
        self.calls.append({
            "prompt_len": len(prompt),
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        if self.errors:
            raise self.errors.pop(0)
        if self.responses:
            return self.responses.pop(0)
        return "(stub generated response)"


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_alarm(
    pool_yaml: Path,
    *,
    llm: Optional[Any] = None,
    audio: Optional[StubAudio] = None,
    improv_timeout: float = 4.0,
) -> tuple[ClownAlarm, StubBus, StubAudio]:
    bus = StubBus()
    audio = audio if audio is not None else StubAudio()
    alarm = ClownAlarm(
        bus=bus,
        audio=audio,
        notifier=None,
        store=None,
        llm=llm,
        responses_yaml=pool_yaml,
        recent_queue_depth=10,
        improv_cache_ttl_seconds=0.0,   # disable cache for tests
        improv_generation_timeout_seconds=improv_timeout,
    )
    # Pretend to be wired to a dispatcher that grants audio ownership.
    alarm._dispatcher = StubDispatcher(owner="clown")
    return alarm, bus, audio


def _write_pool(path: Path, *, military_count: int = 5,
                domestic_count: int = 1, has_improv: bool = True) -> None:
    """Write a pool YAML on demand so tests can craft sparsity."""
    parts: list[str] = ["responses:"]
    for i in range(military_count):
        parts.append(
            f"  - id: mil_{i}\n"
            f"    tone: military\n"
            f'    text: "Curated military entry number {i}."'
        )
    for i in range(domestic_count):
        parts.append(
            f"  - id: dom_{i}\n"
            f"    tone: domestic\n"
            f'    text: "Curated domestic entry number {i}."'
        )
    if has_improv:
        parts.append(
            "  - id: improv_military\n"
            "    tone: improv\n"
            "    generate: true\n"
            "    style_seed: military\n"
        )
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


# ── Tests ───────────────────────────────────────────────────────────────────


async def test_curated_path_no_improv(tmpdir: Path) -> None:
    """Pool with only curated entries - fire selects one, audio
    sequence runs, alarm self-resolves via on_complete."""
    pool_path = tmpdir / "pool_curated.yaml"
    _write_pool(pool_path, military_count=3, has_improv=False)
    alarm, bus, audio = _make_alarm(pool_path)
    await alarm.start()

    await bus.publish("clown.detected", {
        "trigger": "verbal", "evidence": "test", "confidence": 0.9,
    })
    # Audio dispatch is awaited inside _on_detected; on_complete fires
    # immediately in the StubAudio path -> state should be RESOLVED.
    assert alarm.state == AlarmState.RESOLVED, alarm.state
    assert any(c[0] == "play_clown_sequence" for c in audio.calls)
    last = audio.calls[-1]
    assert "Curated military entry" in last[1]["announcement"], last
    print("PASS: curated-only path fires + self-resolves")


async def test_improv_layer1_standard_generation(tmpdir: Path) -> None:
    """>=3 in-style examples -> standard generation path. LLM is called
    once, generated text becomes the announcement."""
    pool_path = tmpdir / "pool_layer1.yaml"
    _write_pool(pool_path, military_count=5, has_improv=True)
    llm = StubLLM(responses=["Generated military response."])
    alarm, bus, audio = _make_alarm(pool_path, llm=llm)
    await alarm.start()

    # Force a generate-improv pick by tightening the dedup queue -
    # use the alarm's pool selection via a deterministic seed.
    import random as _r
    _r.seed(0)
    # Loop until we hit the improv slot (small pool, fast)
    max_attempts = 30
    fired = False
    for _ in range(max_attempts):
        await bus.publish("clown.detected", {
            "trigger": "verbal", "evidence": "test",
        })
        if llm.calls:
            fired = True
            break
        # Reset for next iteration so the alarm refire pathway works.
        alarm.state = AlarmState.INACTIVE
        alarm.fire_id = None
    assert fired, "improv slot never selected after many attempts"
    assert any(c[0] == "play_clown_sequence" for c in audio.calls)
    last = audio.calls[-1]
    assert last[1]["announcement"] == "Generated military response.", last
    # Generation event recorded for dashboard.
    events = alarm.recent_improv_events()
    assert events, "expected an improv event recorded"
    last_ev = events[-1]
    assert last_ev["outcome"] == "generated_standard", last_ev
    assert last_ev["examples_used_count"] >= MIN_EXAMPLES_FOR_GENERATION
    print("PASS: layer 1 - standard generation with >= 3 in-style examples")


async def test_improv_layer2_supplement(tmpdir: Path) -> None:
    """1-2 in-style examples -> supplemented from cross-style. Outcome
    recorded as `generated_with_supplement`."""
    pool_path = tmpdir / "pool_layer2.yaml"
    # Only 2 military, 3 domestic - military slot must supplement.
    _write_pool(pool_path, military_count=2, domestic_count=3,
                has_improv=True)
    llm = StubLLM(responses=["Supplemented response."])
    alarm, _bus, _audio = _make_alarm(pool_path, llm=llm)
    await alarm.start()

    # Drive the improv path directly so we don't depend on random.
    text = await alarm._generate_improv("military")
    assert text == "Supplemented response.", text
    events = alarm.recent_improv_events()
    last_ev = events[-1]
    assert last_ev["outcome"] == "generated_with_supplement", last_ev
    assert last_ev["examples_used_count"] == 2
    assert last_ev["cross_style_supplement_count"] >= 1
    print("PASS: layer 2 - supplement from cross-style when in-style thin")


async def test_improv_layer3_zero_examples_falls_back(tmpdir: Path) -> None:
    """Zero in-style examples -> no LLM call, fall back to curated
    entry. Outcome recorded as `fallback_zero_examples`."""
    pool_path = tmpdir / "pool_layer3.yaml"
    # No military entries at all; only domestic + an improv slot
    # whose style_seed is military.
    pool_path.write_text(
        "responses:\n"
        "  - id: dom_1\n    tone: domestic\n    text: \"Domestic A.\"\n"
        "  - id: dom_2\n    tone: domestic\n    text: \"Domestic B.\"\n"
        "  - id: improv_mil\n"
        "    tone: improv\n    generate: true\n    style_seed: military\n",
        encoding="utf-8",
    )
    llm = StubLLM(responses=["should not be called"])
    alarm, _bus, _audio = _make_alarm(pool_path, llm=llm)
    await alarm.start()

    text = await alarm._generate_improv("military")
    # Must NOT be the LLM response - should be a curated fallback.
    assert text != "should not be called", text
    assert "Domestic" in text or "response pool is empty" in text, text
    assert llm.calls == [], "LLM should not have been called"
    events = alarm.recent_improv_events()
    last_ev = events[-1]
    assert last_ev["outcome"] == "fallback_zero_examples", last_ev
    print("PASS: layer 3 - zero in-style examples -> curated fallback, no LLM")


async def test_improv_generation_failure_falls_back(tmpdir: Path) -> None:
    """LLM raises -> caught + fallback to curated of the same style.
    Outcome recorded as `fallback_generation_failed`."""
    pool_path = tmpdir / "pool_failure.yaml"
    _write_pool(pool_path, military_count=5, has_improv=True)
    llm = StubLLM(errors=[RuntimeError("LLM down")])
    alarm, _bus, _audio = _make_alarm(pool_path, llm=llm)
    await alarm.start()

    text = await alarm._generate_improv("military")
    # Should be a curated military response, not the (failed) generation.
    assert text and "Curated military entry" in text, text
    events = alarm.recent_improv_events()
    last_ev = events[-1]
    assert last_ev["outcome"] == "fallback_generation_failed", last_ev
    assert last_ev["error"] is not None
    print("PASS: LLM failure -> curated fallback of same style")


async def test_voice_cooldown_suppresses_detections(tmpdir: Path) -> None:
    """suppress_for_seconds blocks subsequent fires until the window
    expires."""
    pool_path = tmpdir / "pool_cooldown.yaml"
    _write_pool(pool_path, military_count=3, has_improv=False)
    alarm, bus, audio = _make_alarm(pool_path)
    await alarm.start()

    alarm.suppress_for_seconds(60.0, reason="test cooldown")
    pre_calls = len(audio.calls)
    await bus.publish("clown.detected", {
        "trigger": "verbal", "evidence": "blocked",
    })
    assert alarm.state == AlarmState.INACTIVE, alarm.state
    assert len(audio.calls) == pre_calls, "audio should not have fired"

    # reenable() clears cooldown; next fire goes through.
    alarm.reenable()
    await bus.publish("clown.detected", {
        "trigger": "verbal", "evidence": "fresh",
    })
    assert alarm.state == AlarmState.RESOLVED, alarm.state
    assert len(audio.calls) > pre_calls
    print("PASS: voice cooldown suppresses, reenable() clears")


async def test_parse_cooldown_phrases() -> None:
    """parse_cooldown_phrase covers the spec's natural-language inputs."""
    cases = [
        ("for an hour",         3600.0),
        ("for 30 minutes",      1800.0),
        ("for 30 min",          1800.0),
        ("for 90 seconds",        90.0),
    ]
    for phrase, expected in cases:
        s, _r = parse_cooldown_phrase(phrase)
        assert abs(s - expected) < 0.5, (
            f"{phrase!r} -> {s}s (expected {expected}s)"
        )
    # Indefinite signals via -1 sentinel.
    s, _r = parse_cooldown_phrase("until I say so")
    assert s == -1.0, s
    s, _r = parse_cooldown_phrase("indefinitely")
    assert s == -1.0, s
    # Unparsable returns 0 with a reason.
    s, r = parse_cooldown_phrase("gibberish")
    assert s == 0.0 and r == "unparsable", (s, r)
    print("PASS: parse_cooldown_phrase covers spec examples")


async def test_pool_reload_picks_up_edits(tmpdir: Path) -> None:
    """reload_pool re-reads the YAML - Cole's hot-reload workflow."""
    pool_path = tmpdir / "pool_reload.yaml"
    _write_pool(pool_path, military_count=2, domestic_count=0,
                has_improv=False)
    alarm, _bus, _audio = _make_alarm(pool_path)
    await alarm.start()
    assert len(alarm._pool) == 2, alarm._pool

    _write_pool(pool_path, military_count=5, domestic_count=0,
                has_improv=False)
    count = alarm.reload_pool()
    assert count == 5
    assert len(alarm._pool) == 5
    print("PASS: reload_pool reflects YAML edits")


async def main() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as raw_dir:
        tmpdir = Path(raw_dir)
        await test_curated_path_no_improv(tmpdir)
        await test_improv_layer1_standard_generation(tmpdir)
        await test_improv_layer2_supplement(tmpdir)
        await test_improv_layer3_zero_examples_falls_back(tmpdir)
        await test_improv_generation_failure_falls_back(tmpdir)
        await test_voice_cooldown_suppresses_detections(tmpdir)
        await test_parse_cooldown_phrases()
        await test_pool_reload_picks_up_edits(tmpdir)
    print("\nAll v4.1 clown alarm tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
