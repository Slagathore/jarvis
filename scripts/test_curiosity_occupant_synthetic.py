"""
JARVIS — synthetic test: non-Cole-centric proactive speech
==========================================================
Verifies the CuriosityEngine no longer hardcodes "Cole" into its
proactive-speech prompts, and instead addresses whoever is actually in
the room:

  - A recognized occupant is named in the prompt.
  - An unknown occupant -> generic second-person phrasing, no name.
  - check_async threads an explicit occupants list, and falls back to
    state.present when none is given.
  - REGRESSION GUARD: no topic prompt (or the system prompt) contains
    the word "Cole".

Run:  .venv\\Scripts\\python.exe scripts\\test_curiosity_occupant_synthetic.py
ASCII-only print output (Windows cp1252 console).
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.context.curiosity import CuriosityEngine, TOPIC_ORDER
from modules.context.state import ActivityState


class StubLLM:
    """Captures the messages it is handed so the test can inspect the
    prompt the CuriosityEngine built."""

    def __init__(self) -> None:
        self.last_messages = None

    async def chat(self, messages):
        self.last_messages = messages
        return "a generated line"


def _engine() -> tuple[CuriosityEngine, StubLLM]:
    llm = StubLLM()
    return CuriosityEngine({}, llm), llm


def _user(llm: StubLLM) -> str:
    return llm.last_messages[1]["content"]


def _system(llm: StubLLM) -> str:
    return llm.last_messages[0]["content"]


# ── tests ────────────────────────────────────────────────────────────────

async def test_generate_names_known_occupant() -> None:
    """A recognized occupant is named; "Cole" never appears."""
    engine, llm = _engine()
    await engine._generate("gaming_too_long", ActivityState(), ["Anna"], hours=3.0)
    user = _user(llm)
    assert "Anna" in user, user
    assert "Cole" not in user, user


async def test_generate_generic_when_unknown() -> None:
    """No recognized occupant -> generic, second-person, no name."""
    engine, llm = _engine()
    await engine._generate("nap_checkin", ActivityState(), [], minutes=20.0)
    user = _user(llm)
    assert "Cole" not in user, user
    assert "Anna" not in user, user
    assert "second person" in user.lower(), user


async def test_no_topic_mentions_cole() -> None:
    """REGRESSION GUARD — every topic prompt + the system prompt must be
    free of the hardcoded "Cole"."""
    engine, llm = _engine()
    for topic in TOPIC_ORDER:
        await engine._generate(
            topic, ActivityState(), [], hours=2.5, minutes=18.0
        )
        blob = _system(llm) + " || " + _user(llm)
        assert "Cole" not in blob, f"topic '{topic}' still says Cole: {blob}"


async def test_check_async_threads_occupants() -> None:
    """An explicit occupants list reaches the generated prompt."""
    engine, llm = _engine()
    engine._greeted_today = datetime.now()  # neutralize morning_greeting
    engine._activity_started["napping"] = datetime.now() - timedelta(minutes=20)
    state = ActivityState(activity="napping", interruptibility=0.9)
    result = await engine.check_async(state, occupants=["Anna"])
    assert result is not None, "nap_checkin should have fired"
    assert "Anna" in _user(llm), _user(llm)


async def test_check_async_falls_back_to_present() -> None:
    """With occupants=None, check_async uses state.present."""
    engine, llm = _engine()
    engine._greeted_today = datetime.now()
    engine._activity_started["napping"] = datetime.now() - timedelta(minutes=20)
    state = ActivityState(
        activity="napping", interruptibility=0.9, present=["Bob"]
    )
    result = await engine.check_async(state)  # occupants defaults -> present
    assert result is not None, "nap_checkin should have fired"
    assert "Bob" in _user(llm), _user(llm)


# ── runner ───────────────────────────────────────────────────────────────

async def _run() -> int:
    tests = [
        test_generate_names_known_occupant,
        test_generate_generic_when_unknown,
        test_no_topic_mentions_cole,
        test_check_async_threads_occupants,
        test_check_async_falls_back_to_present,
    ]
    passed = 0
    for test in tests:
        try:
            await test()
            print(f"[OK]   {test.__name__}")
            passed += 1
        except AssertionError as exc:
            print(f"[FAIL] {test.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"[ERR]  {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_run()))
