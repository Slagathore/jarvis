"""
JARVIS — Ambient Home AI
========================
Mission: Phase 2 test script. Validates context-awareness pipeline: PC monitor,
         audio classifier, state fusion, interruptibility scoring, and sleep
         tracker. Runs without a microphone or camera — uses simulated signals.

         Run: python scripts/test_context.py

Modules: scripts/test_context.py
Classes: (none)
Functions:
    test_pc_monitor()        — Get current active window + classify activity
    test_audio_classifier()  — Load YAMNet, run on silence
    test_state_fusion()      — Fuse simulated signals into an ActivityState
    test_interruptibility()  — Verify scoring for known activities
    test_sleep_tracker()     — Simulate posture sequence → sleep detection
    test_curiosity()         — Load engine, verify no crash with UNKNOWN_STATE
    run_tests()              — Run all tests, print summary

#todo: Add test for curiosity topic cooldown (fire once, verify silenced)
#todo: Add test that verifies gaming activity blocks conversation-priority speech
#todo: Add test that verifies appliance tracker transitions (idle→running→done)
"""

import asyncio
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import yaml
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="WARNING", colorize=True)

GREEN = "\033[92m"
RED   = "\033[91m"
CYAN  = "\033[96m"
BOLD  = "\033[1m"
RESET = "\033[0m"


def ok(msg: str):   print(f"  {GREEN}[OK]{RESET} {msg}")
def fail(msg: str): print(f"  {RED}[X]{RESET} {msg}")
def info(msg: str): print(f"  {CYAN}->{RESET} {msg}")


def _load_config() -> dict:
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_pc_monitor() -> bool:
    print(f"\n{BOLD}[1] PC Monitor{RESET}")
    try:
        from modules.activity.pc_monitor import PCMonitor
        config = _load_config()
        monitor = PCMonitor(config=config)

        signal = monitor.get_signal()
        ok(f"PC signal: {signal}")
        return True

    except Exception as e:
        fail(f"PC monitor error: {e}")
        return False


async def test_audio_classifier() -> bool:
    # BUG FIX: AudioClassifier takes config: dict (not flat kwargs).
    # BUG FIX: AudioClassifier.load() is async def — must be awaited in an async function.
    print(f"\n{BOLD}[2] Audio Classifier (YAMNet){RESET}")
    try:
        import numpy as np

        from modules.activity.audio_classifier import AudioClassifier

        config = _load_config()
        classifier = AudioClassifier(config)
        await classifier.load()

        # Classify 3 seconds of silence
        silence = np.zeros(16000 * 3, dtype=np.float32)
        result = classifier.classify(silence)
        ok(f"Audio classification complete: {result}")
        return True

    except Exception as e:
        fail(f"Audio classifier error: {e}")
        return False


async def test_state_fusion() -> bool:
    # BUG FIX: StateFusion.fuse() is async def — must be awaited in an async function.
    print(f"\n{BOLD}[3] State Fusion{RESET}")
    try:
        from modules.context.state_fusion import StateFusion

        config = _load_config()
        fusion = StateFusion(config=config)

        # Simulate a gaming signal
        signals = {
            "pc": {
                "activity": "gaming",
                "process_name": "LeagueofLegends.exe",
                "window_title": "League of Legends",
                "confidence": 0.95,
                "context": {"game": "League of Legends"},
            }
        }
        state = await fusion.fuse(signals, room="office")
        ok(f"Fused state: {state.activity} (confidence={state.confidence:.2f})")
        assert state.activity == "gaming", f"Expected 'gaming', got '{state.activity}'"
        ok("Activity correctly classified as gaming")
        return True

    except Exception as e:
        fail(f"State fusion error: {e}")
        return False


def test_interruptibility() -> bool:
    print(f"\n{BOLD}[4] Interruptibility Scoring{RESET}")
    try:
        from modules.context.interruptibility import InterruptibilityManager
        from modules.context.state import ActivityState

        config = _load_config()
        # BUG FIX: InterruptibilityManager takes config: dict — not flat kwargs
        mgr = InterruptibilityManager(config)

        # Gaming should have low interruptibility
        gaming_score = mgr.get_score("gaming")
        idle_score   = mgr.get_score("idle")
        ok(f"Gaming interruptibility score: {gaming_score}")
        ok(f"Idle interruptibility score: {idle_score}")
        assert gaming_score < idle_score, "Gaming should score lower than idle"
        ok("Scoring order correct (gaming < idle)")

        # Test can_interrupt
        gaming_state = ActivityState(
            activity="gaming",
            location="office",
            interruptibility=gaming_score,
            confidence=0.9,
            signals=["pc"],
        )
        can = mgr.can_interrupt(gaming_state, priority="ambient")
        ok(f"Can interrupt gaming with ambient priority: {can} (expected False)")
        assert not can, "Should NOT be able to interrupt gaming with ambient priority"

        return True

    except Exception as e:
        fail(f"Interruptibility error: {e}")
        return False


def test_sleep_tracker() -> bool:
    print(f"\n{BOLD}[5] Sleep Tracker{RESET}")
    try:
        from modules.context.sleep_tracker import SleepTracker

        config = _load_config()
        # BUG FIX: SleepTracker takes config: dict — not zero args
        tracker = SleepTracker(config)

        # Simulate lying down with lights off
        for _ in range(5):
            tracker.update(posture="lying", lights_on=False, room="bedroom")

        signal = tracker.get_sleep_signal()
        ok(f"Sleep signal after lying posture: {signal}")

        tracker.record_wakeup()
        ok("Wakeup recorded successfully")
        return True

    except Exception as e:
        fail(f"Sleep tracker error: {e}")
        return False


async def test_curiosity() -> bool:
    print(f"\n{BOLD}[6] Curiosity Engine (no LLM call){RESET}")
    try:
        from modules.context.curiosity import CuriosityEngine
        from modules.context.state import UNKNOWN_STATE

        config = _load_config()

        # Use None for LLM — curiosity engine should handle gracefully
        engine = CuriosityEngine(llm=None, config=config)

        # With UNKNOWN_STATE and no cooldowns expired, should return None
        result = await engine.check_async(UNKNOWN_STATE)
        ok(f"Curiosity check with UNKNOWN_STATE returned: {result!r}")
        return True

    except Exception as e:
        fail(f"Curiosity engine error: {e}")
        return False


def print_summary(results: dict) -> None:
    print(f"\n{'=' * 50}")
    print(f"{BOLD}  CONTEXT PIPELINE TEST SUMMARY{RESET}")
    print(f"{'=' * 50}")
    for check, passed in results.items():
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  {status}  {check}")
    print(f"{'=' * 50}")
    if all(results.values()):
        print(f"\n{GREEN}{BOLD}  [OK] All context tests passed.{RESET}\n")
    else:
        failed = sum(1 for v in results.values() if not v)
        print(f"\n{RED}{BOLD}  [X] {failed} test(s) failed.{RESET}\n")


async def run_tests() -> int:
    results = {
        "PC Monitor":        test_pc_monitor(),
        # BUG FIX: test_audio_classifier and test_state_fusion are now async
        "Audio Classifier":  await test_audio_classifier(),
        "State Fusion":      await test_state_fusion(),
        "Interruptibility":  test_interruptibility(),
        "Sleep Tracker":     test_sleep_tracker(),
        "Curiosity Engine":  await test_curiosity(),
    }
    print_summary(results)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(run_tests()))
