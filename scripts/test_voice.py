"""
JARVIS — Ambient Home AI
========================
Mission: Phase 1 test script. Validates the entire voice pipeline end-to-end
         without needing the full orchestrator. Tests wake word detection,
         Whisper STT, Ollama LLM, and Piper TTS in isolation. Each component
         gets its own PASS/FAIL test. Safe to run repeatedly.

         Run: python scripts/test_voice.py

Modules: scripts/test_voice.py
Classes: (none)
Functions:
    test_config()           — Load and validate config.yaml
    test_stt()              — Load Whisper, transcribe a test audio clip
    test_tts()              — Initialize Piper, synthesize a test phrase
    test_llm()              — Connect to Ollama, get a one-sentence response
    test_wake_word()        — Load openWakeWord model (no recording)
    test_audio_devices()    — List available input/output devices
    run_tests()             — Run all tests, print summary

#todo: Add actual audio playback test (play 1 second of silence)
#todo: Add latency measurement for each pipeline component
#todo: Add end-to-end test: play audio → wake word → STT → LLM → TTS
"""

import asyncio
import os
import sys
import time
from typing import Any, cast

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import yaml
from loguru import logger

# Suppress noisy library logs during test
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


def test_config() -> bool:
    print(f"\n{BOLD}[1] Config{RESET}")
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        assert "voice" in config, "Missing 'voice' section"
        assert "ollama" in config, "Missing 'ollama' section"
        assert "system" in config, "Missing 'system' section"
        ok("config.yaml loaded and valid")
        return True
    except Exception as e:
        fail(f"Config error: {e}")
        return False


def test_audio_devices() -> bool:
    print(f"\n{BOLD}[2] Audio Devices{RESET}")
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        inputs = []
        outputs = []
        for device in devices:
            device_info = cast(dict[str, Any], device)
            if int(device_info.get("max_input_channels", 0)) > 0:
                inputs.append(device_info)
            if int(device_info.get("max_output_channels", 0)) > 0:
                outputs.append(device_info)
        ok(f"{len(inputs)} input device(s), {len(outputs)} output device(s)")
        for d in inputs[:2]:
            info(f"  INPUT:  {d.get('name', 'Unknown device')}")
        for d in outputs[:2]:
            info(f"  OUTPUT: {d.get('name', 'Unknown device')}")
        return True
    except Exception as e:
        fail(f"Audio device error: {e}")
        return False


def test_stt() -> bool:
    print(f"\n{BOLD}[3] Whisper STT{RESET}")
    try:
        from modules.voice.stt import WhisperSTT

        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        vcfg = config["voice"]["whisper"]

        # BUG FIX: WhisperSTT takes config: dict — reads its own settings internally
        stt = WhisperSTT(config)

        t0 = time.time()
        stt.load()
        elapsed = time.time() - t0
        ok(f"Whisper '{vcfg['model_size']}' loaded in {elapsed:.1f}s")

        # Transcribe 1 second of silence (should return empty or near-empty)
        silence = np.zeros(16000, dtype=np.float32)
        result = stt.transcribe(silence)
        ok(f"Transcription test complete (empty audio -> {result!r})")
        return True

    except Exception as e:
        fail(f"STT error: {e}")
        return False


def test_tts() -> bool:
    print(f"\n{BOLD}[4] Piper TTS{RESET}")
    try:
        from modules.voice.tts import PiperTTS

        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        vcfg = config["voice"]["tts"]

        # BUG FIX: PiperTTS takes config: dict — reads its own settings internally
        tts = PiperTTS(config)

        t0 = time.time()
        tts.speak("Jarvis online. Voice pipeline operational.")
        elapsed = time.time() - t0
        ok(f"TTS synthesis + playback completed in {elapsed:.1f}s")
        return True

    except Exception as e:
        fail(f"TTS error: {e}")
        return False


async def test_llm() -> bool:
    print(f"\n{BOLD}[5] Ollama LLM{RESET}")
    try:
        from modules.brain.llm import OllamaLLM

        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        ocfg = config["ollama"]

        # BUG FIX: OllamaLLM takes config: dict, not flat kwargs. Also chat() not chat_async().
        llm = OllamaLLM(config)

        t0 = time.time()
        response = await llm.chat([
            {"role": "user", "content": "Say exactly: Voice pipeline test successful."}
        ])
        elapsed = time.time() - t0

        if response:
            ok(f"LLM response in {elapsed:.1f}s: {response!r}")
            return True
        else:
            fail("LLM returned empty response")
            return False

    except Exception as e:
        fail(f"LLM error: {e}")
        return False


def test_wake_word() -> bool:
    print(f"\n{BOLD}[6] Wake Word (model load only){RESET}")
    try:
        from core.event_bus import EventBus
        from modules.voice.wake_word import WakeWordDetector

        with open("config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        wcfg = config["voice"]["wake_word"]

        bus = EventBus()
        # BUG FIX: WakeWordDetector takes (config, bus) — not flat kwargs
        # Params were: model_name=, sensitivity=, cooldown_seconds=, event_bus=
        # Actual constructor: WakeWordDetector(config: dict, bus: EventBus, room=, device=)
        detector = WakeWordDetector(config=config, bus=bus)
        ok(f"WakeWordDetector initialized (model: {wcfg['model']})")
        info("Not starting mic listen — call listen_forever() in orchestrator")
        return True

    except Exception as e:
        fail(f"Wake word init error: {e}")
        return False


def print_summary(results: dict) -> None:
    print(f"\n{'=' * 50}")
    print(f"{BOLD}  VOICE PIPELINE TEST SUMMARY{RESET}")
    print(f"{'=' * 50}")
    for check, passed in results.items():
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  {status}  {check}")
    print(f"{'=' * 50}")
    if all(results.values()):
        print(f"\n{GREEN}{BOLD}  [OK] All voice tests passed.{RESET}")
        print(f"  Run: {CYAN}python main.py{RESET}\n")
    else:
        failed = sum(1 for v in results.values() if not v)
        print(f"\n{RED}{BOLD}  [X] {failed} test(s) failed.{RESET}\n")


async def run_tests() -> int:
    results = {
        "Config":       test_config(),
        "Audio Devices": test_audio_devices(),
        "Whisper STT":  test_stt(),
        "Piper TTS":    test_tts(),
        "Ollama LLM":   await test_llm(),
        "Wake Word":    test_wake_word(),
    }
    print_summary(results)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(run_tests()))
