"""
JARVIS — Ambient Home AI
========================
Mission: First-run setup and environment validation. Ensures all dependencies,
         models, services, and data files are present before JARVIS attempts to
         launch. Idempotent — safe to run as many times as needed.

         Run: python scripts/setup.py

Modules: scripts/setup.py
Classes: (none)
Functions:
    check_python_version()   — Verify Python 3.10+
    check_venv()             — Verify running inside a venv
    check_packages()         — Verify all pip packages importable
    check_cuda()             — Verify GPU/CUDA availability
    check_ollama()           — Verify Ollama is running with required models
    check_mqtt()             — Verify Mosquitto reachable on localhost:1883
    check_microphone()       — Verify at least one audio input device
    check_data_files()       — Download yamnet_labels.csv if missing
    check_database()         — Initialize SQLite schema if missing
    print_summary(results)   — Print colored PASS/FAIL summary table
    main()                   — Run all checks, return exit code

Variables:
    PROJECT_ROOT  — Absolute path to project root
    GREEN/RED/YELLOW/CYAN/BOLD/RESET — ANSI color codes

#todo: Add check for ESPHome CLI installation
#todo: Add check for Piper TTS binary + voice model
#todo: Add check for openWakeWord "hey_jarvis" model download
#todo: Add check for available disk space (models + db can be large)
#todo: Add --fix flag that auto-runs pip install for missing packages
"""

import asyncio
import os
import subprocess
import sys
import urllib.request
from typing import Any, cast

import yaml

# ── Ensure we're running from project root ──────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# ── Color output helpers ────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def ok(msg: str):     print(f"  {GREEN}[OK]{RESET} {msg}")
def fail(msg: str):   print(f"  {RED}[X]{RESET} {msg}")
def warn(msg: str):   print(f"  {YELLOW}[!]{RESET} {msg}")
def info(msg: str):   print(f"  {CYAN}->{RESET} {msg}")
def header(msg: str): print(f"\n{BOLD}{msg}{RESET}")


def _load_config() -> dict:
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_python_version() -> bool:
    header("Python Version")
    major, minor = sys.version_info[:2]
    if major == 3 and minor >= 10:
        ok(f"Python {major}.{minor} (3.10+ required)")
        return True
    else:
        fail(f"Python {major}.{minor} — need 3.10 or higher")
        return False


def check_venv() -> bool:
    header("Virtual Environment")
    in_venv = (
        hasattr(sys, "real_prefix") or
        (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
    )
    if in_venv:
        ok(f"Active venv: {sys.prefix}")
        return True
    else:
        warn("No virtual environment detected")
        warn(r"Run: python -m venv venv && .\venv\Scripts\Activate.ps1")
        return False


def check_packages() -> bool:
    header("Python Packages")
    import importlib

    REQUIRED = {
        "faster_whisper":  "faster-whisper",
        "sounddevice":     "sounddevice",
        "soundfile":       "soundfile",
        "numpy":           "numpy",
        "openwakeword":    "openwakeword",
        "ollama":          "ollama",
        "ultralytics":     "ultralytics",
        "mediapipe":       "mediapipe",
        "cv2":             "opencv-python",
        "PIL":             "Pillow",
        "tensorflow":      "tensorflow",
        "paho":            "paho-mqtt",
        "aiomqtt":         "aiomqtt",
        "fastapi":         "fastapi",
        "uvicorn":         "uvicorn",
        "yaml":            "pyyaml",
        "loguru":          "loguru",
        "aiosqlite":       "aiosqlite",
        "psutil":          "psutil",
        "httpx":           "httpx",
        "aiofiles":        "aiofiles",
    }

    all_ok = True
    for module, package in REQUIRED.items():
        try:
            importlib.import_module(module)
            ok(package)
        except ImportError:
            fail(f"{package} — run: pip install {package}")
            all_ok = False

    # Windows-only
    if sys.platform == "win32":
        try:
            import win32gui  # noqa: F401
            ok("pywin32")
        except ImportError:
            warn("pywin32 not installed — PC activity monitoring disabled")
            warn("Run: pip install pywin32")

    return all_ok


def check_cuda() -> bool:
    header("GPU / CUDA")
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            ok(f"CUDA available: {name} ({vram:.1f}GB VRAM)")
            return True
        else:
            warn("CUDA not available — will use CPU (slower)")
            return True  # Not fatal
    except ImportError:
        # torch not installed; check via ctranslate2 (used by faster-whisper)
        try:
            import ctranslate2
            supported = ctranslate2.get_supported_compute_types("cuda")
            if "float16" in supported or "int8_float16" in supported:
                ok("CTranslate2 CUDA support detected")
                return True
        except Exception:
            pass
        warn("Cannot verify CUDA — will be confirmed on first Whisper run")
        return True


def check_ollama() -> bool:
    header("Ollama")
    import httpx

    config = _load_config()
    ollama_cfg = config.get("ollama", {})
    base_url = str(ollama_cfg.get("base_url", "http://localhost:11434")).rstrip("/")
    text_model = str(ollama_cfg.get("model", "")).strip()
    vision_model = str(ollama_cfg.get("vision_model", text_model)).strip()

    try:
        r = httpx.get(f"{base_url}/api/tags", timeout=5)
        if r.status_code != 200:
            fail("Ollama running but API returned non-200")
            return False

        models = [m["name"] for m in r.json().get("models", [])]
        model_set = {str(model).lower() for model in models}

        has_llm = text_model.lower() in model_set
        if has_llm:
            ok(f"Configured chat model found: {text_model}")
        else:
            fail(f"Configured chat model missing: {text_model}")
            info(f"Run: ollama pull {text_model}")

        has_vision = vision_model.lower() in model_set
        if has_vision:
            if vision_model == text_model:
                ok(f"Configured vision model shares the chat model: {vision_model}")
            else:
                ok(f"Configured vision model found: {vision_model}")
        else:
            warn(f"Configured vision model missing: {vision_model}")
            info(f"Run: ollama pull {vision_model}")

        return has_llm

    except httpx.ConnectError:
        fail("Ollama not running")
        info("Start Ollama: ollama serve")
        if vision_model and vision_model != text_model:
            info(f"Then pull models: ollama pull {text_model} && ollama pull {vision_model}")
        else:
            info(f"Then pull model: ollama pull {text_model}")
        return False


def check_mqtt() -> bool:
    header("MQTT Broker (Mosquitto)")
    import socket

    try:
        sock = socket.create_connection(("localhost", 1883), timeout=3)
        sock.close()
        ok("Mosquitto reachable on localhost:1883")
        return True
    except (ConnectionRefusedError, socket.timeout, OSError):
        fail("Cannot reach Mosquitto on localhost:1883")
        info("Install: https://mosquitto.org/download/")
        info("Start as Windows service or run: mosquitto -v")
        return False


def check_microphone() -> bool:
    header("Microphone")
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devices = []
        for device in devices:
            device_info = cast(dict[str, Any], device)
            if int(device_info.get("max_input_channels", 0)) > 0:
                input_devices.append(device_info)

        if input_devices:
            ok(f"Found {len(input_devices)} input device(s)")
            for d in input_devices[:3]:
                info(f"  -> {d.get('name', 'Unknown device')}")
            return True
        else:
            fail("No input devices found")
            return False

    except Exception as e:
        fail(f"Microphone check failed: {e}")
        return False


def check_data_files() -> bool:
    header("Data Files")
    os.makedirs("data", exist_ok=True)

    # YAMNet labels CSV
    labels_path = "data/yamnet_labels.csv"
    if os.path.exists(labels_path):
        ok("yamnet_labels.csv present")
    else:
        info("Downloading YAMNet labels CSV...")
        try:
            url = (
                "https://raw.githubusercontent.com/tensorflow/models/master/"
                "research/audioset/yamnet/yamnet_class_map.csv"
            )
            urllib.request.urlretrieve(url, labels_path)
            ok("yamnet_labels.csv downloaded")
        except Exception as e:
            warn(f"Could not download YAMNet labels: {e}")
            warn("Audio appliance classification will be limited")

    return True


def check_database() -> bool:
    header("Database")
    import sqlite3

    db_path = "data/jarvis.db"
    try:
        conn = sqlite3.connect(db_path)
        conn.executescript("""
            PRAGMA journal_mode=WAL;

            CREATE TABLE IF NOT EXISTS events (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT    NOT NULL,
                room          TEXT    NOT NULL,
                type          TEXT    NOT NULL,
                content       TEXT    NOT NULL,
                acknowledged  INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS room_baselines (
                room          TEXT PRIMARY KEY,
                baseline_desc TEXT,
                updated_at    TEXT
            );

            CREATE TABLE IF NOT EXISTS user_routines (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                day_of_week TEXT,
                hour        INTEGER,
                activity    TEXT,
                frequency   INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS reminders (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                message        TEXT NOT NULL,
                trigger_time   TEXT,
                recurring      INTEGER DEFAULT 0,
                last_triggered TEXT
            );

            CREATE TABLE IF NOT EXISTS conversation_log (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                room      TEXT NOT NULL,
                role      TEXT NOT NULL,
                content   TEXT NOT NULL
            );
        """)
        conn.commit()
        conn.close()
        ok(f"Database ready: {db_path}")
        return True

    except Exception as e:
        fail(f"Database init failed: {e}")
        return False


def print_summary(results: dict) -> None:
    print(f"\n{'=' * 52}")
    print(f"{BOLD}  SETUP SUMMARY{RESET}")
    print(f"{'=' * 52}")

    for check, passed_check in results.items():
        status = f"{GREEN}PASS{RESET}" if passed_check else f"{RED}FAIL{RESET}"
        print(f"  {status}  {check}")

    print(f"{'=' * 52}")
    passed = sum(1 for v in results.values() if v)
    total  = len(results)

    if passed == total:
        print(f"\n{GREEN}{BOLD}  [OK] All checks passed. Jarvis is ready.{RESET}")
        print(f"  Run: {CYAN}python main.py{RESET}\n")
    else:
        failed = total - passed
        print(f"\n{RED}{BOLD}  [X] {failed} check(s) failed. Fix the issues above.{RESET}\n")


def main() -> int:
    print(f"\n{BOLD}{'=' * 52}{RESET}")
    print(f"{BOLD}  JARVIS SETUP VALIDATOR{RESET}")
    print(f"{BOLD}{'=' * 52}{RESET}")

    results = {
        "Python 3.10+":        check_python_version(),
        "Virtual environment": check_venv(),
        "Python packages":     check_packages(),
        "GPU / CUDA":          check_cuda(),
        "Ollama + models":     check_ollama(),
        "MQTT broker":         check_mqtt(),
        "Microphone":          check_microphone(),
        "Data files":          check_data_files(),
        "Database":            check_database(),
    }

    print_summary(results)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
