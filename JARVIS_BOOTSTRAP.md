# JARVIS — VS CODE AGENT BOOTSTRAP GUIDE
> **Read this entire document before writing a single line of code.**  
> **Owner:** Cole  
> **Reviewer:** Boss  
> **Status:** Ready to build  

---

## WHAT YOU ARE BUILDING

Jarvis is an ambient home AI — not a chatbot, not a smart speaker clone. It:
- Listens for a wake word from any room via ESP32-CAM nodes
- Understands what the user is doing (gaming, sleeping, cooking) and respects it
- Watches rooms via camera for mess, posture, and presence
- Speaks proactively when it has something worth saying
- Remembers conversations, room states, and user patterns across sessions

The architecture is a Python async orchestrator running on a local PC (RTX 4070 Ti),
with ESP32-CAM nodes as dumb audio/video terminals communicating over MQTT.
All ML runs locally. No cloud. No subscriptions.

**This is being built as three things simultaneously:**
1. Personal daily-use tool
2. Demonstrable sellable product
3. Portfolio piece for technical review

Write accordingly. Every file will be read by someone smart.

---

## AGENT INSTRUCTIONS — READ CAREFULLY

You are a senior Python engineer implementing this project from scratch.

Rules:
- **Never use placeholder code.** Every function must be real and working.
- **Comment the WHY, not the WHAT.** Assume the reader can read Python.
- **Config over code.** Any tunable value goes in `config.yaml`, never hardcoded.
- **Async everywhere.** All I/O is non-blocking. Use `asyncio.to_thread()` for blocking calls.
- **Loguru, not print.** Zero `print()` statements in production code. Use `logger`.
- **One responsibility per file.** If a module is doing two things, split it.
- **Test scripts for every phase.** Each phase ends with a passing test script.
- **Type hints on every function signature.** No exceptions.

When you finish a file, immediately move to the next. Do not stop to summarize.
Build in the order defined below. The order matters — later files import earlier ones.

---

## PHASE 0 — SCAFFOLD & ENVIRONMENT
**Build this first. Nothing else works without it.**

### Step 1 — Create project structure

Create every file and folder listed below. Empty `__init__.py` files are fine for now.

```
jarvis/
├── main.py
├── config.yaml
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
├── core/
│   ├── __init__.py
│   ├── orchestrator.py
│   ├── event_bus.py
│   └── exceptions.py
│
├── modules/
│   ├── __init__.py
│   ├── voice/
│   │   ├── __init__.py
│   │   ├── stt.py
│   │   ├── tts.py
│   │   ├── wake_word.py
│   │   └── audio_utils.py
│   ├── brain/
│   │   ├── __init__.py
│   │   ├── llm.py
│   │   ├── session.py
│   │   └── prompt_builder.py
│   ├── context/
│   │   ├── __init__.py
│   │   ├── state.py
│   │   ├── state_fusion.py
│   │   ├── interruptibility.py
│   │   ├── curiosity.py
│   │   └── sleep_tracker.py
│   ├── activity/
│   │   ├── __init__.py
│   │   ├── pc_monitor.py
│   │   ├── audio_classifier.py
│   │   └── appliance_tracker.py
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── camera_manager.py
│   │   ├── object_detector.py
│   │   ├── posture_analyzer.py
│   │   ├── scene_analyzer.py
│   │   └── light_detector.py
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── database.py
│   │   ├── event_log.py
│   │   └── room_baselines.py
│   └── network/
│       ├── __init__.py
│       ├── mqtt_client.py
│       └── node_manager.py
│
├── dashboard/
│   ├── server.py
│   ├── static/
│   │   ├── index.html
│   │   ├── style.css
│   │   └── app.js
│   └── __init__.py
│
├── hardware/
│   └── esphome/
│       ├── secrets.yaml.example
│       ├── node_base.yaml
│       ├── node_office.yaml
│       ├── node_bedroom.yaml
│       └── node_kitchen.yaml
│
├── scripts/
│   ├── setup.py
│   ├── test_voice.py
│   ├── test_context.py
│   ├── test_vision.py
│   └── test_mqtt.py
│
└── data/
    ├── .gitkeep
    └── yamnet_labels.csv   ← downloaded by setup.py
```

---

### Step 2 — Write `requirements.txt`

```
# ── Core ───────────────────────────────────────────────────────────────────
pyyaml==6.0.2
python-dotenv==1.0.1
loguru==0.7.2
httpx==0.27.2
aiofiles==23.2.1

# ── Voice Pipeline ─────────────────────────────────────────────────────────
faster-whisper==1.1.0
sounddevice==0.5.1
soundfile==0.12.1
numpy==1.26.4
openwakeword==0.6.0

# ── LLM / Brain ────────────────────────────────────────────────────────────
ollama==0.3.3

# ── Vision ─────────────────────────────────────────────────────────────────
ultralytics==8.3.0
mediapipe==0.10.14
opencv-python==4.10.0.84
Pillow==10.4.0

# ── Audio Classification ───────────────────────────────────────────────────
tensorflow==2.17.0
tensorflow-hub==0.16.1

# ── MQTT ───────────────────────────────────────────────────────────────────
paho-mqtt==2.1.0
asyncio-mqtt==0.16.2

# ── Database ───────────────────────────────────────────────────────────────
aiosqlite==0.20.0

# ── Windows Process Monitor ────────────────────────────────────────────────
pywin32==306
psutil==6.1.0

# ── Dashboard (real-time web UI) ───────────────────────────────────────────
fastapi==0.115.0
uvicorn==0.32.0
websockets==13.1

# ── Testing ────────────────────────────────────────────────────────────────
pytest==8.3.3
pytest-asyncio==0.24.0
```

---

### Step 3 — Write `config.yaml`

This is the single source of truth for all tunable values.
**Nothing in this file should ever be hardcoded in Python.**

```yaml
# ═══════════════════════════════════════════════════════════════════════════
# JARVIS MASTER CONFIGURATION
# All tunable values live here. Edit this file, not the source code.
# ═══════════════════════════════════════════════════════════════════════════

system:
  name: "Jarvis"
  version: "1.0.0"
  log_level: "INFO"           # DEBUG | INFO | WARNING | ERROR
  data_dir: "data/"
  dashboard_enabled: true
  dashboard_port: 7070
  dashboard_host: "0.0.0.0"  # Set to "127.0.0.1" to restrict to localhost

ollama:
  model: "llama3.1:8b"
  vision_model: "moondream"
  base_url: "http://localhost:11434"
  timeout_seconds: 30
  # Jarvis's core personality and context — edit this to your taste
  system_prompt: |
    You are Jarvis, an ambient home AI assistant for Cole.
    You live in his house and pay attention to what's going on around him.
    You are observant, helpful, and have a dry wit. You know when to stay quiet.
    You speak like a smart friend, not a customer service bot.
    Never say "I notice" or "I observe" — just say what you see.
    Keep responses short unless asked for detail.
    You know Cole has ADHD, works from home, and lives with his partners Anna and Sophie.
    You have five cats in the house. You find them mildly chaotic and entirely worth it.

voice:
  whisper:
    model_size: "base"        # tiny | base | small | medium | large-v3
    device: "cuda"            # cuda | cpu
    compute_type: "float16"   # float16 (GPU) | int8 (CPU)
    language: "en"
    beam_size: 1              # 1 = fastest, 5 = most accurate
  tts:
    engine: "piper"           # piper | xtts
    voice: "en_US-ryan-high"  # Piper voice model name
    speed: 1.0
  wake_word:
    model: "hey_jarvis"       # openWakeWord model name
    sensitivity: 0.5          # 0.0-1.0, higher = more sensitive (more false positives)
    cooldown_seconds: 2
  recording:
    sample_rate: 16000
    silence_threshold_db: -40
    silence_duration_ms: 800   # Stop recording after this much silence
    max_duration_seconds: 30

mqtt:
  broker: "localhost"
  port: 1883
  username: ""                 # Set in .env if using auth: MQTT_USERNAME
  password: ""                 # Set in .env if using auth: MQTT_PASSWORD
  keepalive: 60
  reconnect_delay_seconds: 5
  topics:
    wake:      "jarvis/nodes/{room}/wake"
    audio_in:  "jarvis/nodes/{room}/audio/in"
    audio_out: "jarvis/nodes/{room}/audio/out"
    status:    "jarvis/nodes/{room}/status"
    vision:    "jarvis/vision/{room}/frame"
    reminder:  "jarvis/events/reminder"
    state:     "jarvis/system/state"         # Broadcast current activity state

rooms:
  - id: "office"
    display_name: "Office"
    camera_source: 0           # USB webcam, OpenCV device index
    has_node: false            # Set true when ESP32-CAM is deployed
    node_ip: null
  - id: "bedroom"
    display_name: "Bedroom"
    camera_source: null        # null until ESP32-CAM deployed
    has_node: false
    node_ip: null
  - id: "kitchen"
    display_name: "Kitchen"
    camera_source: null
    has_node: false
    node_ip: null
  - id: "living_room"
    display_name: "Living Room"
    camera_source: null
    has_node: false
    node_ip: null

context:
  vision_scan_interval_minutes: 5
  pc_poll_interval_seconds: 10
  audio_classify_window_seconds: 3
  posture_analysis_fps: 1

interruptibility:
  quiet_hours_start: "22:00"
  quiet_hours_end: "08:00"
  interrupt_cooldown_minutes: 5
  activity_scores:
    sleeping:        0.00
    napping:         0.00
    gaming:          0.15
    coding:          0.20
    reading:         0.30
    watching_media:  0.25
    video_call:      0.05
    cooking:         0.70
    eating:          0.50
    browsing_social: 0.80
    browsing_general: 0.65
    writing:         0.25
    note_taking:     0.40
    ai_image_gen:    0.60
    idle:            0.90
    away:            0.00
    unknown:         0.50

curiosity:
  enabled: true
  topic_cooldowns_hours:
    gaming:    4
    reading:   6
    nap_check: 12
    morning:   24
    cooking:   2
    comfyui:   3
    coding:    8

appliances:
  washer_min_cycle_minutes: 25
  dryer_done_urgency: 0.8
  dishwasher_done_urgency: 0.4
  silence_done_threshold_seconds: 120

memory:
  db_path: "data/jarvis.db"
  max_conversation_turns: 20
  room_baseline_update_hours: 24
```

---

### Step 4 — Write `.env.example`

```bash
# Copy this to .env and fill in your values
# .env is in .gitignore — never commit it

# MQTT credentials (leave empty if Mosquitto has no auth)
MQTT_USERNAME=
MQTT_PASSWORD=

# WiFi credentials for ESPHome nodes
WIFI_SSID=your_network_name
WIFI_PASSWORD=your_wifi_password

# OTA password for ESPHome over-the-air updates
OTA_PASSWORD=change_me_to_something_strong

# MQTT broker IP as seen by ESP32 nodes (your PC's local IP)
MQTT_BROKER_IP=192.168.1.X
```

---

### Step 5 — Write `.gitignore`

```gitignore
# Environment
.env
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python

# Data
data/jarvis.db
data/*.log
data/test_frame_*.jpg

# ESPHome secrets
hardware/esphome/secrets.yaml

# Models (too large for git)
data/voices/
*.pt
*.onnx
*.gguf

# VS Code
.vscode/

# OS
.DS_Store
Thumbs.db
```

---

### Step 6 — Write `scripts/setup.py`

This is the first script run. It validates the full environment,
downloads required models, and creates the database.
It must print a clear PASS/FAIL for every check.

```python
"""
scripts/setup.py

First-run setup and environment validation.
Run: python scripts/setup.py

Checks:
  - Python version
  - Virtual environment active
  - All pip packages installed
  - CUDA / GPU available
  - Ollama running with required models
  - Mosquitto MQTT broker reachable
  - Microphone accessible
  - Required data files present (downloads if missing)
  - Database initialized

This script is safe to run multiple times. It is idempotent.
"""

import sys
import os
import subprocess
import asyncio
import urllib.request

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

def ok(msg: str):    print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg: str):  print(f"  {RED}✗{RESET} {msg}")
def warn(msg: str):  print(f"  {YELLOW}⚠{RESET} {msg}")
def info(msg: str):  print(f"  {CYAN}→{RESET} {msg}")
def header(msg: str): print(f"\n{BOLD}{msg}{RESET}")


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
        hasattr(sys, 'real_prefix') or
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    )
    if in_venv:
        ok(f"Active venv: {sys.prefix}")
        return True
    else:
        warn("No virtual environment detected")
        warn("Run: python -m venv venv && .\\venv\\Scripts\\Activate.ps1")
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
        "fastapi":         "fastapi",
        "uvicorn":         "uvicorn",
        "yaml":            "pyyaml",
        "loguru":          "loguru",
        "aiosqlite":       "aiosqlite",
        "psutil":          "psutil",
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
            import win32gui
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
            warn("Whisper will still work, just slower")
            return True  # Not fatal
    except ImportError:
        # torch not installed — check via faster-whisper's own CUDA check
        try:
            import ctranslate2
            if "cuda" in ctranslate2.get_supported_compute_types("cuda"):
                ok("CTranslate2 CUDA support detected")
                return True
        except Exception:
            pass
        warn("Cannot verify CUDA — will be confirmed on first Whisper run")
        return True


def check_ollama() -> bool:
    header("Ollama")
    import httpx

    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code != 200:
            fail("Ollama running but API returned non-200")
            return False

        models = [m["name"] for m in r.json().get("models", [])]

        # Check for main LLM
        has_llm = any("llama3" in m or "mistral" in m or "phi" in m for m in models)
        if has_llm:
            ok(f"LLM model found: {[m for m in models if any(x in m for x in ['llama', 'mistral', 'phi'])]}")
        else:
            fail("No suitable LLM model found")
            info("Run: ollama pull llama3.1:8b")

        # Check for vision model
        has_vision = any("moondream" in m or "llava" in m for m in models)
        if has_vision:
            ok(f"Vision model found: {[m for m in models if any(x in m for x in ['moondream', 'llava'])]}")
        else:
            warn("No vision model found — scene analysis will be disabled")
            info("Run: ollama pull moondream")

        return has_llm

    except httpx.ConnectError:
        fail("Ollama not running")
        info("Start Ollama: ollama serve")
        info("Then pull models: ollama pull llama3.1:8b && ollama pull moondream")
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
        input_devices = [d for d in devices if d["max_input_channels"] > 0]

        if input_devices:
            ok(f"Found {len(input_devices)} input device(s)")
            for i, d in enumerate(input_devices[:3]):
                info(f"  [{i}] {d['name']}")
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

    # YAMNet labels
    labels_path = "data/yamnet_labels.csv"
    if os.path.exists(labels_path):
        ok("yamnet_labels.csv present")
    else:
        info("Downloading YAMNet labels...")
        try:
            url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
            urllib.request.urlretrieve(url, labels_path)
            ok("yamnet_labels.csv downloaded")
        except Exception as e:
            warn(f"Could not download YAMNet labels: {e}")
            warn("Audio classification will be limited")

    return True


def check_database() -> bool:
    header("Database")
    import sqlite3

    db_path = "data/jarvis.db"
    try:
        conn = sqlite3.connect(db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT    NOT NULL,
                room        TEXT    NOT NULL,
                type        TEXT    NOT NULL,
                content     TEXT    NOT NULL,
                acknowledged INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS room_baselines (
                room            TEXT PRIMARY KEY,
                baseline_desc   TEXT,
                updated_at      TEXT
            );

            CREATE TABLE IF NOT EXISTS user_routines (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                day_of_week TEXT,
                hour        INTEGER,
                activity    TEXT,
                frequency   INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS reminders (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                message         TEXT    NOT NULL,
                trigger_time    TEXT,
                recurring       INTEGER DEFAULT 0,
                last_triggered  TEXT
            );

            CREATE TABLE IF NOT EXISTS conversation_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT    NOT NULL,
                room        TEXT    NOT NULL,
                role        TEXT    NOT NULL,
                content     TEXT    NOT NULL
            );
        """)
        conn.commit()
        conn.close()
        ok(f"Database ready: {db_path}")
        return True

    except Exception as e:
        fail(f"Database init failed: {e}")
        return False


def print_summary(results: dict[str, bool]):
    print(f"\n{'═' * 50}")
    print(f"{BOLD}  SETUP SUMMARY{RESET}")
    print(f"{'═' * 50}")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for check, passed_check in results.items():
        status = f"{GREEN}PASS{RESET}" if passed_check else f"{RED}FAIL{RESET}"
        print(f"  {status}  {check}")

    print(f"{'═' * 50}")

    if passed == total:
        print(f"\n{GREEN}{BOLD}  ✓ All checks passed. Jarvis is ready.{RESET}")
        print(f"  Run: {CYAN}python main.py{RESET}\n")
    else:
        failed = total - passed
        print(f"\n{RED}{BOLD}  ✗ {failed} check(s) failed.{RESET}")
        print(f"  Fix the issues above, then run this script again.\n")


def main():
    print(f"\n{BOLD}{'═' * 50}{RESET}")
    print(f"{BOLD}  JARVIS SETUP VALIDATOR{RESET}")
    print(f"{BOLD}{'═' * 50}{RESET}")

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
```

---

## PHASE 1 — VOICE LOOP

Build these files in order. Each one imports the previous.

### Build order:
```
core/exceptions.py
core/event_bus.py
modules/voice/audio_utils.py
modules/voice/stt.py
modules/voice/tts.py
modules/voice/wake_word.py
modules/brain/llm.py
modules/brain/session.py
modules/brain/prompt_builder.py
modules/memory/database.py
modules/memory/event_log.py
dashboard/server.py              ← Build this before orchestrator
dashboard/static/index.html
dashboard/static/style.css
dashboard/static/app.js
core/orchestrator.py
main.py
scripts/test_voice.py
```

All Phase 1 code was previously defined in JARVIS_BUILD_PLAN.md.
Implement each file exactly as specified there.

The one addition from that plan: the dashboard.
Build it as defined in the DASHBOARD section below.

**Phase 1 milestone:** `python scripts/test_voice.py` passes.
`python main.py` launches Jarvis and the dashboard opens at http://localhost:7070.

---

## PHASE 2 — CONTEXT AWARENESS

### Build order:
```
modules/context/state.py
modules/activity/pc_monitor.py
modules/activity/audio_classifier.py
modules/activity/appliance_tracker.py
modules/context/interruptibility.py
modules/context/sleep_tracker.py
modules/context/state_fusion.py
modules/context/curiosity.py
[UPDATE] core/orchestrator.py      ← Add _init_context(), context_loop()
scripts/test_context.py
```

All Phase 2 code was previously defined in JARVIS_PHASE_2_3.md.

**Phase 2 milestone:** `python scripts/test_context.py` passes.
Interruptibility gate blocks speech during simulated gaming.
Dashboard shows live activity state updates.

---

## PHASE 3 — VISION

### Build order:
```
modules/vision/camera_manager.py
modules/vision/light_detector.py
modules/vision/posture_analyzer.py
modules/vision/object_detector.py
modules/vision/scene_analyzer.py
[UPDATE] modules/context/state_fusion.py   ← Wire vision signals
[UPDATE] core/orchestrator.py              ← Add _init_vision(), vision_loop()
scripts/test_vision.py
```

All Phase 3 code was previously defined in JARVIS_PHASE_2_3.md.

**Phase 3 milestone:** `python scripts/test_vision.py` passes.
Camera feed visible in dashboard. Mess detection triggers speech.

---

## PHASE 4 — MULTI-ROOM (ESP32-CAM NODES)

### Build order:
```
modules/network/mqtt_client.py
modules/network/node_manager.py
hardware/esphome/node_base.yaml
hardware/esphome/node_office.yaml
hardware/esphome/node_bedroom.yaml
hardware/esphome/node_kitchen.yaml
[UPDATE] core/orchestrator.py   ← Add mqtt_loop(), route audio to nodes
scripts/test_mqtt.py
```

**Phase 4 milestone:** `python scripts/test_mqtt.py` passes.
Wake word from ESP32-CAM triggers response from that room's speaker.
Dashboard shows node online/offline status in real time.

---

## THE DASHBOARD — FULL SPECIFICATION

This is the "cool stuff." It runs alongside Jarvis and shows everything
happening in real time. It is how you demo the system to a non-technical
audience. It also makes debugging dramatically easier.

It is a FastAPI + WebSocket server. The frontend is vanilla HTML/CSS/JS —
no React, no build step, opens instantly in any browser.

**URL:** http://localhost:7070

**What it shows:**
- Current activity state (what Jarvis thinks Cole is doing)
- Interruptibility score with visual gauge
- Room cards — one per room showing: node online/offline, camera feed thumbnail, light state, last event
- Live conversation log — every utterance, both sides, timestamped
- Appliance states (washer, dryer, dishwasher) with elapsed time
- Recent Jarvis speech — what it said and why
- System health — Ollama, MQTT, each node's RSSI/latency

**Design direction: Dark, technical, ambient. Like a mission control panel
that someone actually wants to look at.**

- Background: near-black (#0a0a0f)
- Accent: electric blue (#00d4ff) for active states
- Warning: amber (#ffb300) for needs-attention states  
- Font: monospace for data, clean sans for labels
- Subtle scanline texture overlay — not distracting, just atmospheric
- Room cards pulse gently when activity is detected
- Conversation log scrolls in like a terminal

### `dashboard/server.py`

```python
"""
dashboard/server.py

FastAPI + WebSocket server for the Jarvis real-time dashboard.
Runs as an asyncio background task alongside the main orchestrator.

The orchestrator calls dashboard.broadcast(event_dict) whenever state changes.
The dashboard pushes that event to all connected browser clients via WebSocket.

Endpoint:
  GET  /           → serves index.html
  GET  /static/*   → serves CSS, JS
  WS   /ws         → real-time event stream to browser
  GET  /api/state  → current full state snapshot (for page refresh)
  GET  /api/health → system health check
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

STATIC_DIR = Path(__file__).parent / "static"


class DashboardServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 7070):
        self.host = host
        self.port = port
        self.app = FastAPI(title="Jarvis Dashboard", docs_url=None, redoc_url=None)
        self._clients: list[WebSocket] = []
        self._state: dict = self._default_state()
        self._conversation: list[dict] = []  # Last 50 messages
        self._max_conversation = 50

        self._setup_routes()

    def _default_state(self) -> dict:
        """Initial state before any signals arrive."""
        return {
            "activity": "unknown",
            "location": "unknown",
            "interruptibility": 0.5,
            "confidence": 0.0,
            "signals": [],
            "context": {},
            "rooms": {},
            "appliances": {
                "washer":     {"status": "idle", "runtime_minutes": None},
                "dryer":      {"status": "idle", "runtime_minutes": None},
                "dishwasher": {"status": "idle", "runtime_minutes": None},
            },
            "system": {
                "ollama":    {"online": False, "model": ""},
                "mqtt":      {"online": False, "broker": ""},
                "whisper":   {"loaded": False, "model": ""},
                "nodes":     {},
            },
            "last_speech": None,
            "updated_at": datetime.now().isoformat(),
        }

    def _setup_routes(self):
        app = self.app

        # Serve static files
        if STATIC_DIR.exists():
            app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/", response_class=HTMLResponse)
        async def index():
            html_path = STATIC_DIR / "index.html"
            if html_path.exists():
                return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
            return HTMLResponse(content="<h1>Dashboard loading...</h1>")

        @app.websocket("/ws")
        async def websocket_endpoint(ws: WebSocket):
            await ws.accept()
            self._clients.append(ws)
            logger.debug(f"[Dashboard] Client connected ({len(self._clients)} total)")

            # Send current full state immediately on connect
            await ws.send_json({
                "type": "full_state",
                "state": self._state,
                "conversation": self._conversation,
            })

            try:
                while True:
                    # Keep connection alive, receive pings from client
                    await ws.receive_text()
            except WebSocketDisconnect:
                self._clients.remove(ws)
                logger.debug(f"[Dashboard] Client disconnected ({len(self._clients)} remaining)")

        @app.get("/api/state")
        async def get_state():
            return JSONResponse({
                "state": self._state,
                "conversation": self._conversation,
            })

        @app.get("/api/health")
        async def health():
            return JSONResponse({
                "status": "ok",
                "clients": len(self._clients),
                "updated_at": self._state.get("updated_at"),
            })

    async def broadcast(self, event: dict):
        """
        Push an event to all connected browser clients.
        Called by the orchestrator whenever anything changes.

        event types:
          "state_update"    → activity state changed
          "speech"          → Jarvis said something
          "user_speech"     → Cole said something
          "node_status"     → ESP32 node came online/offline
          "appliance"       → appliance state changed
          "system_health"   → Ollama/MQTT status changed
          "vision"          → room camera update
        """
        event["timestamp"] = datetime.now().isoformat()

        # Update internal state cache
        self._update_state(event)

        # Track conversation
        if event.get("type") in ("speech", "user_speech"):
            self._conversation.append({
                "role": "jarvis" if event["type"] == "speech" else "cole",
                "text": event.get("text", ""),
                "room": event.get("room", ""),
                "timestamp": event["timestamp"],
            })
            # Trim to max
            if len(self._conversation) > self._max_conversation:
                self._conversation = self._conversation[-self._max_conversation:]

        # Push to all connected clients
        dead = []
        for client in self._clients:
            try:
                await client.send_json({"type": "event", "event": event})
            except Exception:
                dead.append(client)

        for d in dead:
            self._clients.remove(d)

    def _update_state(self, event: dict):
        """Update the internal state cache based on incoming event."""
        etype = event.get("type")
        self._state["updated_at"] = event.get("timestamp", datetime.now().isoformat())

        if etype == "state_update":
            self._state.update({
                "activity":          event.get("activity", self._state["activity"]),
                "location":          event.get("location", self._state["location"]),
                "interruptibility":  event.get("interruptibility", self._state["interruptibility"]),
                "confidence":        event.get("confidence", self._state["confidence"]),
                "signals":           event.get("signals", self._state["signals"]),
                "context":           event.get("context", self._state["context"]),
            })

        elif etype == "speech":
            self._state["last_speech"] = {
                "text":      event.get("text"),
                "room":      event.get("room"),
                "priority":  event.get("priority"),
                "timestamp": event.get("timestamp"),
            }

        elif etype == "node_status":
            room = event.get("room")
            if room:
                self._state["system"]["nodes"][room] = {
                    "online":    event.get("online", False),
                    "ip":        event.get("ip"),
                    "updated_at": event.get("timestamp"),
                }

        elif etype == "appliance":
            name = event.get("appliance")
            if name and name in self._state["appliances"]:
                self._state["appliances"][name].update({
                    "status":          event.get("status"),
                    "runtime_minutes": event.get("runtime_minutes"),
                })

        elif etype == "system_health":
            self._state["system"].update(event.get("health", {}))

        elif etype == "vision":
            room = event.get("room")
            if room:
                if room not in self._state["rooms"]:
                    self._state["rooms"][room] = {}
                self._state["rooms"][room].update({
                    "lights_on":      event.get("lights_on"),
                    "person_present": event.get("person_present"),
                    "description":    event.get("description"),
                    "updated_at":     event.get("timestamp"),
                })

    async def run(self):
        """Start the dashboard server. Run as a background asyncio task."""
        import uvicorn
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="warning",  # Suppress uvicorn's own logs
            access_log=False,
        )
        server = uvicorn.Server(config)
        logger.info(f"[Dashboard] Running at http://{self.host}:{self.port}")
        await server.serve()
```

---

### `dashboard/static/index.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>JARVIS — Command Center</title>
  <link rel="stylesheet" href="/static/style.css">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Inter:wght@300;400;500&display=swap" rel="stylesheet">
</head>
<body>
  <div class="scanlines"></div>

  <header>
    <div class="logo">
      <span class="logo-j">J</span>ARVIS
      <span class="version">v1.0</span>
    </div>
    <div class="header-status">
      <div class="clock" id="clock">--:--:--</div>
      <div class="ws-status" id="ws-status">
        <span class="dot offline"></span> OFFLINE
      </div>
    </div>
  </header>

  <main>

    <!-- LEFT COLUMN -->
    <div class="col col-left">

      <!-- Activity State Card -->
      <div class="card card-primary" id="activity-card">
        <div class="card-label">ACTIVITY STATE</div>
        <div class="activity-name" id="activity-name">INITIALIZING</div>
        <div class="activity-sub">
          <span id="activity-location">—</span>
          <span class="sep">·</span>
          <span id="activity-context">—</span>
        </div>

        <div class="gauge-wrap">
          <div class="gauge-label">INTERRUPTIBILITY</div>
          <div class="gauge-track">
            <div class="gauge-fill" id="gauge-fill" style="width:50%"></div>
          </div>
          <div class="gauge-value" id="gauge-value">0.50</div>
        </div>

        <div class="signals" id="signals">
          <!-- Signal chips rendered by JS -->
        </div>

        <div class="confidence" id="confidence">Confidence: —</div>
      </div>

      <!-- Appliances Card -->
      <div class="card" id="appliances-card">
        <div class="card-label">APPLIANCES</div>
        <div class="appliance-list">
          <div class="appliance" id="appl-washer">
            <div class="appl-icon">⌁</div>
            <div class="appl-info">
              <div class="appl-name">WASHER</div>
              <div class="appl-status" id="appl-washer-status">idle</div>
            </div>
            <div class="appl-time" id="appl-washer-time">—</div>
          </div>
          <div class="appliance" id="appl-dryer">
            <div class="appl-icon">◎</div>
            <div class="appl-info">
              <div class="appl-name">DRYER</div>
              <div class="appl-status" id="appl-dryer-status">idle</div>
            </div>
            <div class="appl-time" id="appl-dryer-time">—</div>
          </div>
          <div class="appliance" id="appl-dishwasher">
            <div class="appl-icon">≋</div>
            <div class="appl-info">
              <div class="appl-name">DISHWASHER</div>
              <div class="appl-status" id="appl-dishwasher-status">idle</div>
            </div>
            <div class="appl-time" id="appl-dishwasher-time">—</div>
          </div>
        </div>
      </div>

      <!-- System Health Card -->
      <div class="card" id="health-card">
        <div class="card-label">SYSTEM HEALTH</div>
        <div class="health-list" id="health-list">
          <div class="health-item">
            <span class="dot" id="h-ollama"></span>
            <span class="health-name">Ollama LLM</span>
            <span class="health-detail" id="h-ollama-detail">—</span>
          </div>
          <div class="health-item">
            <span class="dot" id="h-mqtt"></span>
            <span class="health-name">MQTT Broker</span>
            <span class="health-detail" id="h-mqtt-detail">—</span>
          </div>
          <div class="health-item">
            <span class="dot" id="h-whisper"></span>
            <span class="health-name">Whisper STT</span>
            <span class="health-detail" id="h-whisper-detail">—</span>
          </div>
        </div>
      </div>

    </div>

    <!-- CENTER COLUMN -->
    <div class="col col-center">

      <!-- Rooms Grid -->
      <div class="card">
        <div class="card-label">ROOMS</div>
        <div class="rooms-grid" id="rooms-grid">
          <!-- Room cards rendered by JS -->
        </div>
      </div>

      <!-- Last Speech -->
      <div class="card card-speech" id="speech-card">
        <div class="card-label">LAST JARVIS OUTPUT</div>
        <div class="speech-text" id="speech-text">—</div>
        <div class="speech-meta">
          <span id="speech-room">—</span>
          <span class="sep">·</span>
          <span id="speech-time">—</span>
        </div>
      </div>

    </div>

    <!-- RIGHT COLUMN -->
    <div class="col col-right">

      <!-- Conversation Log -->
      <div class="card card-tall">
        <div class="card-label">CONVERSATION LOG</div>
        <div class="conv-log" id="conv-log">
          <div class="conv-empty">Waiting for first interaction...</div>
        </div>
      </div>

    </div>

  </main>

  <script src="/static/app.js"></script>
</body>
</html>
```

---

### `dashboard/static/style.css`

```css
/* ── Reset & Base ─────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:          #0a0a0f;
  --bg-card:     #0f0f18;
  --bg-card-alt: #13131f;
  --border:      #1e1e32;
  --border-glow: #00d4ff22;
  --blue:        #00d4ff;
  --blue-dim:    #00d4ff44;
  --green:       #00ff88;
  --amber:       #ffb300;
  --red:         #ff4444;
  --text:        #e0e0f0;
  --text-dim:    #606080;
  --text-muted:  #303050;
  --mono:        'Space Mono', monospace;
  --sans:        'Inter', sans-serif;
  --radius:      4px;
  --glow-blue:   0 0 20px #00d4ff33;
  --glow-green:  0 0 20px #00ff8833;
}

html, body {
  width: 100%; height: 100%;
  background: var(--bg);
  color: var(--text);
  font-family: var(--sans);
  font-size: 14px;
  overflow-x: hidden;
}

/* ── Scanlines ────────────────────────────────────────────────────────────── */
.scanlines {
  position: fixed; inset: 0; z-index: 1000;
  pointer-events: none;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,0,0,0.04) 2px,
    rgba(0,0,0,0.04) 4px
  );
}

/* ── Header ───────────────────────────────────────────────────────────────── */
header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 16px 24px;
  border-bottom: 1px solid var(--border);
  background: var(--bg-card);
}

.logo {
  font-family: var(--mono);
  font-size: 20px;
  font-weight: 700;
  letter-spacing: 6px;
  color: var(--text);
}

.logo-j {
  color: var(--blue);
  text-shadow: var(--glow-blue);
}

.version {
  font-size: 10px;
  color: var(--text-dim);
  letter-spacing: 2px;
  margin-left: 12px;
  vertical-align: middle;
}

.header-status {
  display: flex; align-items: center; gap: 20px;
}

.clock {
  font-family: var(--mono);
  font-size: 13px;
  color: var(--text-dim);
  letter-spacing: 2px;
}

.ws-status {
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 2px;
  display: flex; align-items: center; gap: 6px;
}

/* ── Layout ───────────────────────────────────────────────────────────────── */
main {
  display: grid;
  grid-template-columns: 280px 1fr 300px;
  gap: 16px;
  padding: 16px;
  height: calc(100vh - 60px);
  overflow: hidden;
}

.col { display: flex; flex-direction: column; gap: 12px; overflow: hidden; }
.col-center { overflow-y: auto; }
.col-right { overflow: hidden; }

/* ── Cards ────────────────────────────────────────────────────────────────── */
.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px;
  flex-shrink: 0;
}

.card-primary {
  border-color: var(--border-glow);
  box-shadow: var(--glow-blue);
}

.card-tall {
  flex: 1;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.card-label {
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 3px;
  color: var(--text-dim);
  margin-bottom: 12px;
}

/* ── Activity Card ────────────────────────────────────────────────────────── */
.activity-name {
  font-family: var(--mono);
  font-size: 26px;
  font-weight: 700;
  color: var(--blue);
  text-shadow: var(--glow-blue);
  text-transform: uppercase;
  letter-spacing: 2px;
  margin-bottom: 4px;
  transition: all 0.3s ease;
}

.activity-sub {
  font-size: 12px;
  color: var(--text-dim);
  margin-bottom: 16px;
}

.sep { margin: 0 8px; color: var(--text-muted); }

/* ── Gauge ────────────────────────────────────────────────────────────────── */
.gauge-wrap { margin-bottom: 12px; }

.gauge-label {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 2px;
  color: var(--text-dim);
  margin-bottom: 6px;
}

.gauge-track {
  width: 100%; height: 6px;
  background: var(--bg-card-alt);
  border: 1px solid var(--border);
  border-radius: 3px;
  overflow: hidden;
  margin-bottom: 4px;
}

.gauge-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--blue), var(--green));
  transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
  border-radius: 3px;
}

.gauge-value {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--text-dim);
  text-align: right;
}

/* ── Signals ──────────────────────────────────────────────────────────────── */
.signals {
  display: flex; flex-wrap: wrap; gap: 6px;
  margin-bottom: 10px;
}

.signal-chip {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 1px;
  color: var(--blue);
  border: 1px solid var(--blue-dim);
  border-radius: 2px;
  padding: 2px 6px;
  text-transform: uppercase;
  animation: chipIn 0.2s ease;
}

@keyframes chipIn {
  from { opacity: 0; transform: scale(0.8); }
  to   { opacity: 1; transform: scale(1); }
}

.confidence {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--text-muted);
}

/* ── Appliances ───────────────────────────────────────────────────────────── */
.appliance-list { display: flex; flex-direction: column; gap: 10px; }

.appliance {
  display: flex; align-items: center; gap: 12px;
  padding: 8px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  transition: border-color 0.3s;
}

.appliance.running {
  border-color: var(--amber);
  box-shadow: 0 0 12px #ffb30022;
  animation: pulse-amber 2s ease infinite;
}

.appliance.done {
  border-color: var(--green);
  box-shadow: 0 0 12px #00ff8822;
}

@keyframes pulse-amber {
  0%, 100% { box-shadow: 0 0 8px #ffb30022; }
  50%       { box-shadow: 0 0 20px #ffb30044; }
}

.appl-icon {
  font-size: 18px;
  color: var(--text-dim);
  width: 24px;
  text-align: center;
}

.appl-name {
  font-family: var(--mono);
  font-size: 10px;
  letter-spacing: 2px;
  color: var(--text-dim);
}

.appl-status {
  font-size: 12px;
  color: var(--text);
  text-transform: capitalize;
}

.appl-time {
  margin-left: auto;
  font-family: var(--mono);
  font-size: 11px;
  color: var(--text-dim);
}

/* ── System Health ────────────────────────────────────────────────────────── */
.health-list { display: flex; flex-direction: column; gap: 8px; }

.health-item {
  display: flex; align-items: center; gap: 8px;
  font-size: 12px;
}

.health-name { flex: 1; color: var(--text-dim); }

.health-detail {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--text-muted);
}

/* ── Status Dots ──────────────────────────────────────────────────────────── */
.dot {
  width: 8px; height: 8px;
  border-radius: 50%;
  display: inline-block;
  flex-shrink: 0;
}

.dot.online  { background: var(--green); box-shadow: 0 0 6px var(--green); }
.dot.offline { background: var(--text-muted); }
.dot.warning { background: var(--amber); box-shadow: 0 0 6px var(--amber); }

/* ── Rooms Grid ───────────────────────────────────────────────────────────── */
.rooms-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 10px;
}

.room-card {
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 12px;
  transition: border-color 0.3s, box-shadow 0.3s;
  position: relative;
  overflow: hidden;
}

.room-card.active {
  border-color: var(--blue-dim);
  box-shadow: var(--glow-blue);
}

.room-card.node-online::after {
  content: '●';
  position: absolute;
  top: 8px; right: 8px;
  font-size: 8px;
  color: var(--green);
  text-shadow: 0 0 6px var(--green);
}

.room-name {
  font-family: var(--mono);
  font-size: 11px;
  letter-spacing: 2px;
  color: var(--text-dim);
  margin-bottom: 8px;
  text-transform: uppercase;
}

.room-status {
  font-size: 12px;
  color: var(--text);
  margin-bottom: 4px;
}

.room-meta {
  font-size: 11px;
  color: var(--text-muted);
}

.room-light {
  display: inline-block;
  font-size: 10px;
  padding: 1px 5px;
  border-radius: 2px;
  margin-top: 4px;
}

.room-light.on  { background: #ffb30022; color: var(--amber); border: 1px solid #ffb30044; }
.room-light.off { background: #ffffff08; color: var(--text-muted); border: 1px solid var(--border); }

/* ── Last Speech Card ─────────────────────────────────────────────────────── */
.card-speech {
  border-color: #00ff8822;
}

.speech-text {
  font-size: 16px;
  line-height: 1.5;
  color: var(--green);
  margin-bottom: 8px;
  min-height: 48px;
  font-style: italic;
}

.speech-meta {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--text-muted);
}

/* ── Conversation Log ─────────────────────────────────────────────────────── */
.conv-log {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding-right: 4px;
}

.conv-log::-webkit-scrollbar { width: 3px; }
.conv-log::-webkit-scrollbar-track { background: transparent; }
.conv-log::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

.conv-empty {
  color: var(--text-muted);
  font-family: var(--mono);
  font-size: 11px;
  text-align: center;
  padding: 20px 0;
}

.conv-entry {
  padding: 8px 10px;
  border-radius: var(--radius);
  border-left: 2px solid;
  animation: entryIn 0.3s ease;
}

@keyframes entryIn {
  from { opacity: 0; transform: translateX(10px); }
  to   { opacity: 1; transform: translateX(0); }
}

.conv-entry.jarvis {
  background: #00ff8808;
  border-color: var(--green);
}

.conv-entry.cole {
  background: #00d4ff08;
  border-color: var(--blue);
}

.conv-speaker {
  font-family: var(--mono);
  font-size: 9px;
  letter-spacing: 2px;
  margin-bottom: 4px;
}

.conv-entry.jarvis .conv-speaker { color: var(--green); }
.conv-entry.cole   .conv-speaker { color: var(--blue); }

.conv-text {
  font-size: 12px;
  line-height: 1.5;
  color: var(--text);
}

.conv-time {
  font-family: var(--mono);
  font-size: 9px;
  color: var(--text-muted);
  margin-top: 4px;
}
```

---

### `dashboard/static/app.js`

```javascript
// dashboard/static/app.js
// Real-time dashboard client.
// Connects to WebSocket, applies all state updates to the DOM.
// Zero dependencies — vanilla JS only.

const WS_URL = `ws://${location.host}/ws`;
const RECONNECT_DELAY = 3000;

let ws = null;
let reconnectTimer = null;

// ── WebSocket ──────────────────────────────────────────────────────────────

function connect() {
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    setWsStatus(true);
    clearTimeout(reconnectTimer);
    console.log('[WS] Connected');
  };

  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'full_state') {
      applyFullState(msg.state, msg.conversation);
    } else if (msg.type === 'event') {
      applyEvent(msg.event);
    }
    // Send ping to keep alive
    ws.send('ping');
  };

  ws.onclose = () => {
    setWsStatus(false);
    reconnectTimer = setTimeout(connect, RECONNECT_DELAY);
    console.log('[WS] Disconnected, reconnecting...');
  };

  ws.onerror = () => {
    ws.close();
  };
}

// ── State Application ──────────────────────────────────────────────────────

function applyFullState(state, conversation) {
  updateActivity(state);
  updateAppliances(state.appliances);
  updateHealth(state.system);
  updateRooms(state.rooms);
  if (state.last_speech) updateSpeech(state.last_speech);
  conversation.forEach(entry => appendConversation(entry));
}

function applyEvent(event) {
  switch (event.type) {
    case 'state_update':
      updateActivity(event);
      break;
    case 'speech':
      updateSpeech(event);
      appendConversation({ role: 'jarvis', text: event.text, room: event.room, timestamp: event.timestamp });
      break;
    case 'user_speech':
      appendConversation({ role: 'cole', text: event.text, room: event.room, timestamp: event.timestamp });
      break;
    case 'appliance':
      updateSingleAppliance(event.appliance, event.status, event.runtime_minutes);
      break;
    case 'node_status':
      updateNodeStatus(event.room, event.online);
      break;
    case 'system_health':
      updateHealth(event.health);
      break;
    case 'vision':
      updateRoomVision(event.room, event);
      break;
  }
}

// ── DOM Updaters ───────────────────────────────────────────────────────────

function updateActivity(state) {
  const activity = (state.activity || 'unknown').toUpperCase().replace(/_/g, ' ');
  const interruptibility = state.interruptibility ?? 0.5;
  const confidence = state.confidence ?? 0;
  const signals = state.signals || [];
  const context = state.context || {};

  setText('activity-name', activity);
  setText('activity-location', state.location || '—');

  // Context line (game name, project, etc.)
  const ctxStr = context.game || context.project || context.file || '';
  setText('activity-context', ctxStr || '—');

  // Gauge
  const pct = Math.round(interruptibility * 100);
  const fill = document.getElementById('gauge-fill');
  if (fill) {
    fill.style.width = `${pct}%`;
    // Color: low interruptibility = red, high = green
    if (interruptibility < 0.25) {
      fill.style.background = 'linear-gradient(90deg, #ff4444, #ff6644)';
    } else if (interruptibility < 0.5) {
      fill.style.background = 'linear-gradient(90deg, #ffb300, #ffcc00)';
    } else {
      fill.style.background = 'linear-gradient(90deg, #00d4ff, #00ff88)';
    }
  }
  setText('gauge-value', interruptibility.toFixed(2));

  // Signal chips
  const signalsEl = document.getElementById('signals');
  if (signalsEl) {
    signalsEl.innerHTML = signals
      .map(s => `<span class="signal-chip">${s.replace(/_/g, ' ')}</span>`)
      .join('');
  }

  setText('confidence', `Confidence: ${Math.round(confidence * 100)}%`);

  // Pulse the activity card on change
  pulse('activity-card');
}

function updateAppliances(appliances) {
  if (!appliances) return;
  Object.entries(appliances).forEach(([name, data]) => {
    updateSingleAppliance(name, data.status, data.runtime_minutes);
  });
}

function updateSingleAppliance(name, status, runtimeMinutes) {
  const card = document.getElementById(`appl-${name}`);
  const statusEl = document.getElementById(`appl-${name}-status`);
  const timeEl = document.getElementById(`appl-${name}-time`);
  if (!card) return;

  if (statusEl) statusEl.textContent = status || 'idle';

  card.classList.remove('running', 'done');
  if (status === 'running') card.classList.add('running');
  if (status === 'done')    card.classList.add('done');

  if (timeEl) {
    timeEl.textContent = runtimeMinutes != null
      ? `${Math.round(runtimeMinutes)}m`
      : '—';
  }
}

function updateHealth(system) {
  if (!system) return;

  setDot('h-ollama', system.ollama?.online ? 'online' : 'offline');
  setText('h-ollama-detail', system.ollama?.model || '—');

  setDot('h-mqtt', system.mqtt?.online ? 'online' : 'offline');
  setText('h-mqtt-detail', system.mqtt?.broker || '—');

  setDot('h-whisper', system.whisper?.loaded ? 'online' : 'offline');
  setText('h-whisper-detail', system.whisper?.model || '—');
}

function updateRooms(rooms) {
  const grid = document.getElementById('rooms-grid');
  if (!grid) return;
  grid.innerHTML = '';

  const ROOM_IDS = ['office', 'bedroom', 'kitchen', 'living_room'];

  ROOM_IDS.forEach(roomId => {
    const data = rooms?.[roomId] || {};
    const card = document.createElement('div');
    card.className = 'room-card';
    card.id = `room-${roomId}`;

    const lightsOn = data.lights_on;
    const lightLabel = lightsOn == null ? '' :
      `<span class="room-light ${lightsOn ? 'on' : 'off'}">${lightsOn ? 'LIGHTS ON' : 'LIGHTS OFF'}</span>`;

    card.innerHTML = `
      <div class="room-name">${roomId.replace(/_/g, ' ').toUpperCase()}</div>
      <div class="room-status">${data.person_present ? '● Person detected' : '○ Empty'}</div>
      <div class="room-meta">${data.description || 'No camera data yet'}</div>
      ${lightLabel}
    `;
    grid.appendChild(card);
  });
}

function updateRoomVision(roomId, data) {
  const card = document.getElementById(`room-${roomId}`);
  if (!card) {
    updateRooms({ [roomId]: data });
    return;
  }

  const statusEl = card.querySelector('.room-status');
  if (statusEl) {
    statusEl.textContent = data.person_present ? '● Person detected' : '○ Empty';
  }

  const metaEl = card.querySelector('.room-meta');
  if (metaEl && data.description) {
    metaEl.textContent = data.description;
  }

  const lightEl = card.querySelector('.room-light');
  if (data.lights_on != null) {
    if (!lightEl) {
      const span = document.createElement('span');
      card.appendChild(span);
    }
    const el = card.querySelector('.room-light') || document.createElement('span');
    el.className = `room-light ${data.lights_on ? 'on' : 'off'}`;
    el.textContent = data.lights_on ? 'LIGHTS ON' : 'LIGHTS OFF';
  }

  card.classList.add('active');
  setTimeout(() => card.classList.remove('active'), 2000);
}

function updateNodeStatus(roomId, online) {
  const card = document.getElementById(`room-${roomId}`);
  if (card) {
    card.classList.toggle('node-online', online);
  }
}

function updateSpeech(data) {
  setText('speech-text', `"${data.text || '—'}"`);
  setText('speech-room', data.room ? data.room.toUpperCase() : '—');
  setText('speech-time', formatTime(data.timestamp));
  pulse('speech-card');
}

function appendConversation(entry) {
  const log = document.getElementById('conv-log');
  if (!log) return;

  // Remove empty placeholder
  const empty = log.querySelector('.conv-empty');
  if (empty) empty.remove();

  const el = document.createElement('div');
  el.className = `conv-entry ${entry.role}`;
  el.innerHTML = `
    <div class="conv-speaker">${entry.role === 'jarvis' ? 'JARVIS' : 'COLE'} · ${entry.room?.toUpperCase() || ''}</div>
    <div class="conv-text">${escapeHtml(entry.text)}</div>
    <div class="conv-time">${formatTime(entry.timestamp)}</div>
  `;

  log.appendChild(el);
  log.scrollTop = log.scrollHeight;

  // Keep max 50 entries in DOM
  while (log.children.length > 50) {
    log.removeChild(log.firstChild);
  }
}

// ── Utilities ──────────────────────────────────────────────────────────────

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function setDot(id, state) {
  const el = document.getElementById(id);
  if (el) el.className = `dot ${state}`;
}

function pulse(id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.style.boxShadow = '0 0 30px #00d4ff55';
  setTimeout(() => { el.style.boxShadow = ''; }, 600);
}

function setWsStatus(online) {
  const el = document.getElementById('ws-status');
  if (!el) return;
  el.innerHTML = online
    ? '<span class="dot online"></span> LIVE'
    : '<span class="dot offline"></span> RECONNECTING';
}

function formatTime(isoStr) {
  if (!isoStr) return '—';
  try {
    const d = new Date(isoStr);
    return d.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch { return '—'; }
}

function escapeHtml(str) {
  return (str || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ── Clock ──────────────────────────────────────────────────────────────────

function updateClock() {
  const el = document.getElementById('clock');
  if (el) {
    el.textContent = new Date().toLocaleTimeString('en-US', {
      hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit'
    });
  }
}

setInterval(updateClock, 1000);
updateClock();

// ── Init ───────────────────────────────────────────────────────────────────

connect();
```

---

## WIRING THE DASHBOARD INTO THE ORCHESTRATOR

Add this to `core/orchestrator.py`:

```python
# ── Add to imports ──────────────────────────────────────────────────────────
from dashboard.server import DashboardServer

# ── Add to __init__ ─────────────────────────────────────────────────────────
if self.config["system"].get("dashboard_enabled", True):
    self.dashboard = DashboardServer(
        host=self.config["system"].get("dashboard_host", "0.0.0.0"),
        port=self.config["system"].get("dashboard_port", 7070),
    )
else:
    self.dashboard = None

# ── Broadcast helpers ────────────────────────────────────────────────────────
async def _broadcast(self, event: dict):
    """Send event to dashboard if enabled. Never blocks or raises."""
    if self.dashboard:
        try:
            await self.dashboard.broadcast(event)
        except Exception as e:
            logger.debug(f"[Dashboard] Broadcast error: {e}")

# ── Call _broadcast() at these points in orchestrator ────────────────────────

# After wake word + transcription (in _handle_wake_event):
await self._broadcast({
    "type": "user_speech",
    "text": transcript,
    "room": room_id
})

# After LLM responds and Jarvis speaks:
await self._broadcast({
    "type": "speech",
    "text": response,
    "room": room_id,
    "priority": "conversation"
})

# After state_fusion fuses a new state (in context_loop):
await self._broadcast({
    "type": "state_update",
    "activity": state.activity,
    "location": state.location,
    "interruptibility": state.interruptibility,
    "confidence": state.confidence,
    "signals": state.signals,
    "context": state.context,
})

# After appliance state changes (in _on_appliance_notification):
await self._broadcast({
    "type": "appliance",
    "appliance": appliance_name,
    "status": new_status,
    "runtime_minutes": runtime,
})

# After vision loop runs (in vision_loop):
await self._broadcast({
    "type": "vision",
    "room": room_id,
    "lights_on": lights["lights_on"],
    "person_present": obj_summary["person_present"],
    "description": last_desc,
})

# ── Add to run() ──────────────────────────────────────────────────────────────
async def run(self):
    self._load_models()

    tasks = [
        self.bus.run(),
        self.wake.listen_forever(),
        self.sessions.cleanup_expired(),
    ]

    if self.dashboard:
        tasks.append(self.dashboard.run())

    await asyncio.gather(*tasks)
```

---

## ESPHome HARDWARE CONFIG

### `hardware/esphome/secrets.yaml.example`

```yaml
# Copy to secrets.yaml (NOT committed to git)
wifi_ssid: "YourNetworkName"
wifi_password: "YourWifiPassword"
mqtt_broker_ip: "192.168.1.X"   # Your PC's local IP
mqtt_username: ""
mqtt_password: ""
ota_password: "change_me"

# Per-node static IPs
node_office_ip: "192.168.1.101"
node_bedroom_ip: "192.168.1.102"
node_kitchen_ip: "192.168.1.103"
node_living_room_ip: "192.168.1.104"
```

### `hardware/esphome/node_base.yaml`

```yaml
# Base config included by all room nodes.
# Do not deploy this directly.

esphome:
  name: ${node_name}
  friendly_name: "Jarvis — ${room_display}"
  platform: ESP32
  board: esp32cam          # AI-Thinker ESP32-CAM board

wifi:
  ssid: !secret wifi_ssid
  password: !secret wifi_password
  manual_ip:
    static_ip: ${node_ip}
    gateway: 192.168.1.1
    subnet: 255.255.255.0
  # Fast reconnect
  power_save_mode: none

# OTA updates — after first USB flash, deploy wirelessly
ota:
  - platform: esphome
    password: !secret ota_password

# Web server for debugging (disable in production if needed)
web_server:
  port: 80

logger:
  level: INFO

mqtt:
  broker: !secret mqtt_broker_ip
  port: 1883
  username: !secret mqtt_username
  password: !secret mqtt_password
  # Node announces itself online/offline
  birth_message:
    topic: jarvis/nodes/${room_id}/status
    payload: "online"
    retain: true
  will_message:
    topic: jarvis/nodes/${room_id}/status
    payload: "offline"
    retain: true

# ── I2S Audio Bus ────────────────────────────────────────────────────────────
i2s_audio:
  - id: i2s_bus
    i2s_lrclk_pin: GPIO14    # INMP441 WS + MAX98357A LRC (same pin, shared)
    i2s_bclk_pin: GPIO2      # INMP441 SCK + MAX98357A BCLK (shared)

# ── INMP441 Microphone ───────────────────────────────────────────────────────
microphone:
  - platform: i2s_audio
    id: node_mic
    i2s_audio_id: i2s_bus
    i2s_din_pin: GPIO12      # INMP441 SD — NOTE: tie GPIO12 LOW at boot via 10kΩ to GND
    adc_type: external
    pdm: false

# ── MAX98357A Speaker Amplifier ──────────────────────────────────────────────
speaker:
  - platform: i2s_audio
    id: node_speaker
    i2s_audio_id: i2s_bus
    i2s_dout_pin: GPIO15     # MAX98357A DIN
    dac_type: external
    mode: mono

# ── ESP32-CAM Camera ─────────────────────────────────────────────────────────
esp32_camera:
  name: "${room_display} Camera"
  external_clock:
    pin: GPIO0
    frequency: 20MHz
  i2c_pins:
    sda: GPIO26
    scl: GPIO27
  data_pins: [GPIO5, GPIO18, GPIO19, GPIO21, GPIO36, GPIO39, GPIO34, GPIO35]
  vsync_pin: GPIO25
  href_pin: GPIO23
  pixel_clock_pin: GPIO22
  resolution: 640x480
  jpeg_quality: 15           # Lower = smaller = faster over WiFi
  # Only capture on-demand to save bandwidth
  idle_framerate: 0.1        # 0.1 fps idle (one frame every 10 sec)

# Expose MJPEG stream — Python OpenCV connects here
esp32_camera_web_server:
  - port: 8080
    mode: stream

# ── MQTT Subscriptions (PC → Node) ───────────────────────────────────────────
# PC publishes TTS audio to this topic → node plays it
# This is handled via ESPHome's media_player component or raw MQTT binary

# ── Status LED (GPIO33 on ESP32-CAM) ─────────────────────────────────────────
light:
  - platform: status_led
    name: "Node Status LED"
    pin: GPIO33

# ── Room ID global ───────────────────────────────────────────────────────────
globals:
  - id: room_id_global
    type: std::string
    initial_value: '"${room_id}"'
```

### `hardware/esphome/node_office.yaml`

```yaml
packages:
  base: !include node_base.yaml

substitutions:
  node_name:    "jarvis-node-office"
  room_display: "Office"
  room_id:      "office"
  node_ip:      !secret node_office_ip
```

### `hardware/esphome/node_bedroom.yaml`

```yaml
packages:
  base: !include node_base.yaml

substitutions:
  node_name:    "jarvis-node-bedroom"
  room_display: "Bedroom"
  room_id:      "bedroom"
  node_ip:      !secret node_bedroom_ip
```

### `hardware/esphome/node_kitchen.yaml`

```yaml
packages:
  base: !include node_base.yaml

substitutions:
  node_name:    "jarvis-node-kitchen"
  room_display: "Kitchen"
  room_id:      "kitchen"
  node_ip:      !secret node_kitchen_ip
```

---

## FIRST RUN SEQUENCE

Execute these commands in order. Do not skip steps.

```powershell
# 1. Create and activate virtual environment
cd C:\projects\jarvis
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Run setup validator — must pass before proceeding
python scripts/setup.py

# 4. (If Ollama not running) Start it and pull models
# In a separate terminal:
ollama serve
ollama pull llama3.1:8b
ollama pull moondream

# 5. (If Mosquitto not running) Start it
# In a separate terminal (or configure as Windows service):
mosquitto -v

# 6. Run the voice test — must pass before Phase 2
python scripts/test_voice.py

# 7. Launch Jarvis
python main.py

# 8. Open dashboard in browser
# http://localhost:7070
```

---

## DEMO SCRIPT (for when the boss is watching)

Walk through these in order. Each demonstrates a discrete capability.

```
1. Open http://localhost:7070 in browser
   → Dashboard shows JARVIS command center, all system health indicators

2. Say "Hey Jarvis, what time is it?"
   → Wake word detected, chime plays, response delivered
   → Conversation log updates in real time on dashboard

3. Open a game (or any process in PROCESS_MAP)
   → Activity card changes to "GAMING"
   → Interruptibility gauge drops to red zone
   → Say "Hey Jarvis, [anything]" — Jarvis stays quiet

4. Alt-tab to browser (high interruptibility)
   → Gauge rises to green
   → Jarvis becomes responsive again

5. Point camera at counter with dishes
   → Room card updates with scene description
   → Jarvis says "Kitchen counter's looking a little busy"

6. Cover camera with hand / turn off lights
   → Room card shows LIGHTS OFF
   → Lie down / simulate lying posture
   → Jarvis asks "Taking a nap? I'll keep quiet." once, then goes silent

7. Ask "Hey Jarvis, what do you see in the kitchen?"
   → Jarvis describes what the camera sees in real time

8. Simulate washer sound (play audio near mic)
   → Appliance card shows WASHER: running
   → After timer elapses, Jarvis announces cycle complete
```

---

## WHAT MAKES THIS IMPRESSIVE TO A TECHNICAL REVIEWER

- **Async-first architecture** — every module is non-blocking, nothing starves anything else
- **Event bus decoupling** — modules don't import each other, they publish events
- **Config-driven** — zero magic numbers in source code, everything tunable from YAML
- **Multi-model ML pipeline** — Whisper + YAMNet + YOLOv8 + MediaPipe + Ollama vision, running simultaneously without GPU contention
- **Real-time dashboard** — WebSocket, no frameworks, no build step, instant state visibility
- **State machine for social intelligence** — not just "detect activity" but "should I speak right now" with priority levels, cooldowns, and escalation
- **Production patterns** — loguru structured logging, typed function signatures, custom exceptions, idempotent setup script
- **Hardware integration** — ESP32-CAM nodes with ESPHome, MQTT routing, OTA updates
- **Honest scope management** — phased delivery, each phase independently testable

The code will speak for itself. The dashboard makes it visual.
```
