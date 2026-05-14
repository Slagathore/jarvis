# Jarvis — Ambient Home AI

A local, always-on ambient AI assistant that lives in your house instead of in a cloud.
It watches what you're doing, respects when you're busy, and speaks up when it has something worth saying.

No cloud subscriptions. No wake-word-to-server round trips. Everything runs on your local GPU.

---

## What It Does

- **Wakes on a custom wake word** ("Hey Jarvis") detected locally via [openWakeWord](https://github.com/dscripka/openWakeWord)
- **Transcribes speech** with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CUDA-accelerated)
- **Understands context** — knows if you're gaming, on a call, asleep, or cooking, and adjusts interruptibility accordingly
- **Watches rooms via camera** for mess, light state, posture, and presence using YOLOv8 + MediaPipe
- **Classifies ambient audio** (appliances, music, silence) via TensorFlow/YAMNet
- **Speaks proactively** when curiosity fires and you're interruptible — "Washer's done." / "Kitchen counter's looking a bit busy."
- **Responds to direct questions** via Ollama LLM running locally or configured cloud-backed models
- **Real-time dashboard** at `http://localhost:7070` — activity state, room status, conversation log, appliance tracking, degraded-mode status, wake calibration, and perf/model-call metrics
- **Multi-room support** via ESP32-CAM nodes over MQTT (optional hardware expansion)

---

## Architecture

```text
┌─────────────────────────────────────────────────────┐
│                    Orchestrator                      │
│  (core/orchestrator.py — async task coordinator)    │
└───────────────┬─────────────────────────────────────┘
                │ EventBus (priority pub/sub, no direct imports)
    ┌───────────┼───────────────────────────────┐
    │           │                               │
┌───▼───┐  ┌───▼───────┐  ┌────────┐  ┌───────▼──────┐
│ Voice │  │  Context  │  │ Vision │  │   Network    │
│ STT   │  │  State    │  │ Camera │  │   MQTT       │
│ TTS   │  │  Fusion   │  │ YOLO   │  │   ESP32 nodes│
│ Wake  │  │  PC Mon   │  │ Pose   │  └──────────────┘
└───────┘  │  YAMNet   │  └────────┘
           │  Sleep    │
           └───────────┘
                │
        ┌───────▼───────┐
        │   Brain (LLM) │
        │   Ollama      │
        │   Sessions    │
        └───────────────┘
                │
        ┌───────▼───────┐
        │   Dashboard   │
        │   FastAPI/WS  │
        │   :7070       │
        └───────────────┘
```

Most runtime modules communicate through the async event bus. The bus is bounded, priority-aware, and rate-limits high-volume telemetry topics so wake, safety, alarm, and control events do not sit behind bursts of camera/world/debug traffic.

---

## Hardware

### Minimum (single-room, software only)

- Windows 10/11 PC with a discrete GPU (NVIDIA recommended)
- Microphone
- Speakers

### Full multi-room build

- As above, plus one or more **AI-Thinker ESP32-CAM** nodes per room
- Each node provides: microphone input, speaker output, camera stream, and MQTT heartbeat
- See [hardware/esphome/BUILD_OFFICE_NODE.md](hardware/esphome/BUILD_OFFICE_NODE.md) for the first node build guide

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally with your chosen model
- [Mosquitto MQTT broker](https://mosquitto.org/download/) (for multi-room; optional for single-room)
- NVIDIA GPU with CUDA (strongly recommended — Whisper + YOLO are significantly faster)

---

## Setup

```powershell
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/jarvis.git
cd jarvis

# 2. Create and activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. (CUDA — recommended) Reinstall torch with GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 5. Copy and fill in secrets
copy .env.example .env
# Edit .env with your MQTT credentials if needed

# 6. Start required services
ollama serve
ollama pull YOUR_CHOSEN_MODEL    # e.g. ollama pull llama3.1:8b
mosquitto -v                     # or start as a Windows service

# 7. Run the setup validator — must show all PASS before launching
python scripts/setup.py

# 8. Launch
python main.py
```

Dashboard opens automatically. Browse to `http://localhost:7070`.

---

## Configuration

Everything tunable lives in `config.yaml`. Never hardcode values in source files.

Key sections:

| Section | What to change |
| --- | --- |
| `ollama.model` | Which Ollama model to use for chat and vision |
| `ollama.system_prompt` | Personality and household context — customize to your setup |
| `voice.whisper.model_size` | STT accuracy vs. speed (`base` → `large-v3`) |
| `voice.wake_word.model` | openWakeWord model name |
| `system.event_bus` | Queue size and per-topic telemetry rate limits |
| `rooms` | Per-room `video` / `mic` / `speaker` channels — see `WYZE_SETUP.md` |
| `interruptibility.activity_scores` | How interruptible each detected activity is |
| `curiosity.topic_cooldowns_hours` | How often Jarvis can proactively comment on each topic |
| `process_activity_map` | Map your specific game/app .exe names to activities |

---

## Project Structure

```text
jarvis/
├── main.py                     # Entry point — boots orchestrator
├── config.yaml                 # Single source of truth for all config
├── requirements.txt
│
├── core/
│   ├── orchestrator.py         # Wires all modules together, runs all loops
│   ├── event_bus.py            # Async pub/sub bus
│   └── exceptions.py
│
├── modules/
│   ├── voice/                  # STT (Whisper), TTS (Piper), wake word
│   ├── brain/                  # LLM (Ollama), session memory, prompt builder
│   ├── context/                # State fusion, interruptibility, curiosity, sleep
│   ├── activity/               # PC monitor, audio classifier, appliance tracker
│   ├── vision/                 # Camera, YOLOv8, MediaPipe pose, scene analysis
│   ├── integrations/           # Sensor/actuator plugin contracts
│   ├── memory/                 # SQLite database, event log, room baselines
│   └── network/                # MQTT client, ESP32 node manager
│
├── dashboard/
│   ├── server.py               # FastAPI + WebSocket server
│   └── static/                 # Vanilla JS/CSS — no build step
│
├── hardware/
│   └── esphome/                # ESPHome firmware configs for ESP32-CAM nodes
│
├── scripts/
│   ├── setup.py                # Environment validator (run before first launch)
│   ├── test_voice.py           # Phase 1 test: wake word → STT → TTS
│   ├── test_context.py         # Phase 2 test: activity detection pipeline
│   ├── test_vision.py          # Phase 3 test: camera + detection pipeline
│   └── test_mqtt.py            # Phase 4 test: MQTT + node connectivity
│
└── data/                       # Runtime data (gitignored)
    ├── jarvis.db               # SQLite event/conversation log
    └── voices/                 # Piper TTS voice models
```

---

## ESP32-CAM Nodes (Multi-Room)

Nodes are flashed with [ESPHome](https://esphome.io). Each node provides:

- MJPEG camera stream (OpenCV connects over HTTP)
- MQTT status heartbeat (birth/will messages)
- Microphone and speaker via I2S (INMP441 + MAX98357A)

See the [build guide](hardware/esphome/BUILD_OFFICE_NODE.md) to bring up the first node.

To add a node: copy `hardware/esphome/secrets.yaml.example` to `hardware/esphome/secrets.yaml`, fill in your network details, then set `has_node: true` and `node_ip` for that room in `config.yaml`.

---

## What Each Module Does

### `core/orchestrator.py`

The only file that imports from multiple modules. Everything else communicates via the event bus. Orchestrator starts all async loops, routes wake-word events through the voice pipeline, and calls `_broadcast()` to keep the dashboard current.

### `core/event_bus.py`

Async priority queue. Producers call `await bus.publish(topic, payload)`. Consumers register with `bus.subscribe(topic, handler)`. Wake/safety/control topics are dispatched ahead of world/telemetry/debug topics, high-volume telemetry has token-bucket rate limits, and a crashed handler never takes down the bus.

### `modules/integrations/`

Small plugin-style contracts for future sensors and actuators. New integrations should implement `SensorPlugin` or `ActuatorPlugin`, publish/subscribe through `EventBus`, and register with `IntegrationRegistry` instead of adding more direct wiring to `core/orchestrator.py`.

### Dashboard Operations

The dashboard now includes:

- **Perf tab model tracking** — per provider/model daily calls, cloud calls, average latency, timeout rate, and average tool-loop iterations.
- **Degraded Mode card** — best-effort loaded/degraded/disabled status for wake word, STT, TTS, LLM, MQTT, cameras, identity, world model, open-vocab objects, and registered integrations.
- **Wake Calibration card** — per-room RMS, peak level, wake score, false-positive count, and suggested sensitivity.

### `modules/context/state_fusion.py`

Combines signals from PC monitor, audio classifier, posture detector, and vision into a single `ActivityState`. Weighted voting with confidence scores. The state drives interruptibility decisions.

### `modules/context/interruptibility.py`

Given the current activity state and a speech priority level (`conversation` / `ambient` / `urgent` / `notification`), returns whether Jarvis should speak right now. Enforces quiet hours and inter-interrupt cooldowns.

### `modules/context/curiosity.py`

Topic-based proactive speech engine. Each topic (gaming, cooking, napping, etc.) has a cooldown. When activity matches a topic and the cooldown has elapsed and interruptibility allows, the engine generates a relevant one-liner via LLM and triggers `_speak()`.

---

## Secrets Management

Credentials are **never** committed:

- Copy `.env.example` → `.env` for MQTT credentials
- Copy `hardware/esphome/secrets.yaml.example` → `hardware/esphome/secrets.yaml` for WiFi/OTA credentials
- Both files are in `.gitignore`

---

## Limitations / Roadmap

- **Split the large coordinators** — `core/orchestrator.py` and `dashboard/server.py` are intentionally scheduled for decomposition; see `TODO_SPLIT_ORCHESTRATOR_DASHBOARD.md`.
- **Migrate integrations gradually** — existing camera/audio/MQTT integrations still live in the orchestrator path, but new sensors/actuators should use `modules/integrations/`.
- **Dashboard auth** — keep the dashboard on a trusted network until token auth is added, especially when `dashboard_host` is `0.0.0.0`.
- **YAMNet on CPU** — TensorFlow CUDA support requires additional setup; YAMNet runs on CPU by default.
- **Open-vocab weights** — `open-clip-torch` and `transformers` are now in requirements because `config.yaml` enables open-vocabulary object tracking by default. First boot downloads CLIP/OWLv2 weights to the Hugging Face cache.

---

## License

MIT
