# Jarvis, ambient home AI

A local, always-on assistant that lives in your house instead of in a cloud. It watches what you're doing, stays quiet when you're busy, and speaks up when it has something worth saying.

No cloud required. Nothing has to leave your machine. Free and fully local capable, local or cloud model, configurable. Wake word detection runs locally, so there are no wake word to server round trips.

---

## What it does

Jarvis listens and watches across your rooms and acts on what it notices:

- Wakes on a custom wake word ("Hey Jarvis") detected locally via [openWakeWord](https://github.com/dscripka/openWakeWord).
- Triages ambient speech through a staged voice cascade (VAD, wake, sound event, STT, triage) so it can act on things said near it, not only after a wake word. Every room mic runs the cascade, including the office PC mic.
- Transcribes speech with [faster-whisper](https://github.com/SYSTRAN/faster-whisper), CUDA accelerated.
- Tracks context. It knows if you're gaming, on a call, asleep, or cooking, and adjusts how willing it is to interrupt.
- Watches rooms via camera for mess, light state, posture, and presence using YOLOv8 plus MediaPipe.
- Classifies ambient audio (appliances, music, silence) via TensorFlow/YAMNet.
- Learns object and sound names. Unknowns it keeps seeing or hearing surface in the dashboard Review tab with a photo or audio clip. Name them there or by voice and it recognizes them from then on.
- Identifies residents and pets individually. Face plus voice for people, a per-pet visual identity for cats and dogs. Proactive speech addresses whoever is actually in the room, not one default person.
- Flags behavioral anomalies. It learns each resident's daily routine and surfaces genuinely unusual events (off-hours activity, an unexpected room) in a dashboard review queue.
- Speaks proactively when curiosity fires and you're interruptible. "Washer's done." "Kitchen counter's looking a bit busy." Speech is local via [Piper](https://github.com/rhasspy/piper) or the more expressive [Kokoro](https://github.com/hexgrad/kokoro) backend, selectable per voice.
- Answers direct questions via an Ollama LLM running locally or a configured cloud model.
- Serves a live dashboard at `http://localhost:7070`: activity state, room status, conversation log, appliance tracking, the Review tab (unknown faces, objects, sounds), the behavioral-anomaly queue, degraded-mode status, wake calibration, and per-model call metrics.
- Supports multiple rooms via ESP32-CAM nodes over MQTT (optional hardware expansion).

---

## Architecture

```text
┌─────────────────────────────────────────────────────┐
│                    Orchestrator                      │
│  (core/orchestrator.py - async task coordinator)    │
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

Most runtime modules talk through the async event bus. The bus is bounded, priority aware, and rate-limits high-volume telemetry topics so wake, safety, alarm, and control events do not sit behind bursts of camera, world, and debug traffic.

---

## Hardware

### Minimum (single room, software only)

- Windows 10/11 PC with a discrete GPU (NVIDIA recommended)
- Microphone
- Speakers

### Full multiroom build

- As above, plus one or more AI-Thinker ESP32-CAM nodes per room
- Each node provides microphone input, speaker output, a camera stream, and an MQTT heartbeat
- See [hardware/esphome/BUILD_OFFICE_NODE.md](hardware/esphome/BUILD_OFFICE_NODE.md) for the first node build guide

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally with your chosen model
- [Mosquitto MQTT broker](https://mosquitto.org/download/) for multiroom (optional for single room)
- NVIDIA GPU with CUDA. Strongly recommended, since Whisper and YOLO are much faster on it.

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

# 4. (CUDA, recommended) Reinstall torch with GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 5. Copy and fill in secrets
copy .env.example .env
# Edit .env with your MQTT credentials if needed

# 6. Start required services
ollama serve
ollama pull YOUR_CHOSEN_MODEL    # e.g. ollama pull llama3.1:8b
mosquitto -v                     # or start as a Windows service

# 7. Run the setup validator. It must show all PASS before you launch.
python scripts/setup.py

# 8. Launch
python main.py
```

The dashboard opens automatically. Otherwise browse to `http://localhost:7070`.

---

## Configuration

Everything tunable lives in `config.yaml`. Never hardcode values in source files.

Key sections:

| Section | What to change |
| --- | --- |
| `ollama.model` | Which Ollama model to use for chat and vision |
| `ollama.system_prompt` | Personality and household context. Customize to your setup. |
| `voice.whisper.model_size` | STT accuracy vs. speed (`base` up to `large-v3`) |
| `voice.wake_word.model` | openWakeWord model name |
| `system.event_bus` | Queue size and per-topic telemetry rate limits |
| `rooms` | Per-room video, mic, and speaker channels. See `WYZE_SETUP.md`. |
| `interruptibility.activity_scores` | How interruptible each detected activity is |
| `curiosity.topic_cooldowns_hours` | How often Jarvis can proactively comment on each topic |
| `process_activity_map` | Map your specific game or app .exe names to activities |

---

## Project structure

```text
jarvis/
├── main.py                     # Entry point, boots orchestrator
├── config.yaml                 # Single source of truth for all config
├── requirements.txt
│
├── core/
│   ├── orchestrator.py         # Wires all modules together, runs all loops
│   ├── event_bus.py            # Async pub/sub bus
│   └── exceptions.py
│
├── modules/
│   ├── voice/                  # STT (Whisper), TTS (Piper + Kokoro), wake word + voice cascade
│   ├── brain/                  # LLM (Ollama), session memory, prompt builder, personas
│   ├── context/                # State fusion, interruptibility, curiosity, sleep
│   ├── activity/               # PC monitor, audio classifier, appliance tracker
│   ├── vision/                 # Camera, YOLOv8, MediaPipe pose, object-vocab learner
│   ├── identity/               # Cross-modal person identity (ArcFace face + voice)
│   ├── world_model/            # Persistent entities, belief tracker, pets, anomalies
│   ├── safety/                 # Alarm subsystem (fire, door, cat-escape, clown)
│   ├── integrations/           # Sensor/actuator plugin contracts
│   ├── memory/                 # SQLite database, event log, room baselines
│   └── network/                # MQTT client, ESP32 node manager
│
├── dashboard/
│   ├── server.py               # FastAPI + WebSocket server
│   └── static/                 # Vanilla JS/CSS, no build step
│
├── hardware/
│   └── esphome/                # ESPHome firmware configs for ESP32-CAM nodes
│
├── scripts/
│   ├── setup.py                # Environment validator (run before first launch)
│   ├── test_*.py               # Phase smoke tests (voice/context/vision/mqtt)
│   ├── test_*_synthetic.py     # Assert-based unit suites (run any time, no hardware)
│   ├── diagnose_recognition.py # Read-only face-bank health report
│   └── clean_face_bank.py      # Rebuild a face bank around its coherent core
│
├── start.ps1 / stop.ps1        # Launch / graceful-stop Jarvis
├── force_stop.ps1              # Guaranteed kill, every Jarvis process, nothing else
│
└── data/                       # Runtime data (gitignored)
    ├── jarvis.db               # SQLite event/conversation log
    └── voices/                 # Piper TTS voice models
```

---

## ESP32-CAM nodes (multiroom)

Nodes are flashed with [ESPHome](https://esphome.io). Each node provides:

- An MJPEG camera stream (OpenCV connects over HTTP)
- An MQTT status heartbeat (birth and will messages)
- Microphone and speaker via I2S (INMP441 + MAX98357A)

See the [build guide](hardware/esphome/BUILD_OFFICE_NODE.md) to bring up the first node.

To add a node, copy `hardware/esphome/secrets.yaml.example` to `hardware/esphome/secrets.yaml`, fill in your network details, then set `has_node: true` and `node_ip` for that room in `config.yaml`.

---

## What each module does

### `core/orchestrator.py`

The only file that imports from multiple modules. Everything else talks over the event bus. The orchestrator starts all async loops, routes wake-word events through the voice pipeline, and calls `_broadcast()` to keep the dashboard current.

### `core/event_bus.py`

Async priority queue. Producers call `await bus.publish(topic, payload)`. Consumers register with `bus.subscribe(topic, handler)`. Wake, safety, and control topics dispatch ahead of world, telemetry, and debug topics. High-volume telemetry gets token-bucket rate limits, and a crashed handler never takes down the bus.

### `modules/integrations/`

Small plugin-style contracts for future sensors and actuators. New integrations should implement `SensorPlugin` or `ActuatorPlugin`, publish and subscribe through `EventBus`, and register with `IntegrationRegistry` instead of adding more direct wiring to `core/orchestrator.py`.

### Dashboard operations

The dashboard includes:

- Perf tab model tracking: per provider and model, daily calls, cloud calls, average latency, timeout rate, and average tool-loop iterations.
- Degraded Mode card: best-effort loaded, degraded, or disabled status for wake word, STT, TTS, LLM, MQTT, cameras, identity, world model, open-vocab objects, and registered integrations.
- Wake Calibration card: per-room RMS, peak level, wake score, false-positive count, and suggested sensitivity.
- Anomalies card: the behavioral-anomaly review queue (§25). Events that scored unusual against a resident's learned routine, each with its score, the triggering event, the per-signal breakdown, and a "not unusual" button that feeds threshold auto-tuning.
- Review tab: name or dismiss what Jarvis can't yet identify. Unknown faces (the face-bank pending queue), unknown objects (with the saved crop and where they keep appearing), and unknown sounds (with a playback clip).

### `modules/context/state_fusion.py`

Combines signals from the PC monitor, audio classifier, posture detector, and vision into a single `ActivityState`. Weighted voting with confidence scores. That state drives interruptibility decisions.

### `modules/context/interruptibility.py`

Given the current activity state and a speech priority level (`conversation`, `ambient`, `urgent`, `notification`), it returns whether Jarvis should speak right now. It enforces quiet hours and inter-interrupt cooldowns.

### `modules/context/curiosity.py`

Topic-based proactive speech engine. Each topic (gaming, cooking, napping, and so on) has a cooldown. When activity matches a topic, the cooldown has elapsed, and interruptibility allows it, the engine generates a relevant one-liner via the LLM and triggers `_speak()`.

---

## Secrets management

Credentials are never committed:

- Copy `.env.example` to `.env` for MQTT credentials.
- Copy `hardware/esphome/secrets.yaml.example` to `hardware/esphome/secrets.yaml` for WiFi and OTA credentials.
- Both target files are in `.gitignore`.

---

## Limitations and roadmap

- Large coordinators. `core/orchestrator.py` is split into mixins (init, loops, conversation, tools). `dashboard/server.py` is still one large file and is the remaining decomposition candidate.
- Migrating integrations gradually. Existing camera, audio, and MQTT integrations still live in the orchestrator path, but new sensors and actuators should use `modules/integrations/`.
- Dashboard auth. Keep the dashboard on a trusted network until token auth is added, especially when `dashboard_host` is `0.0.0.0`.
- YAMNet on CPU. TensorFlow CUDA support needs extra setup, so YAMNet runs on CPU by default.
- Open-vocab weights. `open-clip-torch` and `transformers` are in requirements because `config.yaml` enables open-vocabulary object tracking by default. First boot downloads CLIP and OWLv2 weights to the Hugging Face cache.

---

## License

MIT
