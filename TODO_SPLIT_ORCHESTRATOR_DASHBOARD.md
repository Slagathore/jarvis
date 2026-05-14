# Split Orchestrator And Dashboard

The current runtime works, but `core/orchestrator.py` and `dashboard/server.py`
are carrying too many responsibilities. Keep this as an explicit refactor track
so feature work does not keep adding more direct wiring to those files.

## `core/orchestrator.py`

- Extract `VoiceCoordinator`
  - Own STT/TTS/wake setup, wake coalescing, follow-up listening, room audio taps, speaker routing, and voice enrollment.
  - Public surface: `start()`, `stop()`, `register_dashboard(dashboard)`, `speak(...)`, `handle_text(...)`.

- Extract `VisionCoordinator`
  - Own camera manager, light/posture/object/face/scene/anomaly/mess detectors, Wyze camera controls, and camera health.
  - Public surface: `start()`, `stop()`, `capture_snapshot(room)`, `available_rooms()`.

- Extract `WorldModelCoordinator`
  - Own `WorldStore`, `WorldModel`, `ObservationBuilder`, `InteractionMonitor`, pet bootstrap, behavioral profile nightly loop, and world query tools.
  - Public surface: `start()`, `stop()`, `tool_schemas()`, `tool_handlers()`.

- Extract `BrainCoordinator`
  - Own `OllamaLLM`, Gemini direct client lifecycle, Claude client, model registry, memory extraction/retrieval, prompt building, session restore/cleanup, and tool-loop dispatch.
  - Public surface: `process_user_text(text, room)`, `compose_in_character(...)`, `tool_registry()`, `close()`.

- Extract `AlarmCoordinator`
  - Own safety alarm dispatcher, alarm audio, alarm store, and notifier wiring.
  - Public surface: `start()`, `stop()`, `status()`.

- Keep `Orchestrator`
  - Dependency ordering, configuration, lifecycle, bus creation, dashboard wiring, and shutdown only.
  - Target size: under 800 lines after extraction.

## `dashboard/server.py`

- Split route groups into modules under `dashboard/routes/`
  - `core.py`: `/`, `/api/state`, `/api/health`, `/api/degraded`, WebSocket.
  - `voice.py`: voices, speakers, mic status, wake calibration, room audio settings.
  - `vision.py`: cameras, snapshots, streams, reconnect, polygons.
  - `identity.py`: people, samples, pending reviews.
  - `world.py`: events, interactions, pets, clusters, profile rebuilds.
  - `brain.py`: models, model settings, memory, computer/self-edit controls.
  - `ops.py`: perf, logs, tunables, webhooks, restart/shutdown.

- Move state reducers out of `DashboardServer`
  - `dashboard/state.py` should own `_default_state()` and event reducers for `state_update`, `vision`, `audio_level`, `wake_score`, `system_health`, etc.

- Keep `DashboardServer`
  - FastAPI app creation, static mount, shared dependency registration, WebSocket client set, and `broadcast()`.
  - Target size: under 600 lines after extraction.

## Migration Rules

- Do not rewrite behavior while splitting. Move code first, then improve.
- Preserve endpoint paths and JSON shapes so the existing vanilla JS dashboard keeps working during the migration.
- Move one route group or coordinator at a time and run `pyright`, `compileall`, and the synthetic scripts after each step.
- New sensors and actuators should start as `modules/integrations` plugins unless they need to participate in one of the extracted coordinators.
