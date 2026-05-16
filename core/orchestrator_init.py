"""
JARVIS — Ambient Home AI
========================
Mission: InitMixin — extracted from core/orchestrator.py (audit
         roadmap D6 decomposition). Subsystem construction — the _init_* methods called from run().

         Mixed into Orchestrator; every `self.*` resolves against the
         concrete Orchestrator instance at runtime. The full
         orchestrator import block is duplicated below on purpose —
         over-importing is harmless and removes any missing-name risk.

Modules: core/orchestrator_init.py
Classes: InitMixin
"""

from core.orchestrator_base import OrchestratorMixin
import asyncio
import base64
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
from loguru import logger

from core.task_supervisor import TaskPolicy, TaskSupervisor
from core.event_bus import EventBus
from core.exceptions import JarvisError
from dashboard.server import DashboardServer
from modules.integrations import IntegrationContext, IntegrationRegistry
from modules.activity.appliance_tracker import ApplianceTracker
from modules.activity.audio_classifier import AudioClassifier
from modules.activity.pc_monitor import PCMonitor
from modules.brain.llm import OllamaLLM
from modules.brain.model_registry import ModelRegistry
from modules.brain.ask_claude import ClaudeClient
from modules.computer.control import ComputerControl
from modules.selfedit.edit import SelfEditControl
from modules.brain.prompt_builder import PromptBuilder
from modules.brain.session import SessionManager
from modules.context.activity_history import ActivityHistory
from modules.context.curiosity import CuriosityEngine
from modules.context.interruptibility import InterruptibilityManager
from modules.context.sleep_tracker import SleepTracker
from modules.context.state import UNKNOWN_STATE, ActivityState
from modules.context.state_fusion import StateFusion
from modules.memory.database import DatabaseManager
from modules.memory.event_log import EventLogger
from modules.memory.memory_v2 import MemoryStore
from modules.notifications import NotificationDispatcher, NotificationManager
from modules.notifications.channels import build_channels_from_config
from modules.memory.room_baselines import RoomBaselines
from modules.agenda import GoogleCalendar
from modules.network.mqtt_client import MQTTClient
from modules.network.node_manager import NodeManager
from modules.network.webhooks import WebhookManager
from modules.reminders import ReminderScheduler, RemindersStore, parse_reminder
from modules.vision.anomaly_detector import AnomalyDetector
from modules.vision.camera_manager import CameraManager
from modules.vision.face_recognizer import FaceRecognizer
from modules.vision.ignore_zones import filter_detections
from modules.vision.light_detector import LightDetector
from modules.vision.mess_detector import MessDetector
from modules.vision.object_detector import ObjectDetector
from modules.vision.posture_analyzer import PostureAnalyzer
from modules.vision.observation_builder import ObservationBuilder
from modules.layout import DoorMap
from modules.world_model import (
    BehavioralProfileBuilder,
    InteractionMonitor,
    WorldModel,
    WorldQueryTools,
    WorldStore,
    bootstrap_pets_from_config,
)
from modules.world_model.belief import BeliefResolver
from modules.vision.scene_analyzer import SceneAnalyzer
from modules.vision.wyze_cam_control import WyzeCamControl
from core.room_settings import RoomSettings
from modules.brain.persona_manager import PersonaManager
from modules.voice.audio_focus import AudioFocus
from modules.voice.intents import parse_dnd
from modules.voice.mic_manager import MicManager
from modules.voice.speaker_manager import SpeakerManager
from modules.voice.speaker_id import SpeakerIdentifier
from modules.identity.identity_manager import IdentityManager
from modules.voice.sources.null_audio import NullMicSource, NullSpeakerSink
from modules.voice.sources.usb_mic import UsbMicSource
from modules.voice.sources.wake_adapter import MicSourceWakeAdapter
from modules.voice.sources.wyze_ssh_speaker import WyzeSshSpeakerSink
from modules.voice.stt import WhisperSTT
from modules.voice.tts import PiperTTS
from modules.voice.wake_word import WakeWordDetector
from modules.voice.wake_source import WakeSourceManager


# Per-room echo suppression: how long after Jarvis finishes speaking
# in a room we ignore that room's wake events. Tunes the trade-off
# between "Jarvis hears himself and re-wakes" (too short) and "Cole
# can't immediately follow up after a reply" (too long). 1.5s is
# enough for the speaker tail + room reverb to settle on the cam mic
# without noticeably gating real follow-ups (the in-conversation
# follow-up listener is already open separately for that case).
_ECHO_SUPPRESSION_TAIL_S: float = 1.5


from core.orchestrator_tools import ToolsMixin


class InitMixin(OrchestratorMixin):
    """Subsystem construction — the _init_* methods called from run().

    Mixed into Orchestrator — see core/orchestrator_base.py.
    """

    async def _init_database(self) -> None:
        """Initialize SQLite database and logging helpers."""
        # BUG FIX: DatabaseManager takes the full config dict (it reads db_path internally).
        # Was: DatabaseManager(db_path) — wrong, passing a string.
        # Was: await self.db.initialize() — wrong method name, it's init().
        self.db = DatabaseManager(self.config)
        await self.db.init()
        self.event_log = EventLogger(self.db)
        self.notifications = NotificationManager(db=self.db, broadcast=self._broadcast)
        # Phone-alert dispatcher. An absent/empty `notifications:` config
        # block yields zero channels and alarms run local-only — no boot
        # failure. config.notifications.routing maps alarm_type → channels.
        notif_cfg = self.config.get("notifications", {}) or {}
        self.notification_dispatcher = NotificationDispatcher(
            channels=build_channels_from_config(notif_cfg),
            routing=notif_cfg.get("routing", {}) or {},
            db_manager=self.db,
        )
        # BUG FIX: RoomBaselines takes (db, config) — was only receiving db
        self.room_baselines = RoomBaselines(db=self.db, config=self.config)
        self.reminders_store = RemindersStore(self.db)
        self.reminder_scheduler = ReminderScheduler(
            config=self.config,
            store=self.reminders_store,
            event_bus=self.bus,
        )
        logger.info("[Init] Database ready")

    async def _init_voice(self) -> None:
        """Initialize STT, TTS, and wake word detector."""
        # BUG FIX: All voice modules take config: dict, not flat kwargs.
        # The agent wrote the modules one way and the orchestrator another.
        # Every constructor here was wrong — now all pass self.config.

        # Per-room runtime tweaks store. Constructed before the audio +
        # vision managers so they can reference it during their __init__.
        # File path matches data_dir from config so it lives next to the
        # SQLite DB and event logs (sane backup target).
        from pathlib import Path as _Path
        data_dir = _Path(
            self.config.get("system", {}).get("data_dir", "data")
        )
        self.room_settings = RoomSettings(data_dir / "room_settings.json")

        # House layout (door graph). Persisted next to room_settings so a
        # backup of data/ captures both. Empty file is fine — entries
        # accumulate as Cole teaches doors via voice.
        self.door_map = DoorMap(data_dir / "house_layout.json")

        self.stt = WhisperSTT(self.config)
        # WhisperSTT.load() is synchronous (blocking GPU/CPU work) — run in thread
        await asyncio.to_thread(self.stt.load)
        logger.info("[Init] STT (Whisper) loaded")

        self.tts = PiperTTS(self.config)
        # BUG FIX: tts.load() was never called — piper binary + model path never located
        await asyncio.to_thread(self.tts.load)
        logger.info("[Init] TTS (Piper) ready")

        # BUG FIX: WakeWordDetector takes (config, bus) — was passing wrong flat kwargs
        # and using "event_bus" instead of "bus" as the param name
        self.wake = WakeWordDetector(config=self.config, bus=self.bus)
        await asyncio.to_thread(self.wake.load)
        logger.info("[Init] Wake word detector ready")

        # Multi-room wake registry — empty today, populated below from
        # MicManager. Started when the run loop spins up so any sources
        # registered between init and start are picked up.
        self.wake_sources = WakeSourceManager(config=self.config, bus=self.bus)

        # Per-room mic + speaker managers. Built unconditionally — null
        # drivers handle "this room has no audio" cleanly so the rest of
        # the orchestrator doesn't have to None-check at every callsite.
        # SpeakerManager takes RoomSettings so live volume + mute
        # overrides from the dashboard apply on every play() without a
        # restart. MicManager doesn't consume it (no mic-side tweaks
        # yet) but the wiring is parallel for symmetry.
        self.mic_manager = MicManager(self.config)
        self.speaker_manager = SpeakerManager(self.config, room_settings=self.room_settings)
        logger.info(
            f"[Init] MicManager: {len(self.mic_manager.get_rooms())} active rooms; "
            f"SpeakerManager: {len(self.speaker_manager.get_rooms())} active rooms"
        )

        # Bridge per-room mic sources into the wake-word system. Skip:
        #   - Rooms whose mic is the PC's USB device — WakeWordDetector
        #     already consumes that via its own sounddevice grab; a
        #     second listener on the same device would race for samples.
        #   - Rooms with no mic configured (NullMicSource).
        # Everything else (Wyze RTSP audio, ESP MQTT mic) gets a
        # MicSourceWakeAdapter so wake-word detection fires per-room.
        for room_id, src in self.mic_manager._sources.items():
            if isinstance(src, NullMicSource):
                continue
            if isinstance(src, UsbMicSource):
                logger.debug(
                    f"[Init] Skipping wake adapter for '{room_id}' — USB mic is "
                    "owned by the PC WakeWordDetector"
                )
                continue
            try:
                adapter = MicSourceWakeAdapter(src)
                # Audio-level tap — publishes per-room mic RMS / peak to
                # the bus. The existing _on_audio_level subscriber
                # forwards to the dashboard so the UI bars work the same
                # way as the office PC mic (which goes through wake_word).
                # Throttled to ~10Hz inside the adapter.
                async def _level_cb(
                    room: str, rms_db: float, peak_db: float,
                    sample_rate: int,
                ) -> None:
                    await self.bus.publish("audio.level", {
                        "room": room,
                        "db": rms_db,
                        "peak_db": peak_db,
                        "sample_rate": sample_rate,
                    })
                adapter.attach_audio_level_tap(_level_cb)
                self.wake_sources.register(adapter)
                logger.info(f"[Init] Registered wake adapter for '{room_id}'")
            except Exception as e:
                logger.warning(
                    f"[Init] Wake adapter registration for '{room_id}' failed: {e}"
                )

        # Speaker identification — best-effort, fine if it fails to load
        if self.db is not None:
            self.speaker_id = SpeakerIdentifier(self.db)
            try:
                await self.speaker_id.load()
            except Exception as e:
                logger.warning(f"[Init] Speaker ID failed to load: {e}")

        # Audio focus / volume duck — Windows only, no-op elsewhere
        voice_cfg = self.config.get("voice", {})
        focus_cfg = voice_cfg.get("audio_focus", {}) if isinstance(voice_cfg.get("audio_focus"), dict) else {}
        if focus_cfg.get("enabled", True):
            self.audio_focus = AudioFocus(
                duck_factor=float(focus_cfg.get("duck_factor", 0.2)),
            )
            if self.audio_focus.available:
                logger.info("[Init] Audio focus ready (duck other apps while speaking)")
            else:
                logger.debug("[Init] Audio focus unavailable on this platform")

    async def _init_brain(self) -> None:
        """Initialize LLM, session manager, and prompt builder."""
        # BUG FIX: OllamaLLM and SessionManager both take config: dict.
        # Was: OllamaLLM(model=..., base_url=..., timeout=..., system_prompt=...)
        # Was: SessionManager(max_turns=...)
        self.llm = OllamaLLM(self.config)
        self.sessions = SessionManager(self.config)
        self.prompts = PromptBuilder(config=self.config)
        # Persona system. Built from typed config produced by
        # core.config.expand_and_validate(). Wired into PromptBuilder so
        # every LLM call uses the active persona's composed prompt
        # (overlay + persona-specific text). When the personas section
        # is missing from config (legacy), persona stays None and
        # PromptBuilder falls back to ollama.system_prompt.
        typed_personas = self.config.get("_typed_personas")
        if typed_personas:
            self.persona = PersonaManager(
                personas=typed_personas,
                overlay=self.config.get("_persona_overlay", ""),
                revert_cfg=self.config["_persona_revert_cfg"],
                broadcast=self._broadcast,
            )
            self.prompts.attach_persona_manager(self.persona)
            logger.info(
                f"[Init] PersonaManager active "
                f"(default persona = '{self.persona.current_name()}')"
            )
        else:
            logger.info("[Init] No personas section in config — persona system disabled")
        # Load .env so GEMINI_API_KEY / ANTHROPIC_API_KEY are visible.
        try:
            from dotenv import load_dotenv
            from pathlib import Path as _Path
            load_dotenv(_Path(__file__).resolve().parents[1] / ".env")
        except ImportError:
            pass
        self._claude_client = ClaudeClient()

        # Computer control — start DISABLED (Cole flips on from dashboard).
        self.computer = ComputerControl(broadcast=self._broadcast)

        # Self-edit — start DISABLED. Project root is two levels up from this file.
        from pathlib import Path as _Path
        self.selfedit = SelfEditControl(
            project_root=_Path(__file__).resolve().parents[1],
            broadcast=self._broadcast,
        )

        # Memory v2 — semantic store + extraction. Init schema is idempotent.
        # Broadcast callback is wired so auto-extracted memories (background
        # curator after every turn + self-thought loop + explicit LLM tools)
        # all hot-load the dashboard memory card via 'memory.added' events.
        if self.db is not None:
            self.memory = MemoryStore(
                db=self.db, llm=self.llm, broadcast=self._broadcast,
            )
            try:
                await self.memory.init()
            except Exception as e:
                logger.warning(f"[Init] MemoryStore init failed: {e}")
                self.memory = None
        # Model registry — wires the dashboard's LLM selector to the live
        # OllamaLLM. Schema init is idempotent and safe to call before run.
        if self.db is not None:
            self.model_registry = ModelRegistry(
                db=self.db,
                llm=self.llm,
                config=self.config,
                notifier=self.notifications,
                broadcast=self._broadcast,
            )
            try:
                await self.model_registry.init_schema()
            except Exception as e:
                logger.warning(f"[Init] ModelRegistry schema init failed: {e}")
            # Back-reference: lets OllamaLLM look up per-model sampling
            # overrides (temperature, top_k, presence_penalty, etc.) and
            # the thinking-mode toggle on every chat() / chat_with_tools()
            # / vision_query() call. Without this, only modelfile defaults
            # are used.
            if hasattr(self.llm, "set_settings_provider"):
                self.llm.set_settings_provider(self.model_registry)
        logger.info("[Init] Brain (LLM + sessions) ready")

    async def _init_context(self) -> None:
        """Initialize activity detection and context reasoning modules."""
        # BUG FIX: Multiple constructor mismatches fixed here.
        # InterruptibilityManager was getting flat kwargs — takes config: dict
        # AudioClassifier was getting window_seconds kwarg — takes config: dict
        # ApplianceTracker was missing config arg entirely — takes (config, event_bus)
        # SleepTracker was getting no args — takes config: dict
        # CuriosityEngine arg order: (config, llm) not (llm, config) — kwargs so OK but fixed for clarity

        # ActivityHistory feeds StateFusion with time-of-day priors so the
        # fusion gradually becomes informed by Cole's actual routine.
        if self.db is not None:
            self.activity_history = ActivityHistory(self.db)
        self.state_fusion = StateFusion(
            config=self.config,
            activity_history=self.activity_history,
        )
        self.interruptibility = InterruptibilityManager(self.config)
        self.curiosity = CuriosityEngine(config=self.config, llm=self.llm)
        self.sleep_tracker = SleepTracker(self.config)
        self.pc_monitor = PCMonitor(config=self.config)
        self.audio_classifier = AudioClassifier(self.config)
        # BUG FIX: AudioClassifier.load() is `async def` — must await directly, not in thread
        # asyncio.to_thread() on an async function sends a coroutine object to a thread where
        # no event loop exists — it never actually runs.
        await self.audio_classifier.load()
        self.appliance_tracker = ApplianceTracker(config=self.config, event_bus=self.bus)
        logger.info("[Init] Context modules ready")

    async def _init_vision(self) -> None:
        """Initialize camera, vision models, and scene analysis."""
        # BUG FIX: PostureAnalyzer and ObjectDetector take config: dict — were missing it
        # BUG FIX: SceneAnalyzer takes (config, llm) — was receiving flat model/base_url kwargs
        # CameraManager takes RoomSettings so per-room rotation/flip/
        # brightness/contrast applies to every captured frame — both the
        # dashboard preview AND the YOLO/MediaPipe pipelines see the
        # same corrected orientation.
        self.cameras = CameraManager(config=self.config, room_settings=self.room_settings)
        await self.cameras.load()

        # Wyze hardware controls — one per room with video.type wyze_rtsp.
        # SSH host comes from the URL (already env-expanded by core/config),
        # SSH credentials are reused from the room's wyze_ssh_aplay speaker
        # block. Built after CameraManager because the .url field on the
        # video block is the source of truth for the cam IP.
        for room_cfg in self.config.get("rooms", []):
            video = room_cfg.get("video") or {}
            if not isinstance(video, dict) or video.get("type") != "wyze_rtsp":
                continue
            spk = room_cfg.get("speaker") or {}
            host = self._extract_wyze_host(video.get("url", ""))
            if not host:
                logger.warning(
                    f"[Init] Wyze room '{room_cfg.get('id')}' has no host in video.url; "
                    "skipping WyzeCamControl"
                )
                continue
            room_id = room_cfg.get("id", "unknown")
            self.wyze_cam_controls[room_id] = WyzeCamControl(
                room=room_id,
                host=host,
                ssh_user=str(spk.get("ssh_user", "root")),
                ssh_password=spk.get("ssh_password"),
                ssh_key_path=spk.get("ssh_key_path"),
            )
        if self.wyze_cam_controls:
            logger.info(
                f"[Init] WyzeCamControl ready for {len(self.wyze_cam_controls)} room(s): "
                + ", ".join(self.wyze_cam_controls.keys())
            )
        self.light_detector = LightDetector(config=self.config)
        self.posture = PostureAnalyzer(self.config)
        await self.posture.load_async()
        self.object_detector = ObjectDetector(self.config)
        await self.object_detector.load_async()
        self.scene_analyzer = SceneAnalyzer(config=self.config, llm=self.llm)
        self.anomaly_detector = AnomalyDetector(config=self.config, llm=self.llm)
        self.mess_detector = MessDetector(config=self.config, llm=self.llm)

        # Face recognition — best-effort, fine if it can't load
        if self.db is not None:
            self.face_recognizer = FaceRecognizer(self.db)
            try:
                await self.face_recognizer.load()
            except Exception as e:
                logger.warning(f"[Init] Face recognizer failed to load: {e}")

        logger.info("[Init] Vision pipeline ready")

    async def _init_webhooks(self) -> None:
        """Wire the inbound/outbound webhook bridge to the event bus."""
        self.webhooks = WebhookManager(config=self.config, event_bus=self.bus)
        await self.webhooks.load()

    async def _init_calendar(self) -> None:
        """
        Authenticate with Google Calendar. First run opens a browser for OAuth;
        subsequent runs are silent. If credentials.json is missing, calendar
        features are simply disabled — the rest of Jarvis runs fine without it.
        """
        self.calendar = GoogleCalendar(self.config)
        await self.calendar.authenticate()

    async def _init_network(self) -> None:
        """Initialize MQTT client and ESP32 node manager."""
        # BUG FIX: MQTTClient takes (config, event_bus) — was getting flat broker/port/etc kwargs
        # BUG FIX: NodeManager takes (config, mqtt_client) — was using wrong param name "mqtt"
        self.mqtt = MQTTClient(config=self.config, event_bus=self.bus)
        await self.mqtt.connect()
        self.nodes = NodeManager(
            config=self.config, mqtt_client=self.mqtt, event_bus=self.bus,
        )
        await self.nodes.load()
        # Late-bind MQTT into voice managers so esp32_* mic/speaker sources
        # can subscribe/publish. Mic/speaker managers were constructed in
        # _init_voice before MQTT existed.
        if self.mic_manager is not None:
            self.mic_manager.attach_mqtt(self.mqtt)
        if self.speaker_manager is not None:
            self.speaker_manager.attach_mqtt(self.mqtt)
        logger.info("[Init] Network (MQTT + nodes) ready")

    # ── Event Handler Registration ─────────────────────────────────────────
