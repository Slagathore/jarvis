"""
JARVIS — Ambient Home AI
========================
Mission: Central orchestrator that wires every JARVIS module together into a
         single async runtime. This is the top-level coordinator — it instantiates
         all modules, registers event handlers on the bus, starts all background
         loops, and routes data between subsystems.

         Nothing outside this file needs to know about anything else. Modules
         communicate exclusively through the event bus; this file is the only place
         direct module-to-module calls are made.

Modules: core/orchestrator.py
Classes: Orchestrator
Functions:
    Orchestrator.__init__(config)        — Instantiate all modules from config
    Orchestrator.run()                   — Full async entry point; gather all loops
    Orchestrator._load_models()          — Load all ML models (blocking, called once)
    Orchestrator._init_database()        — Initialize database and event logger
    Orchestrator._init_voice()           — Set up STT, TTS, wake word detector
    Orchestrator._init_brain()           — Set up LLM, session manager, prompt builder
    Orchestrator._init_context()         — Set up state fusion, interruptibility, curiosity, sleep
    Orchestrator._init_vision()          — Set up camera, detectors, scene analyzer
    Orchestrator._init_network()         — Set up MQTT client and node manager
    Orchestrator._register_event_handlers() — Subscribe to all event bus topics
    Orchestrator._on_wake_detected(event)   — Wake word handler: record → STT → LLM → TTS
    Orchestrator._context_loop()         — Periodic: PC monitor + audio → state fusion → broadcast
    Orchestrator._vision_loop()          — Periodic: camera → detectors → baselines → broadcast
    Orchestrator._curiosity_loop()       — Periodic: curiosity engine → proactive speech
    Orchestrator._health_broadcast_loop() — Periodic: check Ollama/MQTT → broadcast
    Orchestrator._broadcast(event)       — Safely push event to dashboard
    Orchestrator._on_appliance_changed(event) — Announce finished appliance via TTS
    Orchestrator._on_node_status(event)  — Handle ESP32 node online/offline events
    Orchestrator._speak(text, room, priority) — Full TTS + log pipeline

Variables:
    Orchestrator.config     — Full YAML config dict
    Orchestrator.bus        — EventBus instance (the nervous system)
    Orchestrator.db         — DatabaseManager
    Orchestrator.event_log  — EventLogger
    Orchestrator.wake       — WakeWordDetector
    Orchestrator.stt        — WhisperSTT
    Orchestrator.tts        — PiperTTS
    Orchestrator.llm        — OllamaLLM
    Orchestrator.sessions   — SessionManager
    Orchestrator.prompts    — PromptBuilder
    Orchestrator.state_fusion     — StateFusion
    Orchestrator.interruptibility — InterruptibilityManager
    Orchestrator.curiosity        — CuriosityEngine
    Orchestrator.sleep_tracker    — SleepTracker
    Orchestrator.appliance_tracker — ApplianceTracker
    Orchestrator.pc_monitor        — PCMonitor
    Orchestrator.audio_classifier  — AudioClassifier
    Orchestrator.cameras           — CameraManager
    Orchestrator.light_detector    — LightDetector
    Orchestrator.posture           — PostureAnalyzer
    Orchestrator.object_detector   — ObjectDetector
    Orchestrator.scene_analyzer    — SceneAnalyzer
    Orchestrator.room_baselines    — RoomBaselines
    Orchestrator.mqtt              — MQTTClient
    Orchestrator.nodes             — NodeManager
    Orchestrator.dashboard         — DashboardServer (or None if disabled)
    Orchestrator._current_state    — Last fused ActivityState

#todo: Add persistent reminder system — store reminders in DB, check on timer
#todo: Add face recognition — identify who is in the room using a face model (DeepFace/InsightFace)
#todo: Add voice recognition — identify speaker from voice embedding so "Cole" label is real, not assumed
#todo: Add voice feedback for vision results on user request ("what do you see?")
#todo: Add multi-room audio routing — TTS output goes to the right room's node
#todo: Add calendar integration — pull upcoming events, proactively brief Cole
#todo: Add manual override endpoint — POST /api/activity to force state
#todo: Add graceful shutdown handler for SIGINT/SIGTERM (close DB, disconnect MQTT)
#todo: Add metrics collection — response latency, wake word false positives, etc.
#todo: Add conversation summary at end of day stored to DB
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

import httpx
import numpy as np
from loguru import logger

from core.event_bus import EventBus
from core.exceptions import JarvisError
from dashboard.server import DashboardServer
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
from modules.notifications import NotificationManager
from modules.memory.room_baselines import RoomBaselines
from modules.agenda import GoogleCalendar
from modules.network.mqtt_client import MQTTClient
from modules.network.node_manager import NodeManager
from modules.network.webhooks import WebhookManager
from modules.reminders import ReminderScheduler, RemindersStore, parse_reminder
from modules.vision.anomaly_detector import AnomalyDetector
from modules.vision.camera_manager import CameraManager
from modules.vision.face_recognizer import FaceRecognizer
from modules.vision.light_detector import LightDetector
from modules.vision.mess_detector import MessDetector
from modules.vision.object_detector import ObjectDetector
from modules.vision.posture_analyzer import PostureAnalyzer
from modules.vision.scene_analyzer import SceneAnalyzer
from modules.voice.audio_focus import AudioFocus
from modules.voice.intents import parse_dnd
from modules.voice.speaker_id import SpeakerIdentifier
from modules.identity.identity_manager import IdentityManager
from modules.voice.stt import WhisperSTT
from modules.voice.tts import PiperTTS
from modules.voice.wake_word import WakeWordDetector
from modules.voice.wake_source import WakeSourceManager


class Orchestrator:
    def __init__(self, config: dict):
        self.config = config
        self.bus = EventBus()

        # These are populated in _init_* methods called from run()
        self.db: Optional[DatabaseManager] = None
        self.event_log: Optional[EventLogger] = None
        self.room_baselines: Optional[RoomBaselines] = None
        # Persistent notification inbox surfaced by the dashboard bell.
        # Set up after DB init; passed into IdentityManager so drift/cluster
        # events auto-fire user-visible notifications.
        self.notifications: Optional[NotificationManager] = None
        # Long-term semantic memory (facts, preferences, events, thoughts).
        # LLM-extracted from each turn + retrieved by semantic search and
        # injected into the prompt context. Initialized after the LLM is up.
        self.memory: Optional[MemoryStore] = None

        self.wake: Optional[WakeWordDetector] = None
        # Multi-room wake registry. PC mic continues to flow through `self.wake`
        # (no behavior change). Additional sources — Wyze cam mic streams,
        # ESP node mic publishes — register against this manager so they fire
        # room-tagged 'voice.wake_detected' events on detection.
        self.wake_sources: Optional[WakeSourceManager] = None
        self.stt: Optional[WhisperSTT] = None
        self.tts: Optional[PiperTTS] = None
        self.speaker_id: Optional[SpeakerIdentifier] = None
        # When set, the next wake-recording captured audio is enrolled as a
        # voice sample for this person (instead of being run through STT/LLM).
        # Tuple is (name, prompt_id); cleared after enrollment.
        self._pending_speaker_enrollment: Optional[tuple[str, str]] = None
        self.audio_focus: Optional[AudioFocus] = None

        self.llm: Optional[OllamaLLM] = None
        # Model catalog/registry — surfaces installed models, capabilities,
        # and pull/delete/swap operations to the dashboard.
        self.model_registry: Optional[ModelRegistry] = None
        # Claude API client used by the ask_claude LLM tool (lets the local
        # model escalate hard questions to a stronger reasoning model).
        # ANTHROPIC_API_KEY env var; tool only registers if key present.
        self._claude_client: Optional[ClaudeClient] = None
        # Computer control — kill-switch-gated mouse + keyboard. Default OFF.
        # Toggle from the dashboard. See modules/computer/control.py for the
        # safety architecture (refuse list + confirm queue + pyautogui FAILSAFE).
        self.computer: Optional[ComputerControl] = None
        # Self-edit — Jarvis editing its own codebase. Read tools always
        # available; write/restart tools gated by a kill switch (default OFF).
        # Auto-commits before every write; restart_self pairs with the
        # supervisor wrapper for auto-revert on broken startup.
        self.selfedit: Optional[SelfEditControl] = None
        self.sessions: Optional[SessionManager] = None
        self.prompts: Optional[PromptBuilder] = None

        self.state_fusion: Optional[StateFusion] = None
        self.interruptibility: Optional[InterruptibilityManager] = None
        self.curiosity: Optional[CuriosityEngine] = None
        self.sleep_tracker: Optional[SleepTracker] = None
        self.activity_history: Optional[ActivityHistory] = None

        self.pc_monitor: Optional[PCMonitor] = None
        self.audio_classifier: Optional[AudioClassifier] = None
        self.appliance_tracker: Optional[ApplianceTracker] = None

        self.cameras: Optional[CameraManager] = None
        self.light_detector: Optional[LightDetector] = None
        self.posture: Optional[PostureAnalyzer] = None
        self.object_detector: Optional[ObjectDetector] = None
        self.scene_analyzer: Optional[SceneAnalyzer] = None
        self.face_recognizer: Optional[FaceRecognizer] = None
        # Cross-modal unified identity manager — wraps speaker_id + face_recognizer
        # and persists multiple samples per person across both modalities.
        self.identity: Optional[IdentityManager] = None
        # Last audio buffer captured during a wake event, used by vision loop's
        # opportunistic verify_voice when a face is recognized after the wake.
        # Cleared after one verify cycle.
        self._last_wake_audio: Optional[np.ndarray] = None
        self.anomaly_detector: Optional[AnomalyDetector] = None
        self.mess_detector: Optional[MessDetector] = None

        self.mqtt: Optional[MQTTClient] = None
        self.nodes: Optional[NodeManager] = None
        self.webhooks: Optional[WebhookManager] = None

        self.reminders_store: Optional[RemindersStore] = None
        self.reminder_scheduler: Optional[ReminderScheduler] = None

        self.calendar: Optional[GoogleCalendar] = None
        # Event IDs we've already announced so the proactive alert loop doesn't
        # double-fire as a meeting approaches.
        self._calendar_alerted: set[str] = set()

        self._current_state: ActivityState = UNKNOWN_STATE
        # Last seen pet classes per room — used to dedup pet_seen events so a
        # cat camped on Cole's desk doesn't fire once per minute forever.
        self._last_pets_per_room: dict[str, list[str]] = {}
        # Where Cole most recently demonstrated presence — set by wake events,
        # face recognition, and dashboard chat. Proactive speech (reminders,
        # curiosity, calendar alerts, EOD summary, startup) targets this room
        # so Jarvis follows Cole around instead of always speaking from the
        # office PC. Defaults to "office" because that's where startup happens.
        self._active_user_room: str = "office"
        self._wake_lock = asyncio.Lock()
        self._audio_io_active: bool = False
        # Continuous-conversation follow-up listener: after each TTS reply,
        # we open a window where the user can speak again without re-saying
        # the wake word. The depth counter tracks how nested we are; we don't
        # cap it because natural conversation should keep flowing as long as
        # the user keeps replying. Used for log scoping only.
        self._followup_depth: int = 0
        # Live conversational enrollment state machine. When set, the next
        # follow-up listen captures a name claim (instead of being processed
        # as a normal turn) and seeds a new person from the original audio +
        # the active room's camera frame.
        self._pending_live_enroll: Optional[dict] = None
        self.dashboard: Optional[DashboardServer]

        # Dashboard
        if self.config["system"].get("dashboard_enabled", True):
            self.dashboard = DashboardServer(
                host=self.config["system"].get("dashboard_host", "0.0.0.0"),
                port=self.config["system"].get("dashboard_port", 7070),
            )
        else:
            self.dashboard = None

    # ── Initialization ─────────────────────────────────────────────────────

    async def _init_database(self) -> None:
        """Initialize SQLite database and logging helpers."""
        # BUG FIX: DatabaseManager takes the full config dict (it reads db_path internally).
        # Was: DatabaseManager(db_path) — wrong, passing a string.
        # Was: await self.db.initialize() — wrong method name, it's init().
        self.db = DatabaseManager(self.config)
        await self.db.init()
        self.event_log = EventLogger(self.db)
        self.notifications = NotificationManager(db=self.db, broadcast=self._broadcast)
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

        # Multi-room wake registry — empty today, populated by Wyze cam adapter
        # and ESP node mic firmware once those land. Started when the run loop
        # spins up so any sources registered between init and start are picked up.
        self.wake_sources = WakeSourceManager(config=self.config, bus=self.bus)

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
        self.cameras = CameraManager(config=self.config)
        await self.cameras.load()
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
        self.nodes = NodeManager(config=self.config, mqtt_client=self.mqtt)
        await self.nodes.load()
        logger.info("[Init] Network (MQTT + nodes) ready")

    # ── Event Handler Registration ─────────────────────────────────────────

    def _register_event_handlers(self) -> None:
        """Subscribe to all relevant event bus topics."""
        # BUG FIX: WakeWordDetector publishes to "voice.wake_detected" —
        # orchestrator was subscribing to "wake.detected" (completely different topic).
        # The wake pipeline would have silently never fired.
        self.bus.subscribe("voice.wake_detected", self._on_wake_detected)
        self.bus.subscribe("appliance.state_changed", self._on_appliance_changed)
        self.bus.subscribe("node.status", self._on_node_status)
        self.bus.subscribe("audio.level", self._on_audio_level)
        self.bus.subscribe("reminder.due", self._on_reminder_due)

    async def _on_reminder_due(self, event: dict) -> None:
        """Speak a fired reminder. The scheduler already marked it fired."""
        message = (event.get("message") or "").strip()
        if not message:
            return
        text = await self._compose_in_character(
            prompt=(
                f"You set a reminder for Cole earlier and the time has now arrived. "
                f"The thing he wanted to remember is: \"{message}\". "
                f"Speak the reminder out loud in a single sentence — your usual "
                f"voice, dry and a little witty, referencing the specific task. "
                f"No preamble, no quotation marks, just the spoken line."
            ),
            fallback=f"Heads up — {message}.",
        )
        await self._speak(text, priority="notification")
        await self._broadcast({
            "type": "reminder_fired",
            "id": event.get("id"),
            "message": message,
        })

    async def _compose_in_character(self, prompt: str, fallback: str) -> str:
        """
        Ask the LLM to write a single in-character line using the configured
        system prompt. Returns the LLM line on success; falls back only if the
        LLM is unavailable or the call fails.

        Use this for any user-facing Jarvis speech (announcements, confirmations,
        proactive observations) so phrasings stay in-character instead of
        robotic templates. See feedback memory: no hardcoded user-facing strings.
        """
        if self.llm is None:
            return fallback
        system = self.config["ollama"].get("system_prompt", "You are Jarvis.")
        try:
            response = await self.llm.chat([
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ])
            line = response.strip().strip('"').strip("'").strip()
            if line:
                return line
        except Exception as e:
            logger.warning(f"[LLM] In-character phrasing failed, using fallback: {e}")
        return fallback

    async def _on_audio_level(self, event: dict) -> None:
        """Forward periodic mic dBFS readings from wake_word to the dashboard."""
        await self._broadcast({
            "type": "audio_level",
            "room": event.get("room", "office"),
            "db": event.get("db", -100.0),
        })

    # ── Wake Word + Conversation Pipeline ─────────────────────────────────

    async def _on_wake_detected(self, event: dict) -> None:
        """
        Full pipeline from wake word to spoken response.
        1. Play acknowledgment chime
        2. Record audio until silence
        3. Transcribe with Whisper
        4. Build prompt with context
        5. LLM response
        6. TTS playback
        7. Log everything
        """
        room = event.get("room", "office")
        if self._wake_lock.locked():
            logger.info(f"[Wake] Ignoring duplicate wake in {room} while capture is active")
            return

        logger.info(f"[Wake] Detected in {room}")
        # Wake event = strong presence signal. Update the active room so
        # downstream proactive speech follows Cole.
        self._active_user_room = room

        # Check interruptibility before responding
        # Guard self.interruptibility (Optional) before member access
        if self._current_state and self.interruptibility and not self.interruptibility.can_interrupt(
            self._current_state, priority="conversation"
        ):
            logger.debug("[Wake] Blocked by interruptibility gate")
            return

        # ── CAPTURE PHASE — mic-exclusive, lock held briefly ────────────────
        # Holding the wake_lock across the LLM tool loop (which can run for
        # minutes) used to make subsequent wakes silently queue with no chime.
        # The lock now covers ONLY the chime + record window so a new wake
        # can fire its own beep + capture in parallel with whatever the prior
        # turn's LLM is still doing.
        from modules.voice.audio_utils import (
            SAMPLE_RATE,
            db_from_rms,
            play_chime_async,
            record_until_silence,
        )
        audio_data = None
        async with self._wake_lock:
            was_audio_active = self._audio_io_active
            self._audio_io_active = True
            if self.wake:
                self.wake.suspend()
            try:
                await play_chime_async()
                await asyncio.sleep(0.3)

                recording_cfg = self.config["voice"]["recording"]
                record_device = (
                    self.wake.device if self.wake else recording_cfg.get("device")
                )
                # Use the pre-wake noise floor measured by wake_word's
                # always-on stream — far more reliable than calibrating
                # during the first 400ms of recording (which clips the
                # user mid-sentence if they start talking immediately).
                pre_floor = (
                    self.wake.get_noise_floor_db(
                        fallback_db=recording_cfg["silence_threshold_db"]
                    )
                    if self.wake else recording_cfg["silence_threshold_db"]
                )
                logger.debug(f"[Wake] using pre-calibrated floor: {pre_floor:.1f} dBFS")
                audio_data = await asyncio.to_thread(
                    record_until_silence,
                    silence_threshold_db=pre_floor,
                    silence_duration_ms=recording_cfg["silence_duration_ms"],
                    max_duration_seconds=recording_cfg["max_duration_seconds"],
                    speech_start_timeout_seconds=recording_cfg.get(
                        "speech_start_timeout_seconds",
                        5.0,
                    ),
                    device=record_device,
                    mode=recording_cfg.get("mode", "silence"),
                    fixed_duration_seconds=float(
                        recording_cfg.get("fixed_duration_seconds", 7.0)
                    ),
                    # Adaptive disabled — pre-wake floor is what we use now.
                    adaptive_noise_floor=False,
                )
            finally:
                if self.wake:
                    self.wake.wakeup()
                # Restore prior audio-io state. _speak will re-set it during
                # any TTS playback that follows; the gap between here and
                # there is fine for audio_classifier to read normally.
                self._audio_io_active = was_audio_active
        # ── LOCK RELEASED ──────────────────────────────────────────────────
        # Subsequent wakes can now beep + capture concurrently with the rest
        # of this turn's processing.

        if audio_data is None or len(audio_data) == 0:
            logger.debug("[Wake] No audio recorded")
            return

        try:
            duration_s = len(audio_data) / SAMPLE_RATE
            rms = float(np.sqrt(np.mean(audio_data ** 2))) if len(audio_data) else 0.0
            logger.info(
                f"[Wake] Captured {duration_s:.2f}s of audio "
                f"(rms={db_from_rms(rms):.1f} dBFS)"
            )

            # Voice enrollment fast-path: if dashboard armed an enrollment,
            # route this capture into IdentityManager as a voice_sample for
            # the given name + prompt_id, instead of running STT/LLM.
            if (
                self._pending_speaker_enrollment is not None
                and self.identity is not None
            ):
                name, prompt_id = self._pending_speaker_enrollment
                self._pending_speaker_enrollment = None
                sample_id = await self.identity.enroll_voice(
                    name, audio_data, prompt_id=prompt_id
                )
                ok = sample_id is not None
                await self._broadcast({
                    "type":      "speaker_enrolled",
                    "name":      name,
                    "ok":        ok,
                    "prompt_id": prompt_id,
                })
                confirmation = await self._compose_in_character(
                    prompt=(
                        f"You just successfully recorded a voice sample for '{name}'. "
                        f"Speak a single short in-character acknowledgement that "
                        f"you'll recognize them now. No preamble, no quotes."
                    ) if ok else (
                        f"You tried to enroll a voice sample for '{name}' but it "
                        f"failed. Apologize briefly in your usual voice. No "
                        f"preamble, no quotes."
                    ),
                    fallback=(f"Got it, I'll remember your voice as {name}." if ok
                              else f"Sorry, I couldn't save your voice sample."),
                )
                await self._speak(confirmation, room=room, priority="conversation")
                return

            stt = self.stt
            if stt is None:
                logger.warning("[Wake] STT module not initialized — skipping transcript")
                return
            transcript = await asyncio.to_thread(stt.transcribe, audio_data)
            if not transcript or not transcript.strip():
                logger.info("[Wake] Empty transcript — nothing heard after chime")
                # Dump capture to data/debug/ so we can listen and tell whether
                # the mic actually got speech or just noise.
                if self.config["voice"]["whisper"].get("debug_save_empty", False):
                    try:
                        from pathlib import Path
                        import soundfile as sf
                        debug_dir = (
                            Path(self.config["system"].get("data_dir", "data/")) / "debug"
                        )
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        wav_path = debug_dir / f"empty_transcript_{ts}.wav"
                        sf.write(str(wav_path), audio_data, SAMPLE_RATE, subtype="PCM_16")
                        logger.info(f"[Wake] Saved empty-transcript capture to {wav_path}")
                    except Exception as save_err:
                        logger.debug(f"[Wake] Could not save debug capture: {save_err}")
                return

            logger.info(f"[STT] Transcript: {transcript!r}")

            # Identify the speaker via the unified IdentityManager. Match on
            # voice samples across all enrolled persons. Stash the buffer so
            # the next vision tick can opportunistically verify the face for
            # this person (drift refresh).
            speaker_name: Optional[str] = None
            identified = False
            if self.identity is not None:
                try:
                    match = await self.identity.identify_voice(audio_data)
                    if match is not None:
                        speaker_name = match.name
                        identified = True
                        ambig = " (ambiguous)" if match.is_ambiguous else ""
                        logger.info(
                            f"[Identity/voice] '{match.name}' "
                            f"(sim={match.similarity:.2f}){ambig}"
                        )
                        self._last_wake_audio = audio_data
                    else:
                        logger.debug("[Identity/voice] no match — sample queued in pending")
                except Exception as e:
                    logger.debug(f"[Identity/voice] identify failed: {e}")

            # Live conversational enrollment: voice didn't match anyone but
            # they actually said something. Ask them who they are; the
            # follow-up listener will catch the reply, extract the name,
            # and seed a person row with this original audio + camera face.
            if (
                not identified
                and self.identity is not None
                and self._pending_live_enroll is None
            ):
                self._pending_live_enroll = {
                    "audio": audio_data,
                    "room": room,
                    "first_transcript": transcript,
                }
                question = await self._compose_in_character(
                    prompt=(
                        "You just heard a voice you don't recognize. They "
                        "said: " + repr(transcript) + ". Ask, in one short "
                        "in-character line, what you should call them. No "
                        "preamble, no quotes."
                    ),
                    fallback="I don't think we've met — what should I call you?",
                )
                await self._speak(question, room=room, priority="conversation")
                return

            await self._process_user_text(transcript, room, speaker=speaker_name)

        except Exception as e:
            logger.error(f"[Wake] Pipeline error: {e}")

    async def _process_user_text(self, text: str, room: str, speaker: Optional[str] = None) -> None:
        """
        Core LLM pipeline shared by voice (wake word) and text chat.
        Broadcasts user speech, calls LLM, speaks the response.

        First tries to parse the text as a reminder intent — if it matches,
        creates the reminder and short-circuits the LLM call.
        """
        await self._broadcast({
            "type":    "user_speech",
            "text":    text,
            "room":    room,
            "speaker": speaker,
        })
        if self.event_log:
            log_content = f"[{speaker}] {text}" if speaker else text
            await self.event_log.log_event(room=room, event_type="user_speech", content=log_content)

        # DND intent fast path: "shut up for 30 minutes" / "you can talk again"
        if await self._try_dnd(text, room):
            return

        # Reminder intent fast path: "remind me to X in N minutes" / "at HH:MM"
        if await self._try_create_reminder(text, room):
            return

        # Calendar intents are handled by the LLM via tool calling below —
        # no regex shortcut needed since the model can call calendar_list_events
        # / create_event / delete_event for any phrasing or date range.

        if not self.sessions or not self.prompts or not self.llm:
            logger.warning("[LLM] Brain modules not ready — skipping")
            return

        session = self.sessions.get_session(room)

        # Build activity-history context (predicted remaining + typical-now)
        extras: dict = {}
        if self.activity_history is not None and self._current_state is not None:
            try:
                blurb = await self.activity_history.summary_for_prompt(
                    self._current_state.activity
                )
                if blurb:
                    extras["activity_history"] = blurb
            except Exception as e:
                logger.debug(f"[ActivityHistory] prompt summary failed: {e}")

        # Memory v2: retrieve top-K relevant memories from semantic store and
        # inject as additional system-prompt context so the LLM has long-term
        # knowledge of facts/preferences across conversations.
        if self.memory is not None and self.memory.is_loaded:
            try:
                hits = await self.memory.retrieve(text, k=8)
                if hits:
                    if extras is None:
                        extras = {}
                    lines = ["Relevant memories (use as context, don't recite verbatim):"]
                    for h in hits:
                        subj = f" [{h['subject']}]" if h.get("subject") else ""
                        lines.append(f"  - ({h['kind']}{subj}) {h['content']}")
                    extras["relevant_memories"] = "\n".join(lines)
            except Exception as e:
                logger.debug(f"[MemoryV2] retrieve failed: {e}")

        prompt_context = await self.prompts.build_with_memory(
            user_text=text,
            state=self._current_state,
            session=session,
            room=room,
            db=self.db,
            extras=extras or None,
        )

        # Tool calling: collect every tool currently available — calendar (if
        # authenticated), ask_claude (always available with API key), memory
        # tools, etc. The LLM picks whichever it needs based on the user's
        # question. Empty list → plain chat path.
        tools, handlers = self._build_tool_registry()
        if tools:
            response = await self.llm.chat_with_tools(
                messages=prompt_context,
                tools=tools,
                tool_handlers=handlers,
                action_tool_names=self._ACTION_TOOL_NAMES,
            )
        else:
            response = await self.llm.chat(messages=prompt_context)
        if not response:
            logger.warning("[LLM] Empty response")
            return

        session.add_turn("user", text)
        session.add_turn("assistant", response)
        logger.info(f"[LLM] Response: {response!r}")

        if self.interruptibility is not None:
            self.interruptibility.record_interruption()

        # Fire-and-forget memory extraction. Runs the curator LLM call in the
        # background so the user-facing TTS reply isn't gated on it.
        if self.memory is not None:
            asyncio.create_task(
                self.memory.extract_from_turn(
                    user_text=text, assistant_text=response, room=room
                )
            )

        await self._speak(response, room=room, priority="conversation")

    # ── Calendar tool schemas (Ollama / OpenAI function-calling format) ──────
    # The LLM gets these on every chat call. It decides when to invoke them
    # based on the user's question. Real-time queries — no stale cache, any
    # date range works (today, "first Tuesday of May 2028", historical, etc.)
    _CALENDAR_TOOLS: list[dict] = [
        {
            "type": "function",
            "function": {
                "name": "calendar_list_events",
                "description": (
                    "Look up events on Cole's Google Calendar within a specific "
                    "time range. Use this for ANY question about what's "
                    "scheduled, what's coming up, what's on a particular day or "
                    "date, etc. Always call this rather than guessing — the "
                    "current date is in the system prompt so you can compute "
                    "any range."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_iso": {
                            "type": "string",
                            "description": (
                                "Start of range in ISO 8601 (YYYY-MM-DDTHH:MM:SS). "
                                "Example: 2028-05-02T00:00:00. Defaults to now."
                            ),
                        },
                        "end_iso": {
                            "type": "string",
                            "description": (
                                "End of range in ISO 8601. Example: "
                                "2028-05-02T23:59:59. Defaults to 7 days from now."
                            ),
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calendar_create_event",
                "description": "Create a new event on Cole's Google Calendar.",
                "parameters": {
                    "type": "object",
                    "required": ["title", "start_iso"],
                    "properties": {
                        "title":       {"type": "string"},
                        "start_iso":   {"type": "string", "description": "ISO 8601 start time"},
                        "end_iso":     {"type": "string", "description": "ISO 8601 end time. Defaults to start + 1h."},
                        "description": {"type": "string"},
                        "location":    {"type": "string"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calendar_delete_event",
                "description": (
                    "Delete a specific event by its ID. ALWAYS call "
                    "calendar_list_events first to look up the event ID — "
                    "never guess one."
                ),
                "parameters": {
                    "type": "object",
                    "required": ["event_id"],
                    "properties": {
                        "event_id": {
                            "type": "string",
                            "description": "Event ID returned by calendar_list_events.",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calendar_update_event",
                "description": (
                    "Patch an existing event by its ID. Only fields you pass "
                    "get changed (use this to reschedule, rename, or relocate "
                    "an event). ALWAYS call calendar_list_events first to "
                    "find the right event ID."
                ),
                "parameters": {
                    "type": "object",
                    "required": ["event_id"],
                    "properties": {
                        "event_id":    {"type": "string", "description": "Event ID to patch."},
                        "title":       {"type": "string", "description": "New title (omit to leave unchanged)."},
                        "start_iso":   {"type": "string", "description": "New start time, ISO 8601."},
                        "end_iso":     {"type": "string", "description": "New end time, ISO 8601."},
                        "description": {"type": "string"},
                        "location":    {"type": "string"},
                    },
                },
            },
        },
    ]

    async def _tool_calendar_list_events(
        self,
        start_iso: Optional[str] = None,
        end_iso: Optional[str] = None,
    ) -> dict:
        """Tool handler: list calendar events in an arbitrary time range."""
        if self.calendar is None or not self.calendar.is_authenticated:
            return {"error": "calendar not available"}
        from datetime import timedelta as _td
        now = datetime.now()
        try:
            start = datetime.fromisoformat(start_iso) if start_iso else now
        except ValueError:
            return {"error": f"start_iso not parseable: {start_iso!r}"}
        try:
            end = datetime.fromisoformat(end_iso) if end_iso else (now + _td(days=7))
        except ValueError:
            return {"error": f"end_iso not parseable: {end_iso!r}"}

        # GoogleCalendar.upcoming_events takes hours-from-now; fall back to a
        # direct service.events().list call for an arbitrary range.
        events = await self._calendar_list_in_range(start, end)
        return {"start": start.isoformat(), "end": end.isoformat(), "events": events}

    async def _calendar_list_in_range(self, start: datetime, end: datetime) -> list[dict]:
        """List calendar events between two arbitrary datetimes."""
        cal = self.calendar
        if cal is None or cal._service is None:
            return []
        from datetime import timezone as _tz
        # Google requires ISO with tz; assume local if naive.
        if start.tzinfo is None:
            start = start.astimezone()
        if end.tzinfo is None:
            end = end.astimezone()
        service = cal._service

        def _list() -> list[dict]:
            resp = service.events().list(
                calendarId=cal._calendar_id,
                timeMin=start.isoformat(),
                timeMax=end.isoformat(),
                maxResults=50,
                singleEvents=True,
                orderBy="startTime",
            ).execute()
            return resp.get("items", [])

        try:
            raw = await asyncio.to_thread(_list)
        except Exception as e:
            logger.warning(f"[Calendar] tool list_in_range failed: {e}")
            return []
        return [cal._normalize_event(e) for e in raw]

    async def _tool_calendar_create_event(
        self,
        title: str,
        start_iso: str,
        end_iso: Optional[str] = None,
        description: Optional[str] = None,
        location: Optional[str] = None,
    ) -> dict:
        """Tool handler: create a new event."""
        if self.calendar is None or not self.calendar.is_authenticated:
            return {"error": "calendar not available"}
        try:
            start = datetime.fromisoformat(start_iso)
        except ValueError:
            return {"error": f"start_iso not parseable: {start_iso!r}"}
        end: Optional[datetime] = None
        if end_iso:
            try:
                end = datetime.fromisoformat(end_iso)
            except ValueError:
                return {"error": f"end_iso not parseable: {end_iso!r}"}
        event = await self.calendar.add_event(
            title=title, start=start, end=end,
            description=description, location=location,
        )
        if event is None:
            return {"error": "calendar API rejected create"}
        await self._broadcast({"type": "calendar_added", "event": event})
        return event

    async def _tool_calendar_delete_event(self, event_id: str) -> dict:
        """Tool handler: delete an event by ID."""
        if self.calendar is None or not self.calendar.is_authenticated:
            return {"error": "calendar not available"}
        ok = await self.calendar.delete_event(event_id)
        if ok:
            await self._broadcast({"type": "calendar_deleted", "id": event_id})
        return {"ok": ok, "event_id": event_id}

    async def _tool_calendar_update_event(
        self,
        event_id: str,
        title: Optional[str] = None,
        start_iso: Optional[str] = None,
        end_iso: Optional[str] = None,
        description: Optional[str] = None,
        location: Optional[str] = None,
    ) -> dict:
        """Tool handler: patch an existing event."""
        if self.calendar is None or not self.calendar.is_authenticated:
            return {"error": "calendar not available"}
        start = end = None
        if start_iso:
            try:
                start = datetime.fromisoformat(start_iso)
            except ValueError:
                return {"error": f"start_iso not parseable: {start_iso!r}"}
        if end_iso:
            try:
                end = datetime.fromisoformat(end_iso)
            except ValueError:
                return {"error": f"end_iso not parseable: {end_iso!r}"}
        event = await self.calendar.update_event(
            event_id=event_id, title=title, start=start, end=end,
            description=description, location=location,
        )
        if event is None:
            return {"error": "calendar API rejected update"}
        await self._broadcast({"type": "calendar_updated", "event": event})
        return event

    def _calendar_tool_handlers(self) -> dict:
        """Map tool names to bound async handlers."""
        return {
            "calendar_list_events":   self._tool_calendar_list_events,
            "calendar_create_event":  self._tool_calendar_create_event,
            "calendar_delete_event":  self._tool_calendar_delete_event,
            "calendar_update_event":  self._tool_calendar_update_event,
        }

    # ── ask_claude tool ──────────────────────────────────────────────────────
    # Lets the local LLM escalate hard questions to Claude (a stronger
    # reasoning model). Useful when Jarvis is on a smaller local model and
    # hits a wall on a tricky debug, design choice, or code question.
    _ASK_CLAUDE_TOOL: dict = {
        "type": "function",
        "function": {
            "name": "ask_claude",
            "description": (
                "Ask Anthropic's Claude (a strong reasoning model) a question. "
                "Use this when you need help debugging code, reasoning about a "
                "tricky design, or getting a second opinion. Pass the user's "
                "question or your own. Optionally include a code snippet, "
                "stack trace, or other context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask Claude. Be specific.",
                    },
                    "context": {
                        "type": "string",
                        "description": (
                            "Optional context: code snippet, error message, file "
                            "contents, etc. Goes after the question."
                        ),
                    },
                },
                "required": ["question"],
            },
        },
    }

    async def _tool_ask_claude(self, question: str, context: Optional[str] = None) -> dict:
        if self._claude_client is None:
            return {"error": "ask_claude unavailable — ANTHROPIC_API_KEY not set"}
        if not self._claude_client.has_key:
            return {"error": "ask_claude unavailable — ANTHROPIC_API_KEY not set"}
        answer = await self._claude_client.ask(question, context=context)
        return {"answer": answer}

    # ── Memory tools ────────────────────────────────────────────────────────
    # Let the LLM explicitly save and search long-term memories. Auto-extraction
    # already runs after every turn, but these let the model take direct
    # action when it decides "this specifically should be remembered" or
    # "I need to look something up before answering".
    _MEMORY_TOOLS: list[dict] = [
        {
            "type": "function",
            "function": {
                "name": "remember",
                "description": (
                    "Save a fact, preference, or instruction to long-term memory. "
                    "Use when the user shares something specific worth remembering "
                    "verbatim, or when you've decided something matters enough to "
                    "outlive the current conversation. Auto-extraction already runs "
                    "after every turn — only call this if you want explicit control."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content":    {"type": "string", "description": "The memory as a single declarative sentence."},
                        "kind":       {"type": "string", "description": "fact | preference | event | instruction"},
                        "subject":    {"type": "string", "description": "Person, room, or topic this relates to. Optional."},
                        "importance": {"type": "number", "description": "0.0-1.0. 0.9+ for load-bearing personal info."},
                    },
                    "required": ["content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "recall",
                "description": (
                    "Search long-term memory for facts/preferences/events relevant "
                    "to a query. Use when you suspect Cole has told you something "
                    "before and you want to confirm before guessing. Returns up to "
                    "8 best matches scored by semantic similarity × importance × recency."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What you're looking for."},
                        "k":     {"type": "integer", "description": "Max results (default 8)."},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "record_thought",
                "description": (
                    "Save one of your own reflections or realizations. Use when "
                    "you've noticed something interesting about a pattern, "
                    "drawn a conclusion worth keeping, or want to write down a "
                    "passing observation. These are SEARCHABLE later via recall()."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The thought, written naturally."},
                        "subject": {"type": "string", "description": "What it's about. Optional."},
                    },
                    "required": ["content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "record_question",
                "description": (
                    "Save a question you want answered later — by Cole when he's "
                    "next available, or by Claude via ask_claude. Useful when "
                    "you're curious about something but don't want to interrupt "
                    "the current conversation. The dashboard surfaces unanswered "
                    "questions for review."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content":    {"type": "string", "description": "The question."},
                        "subject":    {"type": "string", "description": "What it's about. Optional."},
                        "importance": {"type": "number", "description": "0.0-1.0. Default 0.5."},
                    },
                    "required": ["content"],
                },
            },
        },
    ]

    async def _tool_remember(
        self, content: str, kind: str = "fact",
        subject: Optional[str] = None, importance: float = 0.7,
    ) -> dict:
        if self.memory is None:
            return {"error": "memory store unavailable"}
        mid = await self.memory.add(
            kind=kind, content=content, subject=subject,
            importance=importance, source_kind="manual",
        )
        return {"ok": mid is not None, "memory_id": mid}

    async def _tool_recall(self, query: str, k: int = 8) -> dict:
        if self.memory is None:
            return {"error": "memory store unavailable"}
        hits = await self.memory.retrieve(query, k=int(k))
        return {"matches": hits}

    async def _tool_record_thought(
        self, content: str, subject: Optional[str] = None,
    ) -> dict:
        if self.memory is None:
            return {"error": "memory store unavailable"}
        mid = await self.memory.record_thought(content, subject=subject)
        return {"ok": mid is not None, "memory_id": mid}

    async def _tool_record_question(
        self, content: str, subject: Optional[str] = None,
        importance: float = 0.5,
    ) -> dict:
        if self.memory is None:
            return {"error": "memory store unavailable"}
        mid = await self.memory.record_question(
            content, subject=subject, importance=importance,
        )
        return {"ok": mid is not None, "memory_id": mid}

    # ── Computer control tools ──────────────────────────────────────────────
    # Mouse + keyboard + screenshot. Gated by the kill switch on
    # self.computer; tools only register when computer.enabled is True so the
    # LLM doesn't even know they exist when control is off.
    _COMPUTER_TOOLS: list[dict] = [
        {
            "type": "function",
            "function": {
                "name": "screenshot",
                "description": "Capture the current desktop. Returns a base64-encoded JPEG plus screen size. Use to see what's on screen before deciding where to click.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "screen_size",
                "description": "Return the screen resolution as {width, height}. Useful for computing relative click positions.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "mouse_click",
                "description": "Click at absolute screen pixel (x, y). Take a screenshot first to find the right coordinates.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer"},
                        "y": {"type": "integer"},
                        "button": {"type": "string", "description": "left | right | middle"},
                        "clicks": {"type": "integer", "description": "1 for click, 2 for double-click"},
                    },
                    "required": ["x", "y"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "mouse_move",
                "description": "Move the cursor to (x, y) without clicking.",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                    "required": ["x", "y"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "keyboard_type",
                "description": "Type a string at the current keyboard focus. Refuses dangerous patterns (rm -rf, format, shutdown, etc.); some patterns require confirmation.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "keyboard_hotkey",
                "description": "Send a key combination, e.g. ['ctrl', 'c']. Refuses dangerous combos (Ctrl+Alt+Del, Win+L, Alt+F4).",
                "parameters": {
                    "type": "object",
                    "properties": {"keys": {"type": "array", "items": {"type": "string"}}},
                    "required": ["keys"],
                },
            },
        },
    ]

    def _computer_tool_handlers(self) -> dict:
        c = self.computer
        if c is None:
            return {}
        return {
            "screenshot":      lambda: c.screenshot(),
            "screen_size":     lambda: c.screen_size(),
            "mouse_click":     c.mouse_click,
            "mouse_move":      c.mouse_move,
            "keyboard_type":   c.keyboard_type,
            "keyboard_hotkey": c.keyboard_hotkey,
        }

    # ── Self-edit tools ─────────────────────────────────────────────────────
    # Read tools always available so the LLM can analyze its own code even
    # when write is disabled. Write/restart tools only register when the
    # selfedit kill switch is ON.
    _SELFEDIT_READ_TOOLS: list[dict] = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file from the project. Path is relative to the project root.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files matching a glob pattern relative to the project root.",
                "parameters": {
                    "type": "object",
                    "properties": {"glob": {"type": "string", "description": "Default: **/*"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "grep_files",
                "description": "Search file contents for a regex pattern. Returns up to 100 matches.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "glob": {"type": "string", "description": "Default: **/*.py"},
                    },
                    "required": ["pattern"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "git_log",
                "description": "Show the last N commits.",
                "parameters": {
                    "type": "object",
                    "properties": {"n": {"type": "integer", "description": "Default 20"}},
                },
            },
        },
    ]
    _SELFEDIT_WRITE_TOOLS: list[dict] = [
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write a complete file. Auto-commits before writing so the change is one git reset away. Some files require dashboard confirmation; the protected list (this file, orchestrator, .env, jarvis.db) is blocked outright.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_file",
                "description": "Find-and-replace exactly one occurrence of old_string with new_string. Auto-commits before editing.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old_string": {"type": "string"},
                        "new_string": {"type": "string"},
                    },
                    "required": ["path", "old_string", "new_string"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "restart_self",
                "description": "Schedule a graceful restart so just-applied edits take effect. Pairs with the supervisor wrapper for auto-revert if the new instance fails to start within 10 seconds.",
                "parameters": {
                    "type": "object",
                    "properties": {"reason": {"type": "string"}},
                    "required": ["reason"],
                },
            },
        },
    ]

    def _selfedit_handlers(self) -> dict:
        s = self.selfedit
        if s is None:
            return {}
        return {
            "read_file":     s.read_file,
            "list_files":    s.list_files,
            "grep_files":    s.grep_files,
            "git_log":       s.git_log,
            "write_file":    s.write_file,
            "edit_file":     s.edit_file,
            "restart_self":  s.restart_self,
        }

    # ── Pattern D: action tools that trigger mid-loop model swap ────────────
    # When the LLM calls any of these tool names, OllamaLLM.chat_with_tools
    # swaps the rest of the loop onto the configured action_model (typically
    # a cheap local model like qwen3.5:4b) so the rate-limit-burning
    # mechanical execution doesn't keep hitting the cloud chat model.
    _ACTION_TOOL_NAMES: set = {
        # Computer control
        "screenshot", "screen_size",
        "mouse_click", "mouse_move",
        "keyboard_type", "keyboard_hotkey",
        # Self-edit (read tools count too — file exploration was a major
        # rate-limit burn pattern Cole observed)
        "read_file", "list_files", "grep_files", "git_log",
        "write_file", "edit_file", "restart_self",
    }

    def _build_tool_registry(self) -> tuple[list[dict], dict]:
        """Aggregate every tool currently available to the LLM.

        Returns (tools_schema, name->handler dict). Each module that adds
        tools (calendar, claude, memory, computer, self-edit) appends here.
        Empty result triggers a plain chat (no-tools) call upstream.
        """
        tools: list[dict] = []
        handlers: dict = {}
        if self.calendar is not None and self.calendar.is_authenticated:
            tools.extend(self._CALENDAR_TOOLS)
            handlers.update(self._calendar_tool_handlers())
        if self._claude_client is not None and self._claude_client.has_key:
            tools.append(self._ASK_CLAUDE_TOOL)
            handlers["ask_claude"] = self._tool_ask_claude
        if self.memory is not None:
            tools.extend(self._MEMORY_TOOLS)
            handlers["remember"]        = self._tool_remember
            handlers["recall"]          = self._tool_recall
            handlers["record_thought"]  = self._tool_record_thought
            handlers["record_question"] = self._tool_record_question
        # Computer tools only appear when the kill switch is ON. Removing
        # them entirely (rather than always-present-with-error-stub) means
        # the LLM doesn't know to attempt them when disabled.
        if self.computer is not None and self.computer.enabled:
            tools.extend(self._COMPUTER_TOOLS)
            handlers.update(self._computer_tool_handlers())
        # Self-edit: read tools always available so the LLM can analyze
        # its own code regardless of the write switch. Write/restart only
        # when the switch is on.
        if self.selfedit is not None:
            tools.extend(self._SELFEDIT_READ_TOOLS)
            handlers.update({
                k: v for k, v in self._selfedit_handlers().items()
                if k in {"read_file", "list_files", "grep_files", "git_log"}
            })
            if self.selfedit.enabled:
                tools.extend(self._SELFEDIT_WRITE_TOOLS)
                handlers.update({
                    k: v for k, v in self._selfedit_handlers().items()
                    if k in {"write_file", "edit_file", "restart_self"}
                })
        return tools, handlers

    async def _try_dnd(self, text: str, room: str) -> bool:
        """
        Handle "do not disturb" / "shut up for X" / "you can talk again" voice
        commands. Returns True if handled. Uses the InterruptibilityManager's
        DND state which gates all proactive speech.
        """
        minutes = parse_dnd(text)
        if minutes is None or self.interruptibility is None:
            return False
        if minutes == 0.0:
            self.interruptibility.clear_dnd()
            await self._broadcast({"type": "dnd", "active": False, "until": None})
            ack = await self._compose_in_character(
                prompt=(
                    "Cole just told you that you can talk again — DND is off. "
                    "Speak a single short in-character acknowledgement. "
                    "No preamble, no quotes."
                ),
                fallback="Back online.",
            )
            await self._speak(ack, room=room, priority="conversation")
            return True
        until = self.interruptibility.set_dnd(minutes)
        await self._broadcast({
            "type":   "dnd",
            "active": True,
            "until":  until.isoformat(),
            "minutes": minutes,
        })
        # Pretty-print duration for the LLM prompt
        if minutes >= 60:
            dur_str = f"{minutes / 60:.1f} hours"
        else:
            dur_str = f"{int(minutes)} minutes"
        ack = await self._compose_in_character(
            prompt=(
                f"Cole just put you in Do Not Disturb mode for {dur_str}. "
                f"Acknowledge in a single short in-character line — you'll "
                f"stay quiet until told otherwise. No preamble, no quotes."
            ),
            fallback=f"Going quiet for {dur_str}.",
        )
        await self._speak(ack, room=room, priority="conversation")
        return True

    async def _try_create_reminder(self, text: str, room: str) -> bool:
        """
        Returns True (and sends confirmation TTS) if `text` parsed as a reminder.
        False otherwise — caller should fall through to the LLM.
        """
        parsed = parse_reminder(text)
        if not parsed or self.reminders_store is None:
            return False
        task, due = parsed
        try:
            rid = await self.reminders_store.add(task, due)
        except Exception as e:
            logger.error(f"[Reminders] Failed to add: {e}")
            return False
        await self._broadcast({
            "type":         "reminder_added",
            "id":           rid,
            "message":      task,
            "trigger_time": due.isoformat(),
        })
        confirmation = await self._compose_in_character(
            prompt=(
                f"Cole just asked you to remind him to '{task}'. The reminder "
                f"will fire at {due.strftime('%A %B %d at %I:%M %p').replace(' 0', ' ')} "
                f"({due.isoformat()}). Speak a single short in-character "
                f"confirmation acknowledging that you've set the reminder. "
                f"No preamble, no quotes."
            ),
            fallback=f"Got it — I'll remind you to {task}.",
        )
        await self._speak(confirmation, room=room, priority="conversation")
        return True

    async def _on_text_chat(self, text: str, room: str = "office") -> None:
        # Dashboard chat = presence signal too — update active room.
        self._active_user_room = room
        """Handle typed messages sent from the dashboard chat input."""
        text = text.strip()
        if not text:
            return
        logger.info(f"[Chat] Dashboard input: {text!r}")
        try:
            await self._process_user_text(text, room)
        except Exception as e:
            logger.error(f"[Chat] Pipeline error: {e}")

    async def _on_voice_change(self, voice_name: str) -> None:
        """Switch TTS voice at runtime from the dashboard dev panel."""
        if not self.tts:
            return
        success = self.tts.set_voice(voice_name)
        if success:
            text = await self._compose_in_character(
                prompt=(
                    f"You just switched your text-to-speech voice to '{voice_name}'. "
                    f"Speak a single short in-character line acknowledging the "
                    f"new voice — note any persona vibes (Wheatley is bumbling "
                    f"British, GLaDOS is sardonic AI). No preamble, no quotes."
                ),
                fallback=f"Voice switched to {voice_name}.",
            )
            await self._speak(text, priority="ambient")

    # ── Background Loops ───────────────────────────────────────────────────

    async def _context_loop(self) -> None:
        """
        Continuously polls PC activity and audio, fuses signals into a state,
        and broadcasts state updates to the dashboard.
        """
        poll_interval = self.config["context"]["pc_poll_interval_seconds"]
        logger.info("[Context] Loop started")

        while True:
            try:
                # Gather signals
                signals = {}

                if self.pc_monitor:
                    pc_signal = await self.pc_monitor.get_signal_async()
                    if pc_signal:
                        signals["pc"] = pc_signal

                if (
                    self.audio_classifier
                    and not self._audio_io_active
                    and self.wake is not None
                ):
                    # Read the last N seconds of audio from wake_word's shared
                    # buffer instead of opening a second InputStream. The old
                    # suspend/wakeup approach killed wake responsiveness — wake
                    # was unavailable ~50% of the time and openWakeWord lost
                    # its prediction context every cycle. Now wake_word holds
                    # the mic continuously and YAMNet just snapshots its buffer.
                    window_s = float(
                        self.config["context"].get("audio_classify_window_seconds", 3)
                    )
                    waveform = self.wake.get_recent_audio(window_s)
                    if waveform is not None and len(waveform) > 0:
                        classifications = await asyncio.to_thread(
                            self.audio_classifier.classify, waveform
                        )
                        if classifications:
                            if self.appliance_tracker:
                                self.appliance_tracker.update(classifications)
                            signals["audio"] = {
                                "activity":   classifications[0]["label"],
                                "confidence": classifications[0]["score"],
                            }

                if self.posture and self.cameras:
                    rooms = self.cameras.get_available_rooms()
                    # Posture / sleep observations come from the camera pointed at Cole's
                    # desk (the USB webcam), not the ESP32 node camera which may be aimed
                    # elsewhere in the room.
                    if "office" in rooms:
                        frame = await self.cameras.capture_frame_async("office")
                        if frame is not None:
                            posture_result = await self.posture.analyze_async(frame)
                            # Posture is context-only — sitting/standing/lying
                            # isn't itself an activity, so we surface it via
                            # state.context for the LLM/dashboard but don't
                            # vote on activity. Sleep tracker consumes posture
                            # directly and emits the actual sleep activity.
                            signals["posture"] = {
                                "context": {"posture": posture_result},
                                "confidence": 0.7 if posture_result != "unknown" else 0.1,
                            }
                            # Update sleep tracker
                            sleep_tracker = self.sleep_tracker
                            if sleep_tracker is not None:
                                lights_on = (
                                    self.light_detector.last_state("office")
                                    if self.light_detector
                                    else None
                                )
                                sleep_tracker.update(
                                    posture=posture_result,
                                    lights_on=lights_on,
                                    room="office",
                                )
                                sleep_signal = sleep_tracker.get_sleep_signal()
                                if sleep_signal:
                                    signals["sleep"] = sleep_signal

                # Track activity transitions for predicted-duration / routine learning
                if (
                    self.activity_history is not None
                    and self._current_state is not None
                    and self._current_state.activity not in ("unknown", "")
                ):
                    try:
                        await self.activity_history.record_change(
                            self._current_state.activity,
                            self._current_state.location,
                        )
                    except Exception as e:
                        logger.debug(f"[ActivityHistory] record_change failed: {e}")

                # Fuse signals into final state
                if self.state_fusion:
                    # Surface what we collected so a stuck "unknown" state can
                    # be diagnosed — log every signal name + activity it voted
                    # for. Empty {} = no source produced a usable signal this
                    # cycle (the cause of activity stuck at unknown / 50%).
                    signal_summary = {
                        s: (sig.get("activity") if isinstance(sig, dict) else "?")
                        for s, sig in signals.items()
                    }
                    logger.debug(f"[Context] Signals this cycle: {signal_summary}")

                    # BUG FIX: StateFusion.fuse() is async def — must be awaited
                    new_state = await self.state_fusion.fuse(signals, room="office")
                    self._current_state = new_state

                    await self._broadcast({
                        "type": "state_update",
                        "activity": new_state.activity,
                        "location": new_state.location,
                        "interruptibility": self.interruptibility.get_score(new_state.activity)
                        if self.interruptibility else 0.5,
                        "confidence": new_state.confidence,
                        "signals": new_state.signals,
                        "context": new_state.context,
                    })

            except Exception as e:
                logger.error(f"[Context] Loop error: {e}")

            await asyncio.sleep(poll_interval)

    async def _vision_loop(self) -> None:
        """
        Periodically captures frames from all available cameras, runs detection
        pipeline, updates room baselines, and broadcasts vision events.
        """
        interval_seconds = self.config["context"]["vision_scan_interval_minutes"] * 60
        logger.info(f"[Vision] Loop started (every {interval_seconds}s)")

        while True:
            try:
                if not self.cameras:
                    await asyncio.sleep(interval_seconds)
                    continue

                for room_id in self.cameras.get_available_rooms():
                    try:
                        frame = await self.cameras.capture_frame_async(room_id)
                        if frame is None:
                            continue

                        # BUG FIX: LightDetector.analyze_async() returns Optional[bool],
                        # not a dict. Calling .get("lights_on") on a bool crashes at runtime.
                        # Use the bool value directly.
                        light_detector = self.light_detector
                        lights_on: Optional[bool] = None
                        if light_detector is not None:
                            lights_on = await light_detector.analyze_async(frame, room=room_id)

                        # Object detection
                        if not self.object_detector:
                            continue
                        detections = await self.object_detector.detect_async(frame)
                        object_summary = self.object_detector.summarize(detections)
                        person_present = self.object_detector.has_person(detections)

                        # Pet detection — broadcast a discrete event so the
                        # dashboard / curiosity engine can react. Tracking
                        # last-seen prevents firing once per scan when a pet
                        # is just camped out.
                        pets_now = self.object_detector.pets(detections)
                        pet_classes_now = sorted({p["class"] for p in pets_now})
                        prev_pets = self._last_pets_per_room.get(room_id, [])
                        if pet_classes_now != prev_pets:
                            self._last_pets_per_room[room_id] = pet_classes_now
                            if pet_classes_now:
                                await self._broadcast({
                                    "type": "pet_seen",
                                    "room": room_id,
                                    "pets": pet_classes_now,
                                })
                                logger.info(
                                    f"[Vision] '{room_id}' pets: {', '.join(pet_classes_now)}"
                                )

                        # Posture + richer per-person state. analyze_full_async
                        # returns a dict with posture, orientation, head_tilt,
                        # arms, lean, gesture, and an activity_hint. The vision
                        # LLM receives this as pre-grounding so its description
                        # benefits from local skeleton data.
                        posture_result: Any = None
                        if self.posture:
                            posture_result = await self.posture.analyze_full_async(frame)

                        # Face recognition — only bother if YOLO actually saw a person.
                        # Goes through IdentityManager so unknown faces feed the
                        # pending-cluster persona builder rather than being dropped.
                        recognized_name: Optional[str] = None
                        recognized_pid: Optional[int] = None
                        if (
                            person_present
                            and self.identity is not None
                        ):
                            try:
                                match = await self.identity.identify_face(frame)
                                if match is not None:
                                    recognized_name = match.name
                                    recognized_pid = match.person_id
                                    ambig = " (ambiguous)" if match.is_ambiguous else ""
                                    logger.info(
                                        f"[Identity/face] '{room_id}' → {match.name} "
                                        f"(sim={match.similarity:.2f}){ambig}"
                                    )
                                    await self._broadcast({
                                        "type":       "person_recognized",
                                        "room":       room_id,
                                        "name":       match.name,
                                        "similarity": match.similarity,
                                        "ambiguous":  match.is_ambiguous,
                                    })
                                    # Presence signal — Cole moved rooms.
                                    # Future proactive speech follows him here.
                                    if room_id != self._active_user_room:
                                        logger.info(
                                            f"[Presence] active room: "
                                            f"{self._active_user_room} → {room_id} "
                                            f"(face: {match.name})"
                                        )
                                        self._active_user_room = room_id
                            except Exception as e:
                                logger.debug(f"[Identity/face] identify failed: {e}")

                        # Drift verify (passive): if a recent wake matched
                        # someone via voice, opportunistically save the face
                        # captured here as a sample for that person.
                        if self.identity is not None:
                            for pid_to_verify, modality in list(
                                self.identity._verify_pending.items()
                            ):
                                if modality != "face":
                                    continue
                                # Only verify in the active user room; if the
                                # camera in this room saw a person, refresh.
                                if room_id != self._active_user_room or not person_present:
                                    continue
                                try:
                                    outcome = await self.identity.verify_face(
                                        pid_to_verify, frame
                                    )
                                    logger.debug(
                                        f"[Identity/drift] verify_face for pid={pid_to_verify}: {outcome}"
                                    )
                                    self.identity._verify_pending.pop(pid_to_verify, None)
                                    if outcome in ("pending_drift", "pending_conflict"):
                                        await self._broadcast({
                                            "type": "identity_pending_added",
                                            "modality": "face",
                                            "outcome": outcome,
                                        })
                                except Exception as e:
                                    logger.debug(f"[Identity/drift] verify_face failed: {e}")

                        # Drift verify reverse direction: if a face was just
                        # matched and we have a recent wake audio buffer
                        # cached, verify the voice matches the same person.
                        if (
                            recognized_pid is not None
                            and self.identity is not None
                            and self._last_wake_audio is not None
                            and self.identity._verify_pending.get(recognized_pid) == "voice"
                        ):
                            try:
                                outcome = await self.identity.verify_voice(
                                    recognized_pid, self._last_wake_audio
                                )
                                logger.debug(
                                    f"[Identity/drift] verify_voice for pid={recognized_pid}: {outcome}"
                                )
                                self.identity._verify_pending.pop(recognized_pid, None)
                                self._last_wake_audio = None
                                if outcome in ("pending_drift", "pending_conflict"):
                                    await self._broadcast({
                                        "type": "identity_pending_added",
                                        "modality": "voice",
                                        "outcome": outcome,
                                    })
                            except Exception as e:
                                logger.debug(f"[Identity/drift] verify_voice failed: {e}")

                        # Scene description — SceneAnalyzer self-gates on local
                        # frame-change detection, so we always call it and let
                        # it decide whether to invoke the vision LLM.
                        if not self.scene_analyzer:
                            continue
                        # Pass identity + posture into the prompt so the vision
                        # model says 'Cole' instead of describing him as 'a
                        # shirtless man', and so the room description benefits
                        # from local pose / activity hints.
                        scene_persons = [recognized_name] if recognized_name else None
                        scene_person_states: Optional[list[dict[str, Any]]] = None
                        if person_present and posture_result is not None:
                            ps_entry: dict[str, Any] = {}
                            if recognized_name:
                                ps_entry["name"] = recognized_name
                            if isinstance(posture_result, dict):
                                for k in ("posture", "orientation", "expression",
                                          "holding", "activity_hint", "gesture"):
                                    if posture_result.get(k):
                                        ps_entry[k] = posture_result[k]
                            elif posture_result:
                                ps_entry["posture"] = str(posture_result)
                            if ps_entry:
                                scene_person_states = [ps_entry]
                        last_desc = await self.scene_analyzer.describe_async(
                            frame, room=room_id, objects=detections,
                            persons=scene_persons,
                            person_states=scene_person_states,
                        )

                        # BUG FIX: update_if_due() doesn't exist on RoomBaselines.
                        # Actual API: needs_update(room) → bool, then update(room, desc).
                        if self.room_baselines and last_desc:
                            if await self.room_baselines.needs_update(room_id):
                                await self.room_baselines.update(room_id, last_desc)

                            # Anomaly scoring — only if we have a baseline + cooldown allows.
                            # Compares current scene to baseline via LLM, fires room_anomaly
                            # event when score exceeds threshold.
                            if (
                                self.anomaly_detector is not None
                                and self.anomaly_detector.should_check(room_id)
                            ):
                                baseline = await self.room_baselines.get(room_id)
                                if baseline and baseline != last_desc:
                                    result = await self.anomaly_detector.score(
                                        room_id, baseline, last_desc
                                    )
                                    if result is not None:
                                        score, reason = result
                                        logger.debug(
                                            f"[Anomaly] '{room_id}' score={score:.1f} reason={reason!r}"
                                        )
                                        if score >= self.anomaly_detector.threshold:
                                            await self._broadcast({
                                                "type":   "room_anomaly",
                                                "room":   room_id,
                                                "score":  score,
                                                "reason": reason,
                                            })
                                            logger.info(
                                                f"[Anomaly] '{room_id}' {score:.1f}/10: {reason}"
                                            )

                            # Mess scoring — independent absolute tidiness check
                            # against the current scene description (no baseline
                            # comparison). Heavier cooldown so we don't burn LLM
                            # calls on a steady-state room.
                            if (
                                self.mess_detector is not None
                                and self.mess_detector.should_check(room_id)
                                and last_desc
                            ):
                                mess_result = await self.mess_detector.score(room_id, last_desc)
                                if mess_result is not None:
                                    mess_score, mess_reason = mess_result
                                    logger.debug(
                                        f"[Mess] '{room_id}' tidiness={mess_score:.1f} reason={mess_reason!r}"
                                    )
                                    if mess_score >= self.mess_detector.threshold:
                                        await self._broadcast({
                                            "type":   "room_messy",
                                            "room":   room_id,
                                            "score":  mess_score,
                                            "reason": mess_reason,
                                        })
                                        logger.info(
                                            f"[Mess] '{room_id}' {mess_score:.1f}/10: {mess_reason}"
                                        )

                        # Broadcast vision state — use lights_on bool directly
                        await self._broadcast({
                            "type": "vision",
                            "room": room_id,
                            "lights_on": lights_on,
                            "person_present": person_present,
                            "person_name":    recognized_name,
                            "objects": object_summary,
                            "description": last_desc,
                        })

                        # Pass vision signal to state fusion — state_fusion still
                        # consumes posture as a string label, so unwrap from the
                        # rich dict if that's what we got.
                        posture_label: Optional[str] = None
                        if isinstance(posture_result, dict):
                            posture_label = posture_result.get("posture")
                        elif isinstance(posture_result, str):
                            posture_label = posture_result
                        if self.state_fusion:
                            self.state_fusion.inject_vision(room_id, {
                                "lights_on": lights_on,
                                "person_present": person_present,
                                "posture": posture_label,
                            })

                    except Exception as room_err:
                        logger.warning(f"[Vision] Room {room_id} error: {room_err}")

            except Exception as e:
                logger.error(f"[Vision] Loop error: {e}")

            await asyncio.sleep(interval_seconds)

    async def _curiosity_loop(self) -> None:
        """
        Periodically checks the curiosity engine for proactive speech opportunities.
        If a topic fires, Jarvis speaks unprompted if interruptibility allows.
        """
        check_interval_seconds = 60  # Check every minute
        logger.info("[Curiosity] Loop started")

        while True:
            await asyncio.sleep(check_interval_seconds)
            try:
                if not self.curiosity or not self._current_state:
                    continue

                utterance = await self.curiosity.check_async(self._current_state)
                if not utterance:
                    continue

                # Only speak if interruptibility allows ambient-priority speech
                if self.interruptibility and not self.interruptibility.can_interrupt(
                    self._current_state, priority="ambient"
                ):
                    logger.debug("[Curiosity] Blocked by interruptibility gate")
                    continue

                await self._speak(utterance, priority="ambient")

            except Exception as e:
                logger.error(f"[Curiosity] Loop error: {e}")

    async def _calendar_alert_loop(self) -> None:
        """
        Poll the calendar every minute and proactively announce meetings
        starting within the next N minutes. Each event is alerted at most once.
        Skips entirely if calendar isn't authenticated.
        """
        cal_cfg = self.config.get("calendar", {}) if isinstance(self.config.get("calendar"), dict) else {}
        lead_minutes = int(cal_cfg.get("alert_lead_minutes", 10))
        poll_seconds = int(cal_cfg.get("alert_poll_seconds", 60))
        logger.info(
            f"[Calendar] Alert loop started (lead={lead_minutes}m, poll={poll_seconds}s)"
        )
        while True:
            try:
                await asyncio.sleep(poll_seconds)
                if self.calendar is None or not self.calendar.is_authenticated:
                    continue
                events = await self.calendar.upcoming_events(hours=1)
                from datetime import datetime as _dt, timedelta as _td
                now = _dt.now().astimezone()
                threshold = now + _td(minutes=lead_minutes)
                for e in events:
                    eid = e.get("id")
                    start_str = e.get("start") or ""
                    if not eid or not start_str or eid in self._calendar_alerted:
                        continue
                    try:
                        start_dt = _dt.fromisoformat(start_str.replace("Z", "+00:00"))
                        if start_dt.tzinfo is None:
                            start_dt = start_dt.astimezone()
                    except ValueError:
                        continue
                    if not (now < start_dt <= threshold):
                        continue
                    # Within the lead window — announce
                    self._calendar_alerted.add(eid)
                    title = e.get("title") or "an event"
                    minutes_away = max(1, int((start_dt - now).total_seconds() // 60))
                    line = await self._compose_in_character(
                        prompt=(
                            f"Cole has a calendar event starting in about "
                            f"{minutes_away} minute{'s' if minutes_away != 1 else ''}: "
                            f"\"{title}\". Give him a single short heads-up in your "
                            f"usual voice. No preamble, no quotes."
                        ),
                        fallback=f"Heads up — {title} in {minutes_away} minutes.",
                    )
                    await self._speak(line, priority="notification")
                    logger.info(f"[Calendar] Alerted on '{title}' (id={eid})")

                # Garbage-collect alerted IDs that have already started so the
                # set doesn't grow unbounded.
                self._calendar_alerted = {
                    eid for eid in self._calendar_alerted
                    if eid in {e.get("id") for e in events}
                }
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Calendar] Alert loop error: {e}")

    async def _eod_summary_loop(self) -> None:
        """
        Once per day at the configured time, ask the LLM to summarise the day's
        events and persist the result to the events table as 'daily_summary'.
        Optionally speak it (off by default — late-night TTS startle is real).
        """
        mem_cfg = self.config.get("memory", {}) if isinstance(self.config.get("memory"), dict) else {}
        summary_hour = int(mem_cfg.get("eod_summary_hour", 23))
        summary_minute = int(mem_cfg.get("eod_summary_minute", 0))
        speak_summary = bool(mem_cfg.get("eod_summary_speak", False))

        last_summary_date = None
        logger.info(
            f"[Summary] EOD loop started (fires daily at {summary_hour:02d}:{summary_minute:02d})"
        )
        while True:
            try:
                await asyncio.sleep(60)
                now = datetime.now()
                if (now.hour, now.minute) >= (summary_hour, summary_minute):
                    today = now.date()
                    if last_summary_date != today:
                        last_summary_date = today
                        await self._generate_daily_summary(today, speak=speak_summary)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Summary] EOD loop error: {e}")

    async def _generate_daily_summary(self, target_date, speak: bool = False) -> None:
        """Generate and persist an LLM-written summary for one day."""
        from datetime import datetime as _dt, timedelta as _td
        if self.db is None or self.llm is None:
            return
        start = _dt.combine(target_date, _dt.min.time())
        end = start + _td(days=1)
        try:
            rows = await self.db.fetchall(
                "SELECT timestamp, type, room, content FROM events "
                "WHERE timestamp >= ? AND timestamp < ? "
                "  AND type != 'daily_summary' "
                "ORDER BY timestamp ASC",
                (start.isoformat(), end.isoformat()),
            )
        except Exception as e:
            logger.warning(f"[Summary] event fetch failed: {e}")
            return
        if not rows:
            logger.info(f"[Summary] No events for {target_date} — skipping")
            return

        # Cap the digest size so we don't blow the prompt budget on a busy day.
        # 200 events with ~120 chars each is ~24KB which fits comfortably.
        lines = []
        for r in rows[:200]:
            ts = (r["timestamp"] or "")[:19]
            content = (r["content"] or "").strip()[:140]
            lines.append(f"[{ts}] {r['room']}/{r['type']}: {content}")
        digest = "\n".join(lines)

        summary = await self._compose_in_character(
            prompt=(
                f"Below is a chronological log of everything Jarvis recorded on "
                f"{target_date.isoformat()} ({len(rows)} events). Write a 2-4 "
                f"sentence end-of-day summary in your usual voice — what Cole "
                f"actually got up to, anything notable. Dry but warm. No "
                f"preamble, no quotes, just the spoken summary.\n\n{digest}"
            ),
            fallback=(
                f"Day summary for {target_date.isoformat()}: {len(rows)} events "
                f"logged across {len({r['room'] for r in rows})} rooms."
            ),
        )
        if self.event_log:
            try:
                await self.event_log.log_event(
                    room="office",
                    event_type="daily_summary",
                    content=summary,
                )
            except Exception as e:
                logger.warning(f"[Summary] persist failed: {e}")
        logger.info(f"[Summary] Day {target_date}: {summary[:120]}...")
        await self._broadcast({
            "type":    "daily_summary",
            "date":    target_date.isoformat(),
            "summary": summary,
        })
        if speak:
            # EOD summary opens the follow-up mic by default — Cole often
            # wants to riff on the recap (ask about a specific entry, add a
            # note for tomorrow). Default behavior of _speak handles this.
            await self._speak(summary, priority="ambient")

    async def _health_broadcast_loop(self) -> None:
        """
        Periodically checks Ollama and MQTT health, broadcasts system_health events.
        """
        interval_seconds = 30
        logger.info("[Health] Broadcast loop started")

        while True:
            try:
                health = {}

                # Check Ollama
                try:
                    async with httpx.AsyncClient(timeout=5) as client:
                        r = await client.get(
                            f"{self.config['ollama']['base_url']}/api/tags"
                        )
                    health["ollama"] = {
                        "online": r.status_code == 200,
                        "model": self.config["ollama"]["model"],
                    }
                except Exception:
                    health["ollama"] = {"online": False, "model": ""}

                # Check MQTT
                mqtt_online = self.mqtt is not None and self.mqtt._connected
                health["mqtt"] = {
                    "online": mqtt_online,
                    "broker": f"{self.config['mqtt']['broker']}:{self.config['mqtt']['port']}",
                }

                # BUG FIX: WhisperSTT uses self.model (public), not self._model (private)
                # The _model attribute doesn't exist — this would always return False.
                health["whisper"] = {
                    "loaded": self.stt is not None and self.stt.model is not None,
                    "model": self.config["voice"]["whisper"]["model_size"],
                }

                await self._broadcast({
                    "type": "system_health",
                    "health": health,
                })

            except Exception as e:
                logger.error(f"[Health] Broadcast error: {e}")

            await asyncio.sleep(interval_seconds)

    # ── Event Handlers ─────────────────────────────────────────────────────

    async def _on_appliance_changed(self, event: dict) -> None:
        """
        When an appliance finishes a cycle, announce it via TTS and broadcast
        to the dashboard.
        """
        appliance = event.get("appliance")
        new_status = event.get("status")
        runtime = event.get("runtime_minutes")
        if not isinstance(appliance, str) or not appliance:
            logger.warning("[Appliance] Missing appliance name in state-change event")
            return

        logger.info(f"[Appliance] {appliance} → {new_status}")

        await self._broadcast({
            "type": "appliance",
            "appliance": appliance,
            "status": new_status,
            "runtime_minutes": runtime,
        })

        if new_status == "done":
            runtime_str = (
                f" (ran about {int(runtime)} minutes)" if isinstance(runtime, (int, float)) else ""
            )
            text = await self._compose_in_character(
                prompt=(
                    f"The {appliance} just finished its cycle{runtime_str}. "
                    f"Tell Cole in a single short in-character line. "
                    f"No preamble, no quotes."
                ),
                fallback=f"The {appliance} just finished.",
            )

            urgency_map = self.config["appliances"]
            urgency = urgency_map.get(f"{appliance}_done_urgency", 0.5)
            priority = "urgent" if urgency >= 0.7 else "notification"

            await self._speak(text, priority=priority)

    async def _on_node_status(self, event: dict) -> None:
        """Handle ESP32 node coming online or going offline."""
        room = event.get("room")
        data = event.get("data")
        if isinstance(data, str):
            online = data.strip().lower() == "online"
            ip = event.get("ip")
        elif isinstance(data, dict):
            status = str(data.get("status", "")).strip().lower()
            online = bool(data.get("online", status == "online"))
            ip = data.get("ip", event.get("ip"))
        else:
            online = bool(event.get("online", False))
            ip = event.get("ip")

        logger.info(f"[Node] {room} → {'online' if online else 'offline'}")

        if self.nodes:
            # NodeManager handles its own state; just broadcast to dashboard
            pass

        await self._broadcast({
            "type": "node_status",
            "room": room,
            "online": online,
            "ip": ip,
        })

    async def _speak_via_node(self, text: str, room: str) -> bool:
        """
        Synthesize text and send the audio bytes to the room's ESP32 node over
        MQTT for playback on the node's speaker. Returns True if the audio was
        published (the node is responsible for playing it). Returns False on
        any failure so the caller falls back to local playback.

        Note: the firmware-side handling of the audio_out topic is still pending
        as of 2026-05-03 — the office node speaker has hardware static issues
        being tracked separately. When it's ready, no Python-side change will
        be needed; the node will just start producing sound.
        """
        if self.tts is None or self.nodes is None:
            return False
        try:
            # Synthesize as float32, convert to 16-bit LE PCM bytes that the
            # ESP I2S speaker can stream. Sentence-level streaming doesn't
            # apply over MQTT (one publish per sentence would be inefficient);
            # we send the whole utterance.
            audio = await self.tts.synthesize_async(text)
        except Exception as e:
            logger.warning(f"[TTS] Node-route synthesis failed: {e}")
            return False
        try:
            import numpy as _np
            pcm_int16 = (_np.clip(audio, -1.0, 1.0) * 32767.0).astype(_np.int16)
            payload = pcm_int16.tobytes()
        except Exception as e:
            logger.warning(f"[TTS] PCM conversion failed: {e}")
            return False
        try:
            sent = await self.nodes.send_audio(room, payload)
        except Exception as e:
            logger.warning(f"[TTS] send_audio('{room}') failed: {e}")
            return False
        if sent:
            logger.info(
                f"[TTS] Routed {len(payload)} bytes to '{room}' node speaker"
            )
        return sent

    def _speaker_sink_for(self, room: str) -> str:
        """
        Return the configured speaker_sink ("local" | "node") for a room.
        Defaults to "local" when no entry or no explicit sink is configured.
        """
        for room_cfg in self.config.get("rooms", []):
            if room_cfg.get("id") == room:
                sink = str(room_cfg.get("speaker_sink", "local")).lower()
                return sink if sink in ("local", "node") else "local"
        return "local"

    # ── Continuous-conversation follow-up listener ─────────────────────────

    async def _listen_followup(self, room: str) -> None:
        """
        Open a short listen window after a conversational reply so the user
        can keep talking without re-saying the wake word. If they say
        anything substantive, route it through the normal STT → identity →
        LLM pipeline as a continuation of the same conversation. If they
        stay quiet for the configured window, exit silently.

        Also handles the live-enrollment second turn: when self._pending_live_enroll
        is set, the captured reply is treated as a name claim instead of a
        normal user turn — Jarvis extracts the name, saves the original
        unknown-speaker audio as their first voice sample, snaps a face
        sample if a person is in frame, and welcomes them.
        """
        recording_cfg = self.config.get("voice", {}).get("recording", {})
        listen_seconds = float(recording_cfg.get("follow_up_listen_seconds", 6.0))
        if listen_seconds <= 0:
            return

        logger.info(f"[Followup] opening listen window ({listen_seconds}s) for room '{room}'")

        # Brief grace so reverb from our own TTS dies before we open the mic.
        await asyncio.sleep(0.4)

        # ── CAPTURE PHASE — mic-exclusive, lock held briefly ────────────────
        # Same lock-scope discipline as _on_wake_detected: hold the lock only
        # for the suspend → record → resume window so a real wake event during
        # processing can fire its own beep + capture in parallel.
        from modules.voice.audio_utils import (
            SAMPLE_RATE,
            record_until_silence,
            db_from_rms,
        )
        audio_data = None
        async with self._wake_lock:
            self._followup_depth += 1
            was_audio_active = self._audio_io_active
            self._audio_io_active = True
            if self.wake:
                self.wake.suspend()
                # Same dance the wake handler does: give wake's InputStream
                # time to actually close before we open ours. Without this
                # WASAPI on Windows can deny the second stream silently or
                # return zero-filled buffers for the first ~second.
                await asyncio.sleep(0.3)
            try:
                record_device = (
                    self.wake.device if self.wake else recording_cfg.get("device")
                )
                pre_floor = (
                    self.wake.get_noise_floor_db(
                        fallback_db=recording_cfg.get("silence_threshold_db", -45.0)
                    )
                    if self.wake
                    else recording_cfg.get("silence_threshold_db", -45.0)
                )
                logger.debug(
                    f"[Followup] record_device={record_device}, pre-floor={pre_floor:.1f} dBFS"
                )
                audio_data = await asyncio.to_thread(
                    record_until_silence,
                    silence_threshold_db=pre_floor,
                    silence_duration_ms=recording_cfg.get("silence_duration_ms", 600),
                    max_duration_seconds=recording_cfg.get("max_duration_seconds", 60.0),
                    speech_start_timeout_seconds=listen_seconds,
                    device=record_device,
                    mode="silence",
                    adaptive_noise_floor=False,
                )
            finally:
                if self.wake:
                    self.wake.wakeup()
                self._audio_io_active = was_audio_active
        # ── LOCK RELEASED ───────────────────────────────────────────────────

        try:
            if audio_data is None or len(audio_data) == 0:
                logger.info("[Followup] no audio captured — window closed silently")
                return

            duration_s = len(audio_data) / SAMPLE_RATE
            import numpy as _np
            rms = float(_np.sqrt(_np.mean(audio_data ** 2))) if len(audio_data) else 0.0
            logger.info(
                f"[Followup] captured {duration_s:.2f}s "
                f"(rms={db_from_rms(rms):.1f} dBFS)"
            )

            # Heuristic: ignore captures that are clearly just our own
            # TTS tail or pure ambient. We rely on speech_start_timeout
            # above to bail when there's no real speech, so anything that
            # gets here had energy in it.
            if duration_s < 0.5:
                logger.info(f"[Followup] Capture too short ({duration_s:.2f}s) — discarding")
                return

            stt = self.stt
            if stt is None:
                return
            transcript = await asyncio.to_thread(stt.transcribe, audio_data)
            if not transcript or not transcript.strip():
                logger.info("[Followup] Empty transcript from Whisper — closing window")
                return

            logger.info(f"[Followup] Captured turn: {transcript!r}")

            # Live-enroll branch: this reply is the user telling us their name.
            if self._pending_live_enroll is not None:
                enroll_state = self._pending_live_enroll
                self._pending_live_enroll = None
                await self._complete_live_enroll(
                    reply_transcript=transcript,
                    original_audio=enroll_state["audio"],
                    room=enroll_state["room"],
                )
                return

            # Normal continuation — same identification + LLM path as a
            # wake-driven turn, just no chime / no wake word required.
            speaker_name: Optional[str] = None
            if self.identity is not None:
                try:
                    match = await self.identity.identify_voice(audio_data)
                    if match is not None:
                        speaker_name = match.name
                        self._last_wake_audio = audio_data
                except Exception as e:
                    logger.debug(f"[Followup] identify failed: {e}")

            await self._process_user_text(transcript, room, speaker=speaker_name)

        except Exception as e:
            logger.warning(f"[Followup] Pipeline error: {e}")
        finally:
            self._followup_depth -= 1

    async def _complete_live_enroll(
        self,
        reply_transcript: str,
        original_audio: Any,
        room: str,
    ) -> None:
        """Second turn of live conversational enrollment.

        The user just told us their name in `reply_transcript`. Extract a
        clean first name via the LLM, persist a voice sample (using the
        ORIGINAL audio that triggered enrollment, not the name-reply audio
        which is too short and contains the name only), grab a face sample
        from the active room camera if a person is in frame, and welcome
        them by name.
        """
        if self.identity is None:
            return

        # LLM extraction: turn "uh, my name's, like, Jordan I guess?" into 'Jordan'.
        # Fallback heuristic: take the longest capitalized-looking token.
        name = await self._extract_name_from_reply(reply_transcript)
        if not name:
            confirmation = await self._compose_in_character(
                prompt=(
                    f"You couldn't make out a name from the reply: {reply_transcript!r}. "
                    "Apologize briefly in character and ask them to try again. "
                    "One short line, no preamble."
                ),
                fallback="Sorry, I didn't catch a name — try again?",
            )
            # Re-arm enrollment so the next reply is treated as the name retry.
            self._pending_live_enroll = {
                "audio": original_audio,
                "room": room,
                "first_transcript": "",
            }
            await self._speak(confirmation, room=room, priority="conversation")
            return

        # Persist voice sample from ORIGINAL audio (longer, has real speech)
        sample_id = await self.identity.enroll_voice(
            name, original_audio, prompt_id="live_question"
        )

        # Snap a face sample if a person is currently in frame in the active
        # room. Best-effort — failures are silent.
        face_saved = False
        if self.cameras is not None:
            try:
                frame = await self.cameras.capture_frame_async(room)
                if frame is not None:
                    fid = await self.identity.enroll_face(name, frame, pose="candid")
                    face_saved = fid is not None
            except Exception as e:
                logger.debug(f"[LiveEnroll] face snap failed: {e}")

        await self._broadcast({
            "type": "identity_live_enrolled",
            "name": name,
            "voice_ok": sample_id is not None,
            "face_ok":  face_saved,
        })

        # Greet in character. Mention the face capture only if it happened so
        # we don't promise something we didn't deliver.
        face_note = (
            " I also got a face capture so I can recognize you on camera."
            if face_saved
            else ""
        )
        greeting = await self._compose_in_character(
            prompt=(
                f"You just learned a new person's name is '{name}'. Greet them "
                f"warmly in one short in-character line.{face_note} No preamble, "
                "no quotes."
            ),
            fallback=f"Nice to meet you, {name}.{face_note}",
        )
        await self._speak(greeting, room=room, priority="conversation")

    async def _extract_name_from_reply(self, reply: str) -> Optional[str]:
        """LLM-extract a first-name from the reply. None if no plausible name."""
        if not self.llm:
            # Heuristic fallback: take the first capitalized token of length >= 2
            for tok in reply.split():
                cleaned = "".join(c for c in tok if c.isalpha())
                if len(cleaned) >= 2 and cleaned[0].isupper():
                    return cleaned
            return None
        try:
            prompt = (
                "Extract the person's first name from this self-introduction. "
                "Respond with ONLY the name, no punctuation, no extra words. "
                "If there is no clear name, respond with the literal word: NONE.\n\n"
                f"Self-introduction: {reply!r}"
            )
            resp = await self.llm.chat([{"role": "user", "content": prompt}])
            extracted = (resp or "").strip().split()
            if not extracted:
                return None
            candidate = extracted[0].strip(".,;:!?'\"")
            if not candidate or candidate.upper() == "NONE":
                return None
            if not candidate[0].isalpha() or len(candidate) > 32:
                return None
            return candidate
        except Exception as e:
            logger.debug(f"[LiveEnroll] name extraction failed: {e}")
            return None

    # ── TTS Helper ─────────────────────────────────────────────────────────

    async def _speak(
        self,
        text: str,
        room: Optional[str] = None,
        priority: str = "ambient",
        expects_response: Optional[bool] = None,
    ) -> None:
        """
        Full speak pipeline: TTS → audio playback → log → broadcast → optional
        follow-up listen window.

        expects_response controls the post-reply mic window:
          - None (default): infer from priority. 'oneway' suppresses follow-up;
            everything else (conversation, reminder, curiosity, calendar)
            opens a listen window. If Jarvis pings Cole proactively, Cole
            should be able to reply without saying the wake word — that's
            the default.
          - True: always open a listen window after speaking.
          - False: never open one (e.g. EOD summary that's purely informational).

        Room resolution:
          - If `room` is None, target Cole's currently-active room (set by
            wake events / face recognition / dashboard chat). Lets proactive
            speech (curiosity, reminders, calendar alerts, EOD summary) follow
            Cole around the house instead of always blasting the office PC.
          - If explicit `room` is passed (e.g. wake response), it overrides.

        Routing:
          - Per-room speaker_sink config drives where audio plays:
              "local"  → PC sound device (the machine running this script)
              "node"   → MQTT to the room's ESP node speaker (falls back to
                         local if node offline or firmware can't play)
          - Default for any room without a sink config is "local".
        """
        try:
            if room is None:
                room = self._active_user_room
            logger.info(f"[TTS] [{priority}] (→{room}) {text!r}")

            if not self.tts:
                logger.warning("[TTS] TTS module not initialized — skipping playback")
                return

            routed_to_node = False
            sink = self._speaker_sink_for(room)
            if sink == "node" and self.nodes is not None and self.nodes.is_online(room):
                routed_to_node = await self._speak_via_node(text, room)

            if not routed_to_node:
                # Local playback. Streaming for multi-sentence; quick path for
                # one-liners. _audio_io_active blocks the audio classifier from
                # reading wake_word's buffer during our own playback. Audio focus
                # ducks other apps' volume so Jarvis isn't drowned out by music.
                was_audio_io_active = self._audio_io_active
                self._audio_io_active = True
                if self.audio_focus is not None and self.audio_focus.available:
                    await self.audio_focus.duck_async()
                try:
                    await self.tts.speak_async(text)
                finally:
                    self._audio_io_active = was_audio_io_active
                    if self.audio_focus is not None and self.audio_focus.available:
                        await self.audio_focus.restore_async()

            # Log to DB
            if self.event_log:
                # BUG FIX: method is log_event() not log()
                await self.event_log.log_event(
                    room=room,
                    event_type="jarvis_speech",
                    content=text,
                )

            # Broadcast to dashboard
            await self._broadcast({
                "type": "speech",
                "text": text,
                "room": room,
                "priority": priority,
            })

            # Open a listen window after speaking unless this was an explicit
            # one-way info dump. Default behavior: ANY speech (conversation
            # reply, reminder, curiosity ping, calendar alert) lets Cole
            # respond hands-free. Only oneway/EOD-summary opt out.
            if expects_response is None:
                wants_followup = priority not in ("oneway",)
            else:
                wants_followup = bool(expects_response)
            if wants_followup:
                # Fire-and-forget — _listen_followup acquires the wake lock
                # internally so it serializes with normal wake captures and
                # any other follow-up that's already running.
                asyncio.create_task(self._listen_followup(room))

        except Exception as e:
            logger.error(f"[TTS] Speak error: {e}")

    # ── Dashboard Broadcast ────────────────────────────────────────────────

    async def _broadcast(self, event: dict) -> None:
        """Send event to dashboard if enabled. Never blocks or raises."""
        if self.dashboard:
            try:
                await self.dashboard.broadcast(event)
            except Exception as e:
                logger.debug(f"[Dashboard] Broadcast error: {e}")

    async def _self_thought_loop(self) -> None:
        """
        Periodic 'time to think' loop.

        When the system is genuinely idle (state has high interruptibility
        and Cole's been quiet), ask the LLM to reflect on recent observations
        + relevant memories, and persist the result as a thought (kind='thought')
        or a question (kind='question'). Questions can later be surfaced to
        Cole or escalated to Claude via the ask_claude tool.

        Conservative cadence: 25-minute base interval, only fires when
        interruptibility >= 0.7, and only ~50% of those firings actually
        produce a thought (so we don't pollute memory with low-value noise).
        """
        import random
        await asyncio.sleep(120)  # don't fire during boot stabilization
        while True:
            try:
                await asyncio.sleep(25 * 60)
                if self.memory is None or self.llm is None:
                    continue
                if self._current_state is not None:
                    interrupt = float(getattr(self._current_state, "interruptibility", 0.5))
                    if interrupt < 0.7:
                        continue
                if random.random() > 0.5:
                    continue
                # Build a small reflection prompt — feed recent memories + state
                ctx_lines = []
                if self._current_state:
                    ctx_lines.append(f"Current activity: {self._current_state.activity}")
                try:
                    recents = await self.memory.list_recent(limit=15)
                    for r in recents[:8]:
                        ctx_lines.append(f"- ({r['kind']}) {r['content']}")
                except Exception:
                    pass
                prompt = (
                    "You're Jarvis with a quiet moment. Reflect on what you've "
                    "noticed recently — patterns in Cole's day, things you're "
                    "curious about, anything you'd like to ask him next time he's "
                    "free.\n\n"
                    "Reply with ONE JSON object: "
                    "{\"kind\": \"thought\"|\"question\", \"content\": \"...\", "
                    "\"importance\": 0.0-1.0, \"subject\": \"...\" or null}\n\n"
                    "If nothing's worth saving, reply with: {}\n\n"
                    + "\n".join(ctx_lines)
                )
                try:
                    raw = await self.llm.chat([{"role": "user", "content": prompt}])
                except Exception as e:
                    logger.debug(f"[SelfThought] LLM call failed: {e}")
                    continue
                import json as _json, re as _re
                match = _re.search(r"\{.*\}", raw or "", _re.DOTALL)
                if not match:
                    continue
                try:
                    obj = _json.loads(match.group(0))
                except Exception:
                    continue
                if not isinstance(obj, dict) or not obj.get("content"):
                    continue
                kind = (obj.get("kind") or "thought").lower()
                if kind == "question":
                    await self.memory.record_question(
                        obj["content"],
                        subject=obj.get("subject"),
                        importance=float(obj.get("importance", 0.6)),
                    )
                    logger.info(f"[SelfThought] +question: {obj['content'][:80]}")
                else:
                    await self.memory.record_thought(
                        obj["content"],
                        subject=obj.get("subject"),
                        importance=float(obj.get("importance", 0.4)),
                    )
                    logger.info(f"[SelfThought] +thought: {obj['content'][:80]}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"[SelfThought] loop error: {e}")

    async def _shutdown(self) -> None:
        """Release long-lived resources cleanly during shutdown."""
        if self.wake:
            self.wake.stop()
        if self.wake_sources is not None:
            try:
                await self.wake_sources.stop()
            except Exception as e:
                logger.debug(f"[Shutdown] wake_sources stop failed: {e}")

        if self.mqtt:
            try:
                await self.mqtt.disconnect()
            except Exception as e:
                logger.debug(f"[Shutdown] MQTT disconnect failed: {e}")

        if self.cameras:
            try:
                await self.cameras.close()
            except Exception as e:
                logger.debug(f"[Shutdown] Camera close failed: {e}")

        if self.db:
            try:
                await self.db.close()
            except Exception as e:
                logger.debug(f"[Shutdown] DB close failed: {e}")

    # ── Main Entry Point ───────────────────────────────────────────────────

    async def run(self) -> None:
        """
        Full async entry point.
        Initializes all modules, registers handlers, then runs all loops concurrently.
        """
        logger.info("[Orchestrator] Starting JARVIS...")

        # Initialize all subsystems in dependency order
        await self._init_database()
        await self._init_voice()
        await self._init_brain()
        await self._init_context()

        try:
            await self._init_vision()
        except Exception as e:
            logger.warning(f"[Init] Vision init failed (continuing without): {e}")

        # Identity v2 manager — wraps speaker_id + face_recognizer with the
        # cross-modal Person abstraction. Migrates legacy speakers/faces rows
        # on first boot, no-op after that. Hooked up to the notification
        # manager so drift/cluster events show on the dashboard bell.
        if self.db is not None:
            self.identity = IdentityManager(
                db=self.db,
                speaker_identifier=self.speaker_id,
                face_recognizer=self.face_recognizer,
                config=self.config,
                notifier=self.notifications,
                broadcast=self._broadcast,
            )
            try:
                await self.identity.init()
            except Exception as e:
                logger.warning(f"[Init] Identity manager init failed: {e}")
                self.identity = None

        try:
            await self._init_network()
        except Exception as e:
            logger.warning(f"[Init] Network init failed (continuing without): {e}")

        try:
            await self._init_calendar()
        except Exception as e:
            logger.warning(f"[Init] Calendar init failed (continuing without): {e}")

        try:
            await self._init_webhooks()
        except Exception as e:
            logger.warning(f"[Init] Webhooks init failed (continuing without): {e}")

        # Restore conversation sessions from the event log so Jarvis remembers
        # what was discussed before the last restart. Idempotent.
        if self.db and self.sessions:
            for room_cfg in self.config.get("rooms", []):
                room_id = room_cfg.get("id")
                if room_id:
                    try:
                        await self.sessions.restore_from_log(room_id, self.db)
                    except Exception as e:
                        logger.warning(f"[Init] Session restore for '{room_id}' failed: {e}")

        # Wire dashboard to config + handlers
        if self.dashboard:
            room_ids = [r["id"] for r in self.config.get("rooms", [])]
            self.dashboard.set_room_ids(room_ids)
            self.dashboard.register_chat_handler(self._on_text_chat)
            if self.tts:
                self.dashboard.register_voice_handler(
                    self._on_voice_change,
                    voices=self.tts.available_voices(),
                    active=self.tts._active_voice,
                )
            if self.cameras:
                self.dashboard.register_camera_manager(self.cameras)
            if self.reminders_store:
                self.dashboard.register_reminders_store(self.reminders_store)
            if self.calendar:
                self.dashboard.register_calendar(self.calendar)
            if self.interruptibility:
                self.dashboard.register_interruptibility(self.interruptibility)
            self.dashboard.register_orchestrator(self)
            if self.speaker_id:
                self.dashboard.register_speaker_id(self.speaker_id)
            if self.face_recognizer:
                self.dashboard.register_face_recognizer(self.face_recognizer)
            if self.identity is not None:
                # Idempotent — register only if dashboard supports the new manager
                if hasattr(self.dashboard, "register_identity"):
                    self.dashboard.register_identity(self.identity)
            if self.notifications is not None and hasattr(self.dashboard, "register_notifications"):
                self.dashboard.register_notifications(self.notifications)
            if self.model_registry is not None and hasattr(self.dashboard, "register_model_registry"):
                self.dashboard.register_model_registry(self.model_registry)
            if self.memory is not None and hasattr(self.dashboard, "register_memory"):
                self.dashboard.register_memory(self.memory)
            if self.computer is not None and hasattr(self.dashboard, "register_computer"):
                self.dashboard.register_computer(self.computer)
            if self.selfedit is not None and hasattr(self.dashboard, "register_selfedit"):
                self.dashboard.register_selfedit(self.selfedit)
            if self.webhooks:
                self.dashboard.register_webhook_manager(self.webhooks)

        # Register event handlers
        self._register_event_handlers()

        logger.info("[Orchestrator] All modules initialized. Running.")

        wake = self.wake
        sessions = self.sessions
        if wake is None or sessions is None:
            raise JarvisError("Core voice modules failed to initialize")

        # Build task list
        tasks = [
            self.bus.run(),
            wake.listen_forever(),
            sessions.cleanup_expired(),
            self._context_loop(),
            self._vision_loop(),
            self._curiosity_loop(),
            self._health_broadcast_loop(),
            self._eod_summary_loop(),
            self._calendar_alert_loop(),
            self._self_thought_loop(),
        ]

        # MQTT monitoring
        if self.mqtt:
            tasks.append(self.mqtt.listen_forever())
        if self.nodes:
            tasks.append(self.nodes.monitor_heartbeats())

        # Multi-room wake — start any pre-registered sources. The manager is
        # safe to start with zero sources; new ones (Wyze, ESP) can register
        # post-start and will spin up on demand.
        if self.wake_sources is not None:
            self.wake_sources.start()

        # Write the heartbeat file the supervisor uses to confirm a clean
        # startup. If we never reach this line, the supervisor will
        # auto-revert the last self-edit.
        try:
            from pathlib import Path as _Path
            heartbeat = _Path(__file__).resolve().parents[1] / "data" / "heartbeat.txt"
            heartbeat.parent.mkdir(parents=True, exist_ok=True)
            heartbeat.write_text(datetime.now(timezone.utc).isoformat(), "utf-8")
        except Exception as e:
            logger.debug(f"[Boot] heartbeat write failed: {e}")

        # Reminder scheduler
        if self.reminder_scheduler:
            tasks.append(self.reminder_scheduler.run())

        # Dashboard
        if self.dashboard:
            tasks.append(self.dashboard.run())

        # Announce startup — let the LLM riff something in character
        startup_line = await self._compose_in_character(
            prompt=(
                "You just finished booting up after a restart. Greet Cole "
                "with a single short in-character line announcing you're "
                "online and ready. No preamble, no quotes."
            ),
            fallback="Jarvis online.",
        )
        # Startup greeting is one-way — Cole rarely talks back to the boot
        # ping, and we don't want a stray fan-noise blip to spawn a wake.
        await self._speak(startup_line, priority="ambient", expects_response=False)

        # Run forever — cancel all tasks cleanly on exit
        gather = asyncio.gather(*tasks, return_exceptions=True)
        try:
            await gather
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            gather.cancel()
            running = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for t in running:
                t.cancel()
            if running:
                await asyncio.gather(*running, return_exceptions=True)
            await self._shutdown()
            logger.info("[Orchestrator] All tasks cancelled. Goodbye.")
