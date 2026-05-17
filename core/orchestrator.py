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
from modules.world_model.patterns import PatternMiner
from modules.world_model.anomaly import AnomalyScorer
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

from core.orchestrator_base import OrchestratorMixin
from core.orchestrator_init import InitMixin
from core.orchestrator_conversation import ConversationMixin
from core.orchestrator_loops import LoopsMixin
class Orchestrator(ToolsMixin, InitMixin, ConversationMixin, LoopsMixin):
    def __init__(self, config: dict):
        self.config = config
        bus_cfg = (config.get("system") or {}).get("event_bus", {})
        rate_limits: dict[str, tuple[float, int]] = {}
        raw_limits = bus_cfg.get("rate_limits", {}) if isinstance(bus_cfg, dict) else {}
        if isinstance(raw_limits, dict):
            for topic, limit in raw_limits.items():
                if not isinstance(limit, dict):
                    continue
                try:
                    rate_limits[str(topic)] = (
                        float(limit.get("per_second", 0.0)),
                        int(limit.get("burst", 1)),
                    )
                except (TypeError, ValueError):
                    continue
        self.bus = EventBus(
            max_queue_size=int(bus_cfg.get("max_queue_size", 1000))
            if isinstance(bus_cfg, dict) else 1000,
            topic_rate_limits=rate_limits or None,
        )

        # These are populated in _init_* methods called from run()
        self.db: Optional[DatabaseManager] = None
        self.event_log: Optional[EventLogger] = None
        self.room_baselines: Optional[RoomBaselines] = None
        # Persistent notification inbox surfaced by the dashboard bell.
        # Set up after DB init; passed into IdentityManager so drift/cluster
        # events auto-fire user-visible notifications.
        self.notifications: Optional[NotificationManager] = None
        # Phone/push fan-out for the §29 alarm subsystem (ntfy / Telegram /
        # Home Assistant). Distinct from `self.notifications` (the dashboard
        # bell). Built in _init_database; passed into every alarm so phone
        # alerts actually go out — previously this was never constructed and
        # alarms silently ran local-only.
        self.notification_dispatcher: Optional[NotificationDispatcher] = None
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
        # Per-room mic/speaker dispatch. MicManager owns one driver per room
        # (USB / Wyze RTSP / ESP MQTT / null); SpeakerManager mirrors it on
        # the output side. The PC mic stays on `self.wake` (WakeWordDetector
        # has its own sounddevice grab); per-room MicSources for non-USB
        # rooms get bridged into wake_sources via MicSourceWakeAdapter so
        # talking near the bedroom Wyze cam fires a 'voice.wake_detected'
        # event tagged with the right room.
        self.mic_manager: Optional[MicManager] = None
        self.speaker_manager: Optional[SpeakerManager] = None
        # Per-room runtime tweaks (rotation, flip, volume, mute) — JSON-
        # persisted at data/room_settings.json. Mutated by the dashboard
        # cog modal; consumed by CameraManager (every frame) and
        # SpeakerManager (every play).
        self.room_settings: Optional[RoomSettings] = None
        # Per-Wyze-room hardware control (night vision, IR LEDs, status
        # LED, reboot). Keyed by room id; only rooms with
        # video.type == wyze_rtsp get an instance. Reuses the SSH
        # credentials from the room's wyze_ssh_aplay speaker config.
        self.wyze_cam_controls: dict[str, WyzeCamControl] = {}
        # Persona system — owns the active system-prompt bundle + the
        # auto-revert state machine (person-entry, away timeout, phone
        # call). Built in _init_brain after the prompt builder so we
        # can wire them together. Optional for legacy configs without
        # a personas section.
        self.persona: Optional[PersonaManager] = None
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
        # House-layout door graph. Built in _init_voice (after data_dir is
        # known) and consumed by _try_layout_teach + future transit-inference.
        self.door_map: Optional[DoorMap] = None
        # World Model — persistent entity registry + bounded-house state
        # machine. Built in run() after IdentityManager so the
        # ObservationBuilder can call into it for face-embedding identity
        # matches. None when init fails (e.g. DB unavailable); the
        # orchestrator runs without it.
        self.world_store: Optional[WorldStore] = None
        self.world_model: Optional[WorldModel] = None
        self.observation_builder: Optional[ObservationBuilder] = None
        self.interaction_monitor: Optional[InteractionMonitor] = None
        # Belief-state tracker (audit roadmap D4). Runs in shadow mode —
        # ingests vision.observation, maintains hypotheses, persists to the
        # belief_* tables, affects nothing live. None when world model is off.
        self.belief_resolver: Optional[BeliefResolver] = None
        # §25 Cross-Day Patterns & Anomalies — built alongside WorldModel.
        self.pattern_miner: Optional[PatternMiner] = None
        self.anomaly_scorer: Optional[AnomalyScorer] = None
        self.alarm_dispatcher: Optional[Any] = None
        # LLM-facing query layer — built alongside WorldModel. The four
        # tools (get_entity_status, list_entities_in_room, who_is_home,
        # search_recent_events) ride into _build_tool_registry when this
        # is non-None.
        self.world_query_tools: Optional[WorldQueryTools] = None
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
        self.integrations = IntegrationRegistry()

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
        # Monotonic timestamp of the last _active_user_room update.
        # Used by the wake coalescer as a vision-disambiguation tiebreaker
        # when multiple mics fire on the same utterance — a recent face
        # detection or wake event in a specific room is a stronger signal
        # of "Cole is THERE" than raw mic confidence (a studio mic in the
        # office can outscore a Wyze mic in the living room even when
        # Cole is clearly in the living room). Stale priors decay: if the
        # room was set >30s ago, the bonus doesn't apply.
        self._active_user_room_ts: float = 0.0
        self._wake_lock = asyncio.Lock()
        self._audio_io_active: bool = False
        # Per-room echo suppression. After Jarvis speaks into a room,
        # that room's mic will hear Jarvis's own voice and may fire a
        # wake. We record the wall-clock time we last finished speaking
        # in each room and ignore wakes from that room until the
        # window expires. Window = TTS duration + ECHO_SUPPRESSION_TAIL_S
        # (set when speech starts; expiry stamped when it ends).
        self._room_speech_until: dict[str, float] = {}
        # Unified scene snapshot per room. Updated by _vision_loop after
        # each per-room pass; consumed by build_scene_extras() for LLM
        # prompts and by the dashboard for the live scene panel. Keys
        # are room ids; each value is a dict of the most recent
        # observations (lights, person, posture, last description, ts).
        self._scene_state: dict[str, dict] = {}
        # D3 unified perception. When enabled, _vision_loop stops re-running
        # its own YOLO/face/posture and consumes the latest vision.observation
        # ObservationBuilder already published — one perception source. The
        # cache is the most recent batch per room; the flag selects the
        # _vision_loop path (false → the verbatim legacy path).
        self._latest_observation: dict[str, dict] = {}
        self._unified_perception: bool = bool(
            (config.get("world_model", {}) or {}).get("unified_perception", True)
        )
        # Wake-event coalescer: when multiple mics hear the same wake word
        # at nearly the same time (Cole says "Hey Jarvis" between the
        # bedroom and kitchen cams), each MicWakeRunner fires its own
        # voice.wake_detected event. Without coalescing we'd start the
        # capture pipeline once per mic, then queue the rest behind the
        # wake_lock. Instead, hold all wake events for a short window and
        # fire ONE capture for the loudest (highest-confidence) room.
        self._pending_wakes: list[dict] = []
        self._wake_window_task: Optional[asyncio.Task] = None
        # Tracks fire-and-forget tasks the orchestrator spawns
        # (memory extraction, listen-followup, scene-narration). Anchors
        # them against GC, logs their exceptions, and gets them cancelled
        # cleanly in _shutdown.
        # Policy-bearing task supervisor. Per-room singleton guards for the
        # scene pipeline and follow-up listeners are expressed via
        # TaskPolicy(singleton_key=...) at the spawn sites rather than
        # hand-rolled dicts.
        self._bg_tasks = TaskSupervisor(label="Orchestrator")
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

    def _register_event_handlers(self) -> None:
        """Subscribe to all relevant event bus topics."""
        # BUG FIX: WakeWordDetector publishes to "voice.wake_detected" —
        # orchestrator was subscribing to "wake.detected" (completely different topic).
        # The wake pipeline would have silently never fired.
        # Coalescer first — it batches per-mic wake events over a short
        # window and forwards only the loudest to _on_wake_detected.
        self.bus.subscribe("voice.wake_detected", self._on_wake_event_raw)
        self.bus.subscribe("voice.wake_score", self._on_wake_score)
        # Voice cascade (Option D) outcomes — published by CascadeWakeRunner.
        self.bus.subscribe("voice.triage_escalate", self._on_triage_escalate)
        self.bus.subscribe("voice.sound_event", self._on_cascade_sound_event)
        self.bus.subscribe("appliance.state_changed", self._on_appliance_changed)
        self.bus.subscribe("node.status", self._on_node_status)
        self.bus.subscribe("audio.level", self._on_audio_level)
        self.bus.subscribe("reminder.due", self._on_reminder_due)
        # D3: cache the latest perception batch per room so _vision_loop's
        # unified path can consume it instead of re-detecting.
        self.bus.subscribe("vision.observation", self._on_vision_observation)

    async def _on_vision_observation(self, payload: dict) -> None:
        """Cache the most recent vision.observation batch per room (D3).
        ObservationBuilder is the single perception source; _vision_loop's
        unified path reads this cache rather than running its own
        detectors."""
        room = payload.get("room")
        if room:
            self._latest_observation[room] = payload

    async def _shutdown(self) -> None:
        """Release long-lived resources cleanly during shutdown."""
        # Cancel any pending wake-coalesce window so it doesn't try to
        # fire the capture pipeline against modules we're about to close.
        if self._wake_window_task is not None and not self._wake_window_task.done():
            self._wake_window_task.cancel()
            try:
                await self._wake_window_task
            except (asyncio.CancelledError, Exception):
                pass
        self._pending_wakes = []
        # Cancel tracked background tasks (memory extraction, follow-up
        # listeners, scene-pipeline jobs). Bounded so a hung task can't
        # stall shutdown — the outer _shutdown is itself wrapped in an
        # 8s asyncio.wait_for.
        try:
            await self._bg_tasks.shutdown(timeout=2.0)
        except Exception as e:
            logger.debug(f"[Shutdown] background-task drain failed: {e}")
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

        # World-model side: stop the observation polling loops, the
        # interaction monitor, and the WorldModel timers. Errors are
        # non-fatal — we're going down anyway.
        if self.alarm_dispatcher is not None:
            try:
                await self.alarm_dispatcher.stop()
            except Exception as e:
                logger.debug(f"[Shutdown] alarm_dispatcher stop failed: {e}")
        if self.belief_resolver is not None:
            try:
                await self.belief_resolver.stop()
            except Exception as e:
                logger.debug(f"[Shutdown] belief_resolver stop failed: {e}")
        if self.anomaly_scorer is not None:
            try:
                await self.anomaly_scorer.stop()
            except Exception as e:
                logger.debug(f"[Shutdown] anomaly_scorer stop failed: {e}")
        if self.interaction_monitor is not None:
            try:
                await self.interaction_monitor.stop()
            except Exception as e:
                logger.debug(f"[Shutdown] interaction_monitor stop failed: {e}")
        if self.observation_builder is not None:
            try:
                await self.observation_builder.stop()
            except Exception as e:
                logger.debug(f"[Shutdown] observation_builder stop failed: {e}")
            # MediaPipe Hands holds GPU buffers on Windows; explicit close.
            hd = getattr(self.observation_builder, "hand_detector", None)
            if hd is not None and hasattr(hd, "close"):
                try:
                    hd.close()
                except Exception:
                    pass
        if self.world_model is not None:
            try:
                await self.world_model.stop()
            except Exception as e:
                logger.debug(f"[Shutdown] world_model stop failed: {e}")

        if self.cameras:
            try:
                await self.cameras.close()
            except Exception as e:
                logger.debug(f"[Shutdown] Camera close failed: {e}")

        # Mic + speaker managers — close after wake_sources so the
        # MicSourceWakeAdapters that hold these sources see the wake side
        # tear down first; otherwise an in-flight callback could fire
        # against a freshly-stopped MicSource and log noise on its way out.
        if self.mic_manager is not None:
            try:
                await self.mic_manager.close()
            except Exception as e:
                logger.debug(f"[Shutdown] MicManager close failed: {e}")
        if self.speaker_manager is not None:
            try:
                await self.speaker_manager.close()
            except Exception as e:
                logger.debug(f"[Shutdown] SpeakerManager close failed: {e}")
        for room, ctrl in self.wyze_cam_controls.items():
            try:
                await ctrl.close()
            except Exception as e:
                logger.debug(f"[Shutdown] WyzeCamControl close for '{room}' failed: {e}")

        if self.webhooks is not None:
            try:
                await self.webhooks.close()
            except Exception as e:
                logger.debug(f"[Shutdown] WebhookManager close failed: {e}")

        if self.notification_dispatcher is not None:
            try:
                await self.notification_dispatcher.close()
            except Exception as e:
                logger.debug(f"[Shutdown] notification dispatcher close failed: {e}")

        try:
            await self.integrations.stop_all()
        except Exception as e:
            logger.debug(f"[Shutdown] integrations stop failed: {e}")

        if self.llm is not None and hasattr(self.llm, "close"):
            try:
                await self.llm.close()
            except Exception as e:
                logger.debug(f"[Shutdown] LLM close failed: {e}")
        if self.llm is not None and hasattr(self.llm, "aclose"):
            try:
                await self.llm.aclose()
            except Exception as e:
                logger.debug(f"[Shutdown] LLM aclose failed: {e}")
        if getattr(self, "triage_gate", None) is not None:
            try:
                await self.triage_gate.aclose()
            except Exception as e:
                logger.debug(f"[Shutdown] triage gate aclose failed: {e}")
        if self._claude_client is not None and hasattr(self._claude_client, "close"):
            try:
                await self._claude_client.close()
            except Exception as e:
                logger.debug(f"[Shutdown] Claude close failed: {e}")

        if self.db:
            try:
                await self.db.close()
            except Exception as e:
                logger.debug(f"[Shutdown] DB close failed: {e}")

    async def _stop_flag_watcher(self) -> None:
        """Poll for data/stop.flag and trigger a graceful shutdown when it
        appears. stop.ps1 writes the flag; this turns it into the same
        KeyboardInterrupt path Ctrl+C uses, so _shutdown actually runs
        (cameras released, DB closed) instead of a hard kill — and the
        supervisor sees a clean exit and stops respawning."""
        from pathlib import Path as _Path
        data_dir = _Path(
            (self.config.get("system") or {}).get("data_dir", "data")
        )
        flag = data_dir / "stop.flag"
        while True:
            await asyncio.sleep(1.0)
            if flag.exists():
                logger.info(
                    "[Orchestrator] stop.flag detected — graceful shutdown"
                )
                try:
                    flag.unlink()
                except OSError:
                    pass
                # Cancel the top-level gather. `await gather` in run() then
                # raises CancelledError, its except/finally run _shutdown.
                # We cancel the gather (not this task / not the main task)
                # so run()'s finally-block awaits still work.
                gather = getattr(self, "_gather", None)
                if gather is not None:
                    gather.cancel()
                return

    async def _supervised(self, name: str, factory) -> None:
        """Run a loop coroutine, restarting it with exponential backoff if
        it crashes at the outer level.

        Each loop already has an inner try/except that survives transient
        per-cycle errors; this catches the rarer case where an exception
        escapes the loop body entirely (the coroutine would otherwise just
        end, and gather(return_exceptions=True) would never restart it).
        A clean return is treated as intentional. CancelledError
        propagates so shutdown still works.
        """
        backoff = 1.0
        while True:
            try:
                await factory()
                return  # clean exit — intentional, stop supervising
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    f"[Supervisor] loop '{name}' crashed — "
                    f"restarting in {backoff:.0f}s"
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

    # ── Smoke test ─────────────────────────────────────────────────────────

    async def smoke_init(self) -> bool:
        """Boot the safe subset and report health — used by `--smoke-init`.

        Exercises the real boot path that matters for catching the bugs
        this audit found (DB schema/migrations, notification dispatcher
        construction, event bus, alarm-audio rendering) WITHOUT opening
        camera/mic hardware or loading heavy ML weights. Returns True iff
        every check passed.
        """
        ok = True

        # DB init + schema migrations + dispatcher construction.
        try:
            await self._init_database()
            logger.info("[Smoke] DB init + schema migrations: OK")
        except Exception as e:
            logger.error(f"[Smoke] DB init FAILED: {e}")
            ok = False

        # Notification dispatcher (constructed inside _init_database).
        if self.notification_dispatcher is not None:
            chans = self.notification_dispatcher.configured_channels
            logger.info(
                f"[Smoke] notification dispatcher: OK "
                f"(channels={chans or 'none enabled'}, "
                f"fire→{self.notification_dispatcher.routes_for('fire')})"
            )
        else:
            logger.error("[Smoke] notification dispatcher NOT constructed")
            ok = False

        # Event bus.
        logger.info(
            f"[Smoke] event bus: OK "
            f"(queue_depth={self.bus.queue_depth}, "
            f"subscribers={self.bus.subscriber_count})"
        )

        # Alarm-audio TTS render — the C1 regression guard. A fake TTS
        # returns the same np.ndarray shape PiperTTS does.
        try:
            import numpy as _np

            from modules.safety.alarms.audio import AlarmAudio

            class _FakeTTS:
                _sample_rate = 22050

                def synthesize(self, text: str):
                    return _np.zeros(2205, dtype=_np.float32)

            aa = AlarmAudio.__new__(AlarmAudio)
            aa._tts = _FakeTTS()  # type: ignore[attr-defined]
            pcm, rate = await aa._render_tts("smoke test")
            if pcm:
                logger.info(
                    f"[Smoke] alarm TTS render: OK ({len(pcm)} bytes @ {rate}Hz)"
                )
            else:
                logger.error("[Smoke] alarm TTS render produced EMPTY audio")
                ok = False
        except Exception as e:
            logger.error(f"[Smoke] alarm TTS render FAILED: {e}")
            ok = False

        # Clean up the DB connection we opened.
        if self.db is not None:
            try:
                await self.db.close()
            except Exception as e:
                logger.debug(f"[Smoke] DB close failed: {e}")

        return ok

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

        # World Model + ObservationBuilder. Built after IdentityManager
        # so person-observation enrichment can route faces through it.
        # Skip if any of the dependencies failed (DB, identity, cameras) —
        # the system degrades to "no persistent presence tracking" rather
        # than crashing.
        if (self.db is not None and self.identity is not None
                and self.cameras is not None and self.object_detector is not None
                and self.face_recognizer is not None):
            try:
                self.world_store = WorldStore(self.db)
                self.world_model = WorldModel(
                    bus=self.bus,
                    store=self.world_store,
                    rooms_config=self.config.get("rooms", []),
                    identity_manager=self.identity,
                    config=self.config.get("world_model", {}),
                )
                # Build a config-shape ObservationBuilder needs from the
                # rooms list, with `world_model` enabled per-room. Default
                # to enabled so vision gets ingested everywhere — opt-out
                # by setting `world_model.enabled: false` on a room.
                ob_rooms = []
                for r in self.config.get("rooms", []):
                    rcopy = dict(r)
                    rcopy["world_model"] = {
                        "enabled": True,
                        **(r.get("world_model") or {}),
                    }
                    ob_rooms.append(rcopy)
                from pathlib import Path as _Path
                data_dir = _Path(
                    self.config.get("system", {}).get("data_dir", "data")
                )
                snapshot_dir = data_dir / "world_snapshots"
                # §24.1 — MediaPipe Hands. Optional; if mediapipe isn't
                # importable in this environment we fall through to no
                # hand detection (interaction events stop firing).
                hand_detector = None
                try:
                    from modules.vision.hand_detector import HandDetector
                    hand_cfg = (
                        self.config.get("world_model", {}) or {}
                    ).get("hand_detector", {}) or {}
                    hand_detector = HandDetector(
                        max_num_hands=int(hand_cfg.get("max_num_hands", 4)),
                        min_detection_confidence=float(
                            hand_cfg.get("min_detection_confidence", 0.5)
                        ),
                    )
                except Exception as e:
                    logger.warning(
                        f"[Init] HandDetector unavailable; interactions "
                        f"will not fire ({e})"
                    )
                # §23 — CLIP encoder + OWLv2 open-vocab detector. Both
                # `try_load` to a Null* fallback when their heavy ML
                # deps are missing; the rest of the pipeline degrades
                # cleanly (no object dedup signal, no open-vocab
                # tracking) without crashing.
                from modules.vision.clip_encoder import (
                    try_load as try_load_clip,
                )
                from modules.vision.openvocab_detector import (
                    try_load as try_load_openvocab,
                )
                clip_cfg = (
                    self.config.get("world_model", {}) or {}
                ).get("clip_encoder", {}) or {}
                openvocab_cfg = (
                    self.config.get("world_model", {}) or {}
                ).get("openvocab_detector", {}) or {}
                clip_encoder = try_load_clip(
                    model_name=clip_cfg.get(
                        "model_name", "ViT-B-32",
                    ),
                    pretrained=clip_cfg.get(
                        "pretrained", "laion2b_s34b_b79k",
                    ),
                    device=clip_cfg.get("device", "cuda"),
                )
                openvocab_detector = try_load_openvocab(
                    model_name=openvocab_cfg.get(
                        "model_name",
                        "google/owlv2-base-patch16-ensemble",
                    ),
                    device=openvocab_cfg.get("device", "cuda"),
                    score_threshold=float(
                        openvocab_cfg.get("score_threshold", 0.20),
                    ),
                )
                tracked_objects_cfg = (
                    self.config.get("tracked_objects", {}) or {}
                )
                tracked_open_vocab = (
                    tracked_objects_cfg.get("open_vocabulary") or []
                )
                openvocab_interval = float(
                    (tracked_objects_cfg.get(
                        "detection_interval_seconds", {},
                    ) or {}).get("open_vocabulary", 30.0)
                )
                # Hand the CLIP encoder to the WorldModel so find_object
                # can encode text queries against the same model.
                self.world_model.clip_encoder = clip_encoder
                self.observation_builder = ObservationBuilder(
                    bus=self.bus,
                    camera_manager=self.cameras,
                    object_detector=self.object_detector,
                    face_recognizer=self.face_recognizer,
                    identity_manager=self.identity,
                    posture_analyzer=self.posture,
                    rooms_config=ob_rooms,
                    snapshot_dir=snapshot_dir,
                    hand_detector=hand_detector,
                    clip_encoder=clip_encoder,
                    openvocab_detector=openvocab_detector,
                    tracked_objects_open_vocab=tracked_open_vocab,
                    tracked_objects_closed_vocab=(
                        tracked_objects_cfg.get("closed_vocabulary") or None
                    ),
                    openvocab_interval_seconds=openvocab_interval,
                    entity_min_confidence=(
                        (self.config.get("vision", {}) or {}).get(
                            "entity_min_confidence", {}
                        ) or {}
                    ),
                )
                # Wire the camera-health publisher so WorldModel can
                # suspend its state machine for offline rooms.
                if hasattr(self.cameras, "attach_bus"):
                    self.cameras.attach_bus(self.bus)
                # Subscribe + spawn timers; ObservationBuilder spawns its
                # per-room polling tasks in its own start().
                await self.world_model.start()
                await self.observation_builder.start()
                # §24.3 InteractionMonitor — subscribes to entity events
                # and correlates INTERACTED_WITH × LOST_VISIBILITY into
                # PICKED_UP, INTERACTED_WITH × FIRST_SEEN into
                # PLACED_DOWN. No-op if hand detector didn't load
                # (no interactions to correlate). Stored on the
                # orchestrator so _shutdown can stop() it.
                self.interaction_monitor = InteractionMonitor(
                    bus=self.bus, world=self.world_model,
                    config=(
                        self.config.get("world_model", {}) or {}
                    ).get("interactions", {}),
                )
                await self.interaction_monitor.start()
                # BeliefResolver (audit roadmap D4a) — subscribes to
                # vision.observation and tracks belief hypotheses in
                # shadow mode. Wrapped in its own try so a belief-layer
                # failure never takes down perception.
                try:
                    belief_cfg = (
                        self.config.get("world_model", {}) or {}
                    ).get("belief_resolver", {}) or {}
                    self.belief_resolver = BeliefResolver(
                        bus=self.bus, db=self.db, config=belief_cfg,
                    )
                    await self.belief_resolver.start()
                except Exception as e:
                    logger.warning(
                        f"[Init] BeliefResolver init failed (continuing "
                        f"without): {e}"
                    )
                    self.belief_resolver = None
                # §25 Cross-Day Patterns & Anomalies — PatternMiner builds
                # nightly behavioral profiles; AnomalyScorer scores live
                # entity events against them and fires world.anomaly.
                try:
                    wm_cfg = self.config.get("world_model", {}) or {}
                    anomaly_cfg = wm_cfg.get("anomaly", {}) or {}
                    self.pattern_miner = PatternMiner(
                        self.world_model,
                        days_back=int(anomaly_cfg.get("pattern_days_back", 30)),
                    )
                    self.anomaly_scorer = AnomalyScorer(
                        bus=self.bus, world=self.world_model, db=self.db,
                        config=anomaly_cfg,
                    )
                    await self.anomaly_scorer.start()
                except Exception as e:
                    logger.warning(
                        f"[Init] §25 anomaly subsystem init failed "
                        f"(continuing without): {e}"
                    )
                    self.pattern_miner = None
                    self.anomaly_scorer = None
                # §29 alarm subsystem — KlaxonLibrary loads per-alarm
                # audio assets, AlarmAudio fans out via the existing
                # speaker manager, AlarmStore persists fires/state.
                # All four alarms register with the dispatcher; only
                # those whose detectors are wired actually fire.
                try:
                    from modules.safety.alarms import (
                        AlarmAudio, AlarmDispatcher, AlarmStore,
                        CatEscapeAlarm, ClownAlarm, DoorOpenAlarm,
                        FireAlarm, KlaxonLibrary, NullAlarmAudio,
                    )
                    klaxons = KlaxonLibrary()
                    klaxons.load_all()
                    if (getattr(self, "speaker_manager", None) is not None
                            and getattr(self, "tts", None) is not None):
                        alarm_audio = AlarmAudio(
                            speaker_manager=self.speaker_manager,
                            tts=self.tts,
                            klaxons=klaxons,
                        )
                    else:
                        alarm_audio = NullAlarmAudio()
                    alarm_store = AlarmStore(self.db)
                    self.alarm_dispatcher = AlarmDispatcher(
                        audio=alarm_audio,
                    )
                    notifier = getattr(self, "notification_dispatcher", None)
                    self.alarm_dispatcher.register(CatEscapeAlarm(
                        bus=self.bus,
                        rooms_config=self.config.get("rooms", []),
                        notifier=notifier,
                        store=alarm_store,
                    ))
                    self.alarm_dispatcher.register(DoorOpenAlarm(
                        bus=self.bus,
                        rooms_config=self.config.get("rooms", []),
                        notifier=notifier,
                        store=alarm_store,
                    ))
                    self.alarm_dispatcher.register(FireAlarm(
                        bus=self.bus,
                        notifier=notifier,
                        store=alarm_store,
                    ))
                    # ClownAlarm gets the LLM client for improv
                    # generation. None is acceptable — improv falls
                    # back to curated entries.
                    self.alarm_dispatcher.register(ClownAlarm(
                        bus=self.bus,
                        audio=alarm_audio,
                        notifier=notifier,
                        store=alarm_store,
                        llm=getattr(self, "llm", None),
                    ))
                    await self.alarm_dispatcher.start()
                    logger.info("[Init] Alarm subsystem started")
                except Exception as e:
                    logger.warning(
                        f"[Init] alarm subsystem init failed (continuing "
                        f"without): {e}"
                    )
                    self.alarm_dispatcher = None
                # Query layer over the live model — registered into the
                # LLM tool registry by _build_tool_registry below.
                self.world_query_tools = WorldQueryTools(self.world_model)
                # Phase 4 (§22): bootstrap declared cats/dogs into entity
                # rows + sync affinities. Idempotent — safe to call every
                # boot. Failures here don't take the world model down.
                try:
                    pets = await bootstrap_pets_from_config(
                        self.world_store, self.config
                    )
                    # Re-hydrate the live entity registry so newly
                    # bootstrapped pets are immediately matchable.
                    if pets:
                        async with self.world_model._lock:
                            for ent in pets:
                                self.world_model.entities[ent.id] = ent
                except Exception as e:
                    logger.warning(f"[Init] pets bootstrap failed: {e}")
                logger.info("[Init] World Model + ObservationBuilder started")
            except Exception as e:
                logger.warning(
                    f"[Init] World Model init failed (continuing without): {e}"
                )
                self.world_store = None
                self.world_model = None
                self.observation_builder = None

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
            if self.mic_manager is not None and hasattr(self.dashboard, "register_mic_manager"):
                self.dashboard.register_mic_manager(self.mic_manager)
            if self.speaker_manager is not None and hasattr(self.dashboard, "register_speaker_manager"):
                self.dashboard.register_speaker_manager(self.speaker_manager)
            if self.room_settings is not None and hasattr(self.dashboard, "register_room_settings"):
                self.dashboard.register_room_settings(self.room_settings)
            if self.wyze_cam_controls and hasattr(self.dashboard, "register_wyze_cam_controls"):
                self.dashboard.register_wyze_cam_controls(self.wyze_cam_controls)
            if self.persona is not None and hasattr(self.dashboard, "register_persona"):
                self.dashboard.register_persona(self.persona)
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

        await self.integrations.start_all(IntegrationContext(
            config=self.config,
            bus=self.bus,
            dashboard=self.dashboard,
            db=self.db,
        ))

        # Register event handlers
        self._register_event_handlers()

        logger.info("[Orchestrator] All modules initialized. Running.")

        wake = self.wake
        sessions = self.sessions
        if wake is None or sessions is None:
            raise JarvisError("Core voice modules failed to initialize")

        # Build task list. The orchestrator's own loops are wrapped in
        # _supervised so an outer-level crash (one that escapes the loop's
        # inner try/except) restarts the loop with backoff instead of
        # silently leaving the system degraded until a full restart.
        tasks = [
            self.bus.run(),
            wake.listen_forever(),
            sessions.cleanup_expired(),
            # Watches for data/stop.flag (written by stop.ps1) and turns an
            # external "stop" into a real graceful shutdown. NOT supervised —
            # it is meant to end the process.
            self._stop_flag_watcher(),
            self._supervised("context_loop", self._context_loop),
            self._supervised("vision_loop", self._vision_loop),
            self._supervised("curiosity_loop", self._curiosity_loop),
            self._supervised("health_broadcast_loop", self._health_broadcast_loop),
            self._supervised("eod_summary_loop", self._eod_summary_loop),
            self._supervised("calendar_alert_loop", self._calendar_alert_loop),
            self._supervised("self_thought_loop", self._self_thought_loop),
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

        # World Model nightly: §22.6 BehavioralProfileBuilder. One pass
        # per resident pet entity, rolling 30-day window. Skipped if
        # the world model didn't come up.
        if self.world_model is not None:
            tasks.append(self._world_model_nightly_loop())

        # Dashboard
        if self.dashboard:
            tasks.append(self.dashboard.run())

        # Greet on startup — kept open-ended on purpose so the LLM doesn't
        # fall into a "I'm back, Cole..." rut. No mention of "online" or
        # "ready" in the prompt; just give it a beat to react however it
        # wants. Fallback is plain in case Ollama is down.
        startup_line = await self._compose_in_character(
            prompt=(
                "You just came back. Say one short line to Cole — whatever "
                "feels right. No preamble, no quotes."
            ),
            fallback="Jarvis online.",
        )
        # Startup greeting is one-way — Cole rarely talks back to the boot
        # ping, and we don't want a stray fan-noise blip to spawn a wake.
        await self._speak(startup_line, priority="ambient", expects_response=False)

        # Wrap each top-level task so exceptions get logged the moment
        # they happen instead of disappearing into the gather's result list.
        # Without this, a task that crashes at startup (e.g. uvicorn config
        # rejecting an unknown kwarg) is silently absorbed by
        # return_exceptions=True and we don't notice until shutdown.
        async def _logged(name: str, coro):
            try:
                return await coro
            except (asyncio.CancelledError, KeyboardInterrupt):
                raise
            except Exception as e:
                logger.exception(f"[Orchestrator] task '{name}' crashed: {e}")
                raise

        labeled_tasks = []
        for c in tasks:
            label = getattr(getattr(c, "cr_code", None), "co_qualname", None) or repr(c)
            labeled_tasks.append(_logged(label, c))

        # Run forever — cancel all tasks cleanly on exit. `_gather` is
        # stashed so _stop_flag_watcher can cancel it (graceful stop path).
        gather = asyncio.gather(*labeled_tasks, return_exceptions=True)
        self._gather = gather
        try:
            await gather
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            # Shutdown order matters. Cancel only the top-level loop gather
            # first, then run _shutdown() *while subsystem tasks still
            # exist* — so each subsystem's own stop() can wind its loops
            # (ObservationBuilder room loops, WorldModel/BeliefResolver
            # loops, dashboard, MQTT, reminders) down cleanly. Cancelling
            # every task up front would yank those loops out from under
            # their owners and leave half-stopped subsystems for _shutdown
            # to close. Only after _shutdown do we sweep orphan tasks.
            gather.cancel()
            # Bound _shutdown — if a Wyze cap.release() or mic worker.join()
            # wedges, fall through rather than hang Ctrl-C.
            try:
                await asyncio.wait_for(self._shutdown(), timeout=8.0)
            except asyncio.TimeoutError:
                logger.warning("[Shutdown] resource teardown exceeded 8s budget")
            # Last-resort sweep: anything the subsystems' own stop() did not
            # already finish gets cancelled now.
            running = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for t in running:
                t.cancel()
            if running:
                # Hard ceiling so a stuck to_thread (cv2 FFmpeg release,
                # ssh teardown) can't hang Ctrl-C indefinitely. Past this
                # window, tasks become daemon-thread garbage and the
                # process exits anyway.
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*running, return_exceptions=True),
                        timeout=5.0,
                    )
                except asyncio.TimeoutError:
                    stuck = [t for t in running if not t.done()]
                    logger.warning(
                        f"[Shutdown] {len(stuck)} orphan task(s) didn't "
                        f"cancel within 5s — exiting anyway"
                    )
            logger.info("[Orchestrator] All tasks cancelled. Goodbye.")
