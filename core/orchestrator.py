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
from datetime import datetime
from typing import Optional

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
from modules.brain.prompt_builder import PromptBuilder
from modules.brain.session import SessionManager
from modules.context.curiosity import CuriosityEngine
from modules.context.interruptibility import InterruptibilityManager
from modules.context.sleep_tracker import SleepTracker
from modules.context.state import UNKNOWN_STATE, ActivityState
from modules.context.state_fusion import StateFusion
from modules.memory.database import DatabaseManager
from modules.memory.event_log import EventLogger
from modules.memory.room_baselines import RoomBaselines
from modules.network.mqtt_client import MQTTClient
from modules.network.node_manager import NodeManager
from modules.vision.camera_manager import CameraManager
from modules.vision.light_detector import LightDetector
from modules.vision.object_detector import ObjectDetector
from modules.vision.posture_analyzer import PostureAnalyzer
from modules.vision.scene_analyzer import SceneAnalyzer
from modules.voice.stt import WhisperSTT
from modules.voice.tts import PiperTTS
from modules.voice.wake_word import WakeWordDetector


class Orchestrator:
    def __init__(self, config: dict):
        self.config = config
        self.bus = EventBus()

        # These are populated in _init_* methods called from run()
        self.db: Optional[DatabaseManager] = None
        self.event_log: Optional[EventLogger] = None
        self.room_baselines: Optional[RoomBaselines] = None

        self.wake: Optional[WakeWordDetector] = None
        self.stt: Optional[WhisperSTT] = None
        self.tts: Optional[PiperTTS] = None

        self.llm: Optional[OllamaLLM] = None
        self.sessions: Optional[SessionManager] = None
        self.prompts: Optional[PromptBuilder] = None

        self.state_fusion: Optional[StateFusion] = None
        self.interruptibility: Optional[InterruptibilityManager] = None
        self.curiosity: Optional[CuriosityEngine] = None
        self.sleep_tracker: Optional[SleepTracker] = None

        self.pc_monitor: Optional[PCMonitor] = None
        self.audio_classifier: Optional[AudioClassifier] = None
        self.appliance_tracker: Optional[ApplianceTracker] = None

        self.cameras: Optional[CameraManager] = None
        self.light_detector: Optional[LightDetector] = None
        self.posture: Optional[PostureAnalyzer] = None
        self.object_detector: Optional[ObjectDetector] = None
        self.scene_analyzer: Optional[SceneAnalyzer] = None

        self.mqtt: Optional[MQTTClient] = None
        self.nodes: Optional[NodeManager] = None

        self._current_state: ActivityState = UNKNOWN_STATE
        self._wake_lock = asyncio.Lock()
        self._audio_io_active: bool = False
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
        # BUG FIX: RoomBaselines takes (db, config) — was only receiving db
        self.room_baselines = RoomBaselines(db=self.db, config=self.config)
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

    async def _init_brain(self) -> None:
        """Initialize LLM, session manager, and prompt builder."""
        # BUG FIX: OllamaLLM and SessionManager both take config: dict.
        # Was: OllamaLLM(model=..., base_url=..., timeout=..., system_prompt=...)
        # Was: SessionManager(max_turns=...)
        self.llm = OllamaLLM(self.config)
        self.sessions = SessionManager(self.config)
        self.prompts = PromptBuilder(config=self.config)
        logger.info("[Init] Brain (LLM + sessions) ready")

    async def _init_context(self) -> None:
        """Initialize activity detection and context reasoning modules."""
        # BUG FIX: Multiple constructor mismatches fixed here.
        # InterruptibilityManager was getting flat kwargs — takes config: dict
        # AudioClassifier was getting window_seconds kwarg — takes config: dict
        # ApplianceTracker was missing config arg entirely — takes (config, event_bus)
        # SleepTracker was getting no args — takes config: dict
        # CuriosityEngine arg order: (config, llm) not (llm, config) — kwargs so OK but fixed for clarity

        self.state_fusion = StateFusion(config=self.config)
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
        logger.info("[Init] Vision pipeline ready")

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

        # Check interruptibility before responding
        # Guard self.interruptibility (Optional) before member access
        if self._current_state and self.interruptibility and not self.interruptibility.can_interrupt(
            self._current_state, priority="conversation"
        ):
            logger.debug("[Wake] Blocked by interruptibility gate")
            return

        async with self._wake_lock:
            self._audio_io_active = True
            try:
                from modules.voice.audio_utils import (
                    SAMPLE_RATE,
                    db_from_rms,
                    play_chime_async,
                    record_until_silence,
                )

                # Suspend wake word mic — prevents dual-InputStream conflict on Windows WASAPI
                if self.wake:
                    self.wake.suspend()

                try:
                    await play_chime_async()
                    await asyncio.sleep(0.3)

                    recording_cfg = self.config["voice"]["recording"]
                    record_device = (
                        self.wake.device if self.wake else recording_cfg.get("device")
                    )
                    audio_data = await asyncio.to_thread(
                        record_until_silence,
                        silence_threshold_db=recording_cfg["silence_threshold_db"],
                        silence_duration_ms=recording_cfg["silence_duration_ms"],
                        max_duration_seconds=recording_cfg["max_duration_seconds"],
                        speech_start_timeout_seconds=recording_cfg.get(
                            "speech_start_timeout_seconds",
                            5.0,
                        ),
                        device=record_device,
                    )
                finally:
                    # Always release the mic, even on exception
                    if self.wake:
                        self.wake.wakeup()

                if audio_data is None or len(audio_data) == 0:
                    logger.debug("[Wake] No audio recorded")
                    return

                duration_s = len(audio_data) / SAMPLE_RATE
                rms = float(np.sqrt(np.mean(audio_data ** 2))) if len(audio_data) else 0.0
                logger.info(
                    f"[Wake] Captured {duration_s:.2f}s of audio "
                    f"(rms={db_from_rms(rms):.1f} dBFS)"
                )

                stt = self.stt
                if stt is None:
                    logger.warning("[Wake] STT module not initialized — skipping transcript")
                    return
                transcript = await asyncio.to_thread(stt.transcribe, audio_data)
                if not transcript or not transcript.strip():
                    logger.info("[Wake] Empty transcript — nothing heard after chime")
                    return

                logger.info(f"[STT] Transcript: {transcript!r}")
                await self._process_user_text(transcript, room)

            except Exception as e:
                logger.error(f"[Wake] Pipeline error: {e}")
            finally:
                self._audio_io_active = False

    async def _process_user_text(self, text: str, room: str) -> None:
        """
        Core LLM pipeline shared by voice (wake word) and text chat.
        Broadcasts user speech, calls LLM, speaks the response.
        """
        await self._broadcast({"type": "user_speech", "text": text, "room": room})
        if self.event_log:
            await self.event_log.log_event(room=room, event_type="user_speech", content=text)

        if not self.sessions or not self.prompts or not self.llm:
            logger.warning("[LLM] Brain modules not ready — skipping")
            return

        session = self.sessions.get_session(room)
        prompt_context = self.prompts.build(
            user_text=text,
            state=self._current_state,
            session=session,
            room=room,
        )

        response = await self.llm.chat(messages=prompt_context)
        if not response:
            logger.warning("[LLM] Empty response")
            return

        session.add_turn("user", text)
        session.add_turn("assistant", response)
        logger.info(f"[LLM] Response: {response!r}")

        if self.interruptibility is not None:
            self.interruptibility.record_interruption()

        await self._speak(response, room=room, priority="conversation")

    async def _on_text_chat(self, text: str, room: str = "office") -> None:
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
            await self._speak(f"Switching to {voice_name}.", room="office", priority="ambient")

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

                if self.audio_classifier and not self._audio_io_active:
                    # BUG FIX: classify_async() returns list[dict] (not a single dict).
                    # Each dict has keys "label", "yamnet_class", "score".
                    # Calling .get("classifications", []) on a list crashes at runtime.
                    # state_fusion expects: signals["audio"] = {"activity": str, "confidence": float}
                    # so we take the top classification from the list.
                    classifications = await self.audio_classifier.classify_async()
                    if classifications:
                        # Feed raw list to appliance tracker (it expects list[dict])
                        if self.appliance_tracker:
                            self.appliance_tracker.update(classifications)
                        # Build state-fusion-compatible dict from top result
                        signals["audio"] = {
                            "activity":   classifications[0]["label"],
                            "confidence": classifications[0]["score"],
                        }

                if self.posture and self.cameras:
                    rooms = self.cameras.get_available_rooms()
                    if "office" in rooms:
                        frame = await self.cameras.capture_frame_async("office")
                        if frame is not None:
                            posture_result = await self.posture.analyze_async(frame)
                            signals["posture"] = {
                                "pose": posture_result,
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

                # Fuse signals into final state
                if self.state_fusion:
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

                        # Posture (for person pose context)
                        posture_result = None
                        if self.posture:
                            posture_result = await self.posture.analyze_async(frame)

                        # Scene description (LLM vision — slower, run last)
                        if not self.scene_analyzer:
                            continue
                        last_desc = self.scene_analyzer.last_description(room_id)
                        should_describe = bool(detections) or last_desc is None
                        if should_describe:
                            description = await self.scene_analyzer.describe_async(
                                frame, room=room_id, objects=detections
                            )
                            last_desc = description or last_desc

                        # BUG FIX: update_if_due() doesn't exist on RoomBaselines.
                        # Actual API: needs_update(room) → bool, then update(room, desc).
                        if self.room_baselines and last_desc:
                            if await self.room_baselines.needs_update(room_id):
                                await self.room_baselines.update(room_id, last_desc)

                        # Broadcast vision state — use lights_on bool directly
                        await self._broadcast({
                            "type": "vision",
                            "room": room_id,
                            "lights_on": lights_on,
                            "person_present": person_present,
                            "objects": object_summary,
                            "description": last_desc,
                        })

                        # Pass vision signal to state fusion — lights_on is already a bool
                        if self.state_fusion:
                            self.state_fusion.inject_vision(room_id, {
                                "lights_on": lights_on,
                                "person_present": person_present,
                                "posture": posture_result,
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

                await self._speak(utterance, room="office", priority="ambient")

            except Exception as e:
                logger.error(f"[Curiosity] Loop error: {e}")

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
            # Map appliance name to a natural announcement
            messages = {
                "washer": f"Hey — the washer's done. That was about {int(runtime or 0)} minutes.",
                "dryer":  f"Dryer's finished. Clothes are ready.",
                "dishwasher": "Dishwasher cycle complete.",
            }
            text = messages.get(appliance, f"{appliance} is done.")

            # Check urgency vs interruptibility
            urgency_map = self.config["appliances"]
            urgency = urgency_map.get(f"{appliance}_done_urgency", 0.5)
            priority = "urgent" if urgency >= 0.7 else "notification"

            await self._speak(text, room="office", priority=priority)

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

    # ── TTS Helper ─────────────────────────────────────────────────────────

    async def _speak(self, text: str, room: str = "office", priority: str = "ambient") -> None:
        """
        Full speak pipeline: TTS → audio playback → log → broadcast.
        Routes audio to appropriate room node if available, otherwise local playback.
        """
        try:
            logger.info(f"[TTS] [{priority}] {text!r}")

            # Guard: tts is Optional — initialized in _init_voice()
            if not self.tts:
                logger.warning("[TTS] TTS module not initialized — skipping playback")
                return

            # Local playback (always available)
            was_audio_io_active = self._audio_io_active
            self._audio_io_active = True
            try:
                await asyncio.to_thread(self.tts.speak, text)
            finally:
                self._audio_io_active = was_audio_io_active

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

    async def _shutdown(self) -> None:
        """Release long-lived resources cleanly during shutdown."""
        if self.wake:
            self.wake.stop()

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

        try:
            await self._init_network()
        except Exception as e:
            logger.warning(f"[Init] Network init failed (continuing without): {e}")

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
        ]

        # MQTT monitoring
        if self.mqtt:
            tasks.append(self.mqtt.listen_forever())
        if self.nodes:
            tasks.append(self.nodes.monitor_heartbeats())

        # Dashboard
        if self.dashboard:
            tasks.append(self.dashboard.run())

        # Announce startup
        await self._speak(
            "Jarvis online.",
            room="office",
            priority="ambient",
        )

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
