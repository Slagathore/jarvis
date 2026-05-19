"""
JARVIS — Ambient Home AI
========================
Mission: LoopsMixin — extracted from core/orchestrator.py (audit
         roadmap D6 decomposition). The periodic background loops + speech / scene / broadcast helpers.

         Mixed into Orchestrator; every `self.*` resolves against the
         concrete Orchestrator instance at runtime. The full
         orchestrator import block is duplicated below on purpose —
         over-importing is harmless and removes any missing-name risk.

Modules: core/orchestrator_loops.py
Classes: LoopsMixin
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

# How long an unanswered §23 object-vocab ask stays "pending" before the
# ask loop gives up and allows a new one. Comfortably past the follow-up
# listen window + STT/LLM round-trip, short enough that the feature
# self-heals within a couple of poll cycles if Cole ignores a question.
_OBJECT_VOCAB_ANSWER_TIMEOUT_S: float = 180.0


from core.orchestrator_tools import ToolsMixin


class LoopsMixin(OrchestratorMixin):
    """The periodic background loops + speech / scene / broadcast helpers.

    Mixed into Orchestrator — see core/orchestrator_base.py.
    """

    def _recent_ambient_audio(self, window_s: float) -> Optional[np.ndarray]:
        """Last `window_s` seconds of office-mic audio for the ambient
        AudioClassifier (YAMNet), as float32 [-1, 1] mono @ 16 kHz.

        The rolling buffer lives on whichever component owns the office
        PC mic: the legacy WakeWordDetector, or — when the office mic is
        routed through the cascade (self.wake is None) — the office
        MicSourceWakeAdapter. Returns None when neither has audio yet.
        """
        if self.wake is not None:
            return self.wake.get_recent_audio(window_s)
        if self.wake_sources is not None:
            src = self.wake_sources.get_source("office")
            getter = getattr(src, "get_recent_audio", None)
            if getter is not None:
                return getter(window_s)
        return None

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
                    # Read the last N seconds of audio from the office mic's
                    # shared rolling buffer instead of opening a second
                    # InputStream. The buffer lives on the legacy
                    # WakeWordDetector, or — when the office mic is routed
                    # through the cascade — on the office MicSourceWakeAdapter;
                    # _recent_ambient_audio picks whichever owns the mic. The
                    # old suspend/wakeup approach killed wake responsiveness —
                    # wake was unavailable ~50% of the time and openWakeWord
                    # lost its prediction context every cycle.
                    window_s = float(
                        self.config["context"].get("audio_classify_window_seconds", 3)
                    )
                    waveform = self._recent_ambient_audio(window_s)
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
                    # Posture surfaced as context for the active room. The
                    # sleep tracker is now updated per-room from
                    # _vision_loop (with identity attached), so we don't
                    # duplicate that work here. We just ask it for the
                    # current active room's signal.
                    if "office" in rooms:
                        frame = await self.cameras.capture_frame_async("office")
                        if frame is not None:
                            posture_result = await self.posture.analyze_async(frame)
                            signals["posture"] = {
                                "context": {"posture": posture_result},
                                "confidence": 0.7 if posture_result != "unknown" else 0.1,
                            }

                # Pull the active room's per-person sleep signal — a
                # napper here means the active state should reflect
                # 'napping'. Other rooms' sleepers don't show up here;
                # they're respected at the speech-gating layer instead.
                sleep_tracker = self.sleep_tracker
                if sleep_tracker is not None:
                    sleep_signal = sleep_tracker.get_room_sleep_signal(
                        self._active_user_room
                    )
                    if sleep_signal:
                        signals["sleep"] = sleep_signal

                # PC activity is Cole's awake signal — clear his sleep
                # state in any room. Don't touch other people.
                if signals.get("pc") and sleep_tracker is not None:
                    sleep_tracker.record_activity(
                        person_name="Cole", signal="pc-active"
                    )

                # Audio classifier voice in the active room = someone is
                # talking there → wake whoever was sleeping in that
                # room. We don't have identity on this signal, so it
                # clears all sleepers in the room (room scope only).
                audio_sig = signals.get("audio") or {}
                if (
                    isinstance(audio_sig, dict)
                    and sleep_tracker is not None
                    and str(audio_sig.get("activity", "")).lower() in {
                        "speech", "speech, female", "speech, male",
                        "conversation", "narration, monologue",
                    }
                ):
                    sleep_tracker.record_activity(
                        room=self._active_user_room, signal="voice-detected"
                    )

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
                    prev_activity = (
                        self._current_state.activity
                        if self._current_state is not None else None
                    )
                    new_state = await self.state_fusion.fuse(signals, room="office")
                    new_state.present = self._room_occupants(new_state.location)
                    self._current_state = new_state

                    # Persona auto-revert hook: feeds the away-timeout
                    # logic. Only fires on transitions (away → not-away
                    # cancels the timer; not-away → away starts it). The
                    # PersonaManager itself decides whether to act based
                    # on the active persona's requires_privacy flag.
                    if self.persona is not None and prev_activity != new_state.activity:
                        try:
                            await self.persona.notify_state_changed(new_state.activity)
                        except Exception as e:
                            logger.debug(f"[Persona] state notify failed: {e}")

                    # Persona phone-call detection — derived from the
                    # current PC process. We treat the room-only PC as
                    # the source of truth (the only place a Cole-driven
                    # call originates today). Window-title keywords add
                    # specificity for ambiguous processes like Slack.
                    if self.persona is not None:
                        await self._persona_check_phone_call(signals)

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
                        # D3: unified path consumes ObservationBuilder's
                        # perception; the verbatim legacy body below runs
                        # only when world_model.unified_perception is false.
                        if self._unified_perception:
                            await self._vision_room_unified(room_id)
                            continue
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
                        # Drop detections inside a configured ignore zone
                        # (a framed painting, a TV, etc). See
                        # modules/vision/ignore_zones.py + the polygon editor.
                        detections = filter_detections(detections, room_id)
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
                                if match is not None and match.is_ambiguous:
                                    # Ambiguous = candidates too close to call.
                                    # Surface it for the dashboard but DO NOT
                                    # commit identity: a wrong "Cole is here"
                                    # poisons presence, persona activation, and
                                    # (soon) the BeliefResolver.
                                    logger.info(
                                        f"[Identity/face] '{room_id}' ambiguous "
                                        f"candidate {match.name} "
                                        f"(sim={match.similarity:.2f}) — not committed"
                                    )
                                    await self._broadcast({
                                        "type":       "person_recognized",
                                        "room":       room_id,
                                        "name":       None,
                                        "candidate":  match.name,
                                        "similarity": match.similarity,
                                        "ambiguous":  True,
                                    })
                                    match = None
                                if match is not None:
                                    recognized_name = match.name
                                    recognized_pid = match.person_id
                                    logger.info(
                                        f"[Identity/face] '{room_id}' → {match.name} "
                                        f"(sim={match.similarity:.2f})"
                                    )
                                    await self._broadcast({
                                        "type":       "person_recognized",
                                        "room":       room_id,
                                        "name":       match.name,
                                        "similarity": match.similarity,
                                        "ambiguous":  False,
                                    })
                                    # Presence signal — Cole moved rooms.
                                    # Future proactive speech follows him here.
                                    if room_id != self._active_user_room:
                                        logger.info(
                                            f"[Presence] active room: "
                                            f"{self._active_user_room} → {room_id} "
                                            f"(face: {match.name})"
                                        )
                                        self._set_active_user_room(room_id)
                            except Exception as e:
                                logger.debug(f"[Identity/face] identify failed: {e}")

                        # Persona system hooks. Two updates per pass:
                        #   1. Room occupancy snapshot — feeds the activation
                        #      privacy gate ("can I activate uwu right now?").
                        #   2. Face-identity notification — triggers the hard
                        #      person-entry revert when a non-Cole face shows
                        #      up in Cole's active room while a private
                        #      persona is on.
                        # person_present is YOLO's "any human in frame" so
                        # it doubles as a count proxy until we have
                        # multi-face counting wired (see project_known_issues).
                        if self.persona is not None:
                            self.persona.notify_room_occupancy(
                                room=room_id,
                                person_count=1 if person_present else 0,
                                cole_present=(recognized_name == "Cole"),
                            )
                            if person_present:
                                # Pass identity even if None — persona_manager
                                # treats None as "unknown face", which is
                                # treated as not-Cole for safety.
                                try:
                                    await self.persona.notify_face_identified(
                                        room=room_id,
                                        identity=recognized_name,
                                    )
                                except Exception as e:
                                    logger.debug(
                                        f"[Persona] face notify failed: {e}"
                                    )

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
                        # Scene description + baseline + anomaly + mess all
                        # do LLM calls that can run 3-6 seconds each. Fire
                        # them as a background task so the room loop doesn't
                        # block on them. The current broadcast uses the
                        # cached description from the prior iteration —
                        # one frame's lag on the scene narration is invisible
                        # to a human and totally worth the unblocked pipeline.
                        prior_state = self._scene_state.get(room_id) or {}
                        last_desc = prior_state.get("description")
                        self._spawn_scene_task(
                            room_id,
                            self._run_scene_pipeline_bg(
                                room_id=room_id,
                                frame=frame,
                                detections=detections,
                                scene_persons=scene_persons,
                                scene_person_states=scene_person_states,
                            ),
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

                        # Refresh the unified scene cache for this room.
                        # build_scene_extras() and the dashboard's scene
                        # panel read from here on demand.
                        scene_posture: Optional[str] = None
                        if isinstance(posture_result, dict):
                            scene_posture = posture_result.get("posture")
                        elif isinstance(posture_result, str):
                            scene_posture = posture_result
                        self._scene_state[room_id] = {
                            "lights_on": lights_on,
                            "person_present": person_present,
                            "person_name": recognized_name,
                            "person_id": recognized_pid,
                            "posture": scene_posture,
                            "objects": object_summary,
                            "description": last_desc,
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                        }

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

                        # Per-person sleep tracking. Posture+identity is
                        # only meaningful when YOLO actually saw someone
                        # in this room; otherwise we'd accumulate ghost
                        # state from a frame with no human in it. The
                        # tracker also handles 'unknown' posture cleanly
                        # (won't move timers), so brief blind moments
                        # don't reset a confirmed nap — that's the
                        # door-disappearance rule in action.
                        if self.sleep_tracker is not None and person_present:
                            self.sleep_tracker.update(
                                room=room_id,
                                posture=posture_label,
                                lights_on=lights_on,
                                person_id=recognized_pid,
                                person_name=recognized_name,
                            )

                    except Exception as room_err:
                        logger.warning(f"[Vision] Room {room_id} error: {room_err}")

            except Exception as e:
                logger.error(f"[Vision] Loop error: {e}")

            await asyncio.sleep(interval_seconds)

    async def _vision_room_unified(self, room_id: str) -> None:
        """D3 unified-perception vision pass.

        Instead of independently re-running YOLO / face / posture, this
        consumes the latest vision.observation ObservationBuilder already
        published (cached by Orchestrator._on_vision_observation), then
        feeds the SAME downstream consumers as the legacy path — pet
        broadcast, presence/active-room, persona, scene narration,
        dashboard broadcast, state-fusion, sleep tracking. One perception
        source; no parallel truth-maker.

        Selected by config.world_model.unified_perception. The verbatim
        legacy path in _vision_loop runs when that flag is false — that is
        the instant revert.
        """
        obs_payload = self._latest_observation.get(room_id)
        if not obs_payload:
            return  # ObservationBuilder has not reported this room yet
        observations = obs_payload.get("observations") or []

        # A frame is still needed for light detection (unique to this
        # loop), the scene pipeline and drift-verify — but NOT for
        # detection; that already happened in ObservationBuilder.
        frame = None
        if self.cameras is not None:
            frame = await self.cameras.capture_frame_async(room_id)

        lights_on: Optional[bool] = None
        if self.light_detector is not None and frame is not None:
            lights_on = await self.light_detector.analyze_async(
                frame, room=room_id)

        # ── Derive perception from the shared observation batch ──────────
        persons = [o for o in observations
                   if getattr(o, "obj_class", None) == "person"]
        pets = [o for o in observations
                if getattr(o, "obj_class", None) in ("cat", "dog")]
        person_present = bool(persons)
        recognized_name: Optional[str] = None
        recognized_pid: Optional[int] = None
        posture_label: Optional[str] = None
        for o in persons:
            pid = getattr(o, "person_id", None)
            if pid is not None and recognized_pid is None:
                recognized_pid = pid
                recognized_name = getattr(o, "person_name", None)
            md = getattr(o, "metadata", {}) or {}
            if posture_label is None and md.get("posture"):
                posture_label = md.get("posture")

        # detection-shaped list for the scene LLM's object grounding.
        # ObservationBuilder emits tracked/open-vocab objects with
        # obj_class="object" and the real label in metadata.detected_class
        # ("cell phone", "backpack"). Surface that label, not the generic
        # "object", or the scene summary loses all object specificity.
        def _disp_class(o: Any) -> str:
            if getattr(o, "obj_class", None) == "object":
                md = getattr(o, "metadata", {}) or {}
                return (md.get("detected_class")
                        or md.get("openvocab_query") or "object")
            return str(getattr(o, "obj_class", "?"))

        detections = [
            {"class": _disp_class(o),
             "box": list(getattr(o, "bbox", ()) or ()),
             "confidence": float(getattr(o, "confidence", 0.0))}
            for o in observations
        ]
        object_summary = (
            self.object_detector.summarize(detections)
            if self.object_detector is not None else ""
        )

        # ── Pet-seen broadcast (deduped — same as legacy) ────────────────
        pet_classes_now = sorted({getattr(o, "obj_class", "") for o in pets})
        if pet_classes_now != self._last_pets_per_room.get(room_id, []):
            self._last_pets_per_room[room_id] = pet_classes_now
            if pet_classes_now:
                await self._broadcast({
                    "type": "pet_seen", "room": room_id,
                    "pets": pet_classes_now,
                })
                logger.info(f"[Vision] '{room_id}' pets: "
                            f"{', '.join(pet_classes_now)}")

        # ── Presence / active-room ───────────────────────────────────────
        if recognized_name is not None:
            await self._broadcast({
                "type": "person_recognized", "room": room_id,
                "name": recognized_name, "ambiguous": False,
            })
            if room_id != self._active_user_room:
                logger.info(f"[Presence] active room: "
                            f"{self._active_user_room} → {room_id} "
                            f"(face: {recognized_name})")
                self._set_active_user_room(room_id)

        # ── Persona hooks ────────────────────────────────────────────────
        if self.persona is not None:
            self.persona.notify_room_occupancy(
                room=room_id,
                person_count=1 if person_present else 0,
                cole_present=(recognized_name == "Cole"),
            )
            if person_present:
                try:
                    await self.persona.notify_face_identified(
                        room=room_id, identity=recognized_name)
                except Exception as e:
                    logger.debug(f"[Persona] face notify failed: {e}")

        # ── Drift verify (passive face/voice sample refresh) ─────────────
        if self.identity is not None and frame is not None:
            for pid_to_verify, modality in list(
                    self.identity._verify_pending.items()):
                if modality != "face":
                    continue
                if room_id != self._active_user_room or not person_present:
                    continue
                try:
                    outcome = await self.identity.verify_face(
                        pid_to_verify, frame)
                    self.identity._verify_pending.pop(pid_to_verify, None)
                    if outcome in ("pending_drift", "pending_conflict"):
                        await self._broadcast({
                            "type": "identity_pending_added",
                            "modality": "face", "outcome": outcome,
                        })
                except Exception as e:
                    logger.debug(f"[Identity/drift] verify_face failed: {e}")
        if (recognized_pid is not None and self.identity is not None
                and self._last_wake_audio is not None
                and self.identity._verify_pending.get(recognized_pid) == "voice"):
            try:
                outcome = await self.identity.verify_voice(
                    recognized_pid, self._last_wake_audio)
                self.identity._verify_pending.pop(recognized_pid, None)
                self._last_wake_audio = None
                if outcome in ("pending_drift", "pending_conflict"):
                    await self._broadcast({
                        "type": "identity_pending_added",
                        "modality": "voice", "outcome": outcome,
                    })
            except Exception as e:
                logger.debug(f"[Identity/drift] verify_voice failed: {e}")

        # ── Scene pipeline (cached desc; bg LLM) ─────────────────────────
        scene_persons = [recognized_name] if recognized_name else None
        scene_person_states: Optional[list[dict[str, Any]]] = None
        if person_present and posture_label:
            ps: dict[str, Any] = {"posture": posture_label}
            if recognized_name:
                ps["name"] = recognized_name
            scene_person_states = [ps]
        prior_state = self._scene_state.get(room_id) or {}
        last_desc = prior_state.get("description")
        if self.scene_analyzer is not None and frame is not None:
            self._spawn_scene_task(
                room_id,
                self._run_scene_pipeline_bg(
                    room_id=room_id, frame=frame, detections=detections,
                    scene_persons=scene_persons,
                    scene_person_states=scene_person_states,
                ),
            )

        # ── Dashboard broadcast + scene cache ────────────────────────────
        await self._broadcast({
            "type": "vision", "room": room_id, "lights_on": lights_on,
            "person_present": person_present, "person_name": recognized_name,
            "objects": object_summary, "description": last_desc,
        })
        self._scene_state[room_id] = {
            "lights_on": lights_on, "person_present": person_present,
            "person_name": recognized_name, "person_id": recognized_pid,
            "posture": posture_label, "objects": object_summary,
            "description": last_desc,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        # ── State fusion + sleep tracking ────────────────────────────────
        if self.state_fusion is not None:
            self.state_fusion.inject_vision(room_id, {
                "lights_on": lights_on, "person_present": person_present,
                "posture": posture_label,
            })
        if self.sleep_tracker is not None and person_present:
            self.sleep_tracker.update(
                room=room_id, posture=posture_label, lights_on=lights_on,
                person_id=recognized_pid, person_name=recognized_name,
            )

    def _room_occupants(self, room: Optional[str]) -> list[str]:
        """Recognized resident names currently in `room`, read from the
        per-room scene cache. Empty when the room has no camera, nobody is
        present, or the face was not recognized. The basis for
        non-Cole-centric proactive speech — curiosity addresses whoever
        this returns, by name."""
        if not room:
            return []
        scene = self._scene_state.get(room) or {}
        name = scene.get("person_name")
        return [name] if name else []

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

                # Resolve who is actually in the room this proactive line
                # will be spoken to, so curiosity addresses them by name (or
                # stays generic) instead of assuming Cole.
                target_room = (
                    self._active_user_room or self._current_state.location
                )
                occupants = self._room_occupants(target_room)
                utterance = await self.curiosity.check_async(
                    self._current_state, occupants=occupants
                )
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

                # Check Ollama — use the LLM module's pooled health client
                # so we get connection reuse + keep-alive across probes
                # instead of paying TLS setup every 30s.
                try:
                    if self.llm is not None:
                        ollama_ok = await self.llm.is_available_async()
                    else:
                        ollama_ok = False
                    health["ollama"] = {
                        "online": ollama_ok,
                        "model": self.config["ollama"]["model"] if ollama_ok else "",
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

                # Push the unified scene snapshot on the same cadence —
                # cheaper than a separate loop, and 30s is plenty for
                # the dashboard's room-overview panel since real-time
                # changes still come through the per-event 'vision'
                # broadcast.
                try:
                    await self._broadcast({
                        "type": "scene_snapshot",
                        "scene": self.get_scene_snapshot(),
                    })
                except Exception as e:
                    logger.debug(f"[Scene] broadcast failed: {e}")

            except Exception as e:
                logger.error(f"[Health] Broadcast error: {e}")

            await asyncio.sleep(interval_seconds)

    async def _world_model_nightly_loop(self) -> None:
        """
        §22.6 BehavioralProfileBuilder. Once per `interval_hours` (24 by
        default), iterate over every resident pet entity and rebuild its
        behavioral profile from the last 30 days of events. The first
        run waits `startup_grace_seconds` to avoid stampeding boot.
        """
        cfg = (self.config.get("world_model") or {}).get("nightly", {})
        interval_hours = float(cfg.get("interval_hours", 24))
        grace_s = float(cfg.get("startup_grace_seconds", 60))
        days_back = int(cfg.get("profile_days_back", 30))

        logger.info(
            f"[WorldModel] nightly profile builder loop starting "
            f"(every {interval_hours}h, {days_back}d window)"
        )
        try:
            await asyncio.sleep(grace_s)
        except asyncio.CancelledError:
            return

        builder = BehavioralProfileBuilder()
        while True:
            try:
                world = self.world_model
                if world is None:
                    await asyncio.sleep(interval_hours * 3600)
                    continue
                # Snapshot the entity list under the lock so we don't
                # rebuild the same dict twice if entities mutate mid-pass.
                async with world._lock:
                    targets = [
                        e for e in world.entities.values()
                        if e.entity_type in ("cat", "dog")
                        and e.is_resident
                        and e.archived_at is None
                    ]
                rebuilt = 0
                for ent in targets:
                    try:
                        await builder.rebuild_for(
                            world, ent, days_back=days_back
                        )
                        rebuilt += 1
                    except Exception as e:
                        logger.warning(
                            f"[WorldModel] profile rebuild failed for "
                            f"'{ent.display_name}': {e}"
                        )
                logger.info(
                    f"[WorldModel] nightly: rebuilt {rebuilt}/"
                    f"{len(targets)} pet profiles"
                )
                # §23.8 — prune stale, untouched object entities. Drops
                # the "every cup is a new cup" stragglers without
                # losing anything Cole ever interacted with.
                try:
                    object_max_age = int(cfg.get(
                        "object_prune_max_age_days", 30,
                    ))
                    pruned = await world.prune_stale_objects(
                        max_age_days=object_max_age,
                    )
                    if pruned:
                        logger.info(
                            f"[WorldModel] nightly: pruned {pruned} "
                            "stale object entit(y/ies)"
                        )
                except Exception as e:
                    logger.warning(
                        f"[WorldModel] object prune failed: {e}"
                    )

                # Entity consolidation — archive per-id duplicates + stale
                # rows. Also runs on startup; the nightly pass keeps a
                # long-running instance from drifting back up.
                try:
                    cons = await world.consolidate_entities()
                    if cons.get("duplicates") or cons.get("stale"):
                        logger.info(
                            f"[WorldModel] nightly: consolidated "
                            f"{cons['duplicates']} duplicate + "
                            f"{cons['stale']} stale entit(ies)"
                        )
                except Exception as e:
                    logger.warning(
                        f"[WorldModel] entity consolidation failed: {e}"
                    )

                # Snapshot retention: drop JPEGs older than 48h that
                # aren't in the per-pet keep-N set. Keeps disk bounded
                # now that we save crops aggressively for interaction
                # precursors. Tunable via world_model.nightly.
                try:
                    ob = getattr(self, "observation_builder", None)
                    snap_dir = getattr(ob, "snapshot_dir", None) if ob else None
                    if snap_dir is not None and world.store is not None:
                        retain_hours = int(cfg.get(
                            "snapshot_retain_hours", 48,
                        ))
                        per_pet_keep = int(cfg.get(
                            "snapshot_per_pet_keep", 20,
                        ))
                        stats = await world.store.prune_snapshot_files(
                            snap_dir,
                            retain_hours=retain_hours,
                            per_pet_keep=per_pet_keep,
                        )
                        if stats.get("deleted"):
                            logger.info(
                                f"[WorldModel] nightly: snapshot prune "
                                f"deleted {stats['deleted']} of {stats['scanned']} "
                                f"(kept {stats['kept']}; "
                                f"window={retain_hours}h, "
                                f"per_pet_keep={per_pet_keep})"
                            )
                except Exception as e:
                    logger.warning(
                        f"[WorldModel] snapshot prune failed: {e}"
                    )

                # DB retention — the append-only / inbox tables that grow
                # without bound (world_entity_events was 150k+ rows / the
                # bulk of a 160 MB DB at audit time). Prune, then VACUUM
                # once so the .db file actually shrinks (DELETE alone never
                # returns pages to the OS).
                try:
                    deleted = 0
                    if world.store is not None:
                        deleted += await world.store.prune_world_events(
                            retain_days=int(cfg.get("event_retention_days", 30))
                        )
                    if self.identity is not None:
                        deleted += await self.identity.prune_resolved_pending()
                    if self.notifications is not None:
                        deleted += await self.notifications.prune_read()
                    if self.belief_resolver is not None:
                        deleted += await self.belief_resolver.prune_evidence(
                            retain_days=int(cfg.get("event_retention_days", 30))
                        )
                    if deleted and self.db is not None:
                        await self.db.vacuum()
                    logger.info(
                        f"[Maintenance] nightly DB prune removed {deleted} row(s)"
                    )
                except Exception as e:
                    logger.warning(f"[Maintenance] DB prune failed: {e}")

                # §25 — rebuild resident behavioral profiles, then let the
                # anomaly scorer auto-tune its threshold against the past
                # week's false-positive rate.
                try:
                    if getattr(self, "pattern_miner", None) is not None:
                        n = await self.pattern_miner.run_nightly()
                        logger.info(
                            f"[Maintenance] PatternMiner rebuilt {n} profile(s)"
                        )
                    if getattr(self, "anomaly_scorer", None) is not None:
                        await self.anomaly_scorer.auto_tune()
                except Exception as e:
                    logger.warning(f"[Maintenance] §25 pattern pass failed: {e}")

                # Recognition bank gardener — quarantine incoherent face
                # samples so per-person cohesion climbs a little each night.
                try:
                    ident = getattr(self, "identity", None)
                    if ident is not None:
                        g = await ident.prune_bank_incoherent()
                        if g.get("quarantined"):
                            logger.info(
                                f"[Maintenance] bank gardener quarantined "
                                f"{g['quarantined']} incoherent face sample(s)"
                            )
                except Exception as e:
                    logger.warning(f"[Maintenance] bank gardener failed: {e}")
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception(
                    "[WorldModel] nightly loop iteration crashed"
                )
            try:
                await asyncio.sleep(interval_hours * 3600)
            except asyncio.CancelledError:
                break

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
        """Handle ESP32 node online/offline TRANSITIONS.

        NodeManager only fires this event on real transitions (offline→
        online, online→offline, or IP/fw change) so the log line below
        is meaningful — it used to fire every 15s per node.
        """
        room = event.get("room")
        data = event.get("data")
        # Prefer the new top-level "online" field (set by NodeManager).
        # Fall back to legacy payload shapes so older producers (tests,
        # other transports) still work.
        if "online" in event:
            online = bool(event.get("online"))
            ip = event.get("ip")
        elif isinstance(data, str):
            online = data.strip().lower() == "online"
            ip = event.get("ip")
        elif isinstance(data, dict):
            status = str(data.get("status", "")).strip().lower()
            online = bool(data.get("online", status == "online"))
            ip = data.get("ip", event.get("ip"))
        else:
            online = False
            ip = event.get("ip")

        logger.info(f"[Node] {room} → {'online' if online else 'offline'}")

        await self._broadcast({
            "type": "node_status",
            "room": room,
            "online": online,
            "ip": ip,
        })

    async def _speak_via_speaker_manager(self, text: str, room: str) -> bool:
        """Route TTS through SpeakerManager — used for non-PC sinks like
        Wyze SSH (audioplay_t20) and any USB device that's NOT the host's
        default output. Synthesizes once, hands the int16 PCM to the
        right driver, and waits for playback to complete (the driver's
        play() blocks until the remote binary exits).

        Returns True on successful playback. False = caller should fall
        back to the local PC speaker path.
        """
        if self.tts is None or self.speaker_manager is None:
            return False
        try:
            audio = await self.tts.synthesize_async(text)
        except Exception as e:
            logger.warning(f"[TTS] SpeakerManager-route synthesis failed: {e}")
            return False
        try:
            import numpy as _np
            pcm_int16 = (_np.clip(audio, -1.0, 1.0) * 32767.0).astype(_np.int16)
            payload = pcm_int16.tobytes()
        except Exception as e:
            logger.warning(f"[TTS] PCM conversion failed: {e}")
            return False
        # PiperTTS sample rate is the rate of the synthesized PCM. The
        # driver (e.g. WyzeSshSpeakerSink) handles any further resampling
        # to the destination's native rate (Wyze speaker is 8kHz).
        rate = int(getattr(self.tts, "_sample_rate", 22050))
        return await self.speaker_manager.play(room, payload, rate)

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

    # Tracks whether we last reported "phone call active" to PersonaManager.
    # Used to dedup the start/end notifications — PersonaManager itself
    # also dedupes but doing it here saves the call cost when state didn't change.
    _persona_call_seen: bool = False

    async def _persona_check_phone_call(self, signals: dict) -> None:
        """Inspect the latest PC signal for an active phone-call process
        and notify PersonaManager on transitions. The detection rule:
            - active process matches one of persona_revert.call_processes
            - AND (the process is unconditional like zoom/teams, OR the
              window title contains one of call_window_keywords)
        Slack and Discord need the title check because they're often open
        without being in a call. Zoom/Teams are usually call-only.
        """
        if self.persona is None:
            return
        revert_cfg = self.config.get("_persona_revert_cfg")
        if revert_cfg is None:
            return
        call_processes = {p.lower() for p in revert_cfg.call_processes}
        title_keywords = [k.lower() for k in revert_cfg.call_window_keywords]

        pc_sig = signals.get("pc") if isinstance(signals.get("pc"), dict) else {}
        proc = str(pc_sig.get("exe", "")).lower() if isinstance(pc_sig, dict) else ""
        title = str(pc_sig.get("window_title", "")).lower() if isinstance(pc_sig, dict) else ""

        in_call = False
        if proc in call_processes:
            # zoom/teams = unconditional call. slack/discord need a title hint.
            unconditional_call_apps = {"zoom.exe", "teams.exe"}
            if proc in unconditional_call_apps:
                in_call = True
            elif title and any(kw in title for kw in title_keywords):
                in_call = True

        if in_call and not self._persona_call_seen:
            self._persona_call_seen = True
            try:
                await self.persona.notify_phone_call_started()
            except Exception as e:
                logger.debug(f"[Persona] phone-start notify failed: {e}")
        elif not in_call and self._persona_call_seen:
            self._persona_call_seen = False
            try:
                await self.persona.notify_phone_call_ended()
            except Exception as e:
                logger.debug(f"[Persona] phone-end notify failed: {e}")

    @staticmethod
    def _extract_wyze_host(url: str) -> Optional[str]:
        """Pull the host out of an rtsp://user:pass@host:port/path URL.
        Used to wire WyzeCamControl from the room's video block without
        asking the user to specify the host twice in config.yaml.
        """
        if not url:
            return None
        try:
            from urllib.parse import urlparse
            return urlparse(url).hostname
        except Exception:
            return None

    def _speaker_sink_for(self, room: str) -> str:
        """
        Return the routing key for a room's speaker: "local" or "node".
        Resolved from the new schema's `room.speaker.type`:
            usb_device_spk / wyze_ssh_aplay / none → "local"
                (all run from the orchestrator process — Wyze SSH speaks
                directly from here without going through the ESP MQTT bus)
            esp32_i2s_spk → "node"
                (audio is published over MQTT for the ESP firmware to play)
        Unknown / missing rooms default to "local".
        """
        for room_cfg in self.config.get("rooms", []):
            if room_cfg.get("id") != room:
                continue
            spk_cfg = room_cfg.get("speaker") or {}
            stype = str(spk_cfg.get("type", "")).lower() if isinstance(spk_cfg, dict) else ""
            if stype == "esp32_i2s_spk":
                return "node"
            return "local"
        return "local"

    # ── Unified scene snapshot ─────────────────────────────────────────────

    def get_scene_snapshot(self) -> dict:
        """Build a single dict describing the current state of the house:
        per-room hardware capabilities (mic / speaker / cam), latest
        observations (presence, posture, lights, last description),
        per-person sleep state, and the active room. Used by the LLM
        prompt and the dashboard's scene panel.

        Pure function over current state — no awaits, no I/O.
        """
        rooms_out: dict[str, dict] = {}
        for room_cfg in self.config.get("rooms", []):
            room_id = str(room_cfg.get("id", "unknown"))
            mic_cfg = room_cfg.get("mic") or {}
            spk_cfg = room_cfg.get("speaker") or {}
            vid_cfg = room_cfg.get("video") or {}
            mic_type = mic_cfg.get("type") if isinstance(mic_cfg, dict) else None
            spk_type = spk_cfg.get("type") if isinstance(spk_cfg, dict) else None
            vid_type = vid_cfg.get("type") if isinstance(vid_cfg, dict) else None

            # Local PC = the only path with cross-app audio ducking.
            # Cam-side speakers (wyze_ssh, esp32) play in isolation.
            local_pc_speaker = spk_type == "usb_device_spk"

            obs = self._scene_state.get(room_id, {})
            sleepers: list[dict] = []
            if self.sleep_tracker is not None:
                sleepers = self.sleep_tracker.get_sleepers_in(room_id)

            # Per-room mute / volume from RoomSettings
            tweaks: dict = {}
            if self.room_settings is not None:
                try:
                    tweaks = self.room_settings.get(room_id) or {}
                except Exception:
                    tweaks = {}

            rooms_out[room_id] = {
                "id": room_id,
                "display_name": room_cfg.get("display_name") or room_id,
                "is_active": room_id == self._active_user_room,
                "capabilities": {
                    "mic": mic_type,
                    "speaker": spk_type,
                    "camera": vid_type,
                    "ducks_pc_audio": local_pc_speaker,
                    "muted": bool(tweaks.get("muted", False)),
                    "volume": tweaks.get("volume"),
                },
                "observations": {
                    "lights_on": obs.get("lights_on"),
                    "person_present": obs.get("person_present"),
                    "person_name": obs.get("person_name"),
                    "posture": obs.get("posture"),
                    "description": obs.get("description"),
                    "objects": obs.get("objects"),
                    "updated_at": obs.get("updated_at"),
                },
                "sleepers": sleepers,
            }

        return {
            "active_room": self._active_user_room,
            "rooms": rooms_out,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def build_scene_extras(self) -> Optional[str]:
        """Render the unified scene as a multi-line string for injection
        into the LLM system prompt via PromptBuilder.extras['scene'].
        Returns None when nothing meaningful has been observed yet (avoid
        cluttering the prompt with empty-room boilerplate at boot).
        """
        snap = self.get_scene_snapshot()
        rooms = snap.get("rooms", {})
        if not rooms:
            return None
        lines = ["House scene snapshot (most recent observation per room):"]
        for room_id, info in rooms.items():
            caps = info["capabilities"]
            obs = info["observations"]
            sleepers = info.get("sleepers") or []

            cap_bits: list[str] = []
            if caps.get("camera"):
                cap_bits.append(f"cam={caps['camera']}")
            if caps.get("mic"):
                cap_bits.append(f"mic={caps['mic']}")
            if caps.get("speaker"):
                cap_bits.append(f"spk={caps['speaker']}")
            if caps.get("muted"):
                cap_bits.append("MUTED")
            cap_str = ", ".join(cap_bits) if cap_bits else "no hardware"

            obs_bits: list[str] = []
            if obs.get("lights_on") is True:
                obs_bits.append("lights on")
            elif obs.get("lights_on") is False:
                obs_bits.append("lights off")
            if obs.get("person_present"):
                who = obs.get("person_name") or "unknown person"
                obs_bits.append(f"{who} here")
                if obs.get("posture"):
                    obs_bits.append(f"posture={obs['posture']}")
            if sleepers:
                names = ", ".join(
                    s.get("person_name") or "someone" for s in sleepers
                )
                kinds = {s.get("kind") for s in sleepers}
                kind = "asleep" if "sleeping" in kinds else "napping"
                obs_bits.append(f"{names} {kind}")

            tag = " [active]" if info.get("is_active") else ""
            tail = " | " + "; ".join(obs_bits) if obs_bits else ""
            lines.append(f"  - {info['display_name']}{tag} ({cap_str}){tail}")
            if obs.get("description"):
                desc = str(obs["description"])[:140]
                lines.append(f"      last seen: {desc}")

        # Append the door-disappearance rule reminder so the LLM doesn't
        # forget it between turns.
        lines.append("")
        lines.append(
            "Note: if a known person disappears from a camera but didn't "
            "leave through a known door, assume they're still in the room. "
            "A blank frame is not an exit."
        )
        return "\n".join(lines)

    # ── Continuous-conversation follow-up listener ─────────────────────────

    async def _play_followup_beep(self, room: str) -> None:
        """Play the reply-window beep into `room` via its configured speaker
        sink, falling back to the PC chime if the room has no usable speaker.
        Same routing rule as wake-chime: if SpeakerManager has a real sink
        (anything but Null) for the room, push the chime there; otherwise
        the local PC speaker is the only option Cole will hear.
        """
        from modules.voice.audio_utils import chime_bytes, play_chime_async
        played_in_room = False
        try:
            if self.speaker_manager is not None:
                spk_type = self.speaker_manager.get_speaker_type(room)
                if spk_type not in ("none",):
                    played_in_room = await self.speaker_manager.play_chime(room)
        except Exception as e:
            logger.debug(f"[Followup] in-room beep for '{room}' raised: {e}")
            played_in_room = False
        if not played_in_room:
            try:
                await play_chime_async()
            except Exception as e:
                logger.debug(f"[Followup] PC fallback beep failed: {e}")

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

        # Pre-record cadence: 1s pause after Jarvis speech (lets reverb die +
        # gives natural pacing) → in-room beep so the user knows the mic is
        # hot wherever they are → 300ms gap so the beep itself doesn't bleed
        # into the recorded audio. Mirrors the wake-chime routing so a
        # bedroom followup beeps in the bedroom, not on the office PC.
        await asyncio.sleep(1.0)
        await self._play_followup_beep(room)
        await asyncio.sleep(0.3)

        # ── CAPTURE PHASE — mic-exclusive, lock held briefly ────────────────
        # Same lock-scope discipline as _on_wake_detected: hold the lock only
        # for the suspend → record → resume window so a real wake event during
        # processing can fire its own beep + capture in parallel.
        from modules.voice.audio_utils import (
            SAMPLE_RATE,
            record_until_silence,
            record_until_silence_from_chunks,
            db_from_rms,
        )
        audio_data = None
        async with self._wake_lock:
            self._followup_depth += 1
            was_audio_active = self._audio_io_active
            self._audio_io_active = True
            # Mirror the routing logic from _on_wake_detected: tap the
            # room's wake adapter when one exists, fall back to the office
            # PC mic only when there isn't one (i.e. office room).
            room_adapter = (
                self.wake_sources.get_source(room)
                if self.wake_sources is not None else None
            )
            use_room_tap = room_adapter is not None and hasattr(
                room_adapter, "attach_recording_tap"
            )
            logger.info(
                f"[Followup] lock acquired for '{room}' "
                f"(use_room_tap={use_room_tap}, depth={self._followup_depth})"
            )
            if self.wake and not use_room_tap:
                self.wake.suspend()
                logger.debug("[Followup] PC wake mic suspended")
                # Same dance the wake handler does: give wake's InputStream
                # time to actually close before we open ours. Without this
                # WASAPI on Windows can deny the second stream silently or
                # return zero-filled buffers for the first ~second.
                await asyncio.sleep(0.3)
            try:
                if not use_room_tap:
                    record_device = (
                        self.wake.device if self.wake
                        else recording_cfg.get("device")
                    )
                    pre_floor = (
                        self.wake.get_noise_floor_db(
                            fallback_db=recording_cfg.get(
                                "silence_threshold_db", -45.0
                            )
                        )
                        if self.wake
                        else recording_cfg.get("silence_threshold_db", -45.0)
                    )
                    logger.info(
                        f"[Followup] starting record_until_silence "
                        f"(device={record_device}, pre-floor={pre_floor:.1f} dBFS, "
                        f"speech_start_timeout={listen_seconds}s, "
                        f"max_dur={recording_cfg.get('max_duration_seconds', 60.0)}s)"
                    )
                    try:
                        audio_data = await asyncio.to_thread(
                            record_until_silence,
                            silence_threshold_db=pre_floor,
                            silence_duration_ms=recording_cfg.get(
                                "silence_duration_ms", 600
                            ),
                            max_duration_seconds=recording_cfg.get(
                                "max_duration_seconds", 60.0
                            ),
                            speech_start_timeout_seconds=listen_seconds,
                            device=record_device,
                            mode="silence",
                            adaptive_noise_floor=False,
                        )
                        logger.info(
                            f"[Followup] record_until_silence returned: "
                            f"type={type(audio_data).__name__}, "
                            f"len={len(audio_data) if audio_data is not None else 'N/A'}"
                        )
                    except Exception as rec_e:
                        # Catch HERE so the exception type is visible —
                        # without this, an AudioError raised by
                        # record_until_silence (e.g. portaudio failure)
                        # propagates past the inner try and the outer
                        # except at "Pipeline error" only sees the
                        # message, losing the type info that's diagnostic.
                        logger.warning(
                            f"[Followup] record_until_silence raised "
                            f"{type(rec_e).__name__}: {rec_e}"
                        )
                        audio_data = None
                else:
                    # Per-room adaptive floor (see _on_wake_detected): the
                    # room's wake adapter calibrates the silence threshold
                    # from its own rolling ambient. This is the fix for the
                    # office follow-up capturing 60 s of room noise and
                    # handing Whisper an empty transcript — office ambient
                    # (~-37 dBFS) sat above the old static -45 floor.
                    static_db = float(
                        recording_cfg.get("silence_threshold_db", -45.0)
                    )
                    floor_fn = getattr(
                        room_adapter, "get_noise_floor_db", None
                    )
                    if recording_cfg.get("adaptive_noise_floor", True) and floor_fn:
                        threshold_db = floor_fn(
                            margin_db=float(
                                recording_cfg.get("noise_floor_margin_db", 8.0)
                            ),
                            fallback_db=static_db,
                        )
                        logger.debug(
                            f"[Followup] room='{room}' adaptive floor: "
                            f"{threshold_db:.1f} dBFS "
                            f"(static fallback {static_db:.1f})"
                        )
                    else:
                        threshold_db = static_db
                        logger.debug(
                            f"[Followup] room='{room}' static floor: "
                            f"{threshold_db:.1f} dBFS"
                        )
                    assert room_adapter is not None  # use_room_tap implies non-None
                    try:
                        audio_data = await record_until_silence_from_chunks(
                            attach_tap=room_adapter.attach_recording_tap,
                            detach_tap=room_adapter.detach_recording_tap,
                            silence_threshold_db=threshold_db,
                            silence_duration_ms=recording_cfg.get(
                                "silence_duration_ms", 600
                            ),
                            max_duration_seconds=recording_cfg.get(
                                "max_duration_seconds", 60.0
                            ),
                            speech_start_timeout_seconds=listen_seconds,
                            chunk_sample_rate=getattr(
                                room_adapter, "recording_sample_rate", 16000
                            ),
                        )
                    except Exception as e:
                        logger.warning(
                            f"[Followup] Tap capture for room '{room}' "
                            f"failed: {e}"
                        )
                        audio_data = None
            finally:
                if self.wake and not use_room_tap:
                    self.wake.wakeup()
                self._audio_io_active = was_audio_active
        # ── LOCK RELEASED ───────────────────────────────────────────────────

        try:
            if audio_data is None:
                logger.info(
                    "[Followup] audio_data is None — recording call either "
                    "raised, returned None, or was never reached"
                )
                return
            if len(audio_data) == 0:
                logger.info(
                    "[Followup] audio_data is empty (0 samples) — recording "
                    "started but timed out before any speech was captured"
                )
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

            # Object-vocab branch: this reply is Cole answering the §23
            # "what's that object?" ask — name it or dismiss it.
            if self._pending_object_question is not None:
                obj_q = self._pending_object_question
                self._pending_object_question = None
                await self._complete_object_vocab_answer(
                    reply_transcript=transcript,
                    key=obj_q.get("key", ""),
                    room=obj_q.get("room") or room,
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

    # ── Unknown-sound review ───────────────────────────────────────────────

    async def _on_unknown_sound(self, payload: dict) -> None:
        """A voice.unknown_sound from the cascade — a VAD segment that was
        not a wake word, a watched safety event, or speech. Save a clip
        and note it for the dashboard Review tab. Throttled per room so a
        persistently noisy room cannot flood the store."""
        sv = getattr(self, "sound_vocab", None)
        if sv is None:
            return
        room = payload.get("room") or "?"
        segment = payload.get("segment")
        import time as _t
        now = _t.monotonic()
        # One capture per room per 20 s — enough to catch recurring
        # mystery sounds without logging every cough and chair scrape.
        if now - self._unknown_sound_last.get(room, 0.0) < 20.0:
            return
        self._unknown_sound_last[room] = now
        clip_path = None
        dur = 0.0
        if segment is not None and len(segment):
            dur = len(segment) / 16000.0
            clip_path = self._save_sound_clip(room, segment)
        sv.note_unknown(room, clip_path=clip_path, duration_s=dur)
        logger.info(
            f"[UnknownSound] unidentified sound in '{room}' ({dur:.1f}s)"
        )
        try:
            await self._broadcast({"type": "unknown_sound", "room": room})
        except Exception:
            pass

    def _save_sound_clip(self, room: str, segment) -> Optional[str]:
        """Write a VAD segment (float32 [-1,1] @ 16 kHz) to a mono 16-bit
        WAV under data/sound_snapshots/. Returns the path, or None."""
        try:
            import uuid
            import wave
            from datetime import datetime as _dt
            from pathlib import Path
            import numpy as _np

            out_dir = Path("data/sound_snapshots")
            out_dir.mkdir(parents=True, exist_ok=True)
            seg = _np.asarray(segment, dtype=_np.float32).flatten()
            pcm = _np.clip(seg * 32768.0, -32768, 32767).astype(_np.int16)
            path = out_dir / (
                f"unknown_{room}_{_dt.now().strftime('%Y%m%dT%H%M%S')}_"
                f"{uuid.uuid4().hex[:6]}.wav"
            )
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(pcm.tobytes())
            return str(path)
        except Exception as e:
            logger.debug(f"[UnknownSound] clip save failed: {e}")
            return None

    # ── §23 ask-to-learn object vocabulary ─────────────────────────────────

    async def _object_vocab_loop(self) -> None:
        """§23 ask-to-learn loop. Periodically checks the ObjectVocabLearner
        for a recurring un-tracked object worth asking Cole to name, and —
        when interruptibility allows — speaks the question. Cole's reply is
        caught by _listen_followup via the _pending_object_question flag."""
        learner = getattr(self, "object_vocab", None)
        if learner is None or not getattr(learner, "enabled", False):
            logger.info("[ObjectVocab] ask loop disabled (no learner)")
            return
        poll_s = float(
            ((self.config.get("vision", {}) or {})
             .get("object_vocab", {}) or {}).get("ask_poll_seconds", 90)
        )
        logger.info(f"[ObjectVocab] ask loop started (poll {poll_s:.0f}s)")
        while True:
            await asyncio.sleep(poll_s)
            try:
                # Waiting on an answer to a previous ask? Don't stack a
                # second one. But if it has gone unanswered well past the
                # follow-up window (Cole ignored the question, or the
                # capture caught nothing), clear it so the loop self-heals
                # instead of wedging the feature permanently.
                pending = self._pending_object_question
                if pending is not None:
                    import time as _time
                    age = _time.monotonic() - pending.get("asked_ts", 0.0)
                    if age < _OBJECT_VOCAB_ANSWER_TIMEOUT_S:
                        continue
                    logger.info(
                        f"[ObjectVocab] previous ask unanswered ({age:.0f}s)"
                        " — clearing so a new one can fire"
                    )
                    self._pending_object_question = None
                question = learner.pending_question()
                if not question:
                    continue
                if self._current_state is None:
                    continue  # no fused state yet — stay quiet
                # Same ambient-interruptibility gate the curiosity loop uses.
                if (self.interruptibility is not None
                        and not self.interruptibility.can_interrupt(
                            self._current_state, priority="ambient")):
                    logger.debug(
                        "[ObjectVocab] ask deferred — interruptibility gate"
                    )
                    continue
                await self._ask_about_unknown_object(question)
            except Exception as e:
                logger.error(f"[ObjectVocab] ask loop error: {e}")

    async def _ask_about_unknown_object(self, question: dict) -> None:
        """Speak a §23 'what's that object?' ask and arm the follow-up
        listener (via _pending_object_question) to route Cole's reply."""
        learner = getattr(self, "object_vocab", None)
        if learner is None:
            return
        key = question.get("key", "")
        # The object's room goes in the question TEXT; the question is
        # spoken to where Cole actually is (his active room), not into the
        # empty room the object happens to sit in.
        object_room = question.get("room") or "the house"
        target_room = self._active_user_room or "office"
        descriptor = question.get("descriptor") or {}
        yolo_class = str(descriptor.get("yolo_class") or "object")
        count = int(question.get("count", 0))
        # mark_asked first so the loop won't re-pick this one if the speak
        # path is slow; _pending_object_question routes the reply back.
        # asked_ts lets _object_vocab_loop time out an unanswered ask.
        import time as _time
        learner.mark_asked(key)
        self._pending_object_question = {
            "key": key, "room": target_room, "asked_ts": _time.monotonic(),
        }
        logger.info(
            f"[ObjectVocab] asking Cole (in '{target_room}') about "
            f"'{yolo_class}' seen in '{object_room}' ({count}x)"
        )
        line = await self._compose_in_character(
            prompt=(
                f"The vision system keeps seeing an object it labels "
                f"'{yolo_class}' in the {object_room} — about {count} times "
                f"— but it is not in your tracked-objects list. Ask Cole, "
                f"in one short in-character line, what it actually is. "
                f"No preamble, no quotes."
            ),
            fallback=(
                f"I keep noticing a {yolo_class} in the {object_room} — "
                f"what is that, exactly?"
            ),
        )
        await self._speak(
            line, room=target_room, priority="curiosity",
            expects_response=True,
        )

    async def _complete_object_vocab_answer(
        self, reply_transcript: str, key: str, room: str,
    ) -> None:
        """Cole answered the §23 'what's that?' ask. Dismiss the unknown
        if he waved it off, else record the name as a learned object so
        the open-vocab detector tracks it from now on."""
        learner = getattr(self, "object_vocab", None)
        if learner is None:
            return
        reply = (reply_transcript or "").strip()
        low = reply.lower()
        dismiss_cues = (
            "nothing", "never mind", "nevermind", "ignore it", "ignore that",
            "don't track", "dont track", "not important", "no idea",
            "leave it", "forget it", "skip it", "don't worry",
        )
        if not reply or any(cue in low for cue in dismiss_cues):
            learner.dismiss(key)
            logger.info(f"[ObjectVocab] dismissed unknown '{key}'")
            line = await self._compose_in_character(
                prompt=(
                    "Cole told you that recurring object is not worth "
                    "tracking. Acknowledge in one short in-character line."
                ),
                fallback="Noted — I'll stop watching that one.",
            )
            await self._speak(line, room=room, priority="conversation")
            return
        name = await self._extract_object_name_from_reply(reply)
        if not name:
            logger.info(f"[ObjectVocab] no object name parsed from {reply!r}")
            line = await self._compose_in_character(
                prompt=(
                    f"You could not make out an object name from Cole's "
                    f"reply: {reply!r}. Acknowledge briefly — you will ask "
                    f"again later if you keep seeing it. One short line."
                ),
                fallback="Hm, I didn't catch what that was — I'll ask later.",
            )
            await self._speak(line, room=room, priority="conversation")
            return
        entry = learner.record_answer(key, name)
        learned_name = (entry or {}).get("name", name)
        logger.info(f"[ObjectVocab] learned object '{learned_name}'")
        await self._broadcast({
            "type": "object_vocab_learned",
            "name": learned_name,
            "room": room,
        })
        line = await self._compose_in_character(
            prompt=(
                f"Cole just told you that recurring object is a "
                f"'{learned_name}'. You will now track it by name. Give one "
                f"short in-character line of acknowledgement. No quotes."
            ),
            fallback=f"Got it — a {learned_name}. I'll keep track of it now.",
        )
        await self._speak(line, room=room, priority="conversation")

    async def _extract_object_name_from_reply(
        self, reply: str,
    ) -> Optional[str]:
        """LLM-extract a short object name from Cole's reply, with a
        leading-filler-strip heuristic fallback. None when nothing
        plausible is found."""
        reply = (reply or "").strip()
        if not reply:
            return None
        if self.llm is not None:
            try:
                prompt = (
                    "Cole was asked what an object is. Extract a short "
                    "object name (1-3 words, lowercase, singular) from his "
                    "reply. Respond with ONLY the name — no punctuation, no "
                    "extra words. If there is no clear object, respond with "
                    f"the literal word: NONE.\n\nReply: {reply!r}"
                )
                resp = await self.llm.chat(
                    [{"role": "user", "content": prompt}]
                )
                candidate = (resp or "").strip().strip(".,;:!?'\"").lower()
                if (candidate and candidate.upper() != "NONE"
                        and len(candidate) <= 40
                        and any(c.isalpha() for c in candidate)):
                    return " ".join(candidate.split()[:3])
            except Exception as e:
                logger.debug(f"[ObjectVocab] name extraction failed: {e}")
        # Heuristic fallback: strip leading filler / articles / possessives,
        # then take up to three words.
        import re as _re
        cleaned = _re.sub(
            r"^\s*(it'?s|that'?s|it is|that is|just|probably)\b\s*",
            "", reply.lower(),
        )
        cleaned = _re.sub(
            r"^\s*(a|an|the|my|your|our|his|her)\b\s*", "", cleaned,
        ).strip(".,;:!?'\" ")
        if cleaned and any(c.isalpha() for c in cleaned):
            return " ".join(cleaned.split()[:3])
        return None

    # ── TTS Helper ─────────────────────────────────────────────────────────

    async def _play_tts(self, text: str, room: str) -> None:
        """Play TTS on the local speaker.

        voice.speech_start / voice.speech_end ALWAYS bracket playback —
        a CascadeWakeRunner keys off them for echo suppression so a room
        mic driving the cascade (e.g. the office mic) does not transcribe
        and triage Jarvis's own voice. With barge-in additionally
        enabled, playback runs as a cancellable tracked task so a
        voice.barge_in event can cut it off at the next sentence."""
        await self.bus.publish("voice.speech_start", {"room": room})
        try:
            if not self._barge_in_enabled:
                await self.tts.speak_async(text)
            else:
                task = asyncio.create_task(self.tts.speak_async(text))
                self._active_speech[room] = task
                try:
                    await task
                except asyncio.CancelledError:
                    logger.info(
                        f"[BargeIn] speech in '{room}' cut off by the user"
                    )
                finally:
                    if self._active_speech.get(room) is task:
                        self._active_speech.pop(room, None)
        finally:
            await self.bus.publish("voice.speech_end", {"room": room})

    async def _on_barge_in(self, event: dict) -> None:
        """A voice.barge_in event — the user talked over Jarvis. Cancel
        the in-flight TTS task for that room (if any)."""
        room = event.get("room", "")
        task = self._active_speech.get(room)
        if task is not None and not task.done():
            logger.info(f"[BargeIn] cancelling speech in '{room}'")
            task.cancel()

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
            # _active_user_room always carries a real room id ("office" by
            # default); the `or` is a type-narrowing belt so `room` is a
            # definite str for the rest of _speak (speaker routing, the
            # echo-suppression dict keys, the follow-up listener).
            room = room or "office"

            # Sleep deference: if anyone is napping/sleeping in the
            # target room, suppress proactive speech. Direct
            # 'conversation' replies still go through — if someone in
            # the bedroom said "hey jarvis" while Anna's napping there,
            # not responding would be worse than the disturbance. Other
            # priorities (ambient/curiosity/notification/reminder) are
            # gated. Whisper-mode is a future TODO.
            if (
                self.sleep_tracker is not None
                and self.sleep_tracker.is_anyone_sleeping_in(room)
                and priority != "conversation"
            ):
                sleepers = self.sleep_tracker.get_sleepers_in(room)
                names = ", ".join(
                    s.get("person_name") or "someone" for s in sleepers
                ) or "someone"
                logger.info(
                    f"[TTS] suppressed {priority} speech in '{room}' "
                    f"— {names} sleeping there: {text!r}"
                )
                return

            # Output leak filter — when persona system is wired, scrub
            # any hidden-persona name mentions before they reach TTS or
            # the dashboard log. The filter is a no-op when the active
            # persona is itself hidden (Cole's already in on it) or
            # when Cole is reliably alone. Costs ~one regex.subn per
            # utterance.
            if self.persona is not None:
                text = self.persona.filter_output(text)

            logger.info(f"[TTS] [{priority}] (→{room}) {text!r}")

            if not self.tts:
                logger.warning("[TTS] TTS module not initialized — skipping playback")
                return

            # Three-way routing:
            #   1. esp32_i2s_spk → MQTT to ESP node (legacy "node" path)
            #   2. wyze_ssh_aplay → SSH SFTP to cam, audioplay_t20 (new)
            #   3. anything else (usb_device_spk, none, missing config) →
            #      local PC sound device with audio_focus + classifier coordination
            # The local-PC path stays the default because that's where the
            # office speaker lives and it has the rich ducking/focus
            # integration; routing it through SpeakerManager would lose that.
            routed = False
            speaker_type = (
                self.speaker_manager.get_speaker_type(room)
                if self.speaker_manager is not None
                else "none"
            )
            # Echo suppression for non-local speakers: any room whose
            # mic will hear Jarvis's own voice (Wyze cam, ESP node)
            # needs the wake handler to ignore that room while we
            # speak + a short tail after. Local PC speech doesn't need
            # this — the office mic + _audio_io_active flag already
            # handle it. We stamp a far-future expiry up front, then
            # rewrite to "now + tail" once speech completes.
            import time as _time
            uses_room_speaker = speaker_type in ("esp32_i2s_spk", "wyze_ssh_aplay")
            if uses_room_speaker:
                self._room_speech_until[room] = _time.monotonic() + 60.0

            try:
                if speaker_type == "esp32_i2s_spk" and self.nodes is not None and self.nodes.is_online(room):
                    routed = await self._speak_via_node(text, room)
                elif speaker_type == "wyze_ssh_aplay":
                    routed = await self._speak_via_speaker_manager(text, room)
            finally:
                if uses_room_speaker:
                    self._room_speech_until[room] = (
                        _time.monotonic() + _ECHO_SUPPRESSION_TAIL_S
                    )

            if not routed:
                # Local playback. Streaming for multi-sentence; quick path for
                # one-liners. _audio_io_active blocks the audio classifier from
                # reading wake_word's buffer during our own playback. Audio focus
                # ducks other apps' volume so Jarvis isn't drowned out by music.
                was_audio_io_active = self._audio_io_active
                self._audio_io_active = True
                if self.audio_focus is not None and self.audio_focus.available:
                    await self.audio_focus.duck_async()
                try:
                    await self._play_tts(text, room)
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
                # One follow-up listener per room — the newest speech owns
                # the listen window (cancel_previous). Without this, two TTS
                # replies in the same room stack two listeners that then
                # queue behind the wake lock. _listen_followup acquires the
                # wake lock internally so it still serializes with wake
                # captures.
                self._bg_tasks.spawn(
                    self._listen_followup(room),
                    name=f"listen_followup:{room}",
                    policy=TaskPolicy(
                        singleton_key=f"followup:{room}",
                        cancel_previous=True,
                    ),
                )

        except Exception as e:
            logger.error(f"[TTS] Speak error: {e}")

    # ── Dashboard Broadcast ────────────────────────────────────────────────

    def _set_active_user_room(self, room: str) -> None:
        """Update the active-user-room tracker AND its timestamp atomically.

        Keep both fields in lockstep so the wake coalescer's freshness
        check (`now - ts < window`) is meaningful. Call this from any
        path that has a strong "Cole is in <room>" signal — wake events,
        face recognition, dashboard chat. Don't touch `_active_user_room`
        directly.
        """
        import time as _time
        self._active_user_room = room
        self._active_user_room_ts = _time.monotonic()

    def _spawn_scene_task(self, room_id: str, coro) -> None:
        """Spawn the scene pipeline for a room, at most one at a time.

        The vision loop reaches every room every cycle; the scene
        pipeline does 3-6s LLM calls. The singleton policy drops this
        cycle's coro if the prior one for the room is still running
        (one frame's lag on scene narration is invisible).
        """
        self._bg_tasks.spawn(
            coro,
            name=f"scene_bg:{room_id}",
            policy=TaskPolicy(singleton_key=f"scene:{room_id}"),
        )

    async def _run_scene_pipeline_bg(
        self,
        room_id: str,
        frame,
        detections: list,
        scene_persons,
        scene_person_states,
    ) -> None:
        """Run the heavy LLM portion of the per-room scene pipeline as
        a background task. Updates _scene_state[room_id]['description']
        and broadcasts the room_anomaly / room_messy events when fired.
        Failures are logged and swallowed — the room loop never sees
        them. Cole asked for this after Perf showed scene_llm at 3-6
        seconds per call, blocking the room loop."""
        if not self.scene_analyzer:
            return
        try:
            # Hard timeout so a wedged Ollama call can't pin this room's
            # singleton slot (see _spawn_scene_task) indefinitely.
            last_desc = await asyncio.wait_for(
                self.scene_analyzer.describe_async(
                    frame, room=room_id, objects=detections,
                    persons=scene_persons,
                    person_states=scene_person_states,
                ),
                timeout=20.0,
            )
        except Exception as e:
            logger.debug(
                f"[SceneBg] '{room_id}' describe_async failed: {e}"
            )
            return
        # Update cache + push a follow-up vision broadcast so the
        # dashboard description field is refreshed once the LLM call
        # completes (even though the room loop already broadcast the
        # cached version).
        if last_desc:
            cur = self._scene_state.get(room_id) or {}
            cur["description"] = last_desc
            cur["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._scene_state[room_id] = cur
            await self._broadcast({
                "type": "vision",
                "room": room_id,
                "description": last_desc,
            })
        # Baseline + anomaly + mess (each may hit the LLM again).
        if self.room_baselines and last_desc:
            try:
                if await self.room_baselines.needs_update(room_id):
                    await self.room_baselines.update(room_id, last_desc)
            except Exception as e:
                logger.debug(f"[SceneBg] '{room_id}' baseline failed: {e}")
            if (self.anomaly_detector is not None
                    and self.anomaly_detector.should_check(room_id)):
                try:
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
                except Exception as e:
                    logger.debug(f"[SceneBg] '{room_id}' anomaly failed: {e}")
            if (self.mess_detector is not None
                    and self.mess_detector.should_check(room_id)
                    and last_desc):
                try:
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
                except Exception as e:
                    logger.debug(f"[SceneBg] '{room_id}' mess failed: {e}")

    async def _broadcast(self, event: dict) -> None:
        """Send event to dashboard if enabled. Never blocks or raises."""
        if self.dashboard:
            try:
                await self.dashboard.broadcast(event)
            except Exception as e:
                logger.debug(f"[Dashboard] Broadcast error: {e}")

    async def _emit_issue(
        self,
        level: str,
        source: str,
        message: str,
        room: Optional[str] = None,
    ) -> None:
        """User-facing issue notification.

        Thin wrapper around the existing NotificationManager so callers
        don't need to know whether the inbox is wired in. Logs at the
        matching level (so the issue lands in the file log unconditionally),
        then persists + broadcasts via `self.notifications.notify()` —
        which feeds the bell-icon dropdown the dashboard already renders.

        Use for things Cole should know about as they happen — chime
        fallback when a room speaker is missing or rejects audio,
        recurring RTSP drops, etc. Don't use for routine status.

        level:   "info" | "warning" | "error" — passed through as
                 NotificationManager severity and used to pick log level
        source:  short kind identifier (e.g. "chime_fallback",
                 "rtsp_drop") — recorded as the notification kind
        message: human-readable explanation
        room:    optional room ID, surfaced in the title for context
        """
        log_line = f"[Issue/{source}]" + (f" ({room})" if room else "") + f" {message}"
        if level == "error":
            logger.error(log_line)
        elif level == "warning":
            logger.warning(log_line)
        else:
            logger.info(log_line)
        if self.notifications is None:
            # Inbox not wired (test harness, partial init). The log line
            # above is the only surface we can guarantee — that's fine.
            return
        title = source.replace("_", " ").title()
        if room:
            title = f"{title} — {room}"
        try:
            await self.notifications.notify(
                kind=source,
                title=title,
                message=message,
                severity=level,
            )
        except Exception as e:
            logger.debug(f"[Issue] notify dispatch failed: {e}")

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
