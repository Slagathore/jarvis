"""
JARVIS — Ambient Home AI
========================
Mission: ConversationMixin — extracted from core/orchestrator.py (audit
         roadmap D6 decomposition). Wake handling, user-text processing, conversational intents.

         Mixed into Orchestrator; every `self.*` resolves against the
         concrete Orchestrator instance at runtime. The full
         orchestrator import block is duplicated below on purpose —
         over-importing is harmless and removes any missing-name risk.

Modules: core/orchestrator_conversation.py
Classes: ConversationMixin
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


class ConversationMixin(OrchestratorMixin):
    """Wake handling, user-text processing, conversational intents.

    Mixed into Orchestrator — see core/orchestrator_base.py.
    """

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
        """Forward periodic mic dBFS readings from wake_word + wake adapters
        to the dashboard. Wake_word emits {room, db}; wake adapters emit
        the richer {room, db, peak_db, sample_rate}. Forward whatever's
        present so the bar widgets see both fields when available."""
        from modules.context.perf_tracker import perf
        room = event.get("room", "office")
        perf().increment(f"audio_level.{room}")
        payload = {
            "type": "audio_level",
            "room": room,
            "db": event.get("db", -100.0),
        }
        if "peak_db" in event:
            payload["peak_db"] = event["peak_db"]
        if "sample_rate" in event:
            payload["sample_rate"] = event["sample_rate"]
        await self._broadcast(payload)

    async def _on_wake_score(self, event: dict) -> None:
        """Forward low-rate wake-word scores to the dashboard calibration UI."""
        await self._broadcast({
            "type": "wake_score",
            "room": event.get("room", "office"),
            "model": event.get("model", ""),
            "score": float(event.get("score", 0.0) or 0.0),
            "sensitivity": float(event.get("sensitivity", 0.5) or 0.5),
        })

    # ── Wake Word + Conversation Pipeline ─────────────────────────────────

    async def _on_wake_event_raw(self, event: dict) -> None:
        """Coalesce concurrent wake-detection events. When Cole says
        "Hey Jarvis" between two cams, every mic that hears it fires its
        own voice.wake_detected — without coalescing the orchestrator
        would start the capture pipeline once per mic and have to serialize
        them behind _wake_lock. Instead, batch all events for a short
        window after the first one arrives and dispatch ONE capture.

        Winner selection (see _fire_pending_wake_after):
          - Default: highest OWW confidence wins.
          - Vision-disambiguation tiebreaker: if `_active_user_room` was
            updated within the last 30s AND is one of the candidates,
            that room wins regardless of OWW confidence (the studio-mic
            in the office can outscore Wyze mics elsewhere even when
            Cole is clearly elsewhere — vision presence is a stronger
            localizer than raw mic confidence in that case).

        The coalesce window is set in config.voice.wake_word.coalesce_window_ms
        (default 250 ms). Office-first wakes get a longer guard window when
        vision recently saw Cole elsewhere, because the office studio mic can
        hear and fire before a Wyze room mic gets its vote onto the bus.
        """
        # Per-room echo suppression: drop wakes from a room we just
        # spoke into. The Wyze/ESP path puts audio out the cam speaker,
        # so the SAME cam's mic hears Jarvis and tries to wake on him.
        room = event.get("room")
        if room is not None:
            import time as _time
            until = self._room_speech_until.get(room, 0.0)
            if _time.monotonic() < until:
                logger.debug(
                    f"[Wake] echo-suppressed: '{room}' is in post-speech "
                    f"quiet window"
                )
                return

        self._pending_wakes.append(event)
        if self._wake_window_task is None or self._wake_window_task.done():
            wake_cfg = self.config.get("voice", {}).get("wake_word", {}) or {}
            window_ms = int(wake_cfg.get("coalesce_window_ms", 250))
            prior = self._wake_room_prior()
            if (
                event.get("room") == "office"
                and prior is not None
                and prior[0] != "office"
            ):
                window_ms = max(
                    window_ms,
                    int(wake_cfg.get("office_bleed_guard_ms", 1800)),
                )
                logger.debug(
                    f"[Wake] office-first wake; holding {window_ms}ms for "
                    f"possible local room contender ({prior[0]} via {prior[1]})"
                )
            window_s = max(0.0, window_ms / 1000.0)
            self._wake_window_task = asyncio.create_task(
                self._fire_pending_wake_after(window_s)
            )

    def _wake_room_prior(self) -> Optional[tuple[str, str, float]]:
        """Return a recent non-audio room-localization prior for wake routing.

        Priority:
          1. `_active_user_room` from recognized face / dashboard chat.
          2. A single non-office room in `_scene_state` with recent YOLO
             person presence.

        The second path is deliberately conservative: if two non-office rooms
        both recently had a person, return None and let OWW confidence decide.
        """
        import time as _time

        wake_cfg = self.config.get("voice", {}).get("wake_word", {}) or {}
        now = _time.monotonic()
        active_freshness_s = float(wake_cfg.get("vision_prior_freshness_s", 90.0))
        if (
            self._active_user_room
            and (now - self._active_user_room_ts) < active_freshness_s
        ):
            return (
                self._active_user_room,
                "active",
                now - self._active_user_room_ts,
            )

        scene_freshness_s = float(wake_cfg.get("person_presence_prior_freshness_s", 90.0))
        candidates: list[tuple[str, float]] = []
        wall_now = datetime.now(timezone.utc)
        for room_id, obs in self._scene_state.items():
            if room_id == "office" or not obs.get("person_present"):
                continue
            updated_raw = obs.get("updated_at")
            if not isinstance(updated_raw, str):
                continue
            try:
                updated_at = datetime.fromisoformat(updated_raw)
            except ValueError:
                continue
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=timezone.utc)
            age_s = (wall_now - updated_at).total_seconds()
            if 0.0 <= age_s <= scene_freshness_s:
                candidates.append((room_id, age_s))
        if len(candidates) == 1:
            room_id, age_s = candidates[0]
            return (room_id, "vision_person", age_s)
        return None

    async def _fire_pending_wake_after(self, delay_s: float) -> None:
        """Sleeps the coalesce window, then fires the winning pending wake.

        Winner = max(score) where score = OWW confidence + vision-prior
        bonus. The vision-prior bonus applies when `_active_user_room`
        was updated recently (within VISION_PRIOR_FRESHNESS_S) and the
        candidate room matches it. Otherwise scoring is pure OWW
        confidence. See _on_wake_event_raw docstring for rationale.

        On cancellation (orchestrator shutting down) drops the pending
        list silently — the capture pipeline isn't safe to start mid-shutdown.
        """
        try:
            await asyncio.sleep(delay_s)
        except asyncio.CancelledError:
            self._pending_wakes = []
            self._wake_window_task = None
            return

        pending = self._pending_wakes
        self._pending_wakes = []
        self._wake_window_task = None
        if not pending:
            return

        # Vision/person-presence disambiguation: a recent room-local presence
        # signal is a stronger localizer than raw OWW confidence when mic
        # quality varies wildly between rooms. The office PC mic is much
        # louder/cleaner than Wyze room mics; without an override it keeps
        # winning arbitration even when Cole is clearly elsewhere on camera.
        wake_cfg = self.config.get("voice", {}).get("wake_word", {}) or {}
        prior = self._wake_room_prior()
        prior_room = prior[0] if prior is not None else None
        prior_source = prior[1] if prior is not None else None
        prior_age_s = prior[2] if prior is not None else None
        vision_prior_bonus = float(wake_cfg.get("vision_prior_bonus", 0.5))
        # When True, a fresh vision prior forces a hard override: if the
        # prior_room has ANY wake event in the coalesce window, it wins
        # regardless of OWW confidence. Other rooms drop to 0 effectively.
        # Configurable so users can revert to soft (additive) behavior.
        vision_prior_hard_override = bool(
            wake_cfg.get("vision_prior_hard_override", True)
        )
        # Minimum OWW confidence the prior_room must show before the hard
        # override kicks in. Set low so even a faint Wyze mic pickup wins,
        # but not so low that we route to a room that didn't actually
        # hear anything.
        hard_override_min_conf = float(
            wake_cfg.get("vision_prior_hard_min_confidence", 0.10)
        )

        prior_event = None
        if prior_room is not None and vision_prior_hard_override:
            for e in pending:
                if e.get("room") == prior_room:
                    if float(e.get("confidence", 0.0)) >= hard_override_min_conf:
                        prior_event = e
                        break

        def _score(e: dict) -> float:
            base = float(e.get("confidence", 0.0))
            if prior_room is not None and e.get("room") == prior_room:
                return base + vision_prior_bonus
            return base

        if prior_event is not None:
            # Hard override path — prior_room has a real wake; everything
            # else gets squashed.
            winner = prior_event
        else:
            # Vision says Cole is NOT in office, but only office heard the
            # wake. This is almost always a false trigger from PC speakers,
            # TV audio, or someone in living_room being picked up by the
            # studio mic. Drop it rather than fire in the wrong room.
            suppress_office_false_positive = bool(
                wake_cfg.get("suppress_office_false_positive", True)
            )
            if (suppress_office_false_positive
                    and prior_room is not None
                    and prior_room != "office"):
                non_office = [e for e in pending if e.get("room") != "office"]
                if not non_office:
                    logger.info(
                        f"[Wake] Suppressed office-only wake — vision puts "
                        f"Cole in '{prior_room}' (age={prior_age_s:.1f}s). "
                        f"office@{pending[0].get('confidence', 0):.3f} dropped "
                        f"as likely false trigger."
                    )
                    return
                # Vision elsewhere AND some non-office room also heard it
                # — let those compete on score, but exclude office.
                pending = non_office
            winner = max(pending, key=_score)
        if len(pending) > 1:
            # Per-room rundown so the log makes the choice obvious. Show
            # base confidence + (bonus) when the prior applied, so it's
            # visible WHY a lower-confidence room won.
            parts = []
            for e in pending:
                room_id = e.get("room", "?")
                conf = float(e.get("confidence", 0.0))
                bonus_tag = (
                    "+vision" if prior_room is not None and room_id == prior_room
                    else ""
                )
                parts.append(f"{room_id}@{conf:.3f}{bonus_tag}")
            rundown = ", ".join(parts)
            prior_age = (
                f", prior={prior_room}/{prior_source}@{prior_age_s:.1f}s"
                if prior_room and prior_age_s is not None else ""
            )
            override_tag = " [HARD-OVERRIDE]" if prior_event is not None else ""
            logger.info(
                f"[Wake] Coalesced {len(pending)} simultaneous wakes "
                f"[{rundown}{prior_age}] → winner '{winner.get('room')}'{override_tag}"
            )
        await self._on_wake_detected(winner)

    async def _on_triage_escalate(self, event: dict) -> None:
        """Voice cascade Stage 4 decided an un-waked utterance is actually
        for Jarvis. The transcript already exists — feed it straight into
        the normal text-utterance path (no chime, no re-record)."""
        room = event.get("room", "office")
        text = (event.get("text") or "").strip()
        if not text:
            return
        logger.info(f"[Cascade] triage-escalated utterance in {room}: {text!r}")
        try:
            await self._process_user_text(text, room)
        except Exception as e:
            logger.warning(f"[Cascade] triage escalation handling failed: {e}")

    async def _on_cascade_sound_event(self, event: dict) -> None:
        """Voice cascade Stage 2b detected a watched non-speech sound
        (alarm, glass, cry, siren). Log it and surface it on the
        dashboard. Routing into the alarm/safety dispatcher is a
        follow-up — for now this makes the detection visible."""
        room = event.get("room", "?")
        category = event.get("category", "?")
        logger.info(f"[Cascade] sound event '{category}' in {room}")
        try:
            await self._broadcast({
                "type": "sound_event", "room": room, "category": category,
                "detail": event.get("detail", {}),
            })
        except Exception as e:
            logger.debug(f"[Cascade] sound_event broadcast failed: {e}")

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
        self._set_active_user_room(room)
        # Wake = whoever spoke is awake. We don't yet know who (voice ID
        # happens after STT), so for now clear everyone in this room.
        # The post-identify path below will be more surgical.
        if self.sleep_tracker is not None:
            self.sleep_tracker.record_activity(room=room, signal="wake-word")

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
            record_until_silence_from_chunks,
        )
        audio_data = None
        async with self._wake_lock:
            was_audio_active = self._audio_io_active
            self._audio_io_active = True
            # Decide which mic owns this capture:
            #   - If the room has a wake adapter registered (Wyze RTSP /
            #     ESP MQTT mics), tap THAT adapter for chunks. Suspending
            #     the office PC's wake stream is unnecessary and just
            #     blinds office wake detection while we capture elsewhere.
            #   - If the room has no wake adapter, it's the office (the
            #     only room whose mic the PC WakeWordDetector owns).
            #     Existing sounddevice-based path with self.wake.suspend().
            room_adapter = (
                self.wake_sources.get_source(room)
                if self.wake_sources is not None else None
            )
            use_room_tap = room_adapter is not None and hasattr(
                room_adapter, "attach_recording_tap"
            )
            if self.wake and not use_room_tap:
                self.wake.suspend()
            try:
                # Chime routing: if this is a non-office wake AND the room
                # has a working speaker sink, push the chime to THAT room
                # so Cole hears it where he is, not faintly from the office.
                # Fall back to PC sounddevice when (a) the wake is in the
                # office, or (b) the room has no speaker configured (sink
                # type "none"), or (c) the in-room playback raises. Cases
                # (b) and (c) are surfaced as dashboard notifications —
                # the user explicitly wants to know when this happens
                # because the chime ends up in the wrong room.
                chimed_in_room = False
                if use_room_tap and self.speaker_manager is not None:
                    spk_type = self.speaker_manager.get_speaker_type(room)
                    if spk_type == "none":
                        await self._emit_issue(
                            level="warning",
                            source="chime_fallback",
                            room=room,
                            message=(
                                f"Wake fired in '{room}' but the room has "
                                "no speaker configured — chiming on the "
                                "PC speakers instead."
                            ),
                        )
                    else:
                        try:
                            # play_chime() takes the pre-staged fast path
                            # for sinks that support it (Wyze SSH today),
                            # cutting ~300-500 ms of SCP latency off the
                            # wake-chime → record-start sequence. Falls
                            # back to plain play() with the same PCM bytes
                            # for sinks that don't.
                            played = await self.speaker_manager.play_chime(room)
                            chimed_in_room = bool(played)
                            if not chimed_in_room:
                                await self._emit_issue(
                                    level="warning",
                                    source="chime_fallback",
                                    room=room,
                                    message=(
                                        f"In-room chime to '{room}' was "
                                        "rejected by the speaker driver — "
                                        "chiming on the PC speakers instead."
                                    ),
                                )
                        except Exception as e:
                            logger.warning(
                                f"[Wake] in-room chime to '{room}' failed: {e}"
                                " — falling back to PC chime"
                            )
                            await self._emit_issue(
                                level="error",
                                source="chime_fallback",
                                room=room,
                                message=(
                                    f"In-room chime to '{room}' raised: "
                                    f"{e}. Chiming on PC speakers instead."
                                ),
                            )
                if not chimed_in_room:
                    await play_chime_async()
                await asyncio.sleep(0.3)

                recording_cfg = self.config["voice"]["recording"]
                if not use_room_tap:
                    # Existing path: PC sounddevice mic with the wake
                    # detector's pre-calibrated noise floor.
                    record_device = (
                        self.wake.device if self.wake
                        else recording_cfg.get("device")
                    )
                    pre_floor = (
                        self.wake.get_noise_floor_db(
                            fallback_db=recording_cfg["silence_threshold_db"]
                        )
                        if self.wake
                        else recording_cfg["silence_threshold_db"]
                    )
                    logger.debug(
                        f"[Wake] using pre-calibrated floor: {pre_floor:.1f} dBFS"
                    )
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
                else:
                    # Wyze room: wake fired from a MicSourceWakeAdapter. Tap
                    # its chunk stream so we capture from the SAME mic that
                    # heard the wake, not the office PC mic. Without this,
                    # bedroom wake → office mic recording → STT picks up
                    # silence/distant noise, and Cole's actual followup is
                    # never transcribed (real bug observed 2026-05-09).
                    # No pre-wake floor available for non-office rooms
                    # (the wake detector isn't tracking ambient there like
                    # wake_word does in the office). Use the configured
                    # static threshold; works fine for Wyze RTSP audio
                    # which is consistently moderate.
                    threshold_db = float(
                        recording_cfg["silence_threshold_db"]
                    )
                    logger.debug(
                        f"[Wake] room='{room}' using static floor: "
                        f"{threshold_db:.1f} dBFS"
                    )
                    assert room_adapter is not None  # use_room_tap implies non-None
                    try:
                        audio_data = await record_until_silence_from_chunks(
                            attach_tap=room_adapter.attach_recording_tap,
                            detach_tap=room_adapter.detach_recording_tap,
                            silence_threshold_db=threshold_db,
                            silence_duration_ms=recording_cfg[
                                "silence_duration_ms"
                            ],
                            max_duration_seconds=recording_cfg[
                                "max_duration_seconds"
                            ],
                            speech_start_timeout_seconds=recording_cfg.get(
                                "speech_start_timeout_seconds",
                                5.0,
                            ),
                            chunk_sample_rate=getattr(
                                room_adapter, "recording_sample_rate", 16000
                            ),
                        )
                    except Exception as e:
                        logger.warning(
                            f"[Wake] Tap capture for room '{room}' "
                            f"failed: {e}"
                        )
                        audio_data = None
            finally:
                if self.wake and not use_room_tap:
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
                        # Now we know who spoke — clear their sleep
                        # state across all rooms (they may have an
                        # entry in another cam from earlier).
                        if self.sleep_tracker is not None:
                            self.sleep_tracker.record_activity(
                                person_id=match.person_id,
                                person_name=match.name,
                                signal="voice-id",
                            )
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

        # House-layout teaching: "this door goes to the kitchen" / "list
        # doors" / "forget all doors here". Persisted into door_map so a
        # future transit-inference pass can use them.
        if await self._try_layout_teach(text, room):
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

        # Unified scene snapshot — gives the LLM per-room hardware
        # capability + presence + sleep-state awareness in one block, so
        # it can reason about "is anyone in the bedroom?", "who's
        # napping?", "is the kitchen mic actually a thing here?" without
        # us inventing a tool call for each.
        try:
            scene_extra = self.build_scene_extras()
            if scene_extra:
                extras["scene"] = scene_extra
        except Exception as e:
            logger.debug(f"[Scene] build_scene_extras failed: {e}")

        if self.activity_history is not None and self._current_state is not None:
            try:
                blurb = await self.activity_history.summary_for_prompt(
                    self._current_state.activity
                )
                if blurb:
                    extras["activity_history"] = blurb
            except Exception as e:
                logger.debug(f"[ActivityHistory] prompt summary failed: {e}")

        # World Model snapshot — top-N currently-tracked entities + last
        # few state changes. Capped at ~200 tokens (8 entities, 3 events
        # by default) so it doesn't crowd out memories / scene / activity.
        # Skipped when the world is empty (no entities + no events) so
        # cold-boot prompts don't get a useless header.
        if self.world_model is not None:
            try:
                world_snapshot = await self.world_model.build_snapshot_for_prompt()
                if world_snapshot:
                    extras["world_snapshot"] = world_snapshot
            except Exception as e:
                logger.debug(f"[WorldModel] prompt snapshot failed: {e}")

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

        # Attach a FRESH snapshot from the wake room to the latest user
        # message. The LLM is multimodal (Gemini 3 / vision-capable
        # Ollama); giving it the actual scene at the moment of the
        # utterance is dramatically more useful than only the periodic
        # vision-loop description from up to a minute ago. We only attach
        # to the LATEST user message — historical turns stay text-only so
        # context doesn't bloat with stale images.
        await self._attach_room_snapshot(prompt_context, room)

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
            self._bg_tasks.spawn(
                self.memory.extract_from_turn(
                    user_text=text, assistant_text=response, room=room
                ),
                name=f"memory.extract:{room}",
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

    # ── House-layout teaching ─────────────────────────────────────────────

    def _resolve_room_phrase(self, phrase: str) -> Optional[str]:
        """Map a spoken phrase like "the kitchen" or "office" to a room id
        from config.yaml. Returns None if no obvious match — caller asks
        the user to try again rather than silently writing the wrong id.

        Match is loose on purpose: spaces vs underscores, partial substring
        ("bed" → "bedroom"), and config display_name matching all win.
        """
        if not phrase:
            return None
        needle = phrase.lower().strip().replace(" ", "_")
        rooms = self.config.get("rooms", []) or []
        # Pass 1: exact id match.
        for r in rooms:
            rid = str(r.get("id", "")).lower()
            if rid and rid == needle:
                return r["id"]
        # Pass 2: display_name (case-insensitive).
        bare = phrase.lower().strip()
        for r in rooms:
            disp = str(r.get("display_name", "")).lower()
            if disp and disp == bare:
                return r["id"]
        # Pass 3: substring match (single hit only — multiple = ambiguous,
        # better to reject than guess).
        candidates = []
        for r in rooms:
            rid = str(r.get("id", "")).lower()
            disp = str(r.get("display_name", "")).lower()
            if needle and (needle in rid or needle in disp.replace(" ", "_")):
                candidates.append(r["id"])
            elif bare and bare in disp:
                candidates.append(r["id"])
        if len(candidates) == 1:
            return candidates[0]
        return None

    async def _capture_pointing_xy(self, room: str) -> tuple[float, float]:
        """Snap a frame in `room` and pull the pointing wrist's normalized
        (x, y) from PostureAnalyzer. Returns (0.5, 0.5) if no pointing
        gesture is detected — frame center is a sensible fallback that
        the dashboard can render and the user can adjust visually later.
        """
        default = (0.5, 0.5)
        if self.cameras is None or self.posture is None:
            return default
        try:
            frame = await self.cameras.capture_frame_async(room)
        except Exception as e:
            logger.debug(f"[Layout] capture_frame for '{room}' failed: {e}")
            return default
        if frame is None:
            return default
        try:
            full = await self.posture.analyze_full_async(frame)
        except Exception as e:
            logger.debug(f"[Layout] posture analyze failed: {e}")
            return default
        if not full or full.get("gesture") != "pointing":
            return default
        # PostureAnalyzer doesn't currently expose the actual landmark
        # coordinates of the pointing wrist; only the gesture label. Until
        # we wire that through, returning the default still saves the
        # door entry — the user can refine via the dashboard. Logging
        # here so we don't silently lose the pointing context.
        logger.debug(
            f"[Layout] pointing detected in '{room}' but exact xy not "
            "yet exported by PostureAnalyzer — using frame center."
        )
        return default

    async def _try_layout_teach(self, text: str, room: str) -> bool:
        """Handle "this door goes to X" / "list doors" / "forget all
        doors". Returns True if handled (caller short-circuits the LLM).
        """
        from modules.voice.intents import parse_layout_command
        if self.door_map is None:
            return False
        intent = parse_layout_command(text)
        if intent is None:
            return False

        action = intent.get("action")

        if action == "list":
            doors = self.door_map.get_doors(room)
            if not doors:
                summary = f"No doors taught in {room} yet."
            else:
                lines = [f"{len(doors)} door(s) in {room}:"]
                for d in doors:
                    target = d.get("neighbor_room") or "unknown"
                    lines.append(
                        f"  • '{d.get('label', '?')}' → {target}"
                    )
                summary = "\n".join(lines)
            ack = await self._compose_in_character(
                prompt=(
                    f"Cole just asked you to list the doors taught for the "
                    f"current room '{room}'. Here is the data — speak a "
                    f"single short in-character paraphrase so it sounds "
                    f"natural, no preamble:\n{summary}"
                ),
                fallback=summary,
            )
            await self._speak(ack, room=room, priority="conversation")
            return True

        if action == "clear_all":
            n = await self.door_map.clear_room(room)
            ack = await self._compose_in_character(
                prompt=(
                    f"Cole just told you to forget every door in '{room}'. "
                    f"You cleared {n} entries. Speak a single short "
                    f"in-character acknowledgement, no preamble."
                ),
                fallback=f"Cleared {n} doors in {room}.",
            )
            await self._speak(ack, room=room, priority="conversation")
            return True

        if action == "clear_one":
            phrase = intent.get("room_phrase", "")
            entry = self.door_map.find_by_label(room, phrase)
            if entry is None:
                ack = await self._compose_in_character(
                    prompt=(
                        f"Cole asked you to forget the '{phrase}' door in "
                        f"'{room}', but no matching entry was taught. "
                        f"Speak a single short in-character note explaining "
                        f"there's nothing to remove, no preamble."
                    ),
                    fallback=f"No '{phrase}' door taught in {room}.",
                )
                await self._speak(ack, room=room, priority="conversation")
                return True
            await self.door_map.remove_door(room, str(entry.get("id", "")))
            ack = await self._compose_in_character(
                prompt=(
                    f"Cole told you to forget the '{phrase}' door in "
                    f"'{room}'. You removed it. Speak a single short "
                    f"in-character acknowledgement, no preamble."
                ),
                fallback=f"Forgot the {phrase} door in {room}.",
            )
            await self._speak(ack, room=room, priority="conversation")
            return True

        # action == "add"
        phrase = intent.get("room_phrase", "")
        neighbor_id = self._resolve_room_phrase(phrase)
        # Capture pointing xy in parallel-friendly order (sequential is
        # fine — single frame fetch + a fast pose pass).
        fx, fy = await self._capture_pointing_xy(room)
        label = phrase if phrase else "unlabeled"
        await self.door_map.add_door(
            room=room,
            label=label,
            neighbor_room=neighbor_id,
            fx=fx,
            fy=fy,
        )
        await self._broadcast({
            "type":          "layout_door_added",
            "room":          room,
            "label":         label,
            "neighbor_room": neighbor_id,
            "fx":            fx,
            "fy":            fy,
        })
        if neighbor_id is None:
            ack = await self._compose_in_character(
                prompt=(
                    f"Cole just taught you a new door in '{room}' that he "
                    f"called '{phrase}', but no room with that name exists "
                    f"in the config. You saved the door anyway, but you "
                    f"don't know where it leads yet. Speak a single short "
                    f"in-character acknowledgement that mentions the room "
                    f"name didn't match anything you know, no preamble."
                ),
                fallback=(
                    f"Saved a door in {room}, but '{phrase}' isn't a room "
                    f"I recognize."
                ),
            )
        else:
            ack = await self._compose_in_character(
                prompt=(
                    f"Cole just taught you a new door in '{room}' leading "
                    f"to '{neighbor_id}'. Speak a single short in-character "
                    f"acknowledgement, no preamble."
                ),
                fallback=f"Got it — door from {room} to {neighbor_id} saved.",
            )
        await self._speak(ack, room=room, priority="conversation")
        return True

    async def _on_text_chat(self, text: str, room: str = "office") -> None:
        # Dashboard chat = presence signal too — update active room.
        self._set_active_user_room(room)
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
