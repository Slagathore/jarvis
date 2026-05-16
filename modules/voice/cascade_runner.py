"""
JARVIS — Ambient Home AI
========================
Mission: CascadeWakeRunner — the per-room driver that runs the Option-D
         voice cascade against one WakeSource's audio stream.

         It is the cascade-enabled counterpart of MicWakeRunner. Two
         things run off the same 80 ms chunk stream:

         1. STREAMING wake-word — openWakeWord predicts every chunk, so
            "Hey Jarvis" fires with the same low latency as the legacy
            runner and the existing wake → record → respond flow takes
            over unchanged.

         2. The CASCADE — every chunk also feeds VoiceCascade for VAD
            segmentation. When a speech segment closes, it runs Stages
            2b-4 (sound-event → STT → triage). The outcome is published:
              EVENT            → voice.sound_event
              TRIAGE_ESCALATE  → voice.triage_escalate (orchestrator
                                 then runs it through _process_user_text)
              LOG_ONLY / DROP  → logged only

         For `wake_suppress_s` after any wake, cascade segments are
         skipped — the orchestrator owns that conversation, and the
         command half of a "Hey Jarvis ... <command>" utterance must not
         be independently triaged.

Modules: modules/voice/cascade_runner.py
Classes: CascadeWakeRunner
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

import numpy as np
from loguru import logger

from core.event_bus import EventBus
from modules.voice.cascade import CascadeAction, VoiceCascade
from modules.voice.vad import VoiceActivityDetector
from modules.voice.wake_source import OWW_CHUNK_SIZE, WakeSource


class CascadeWakeRunner:
    """Cascade-enabled per-room runner. Drop-in for MicWakeRunner — same
    room property, run()/stop() lifecycle — so WakeSourceManager can pick
    one or the other by the `voice.cascade.enabled` flag."""

    def __init__(
        self,
        config: dict,
        bus: EventBus,
        source: WakeSource,
        *,
        sound_classifier: Any = None,
        stt: Any = None,
        triage_gate: Any = None,
    ) -> None:
        voice_cfg = config["voice"]
        self._wake_cfg = voice_cfg["wake_word"]
        self._cascade_cfg = voice_cfg.get("cascade", {}) or {}
        self._vad_cfg = voice_cfg.get("vad", {}) or {}
        self._bus = bus
        self._source = source
        self._sound_classifier = sound_classifier
        self._stt = stt
        self._triage = triage_gate
        self._wake_suppress_s = float(self._cascade_cfg.get("wake_suppress_s", 30.0))

        self._oww: Optional[Any] = None
        self._vad = VoiceActivityDetector(self._vad_cfg)
        self._cascade: Optional[VoiceCascade] = None
        self._task: Optional[asyncio.Task] = None
        self._last_detection = 0.0
        self._last_score_emit = 0.0
        self._wake_suppress_until = 0.0
        self._running = False

    @property
    def room(self) -> str:
        return self._source.room

    def load(self) -> None:
        """Load the OWW model + VAD and build the cascade. Blocking."""
        from openwakeword.model import Model

        model_name = self._wake_cfg.get("model", "hey_jarvis")
        self._oww = Model(wakeword_models=[model_name], inference_framework="onnx")
        self._vad.load()
        self._cascade = VoiceCascade(
            vad=self._vad,
            sound_classifier=self._sound_classifier,
            stt=self._stt,
            triage_gate=self._triage,
            wake_detector=None,        # wake is handled streaming, below
            config=self._cascade_cfg,
        )
        logger.info(
            f"[CascadeWake] runner ready for room '{self.room}' "
            f"(wake='{model_name}', VAD={self._vad.backend})"
        )

    async def run(self) -> None:
        """Consume the source stream: streaming wake + cascade per chunk."""
        if self._oww is None:
            await asyncio.to_thread(self.load)
        sensitivity = float(self._wake_cfg.get("sensitivity", 0.5))
        cooldown = float(self._wake_cfg.get("cooldown_seconds", 2))
        self._running = True
        try:
            async for chunk in self._source.stream():
                if not self._running:
                    break
                if chunk is None:
                    continue
                if chunk.dtype != np.int16:
                    chunk = chunk.astype(np.int16)
                if chunk.shape != (OWW_CHUNK_SIZE,):
                    continue
                await self._streaming_wake(chunk, sensitivity, cooldown)
                # Feed the cascade in parallel (float32 in [-1, 1]).
                f32 = chunk.astype(np.float32) / 32768.0
                if self._cascade is not None:
                    segment = self._cascade.feed_chunk(f32)
                    if segment is not None:
                        await self._on_segment(segment)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"[CascadeWake:{self.room}] run loop error: {e}")
        finally:
            self._running = False

    async def _streaming_wake(
        self, chunk: np.ndarray, sensitivity: float, cooldown: float
    ) -> None:
        """openWakeWord on one chunk — unchanged behaviour vs MicWakeRunner."""
        oww = self._oww
        if oww is None:
            return
        try:
            preds = await asyncio.to_thread(oww.predict, chunk)
        except Exception as e:
            logger.debug(f"[CascadeWake:{self.room}] predict failed: {e}")
            return
        if isinstance(preds, tuple):
            preds = preds[0]
        if not isinstance(preds, dict):
            return
        now = time.monotonic()
        for model_name, score in preds.items():
            score_v = float(score)
            if now - self._last_score_emit >= 0.5:
                self._last_score_emit = now
                await self._bus.publish("voice.wake_score", {
                    "room": self.room, "model": model_name,
                    "score": score_v, "sensitivity": sensitivity,
                })
            if score_v < sensitivity:
                continue
            # Heard a wake — mute cascade triage so the orchestrator owns
            # this conversation (and the command half of a multi-segment
            # "Hey Jarvis ... <command>" is not separately triaged).
            self._wake_suppress_until = now + self._wake_suppress_s
            if (now - self._last_detection) < cooldown:
                continue
            self._last_detection = now
            logger.info(
                f"[CascadeWake] wake '{model_name}' "
                f"(score={score_v:.3f}) in room '{self.room}'"
            )
            await self._bus.publish("voice.wake_detected", {
                "room": self.room, "confidence": score_v, "model": model_name,
            })

    async def _on_segment(self, segment: np.ndarray) -> None:
        """A speech segment closed — run Stages 2b-4 and publish the outcome."""
        if time.monotonic() < self._wake_suppress_until:
            return  # within the post-wake window — orchestrator owns this
        cascade = self._cascade
        if cascade is None:
            return
        decision = await cascade.evaluate_segment(segment, room=self.room)
        if decision.action == CascadeAction.EVENT:
            logger.info(
                f"[CascadeWake:{self.room}] sound event -> "
                f"{decision.event_category}"
            )
            await self._bus.publish("voice.sound_event", {
                "room": self.room, "category": decision.event_category,
                "detail": decision.detail,
            })
        elif decision.action == CascadeAction.TRIAGE_ESCALATE:
            logger.info(
                f"[CascadeWake:{self.room}] triage escalate -> "
                f"{decision.transcript!r}"
            )
            await self._bus.publish("voice.triage_escalate", {
                "room": self.room, "text": decision.transcript,
            })
        elif decision.action == CascadeAction.LOG_ONLY:
            logger.debug(
                f"[CascadeWake:{self.room}] ambient (logged only) -> "
                f"{decision.transcript!r}"
            )
        # CascadeAction.DROP / WAKE — nothing to do here.

    def stop(self) -> None:
        self._running = False
        if self._task is not None and not self._task.done():
            self._task.cancel()
