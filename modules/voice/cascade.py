"""
JARVIS — Ambient Home AI
========================
Mission: VoiceCascade — the coordinator that turns the always-on
         microphone into a cheap-stages-gate-expensive-stages decision
         pipeline (Option D):

           Stage 0  always-on mic        (caller feeds chunks)
           Stage 1  VAD                  — is there speech-like audio?
           Stage 2a wake-word            — did they say "Hey Jarvis"?
           Stage 2b sound-event          — alarm / glass / cry / siren?
           Stage 3  STT                  — what did they actually say?
           Stage 4  triage LLM           — is this for Jarvis?

         The cascade has two jobs:
           feed_chunk()      — VAD-segments the stream, with an 800 ms
                               pre-roll so the first word is never
                               clipped; yields a complete segment when
                               speech ends.
           evaluate_segment()— runs Stages 2a-4 on that segment and
                               returns a CascadeDecision.

         Components are injected (VAD, sound classifier, STT, triage,
         wake detector) so the cascade stays decoupled and unit-testable.
         A missing/unloaded component is skipped — the cascade degrades
         to whatever stages are available.

Modules: modules/voice/cascade.py
Classes: CascadeAction, CascadeDecision, VoiceCascade
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from loguru import logger

from modules.voice.vad import VAD_SAMPLE_RATE


class CascadeAction:
    """Terminal decisions the cascade can reach for one audio segment."""

    DROP = "drop"                       # nothing actionable
    WAKE = "wake"                       # wake word fired → full assistant
    EVENT = "event"                     # watched sound event → safety path
    TRIAGE_ESCALATE = "triage_escalate"  # triage said this is for Jarvis
    LOG_ONLY = "log_only"               # transcribed, but ambient — memory only


@dataclass
class CascadeDecision:
    """Outcome of evaluating one segment. `action` is a CascadeAction."""

    action: str
    transcript: str = ""
    event_category: str = ""
    stage: str = ""                     # which stage produced the decision
    detail: dict = field(default_factory=dict)


class VoiceCascade:
    """Staged voice-pipeline coordinator. See module docstring.

    Config keys (from config["voice"]["cascade"], all optional):
        preroll_ms:      audio kept before VAD opens (default 800)
        max_segment_s:   hard cap on one segment's length (default 20)
        min_segment_ms:  segments shorter than this are dropped (default 250)
    """

    def __init__(
        self,
        *,
        vad: Any = None,
        sound_classifier: Any = None,
        stt: Any = None,
        triage_gate: Any = None,
        wake_detector: Optional[Callable[[np.ndarray], bool]] = None,
        config: Optional[dict] = None,
    ) -> None:
        cfg = (config or {})
        self._vad = vad
        self._sounds = sound_classifier
        self._stt = stt
        self._triage = triage_gate
        self._wake_detector = wake_detector

        self._preroll_n = int(VAD_SAMPLE_RATE * cfg.get("preroll_ms", 800) / 1000)
        self._max_segment_n = int(VAD_SAMPLE_RATE * cfg.get("max_segment_s", 20))
        self._min_segment_n = int(VAD_SAMPLE_RATE * cfg.get("min_segment_ms", 250) / 1000)

        # Segmentation state
        self._preroll = np.zeros(0, dtype=np.float32)
        self._segment: list[np.ndarray] = []
        self._in_segment = False

    # ── Stage 0/1: stream segmentation ───────────────────────────────────────

    def feed_chunk(self, chunk: np.ndarray) -> Optional[np.ndarray]:
        """Feed one mic chunk (float32 mono 16 kHz). Returns a complete
        speech segment (pre-roll + speech) when VAD closes a segment,
        else None. The pre-roll ring buffer means the segment includes
        ~800 ms of audio from *before* VAD opened, so the leading word
        is never clipped."""
        chunk = np.asarray(chunk, dtype=np.float32).flatten()
        if chunk.size == 0:
            return None

        speech = self._vad.update(chunk) if self._vad is not None else True

        if speech:
            if not self._in_segment:
                # Open a segment, seeding it with the pre-roll buffer.
                self._in_segment = True
                self._segment = [self._preroll.copy()]
            self._segment.append(chunk)
            if self._segment_len() >= self._max_segment_n:
                return self._close_segment()  # safety cap on a runaway segment
            return None

        # Not speech — keep the rolling pre-roll buffer fresh.
        self._push_preroll(chunk)
        if self._in_segment:
            return self._close_segment()
        return None

    def _push_preroll(self, chunk: np.ndarray) -> None:
        buf = np.concatenate([self._preroll, chunk])
        self._preroll = buf[-self._preroll_n:] if buf.size > self._preroll_n else buf

    def _segment_len(self) -> int:
        return sum(int(c.size) for c in self._segment)

    def _close_segment(self) -> Optional[np.ndarray]:
        seg = np.concatenate(self._segment) if self._segment else np.zeros(0, np.float32)
        self._segment = []
        self._in_segment = False
        if seg.size < self._min_segment_n:
            return None  # too short to be real speech
        return seg

    def reset(self) -> None:
        """Drop any in-progress segment and pre-roll — call between streams."""
        self._preroll = np.zeros(0, dtype=np.float32)
        self._segment = []
        self._in_segment = False
        if self._vad is not None:
            self._vad.reset()

    # ── Stages 2a-4: segment evaluation ──────────────────────────────────────

    async def evaluate_segment(
        self, audio: np.ndarray, *, room: str = ""
    ) -> CascadeDecision:
        """Run the staged decision flow on one complete speech segment.
        Each stage short-circuits: the first one to fire wins."""
        audio = np.asarray(audio, dtype=np.float32).flatten()
        if audio.size == 0:
            return CascadeDecision(CascadeAction.DROP, stage="empty")

        # Stage 2a — wake word.
        if self._wake_detector is not None:
            try:
                if self._wake_detector(audio):
                    logger.debug(f"[Cascade] wake word ({room or 'room?'})")
                    return CascadeDecision(CascadeAction.WAKE, stage="wake")
            except Exception as e:
                logger.debug(f"[Cascade] wake detector failed: {e}")

        # Stage 2b — watched sound event (alarm / glass / cry / siren).
        if self._sounds is not None and getattr(self._sounds, "loaded", False):
            try:
                events = await self._sounds.detect_events_async(audio)
                if events:
                    top = events[0]
                    logger.info(
                        f"[Cascade] sound event: {top['category']} "
                        f"({top['class_name']} {top['score']:.2f})"
                    )
                    return CascadeDecision(
                        CascadeAction.EVENT, event_category=top["category"],
                        stage="sound_event", detail={"events": events})
            except Exception as e:
                logger.debug(f"[Cascade] sound classify failed: {e}")

        # Stage 3 — STT.
        transcript = ""
        if self._stt is not None and getattr(self._stt, "loaded", False):
            try:
                transcript = (await self._stt.transcribe_async(audio)) or ""
            except Exception as e:
                logger.debug(f"[Cascade] STT failed: {e}")
        if not transcript.strip():
            return CascadeDecision(CascadeAction.DROP, stage="stt_empty")

        # Stage 4 — triage LLM. Fail-closed inside the gate: on any
        # failure it returns escalate=False, so we land on LOG_ONLY.
        if self._triage is not None and getattr(self._triage, "enabled", False):
            try:
                verdict = await self._triage.should_escalate(transcript)
                if verdict.get("escalate"):
                    logger.info(f"[Cascade] triage ESCALATE — {transcript!r}")
                    return CascadeDecision(
                        CascadeAction.TRIAGE_ESCALATE, transcript=transcript,
                        stage="triage", detail=verdict)
            except Exception as e:
                logger.debug(f"[Cascade] triage failed: {e}")

        # Transcribed but ambient — keep it for memory, do not wake Jarvis.
        return CascadeDecision(CascadeAction.LOG_ONLY, transcript=transcript,
                               stage="triage")
