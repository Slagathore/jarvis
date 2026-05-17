"""
JARVIS — Brain / Personality
============================
Mission: PersonalityModel — a stateful temperament layer for the active
         persona (audit roadmap B3). Two independent, individually
         toggleable models:

         HEXACO — six STABLE personality traits (Honesty-Humility,
         Emotionality, eXtraversion, Agreeableness, Conscientiousness,
         Openness). Set per config, they do not drift; they describe who
         the persona fundamentally is.

         PAD — a REACTIVE mood in three axes (Pleasure, Arousal,
         Dominance, each -1..1). It is nudged by events (an anomaly
         raises arousal and lowers pleasure; a wake engages it) and
         continuously decays back toward a configured baseline. It
         describes how the persona feels right now.

         prompt_fragment() turns the current state into a short cue the
         PersonaManager appends to the system prompt — so the persona's
         replies are coloured by a real state machine instead of being
         re-improvised every turn. Either model can be switched off in
         config; off → it contributes nothing.

Modules: modules/brain/personality.py
Classes: PersonalityModel
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from loguru import logger

# HEXACO trait → (low descriptor, high descriptor). Emitted when a trait
# sits clearly below 0.35 or above 0.65; the mid-band says nothing.
_HEXACO_WORDS: dict[str, tuple[str, str]] = {
    "honesty_humility": ("self-serving and a little manipulative",
                         "sincere and unassuming"),
    "emotionality":     ("emotionally cool and unflappable",
                         "sensitive and easily moved"),
    "extraversion":     ("reserved and low-key", "outgoing and lively"),
    "agreeableness":    ("quick to criticize, slow to forgive",
                         "patient and forgiving"),
    "conscientiousness": ("loose and improvisational", "meticulous and precise"),
    "openness":         ("conventional and concrete",
                         "curious and intellectually playful"),
}

# PAD octant (sign of pleasure, arousal, dominance) → mood word
# (Mehrabian's emotion octants).
_PAD_OCTANTS: dict[tuple[bool, bool, bool], str] = {
    (True,  True,  True):  "exuberant",
    (True,  True,  False): "eager",
    (True,  False, True):  "relaxed and in control",
    (True,  False, False): "placid",
    (False, True,  True):  "hostile",
    (False, True,  False): "anxious",
    (False, False, True):  "coldly disdainful",
    (False, False, False): "bored",
}

# Bus topic → (pleasure, arousal, dominance) nudge applied on each event.
_EVENT_NUDGES: dict[str, tuple[float, float, float]] = {
    "world.anomaly":        (-0.20, 0.30, 0.05),   # concern
    "voice.wake_detected":  (0.05, 0.20, 0.0),     # engaged
    "voice.sound_event":    (-0.10, 0.35, 0.0),    # something happened
    "alarm.fired":          (-0.30, 0.55, 0.10),   # alarm
}


def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


class PersonalityModel:
    """Stable HEXACO traits + a reactive PAD mood. Both toggleable.

    Config (from config["personality"], all optional):
        pad_enabled / hexaco_enabled : bool master switches
        hexaco:        {trait: 0..1} stable trait values
        pad_baseline:  {pleasure, arousal, dominance} the mood decays to
        pad_decay:     fraction of the gap to baseline closed per tick (0.15)
        decay_interval_s: seconds between decay ticks (60)
    """

    def __init__(self, bus: Any = None, config: Optional[dict] = None) -> None:
        cfg = config or {}
        self._bus = bus
        self.pad_enabled = bool(cfg.get("pad_enabled", True))
        self.hexaco_enabled = bool(cfg.get("hexaco_enabled", True))
        self._decay = float(cfg.get("pad_decay", 0.15))
        self._decay_interval_s = float(cfg.get("decay_interval_s", 60))

        hx = cfg.get("hexaco", {}) or {}
        self.hexaco: dict[str, float] = {
            t: _clamp(float(hx.get(t, 0.5)), 0.0, 1.0) for t in _HEXACO_WORDS
        }
        base = cfg.get("pad_baseline", {}) or {}
        self._baseline = {
            "pleasure": _clamp(float(base.get("pleasure", 0.0))),
            "arousal": _clamp(float(base.get("arousal", 0.0))),
            "dominance": _clamp(float(base.get("dominance", 0.0))),
        }
        # Current mood starts at baseline.
        self.pad: dict[str, float] = dict(self._baseline)
        self._subs: list[Any] = []
        self._decay_task: Optional[asyncio.Task] = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Subscribe for auto-nudging and start the decay loop. No-op if
        PAD is disabled (HEXACO is static — it needs no runtime)."""
        if not self.pad_enabled:
            logger.info("[Personality] PAD disabled — mood is static")
            return
        if self._bus is not None:
            for topic, nudge in _EVENT_NUDGES.items():
                try:
                    self._subs.append(
                        self._bus.subscribe(topic, self._handler_for(nudge)))
                except Exception:
                    pass
        self._decay_task = asyncio.create_task(
            self._decay_loop(), name="personality:decay")
        logger.info(
            f"[Personality] started (PAD on, HEXACO "
            f"{'on' if self.hexaco_enabled else 'off'})"
        )

    async def stop(self) -> None:
        for sub in self._subs:
            try:
                sub.unsubscribe()
            except Exception:
                pass
        self._subs.clear()
        if self._decay_task is not None and not self._decay_task.done():
            self._decay_task.cancel()
            try:
                await self._decay_task
            except (asyncio.CancelledError, Exception):
                pass
            self._decay_task = None

    async def _decay_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._decay_interval_s)
                self.decay()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.debug("[Personality] decay tick failed")

    # ── Mood dynamics ────────────────────────────────────────────────────────

    def nudge(self, pleasure: float = 0.0, arousal: float = 0.0,
              dominance: float = 0.0) -> None:
        """Shift the current PAD mood. Clamped to [-1, 1]."""
        if not self.pad_enabled:
            return
        self.pad["pleasure"] = _clamp(self.pad["pleasure"] + pleasure)
        self.pad["arousal"] = _clamp(self.pad["arousal"] + arousal)
        self.pad["dominance"] = _clamp(self.pad["dominance"] + dominance)

    def decay(self) -> None:
        """Pull the mood a fraction of the way back toward baseline."""
        if not self.pad_enabled:
            return
        for axis, base in self._baseline.items():
            self.pad[axis] = round(
                self.pad[axis] + (base - self.pad[axis]) * self._decay, 4)

    def _handler_for(self, nudge: tuple[float, float, float]):
        """Build a bus handler that applies a fixed PAD nudge. One per
        topic — the closure captures the topic's nudge so a single shared
        handler does not have to recover the topic from the payload."""
        async def _handler(_payload: dict) -> None:
            self.nudge(*nudge)
        return _handler

    def apply_event(self, topic: str) -> None:
        """Explicit event nudge (for callers that know the topic)."""
        if topic in _EVENT_NUDGES:
            self.nudge(*_EVENT_NUDGES[topic])

    # ── Prompt projection ────────────────────────────────────────────────────

    def _mood_word(self) -> str:
        octant = (self.pad["pleasure"] >= 0.0,
                  self.pad["arousal"] >= 0.0,
                  self.pad["dominance"] >= 0.0)
        return _PAD_OCTANTS.get(octant, "even-tempered")

    def _hexaco_phrase(self) -> str:
        bits: list[str] = []
        for trait, value in self.hexaco.items():
            low, high = _HEXACO_WORDS[trait]
            if value <= 0.35:
                bits.append(low)
            elif value >= 0.65:
                bits.append(high)
        return "; ".join(bits)

    def prompt_fragment(self) -> str:
        """A short temperament cue for the system prompt. Empty string
        when both models are off (then it adds nothing to the prompt)."""
        lines: list[str] = []
        if self.hexaco_enabled:
            phrase = self._hexaco_phrase()
            if phrase:
                lines.append(f"Temperament: you are {phrase}.")
        if self.pad_enabled:
            lines.append(
                f"Right now you feel {self._mood_word()}. Let that colour "
                f"your tone — do not state it outright."
            )
        return "\n".join(lines)

    def snapshot(self) -> dict:
        """Dashboard view of the live personality state."""
        return {
            "pad_enabled": self.pad_enabled,
            "hexaco_enabled": self.hexaco_enabled,
            "pad": dict(self.pad),
            "mood": self._mood_word() if self.pad_enabled else None,
            "hexaco": dict(self.hexaco),
        }
