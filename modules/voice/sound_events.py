"""
JARVIS — Ambient Home AI
========================
Mission: SoundEventClassifier — Stage 2b of the voice cascade. When the
         VAD gate opens but the wake word did NOT fire, the audio might
         still be something Jarvis must react to: a smoke alarm, glass
         breaking, a baby crying, a siren. This stage classifies the
         non-speech (or speech-adjacent) audio against the AudioSet
         ontology and reports any hit on a watched category.

         MODEL: Google's YAMNet via tensorflow_hub — a ~17 MB MobileNet
         trained on AudioSet's 521 sound classes. Device is config-driven
         (sound_events.device): "cuda" runs it on the GPU, "cpu" forces
         it off the GPU so it cannot contend with the YOLO/Whisper torch
         stack for VRAM. YAMNet is tiny either way.

         Watched categories map a Jarvis-meaningful event name to a set
         of AudioSet display-name substrings. Config can override the
         map and the per-category score threshold.

Modules: modules/voice/sound_events.py
Classes: SoundEventClassifier
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

import numpy as np
from loguru import logger

YAMNET_SAMPLE_RATE = 16000

# Jarvis-meaningful event → AudioSet display-name substrings (lowercased).
# A category fires if ANY of its substrings matches a class scoring above
# threshold. Categories are checked in declaration order and the first
# match wins, so the specific `fire_alarm` precedes the broad `alarm`.
# Substrings are deliberately broad — YAMNet's bare classes are "Alarm",
# "Beep, bleep", etc., so over-specific needles ("alarm clock") miss.
_DEFAULT_WATCHLIST: dict[str, list[str]] = {
    "fire_alarm":  ["fire alarm", "smoke detector", "smoke alarm"],
    "alarm":       ["alarm", "siren", "buzzer", "air horn", "foghorn",
                    "beep, bleep"],
    "glass_break": ["glass", "shatter"],
    "baby_cry":    ["baby cry", "infant cry", "crying, sobbing"],
    "scream":      ["screaming", "shout", "yell"],
    "dog":         ["bark", "howl", "growling", "bow-wow"],
    "knock":       ["knock", "doorbell", "ding-dong"],
}

# Natural-language prompts for the CLAP open-vocab fallback, one per
# watched category. CLAP scores a clip against these (free text — no
# retraining), so a sound outside YAMNet's 521 classes can still be
# recognised.
_CLAP_CATEGORY_PROMPTS: dict[str, str] = {
    "fire_alarm":  "a fire alarm or smoke detector alarm sounding",
    "alarm":       "an alarm, siren, or loud electronic buzzer",
    "glass_break": "glass shattering or a window breaking",
    "baby_cry":    "a baby or young child crying",
    "scream":      "a person screaming or yelling in distress",
    "dog":         "a dog barking or howling",
    "knock":       "knocking on a door or a doorbell ringing",
}

# Neutral anchors — if CLAP picks one of these over every category
# prompt, the sound is genuinely nothing Jarvis needs to act on.
_CLAP_ANCHORS: list[str] = [
    "ordinary quiet household background noise",
    "people talking in normal conversation",
    "music or television playing",
    "silence",
]


class SoundEventClassifier:
    """AudioSet sound-event classifier for cascade Stage 2b.

    Config keys (from config["voice"]["sound_events"], all optional):
        enabled:    bool — master switch (default True)
        device:     "cuda" | "cpu" — GPU or CPU inference (default cuda)
        threshold:  float — min class score to count as a hit (0.30)
        watchlist:  dict — overrides _DEFAULT_WATCHLIST entirely if given
        top_k:      int — how many raw classes classify() returns (5)
        clap_escalate_below: float — when YAMNet's top score is under this
                    and nothing watched fired, escalate to CLAP (0.35)
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", True))
        self._device = str(cfg.get("device", "cuda")).lower()
        self._threshold = float(cfg.get("threshold", 0.30))
        self._top_k = int(cfg.get("top_k", 5))
        self._clap_escalate_below = float(cfg.get("clap_escalate_below", 0.35))
        self._watchlist: dict[str, list[str]] = {
            k: [s.lower() for s in v]
            for k, v in (cfg.get("watchlist") or _DEFAULT_WATCHLIST).items()
        }
        self._model: Any = None
        self._class_names: list[str] = []
        self._clap: Any = None          # open-vocab fallback, attached later
        self.loaded = False

    def attach_clap(self, clap: Any) -> None:
        """Wire in a ClapClassifier as the open-vocab fallback. When
        YAMNet is confidently lost (no watched hit + a low top score),
        detect_events() asks CLAP for a free-text second opinion."""
        self._clap = clap

    # ── Loading ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Acquire the shared YAMNet model. Blocking — call once at
        startup. Never raises: on failure loaded=False and detect_events()
        returns nothing. The model is shared with AudioClassifier via
        yamnet_loader so YAMNet is loaded exactly once per process."""
        if not self.enabled:
            logger.info("[SoundEvents] disabled by config")
            return
        try:
            from modules.voice.yamnet_loader import load_yamnet
            self._model, self._class_names = load_yamnet(device=self._device)
            self.loaded = self._model is not None and bool(self._class_names)
            if self.loaded:
                logger.info(
                    f"[SoundEvents] ready "
                    f"({len(self._class_names)} classes, "
                    f"{len(self._watchlist)} watched categories)"
                )
            else:
                logger.warning("[SoundEvents] shared YAMNet unavailable "
                                "— stage disabled")
        except Exception as e:
            logger.warning(f"[SoundEvents] load failed ({e}) — stage disabled")
            self.loaded = False

    # ── Classification ───────────────────────────────────────────────────────

    def classify(self, audio: np.ndarray) -> list[tuple[str, float]]:
        """Return the top-K (class_name, score) for one audio clip.
        `audio` is float32 mono at 16 kHz in [-1, 1]."""
        if not self.loaded or audio.size == 0:
            return []
        wav = np.asarray(audio, dtype=np.float32).flatten()
        try:
            scores, _, _ = self._model(wav)  # also yields embeddings, spectrogram
            # scores is (frames, 521); a sound that occurs in any frame
            # matters, so reduce across frames with max, not mean.
            clip = np.max(scores.numpy(), axis=0)
        except Exception as e:
            logger.debug(f"[SoundEvents] inference failed: {e}")
            return []
        top = np.argsort(clip)[::-1][: self._top_k]
        return [(self._class_names[i], float(clip[i])) for i in top]

    def detect_events(self, audio: np.ndarray) -> list[dict]:
        """Classify, then report any watched category that fired.
        Returns [{category, class_name, score, source}], strongest first.

        If YAMNet finds nothing watched AND its own top score is low
        (it is confidently lost), escalate to the open-vocab CLAP
        fallback for a free-text second opinion."""
        raw = self.classify(audio)
        hits: list[dict] = []
        for class_name, score in raw:
            if score < self._threshold:
                continue
            low = class_name.lower()
            for category, needles in self._watchlist.items():
                if any(n in low for n in needles):
                    hits.append({
                        "category": category, "class_name": class_name,
                        "score": round(score, 4), "source": "yamnet",
                    })
                    break
        if hits:
            hits.sort(key=lambda h: h["score"], reverse=True)
            return hits

        # Nothing watched fired. If YAMNet was also UNCERTAIN (low top
        # score) the sound is outside its 521-class vocabulary — ask CLAP.
        yamnet_top = raw[0][1] if raw else 0.0
        if (self._clap is not None and getattr(self._clap, "loaded", False)
                and yamnet_top < self._clap_escalate_below):
            clap_hit = self._clap_escalate(audio)
            if clap_hit is not None:
                hits.append(clap_hit)
        return hits

    def _clap_escalate(self, audio: np.ndarray) -> Optional[dict]:
        """Run CLAP against descriptive prompts for each watched category
        plus neutral anchors. Returns a hit dict only if CLAP picks a
        category prompt over the anchors with enough confidence."""
        prompts = list(_CLAP_CATEGORY_PROMPTS.items())     # (category, text)
        labels = [text for _, text in prompts] + _CLAP_ANCHORS
        try:
            match = self._clap.best_match(audio, labels, src_rate=16000)
        except Exception as e:
            logger.debug(f"[SoundEvents] CLAP escalation failed: {e}")
            return None
        if match is None:
            return None
        label, score = match
        for category, text in prompts:
            if text == label:      # CLAP picked a real category, not an anchor
                logger.info(
                    f"[SoundEvents] CLAP escalation -> {category} "
                    f"({score:.2f}) — YAMNet was uncertain"
                )
                return {
                    "category": category, "class_name": label,
                    "score": round(float(score), 4), "source": "clap",
                }
        return None  # CLAP picked a neutral anchor — genuinely nothing

    async def detect_events_async(self, audio: np.ndarray) -> list[dict]:
        """Non-blocking detect_events — inference runs in a thread."""
        return await asyncio.to_thread(self.detect_events, audio)
