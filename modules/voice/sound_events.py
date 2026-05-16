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
import csv
from typing import Any, Optional

import numpy as np
from loguru import logger

YAMNET_SAMPLE_RATE = 16000
_YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"

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


class SoundEventClassifier:
    """AudioSet sound-event classifier for cascade Stage 2b.

    Config keys (from config["voice"]["sound_events"], all optional):
        enabled:    bool — master switch (default True)
        device:     "cuda" | "cpu" — GPU or CPU inference (default cuda)
        threshold:  float — min class score to count as a hit (0.30)
        watchlist:  dict — overrides _DEFAULT_WATCHLIST entirely if given
        top_k:      int — how many raw classes classify() returns (5)
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", True))
        self._device = str(cfg.get("device", "cuda")).lower()
        self._threshold = float(cfg.get("threshold", 0.30))
        self._top_k = int(cfg.get("top_k", 5))
        self._watchlist: dict[str, list[str]] = {
            k: [s.lower() for s in v]
            for k, v in (cfg.get("watchlist") or _DEFAULT_WATCHLIST).items()
        }
        self._model: Any = None
        self._class_names: list[str] = []
        self.loaded = False

    # ── Loading ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load YAMNet (CPU-only). Blocking — call once at startup.
        Never raises: on failure the classifier reports loaded=False and
        detect_events() simply returns nothing."""
        if not self.enabled:
            logger.info("[SoundEvents] disabled by config")
            return
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
            if self._device == "cpu":
                # Force YAMNet off the GPU so it cannot contend with the
                # YOLO / Whisper torch stack for VRAM.
                try:
                    tf.config.set_visible_devices([], "GPU")
                except Exception:
                    pass
            gpus = tf.config.list_physical_devices("GPU")
            self._model = hub.load(_YAMNET_HANDLE)
            self._class_names = self._load_class_names()
            self.loaded = bool(self._class_names)
            if self.loaded:
                on_gpu = bool(gpus) and self._device != "cpu"
                logger.info(
                    f"[SoundEvents] YAMNet ready on "
                    f"{'GPU' if on_gpu else 'CPU'} "
                    f"({len(self._class_names)} classes, "
                    f"{len(self._watchlist)} watched categories)"
                )
        except Exception as e:
            logger.warning(f"[SoundEvents] load failed ({e}) — stage disabled")
            self.loaded = False

    def _load_class_names(self) -> list[str]:
        """YAMNet ships a CSV (index, mid, display_name) at class_map_path()."""
        try:
            import tensorflow as tf
            path = self._model.class_map_path().numpy().decode("utf-8")
            names: list[str] = []
            with tf.io.gfile.GFile(path) as f:
                for row in csv.DictReader(f):
                    names.append(row["display_name"])
            return names
        except Exception as e:
            logger.warning(f"[SoundEvents] class map load failed: {e}")
            return []

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
        Returns [{category, class_name, score}], strongest first."""
        hits: list[dict] = []
        for class_name, score in self.classify(audio):
            if score < self._threshold:
                continue
            low = class_name.lower()
            for category, needles in self._watchlist.items():
                if any(n in low for n in needles):
                    hits.append({
                        "category": category,
                        "class_name": class_name,
                        "score": round(score, 4),
                    })
                    break
        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits

    async def detect_events_async(self, audio: np.ndarray) -> list[dict]:
        """Non-blocking detect_events — inference runs in a thread."""
        return await asyncio.to_thread(self.detect_events, audio)
