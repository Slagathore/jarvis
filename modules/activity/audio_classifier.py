"""
JARVIS — Ambient Home AI
========================
Mission: Classify ambient sounds in the home using Google's YAMNet model
         to identify running appliances, activity sounds, and environmental
         audio events. Records a short audio window, runs it through YAMNet
         (via TensorFlow Hub), and returns ranked sound categories mapped
         to Jarvis-relevant labels (washer, dryer, dishwasher, TV, etc.).

         YAMNet classifies 521 AudioSet sound classes. We filter to the
         Jarvis-relevant subset and emit structured signal dicts.

Modules: modules/activity/audio_classifier.py
Classes: AudioClassifier
Functions:
    AudioClassifier.__init__(config)    — Initialize with config
    AudioClassifier.load()              — Load YAMNet from TensorFlow Hub
    AudioClassifier.classify()          — Blocking: record and classify audio
    AudioClassifier.classify_async()    — Async wrapper
    AudioClassifier._record_window()    — Record N seconds of audio via sounddevice
    AudioClassifier._run_yamnet()       — Run inference on the waveform
    AudioClassifier._map_to_jarvis()    — Filter to Jarvis-relevant categories
    AudioClassifier.is_loaded          — Property: True if model is loaded

Variables:
    AudioClassifier._model            — YAMNet TF Hub model
    AudioClassifier._class_names      — List of 521 YAMNet class labels
    AudioClassifier._window_seconds   — How many seconds to record per classification
    AudioClassifier._sample_rate      — 16000 Hz (YAMNet native)
    YAMNET_MODEL_URL                  — TensorFlow Hub YAMNet handle
    JARVIS_CATEGORY_MAP               — Maps YAMNet labels to Jarvis appliance labels

#todo: Add real-time streaming inference instead of fixed windows
#todo: Add peak detection — high-confidence event detection mid-window
#todo: Cache model to disk after first download for offline use
#todo: Add ambient baseline subtraction (ignore constant background hum)
"""

import asyncio
from pathlib import Path
from typing import Any, Optional

import numpy as np
from loguru import logger

YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"

# Map subsets of YAMNet's 521 classes to Jarvis-meaningful categories.
# Keys are substrings of YAMNet class names (lowercased); values are Jarvis labels.
JARVIS_CATEGORY_MAP: dict[str, str] = {
    # Appliances
    "washing machine":  "washer",
    "laundry":          "washer",
    "dishwasher":       "dishwasher",
    "dryer":            "dryer",
    "microwave oven":   "microwave",
    "vacuum cleaner":   "vacuum",
    "blender":          "cooking",
    "knife":            "cooking",
    "sizzle":           "cooking",

    # Media
    "television":       "watching_media",
    "television set":   "watching_media",
    "video game":       "gaming",
    "music":            "music",
    "singing":          "music",

    # Voice / Social
    "speech":           "conversation",
    "conversation":     "conversation",

    # Alerts
    "smoke detector":   "alarm",
    "fire alarm":       "alarm",
    "alarm clock":      "alarm",
    "alarm":            "alarm",
}


class AudioClassifier:
    """
    Records a short audio window and classifies ambient sounds via YAMNet.

    Results are returned as a list of dicts with "label" (Jarvis category),
    "yamnet_class" (raw YAMNet label), and "score" (0–1 confidence).
    """

    def __init__(self, config: dict) -> None:
        context_cfg = config.get("context", {})
        self._window_seconds: float = float(
            context_cfg.get("audio_classify_window_seconds", 3)
        )
        self._sample_rate: int = 16000  # YAMNet requires 16kHz mono float32
        self._model: Optional[Any] = None
        self._class_names: list[str] = []
        self._top_n: int = 5

    @property
    def is_loaded(self) -> bool:
        """True if YAMNet model has been successfully loaded."""
        return self._model is not None

    async def load(self) -> None:
        """Load YAMNet from TensorFlow Hub (downloads on first call, ~25 MB)."""
        await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        """Synchronous model load. Runs in thread pool."""
        try:
            import tensorflow as tf
            import tensorflow_hub as hub

            logger.info("[AudioClassifier] Loading YAMNet from TensorFlow Hub...")
            self._model = hub.load(YAMNET_MODEL_URL)
            model = self._model
            if model is None:
                logger.error("[AudioClassifier] YAMNet returned no model object")
                return

            # Load class names from local file or from hub
            labels_path = Path("data/yamnet_labels.csv")
            if labels_path.exists():
                with open(labels_path, "r") as f:
                    self._class_names = [
                        line.strip().split(",")[2].strip('"')
                        for line in f.readlines()[1:]  # skip header
                    ]
            else:
                # Fallback: get class names from model asset
                logger.warning(
                    "[AudioClassifier] yamnet_labels.csv not found, "
                    "using model-embedded class names"
                )
                class_map_path = model.class_map_path().numpy().decode("utf-8")
                with tf.io.gfile.GFile(class_map_path) as f:
                    self._class_names = [
                        line.strip().split(",")[2].strip('"')
                        for line in f.readlines()[1:]
                    ]

            logger.info(
                f"[AudioClassifier] YAMNet loaded — {len(self._class_names)} classes"
            )
        except ImportError:
            logger.error(
                "[AudioClassifier] tensorflow/tensorflow_hub not installed — "
                "audio classification disabled"
            )
        except Exception as e:
            logger.error(f"[AudioClassifier] Load failed: {e}")

    def classify(self, waveform: Optional[np.ndarray] = None) -> list[dict]:
        """
        Blocking: record a short audio window and classify it.
        Returns list of dicts: [{label, yamnet_class, score}, ...]
        """
        if not self.is_loaded:
            return []

        try:
            active_waveform = waveform if waveform is not None else self._record_window()
            return self._run_yamnet(active_waveform)
        except Exception as e:
            logger.warning(f"[AudioClassifier] Classification failed: {e}")
            return []

    async def classify_async(self) -> list[dict]:
        """Async wrapper — runs blocking classify() in a thread pool."""
        return await asyncio.to_thread(self.classify)

    def _record_window(self) -> np.ndarray:
        """
        Record audio_classify_window_seconds of audio.
        Returns float32 numpy array at 16kHz mono.
        """
        import sounddevice as sd

        n_samples = int(self._window_seconds * self._sample_rate)
        recording = sd.rec(
            n_samples,
            samplerate=self._sample_rate,
            channels=1,
            dtype=np.float32,
        )
        sd.wait()
        return recording.flatten()

    def _run_yamnet(self, waveform: np.ndarray) -> list[dict]:
        """
        Run YAMNet inference and return top Jarvis-relevant classifications.

        Args:
            waveform: Float32 array at 16kHz mono.

        Returns:
            List of {label, yamnet_class, score} dicts, filtered to Jarvis categories.
        """
        import tensorflow as tf

        model = self._model
        if model is None:
            return []

        waveform_tf = tf.constant(waveform, dtype=tf.float32)
        scores, embeddings, spectrogram = model(waveform_tf)

        # Average scores across time frames
        mean_scores = tf.reduce_mean(scores, axis=0).numpy()

        # Get top N overall class indices
        top_indices = np.argsort(mean_scores)[::-1][: self._top_n * 3]

        results = []
        seen_jarvis_labels: set[str] = set()

        for idx in top_indices:
            score = float(mean_scores[idx])
            if score < 0.05:
                break

            yamnet_class = (
                self._class_names[idx] if idx < len(self._class_names) else str(idx)
            ).lower()

            jarvis_label = self._map_to_jarvis(yamnet_class)
            if jarvis_label and jarvis_label not in seen_jarvis_labels:
                seen_jarvis_labels.add(jarvis_label)
                results.append(
                    {
                        "label":        jarvis_label,
                        "yamnet_class": yamnet_class,
                        "score":        score,
                    }
                )
                if len(results) >= self._top_n:
                    break

        if results:
            logger.debug(
                "[AudioClassifier] Top sounds: "
                + ", ".join(f"{r['label']}({r['score']:.2f})" for r in results)
            )
        return results

    @staticmethod
    def _map_to_jarvis(yamnet_class: str) -> Optional[str]:
        """Map a YAMNet class name to a Jarvis activity/appliance label, or None."""
        for keyword, label in JARVIS_CATEGORY_MAP.items():
            if keyword in yamnet_class:
                return label
        return None
