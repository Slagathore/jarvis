"""
JARVIS — Ambient Home AI
========================
Mission: Shared YAMNet loader — one model instance for the whole process.

         YAMNet is used by two subsystems: AudioClassifier (appliance /
         activity sounds) and SoundEventClassifier (cascade Stage 2b
         safety events). Each used to call tensorflow_hub.load()
         independently — loading the model, the TF graph, and ~17 MB of
         weights twice.

         load_yamnet() loads it exactly once (thread-safe) and hands the
         same (model, class_names) to every caller. The first caller's
         `device` preference wins — TF device visibility is process-wide
         anyway, so it cannot be per-instance.

Modules: modules/voice/yamnet_loader.py
Functions: load_yamnet
"""

from __future__ import annotations

import csv
import threading
from typing import Any, Optional

from loguru import logger

_YAMNET_HANDLE = "https://tfhub.dev/google/yamnet/1"

_lock = threading.Lock()
_model: Any = None
_class_names: list[str] = []
_loaded = False


def load_yamnet(device: str = "cuda") -> tuple[Any, list[str]]:
    """Return the shared (yamnet_model, class_names). Loads on first call.

    device: "cpu" forces YAMNet off the GPU (so it cannot contend with
    the YOLO/Whisper torch stack for VRAM); anything else leaves TF's
    device selection alone. Thread-safe and idempotent — later calls
    return the cached pair and ignore `device`.

    On failure returns (None, []) — callers degrade gracefully.
    """
    global _model, _class_names, _loaded
    with _lock:
        if _loaded:
            return _model, _class_names
        _loaded = True  # mark attempted — a hard failure should not retry-loop
        try:
            import tensorflow as tf
            import tensorflow_hub as hub

            if str(device).lower() == "cpu":
                try:
                    tf.config.set_visible_devices([], "GPU")
                except Exception:
                    pass
            _model = hub.load(_YAMNET_HANDLE)
            _class_names = _read_class_names(tf, _model)
            logger.info(
                f"[YAMNet] shared model loaded ({len(_class_names)} classes)"
            )
        except Exception as e:
            logger.warning(f"[YAMNet] shared load failed: {e}")
            _model, _class_names = None, []
        return _model, _class_names


def _read_class_names(tf: Any, model: Any) -> list[str]:
    """Parse YAMNet's bundled class map (index, mid, display_name)."""
    try:
        path = model.class_map_path().numpy().decode("utf-8")
        names: list[str] = []
        with tf.io.gfile.GFile(path) as f:
            for row in csv.DictReader(f):
                names.append(row["display_name"])
        return names
    except Exception as e:
        logger.warning(f"[YAMNet] class map parse failed: {e}")
        return []


def is_loaded() -> bool:
    return _model is not None
