"""
JARVIS — Ambient Home AI
========================
Mission: Run YOLOv8 object detection on camera frames to identify objects
         present in a room. Provides a structured summary of what's in the
         scene — people, food, animals, devices — that feeds into the LLM
         scene analyzer and the curiosity engine. Five cats show up a lot.

Modules: modules/vision/object_detector.py
Classes: ObjectDetector
Functions:
    ObjectDetector.__init__(config)        — Initialize with config
    ObjectDetector.load()                  — Load YOLOv8n model
    ObjectDetector.detect(frame)           — Sync: run detection, return list
    ObjectDetector.detect_async(frame)     — Async wrapper
    ObjectDetector.summarize(detections)   — Convert detections to text summary
    ObjectDetector.has_person(detections)  — True if a person is detected
    ObjectDetector.is_loaded               — Property

Variables:
    ObjectDetector._model         — ultralytics YOLO instance
    ObjectDetector._conf_threshold — Minimum confidence for results
    ObjectDetector._model_name    — "yolov8n.pt" (nano, fastest)

Detection result dict:
    {
        "class":      "person",
        "confidence": 0.94,
        "box":        [x1, y1, x2, y2],   # pixel coords
        "label":      "person 0.94",       # formatted label string
    }

#todo: Add person re-identification to track which person (Cole vs Anna vs Sophie)
#todo: Add object persistence — if phone is on desk for 30+ min, note it
#todo: Add custom classes via fine-tuning (specific cats' silhouettes)
#todo: Store detection history for "where did I put X" queries
"""

import asyncio
from typing import Any, Optional

import numpy as np
from loguru import logger

YOLO = None

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    logger.warning("[ObjectDetector] ultralytics not available — object detection disabled")


# Objects Jarvis particularly cares about summarizing
NOTABLE_CLASSES: set[str] = {
    "person", "cat", "dog", "laptop", "tv", "cell phone",
    "cup", "bottle", "book", "bed", "couch", "chair",
    "food", "pizza", "sandwich", "bowl",
}


class ObjectDetector:
    """
    YOLOv8n object detector for room scene understanding.

    Uses the nano model by default for maximum speed on CPU.
    Switches to GPU automatically if CUDA is available via ultralytics.
    """

    def __init__(self, config: dict) -> None:
        vision_cfg = config.get("vision", {})
        self._model_name: str = vision_cfg.get("yolo_model", "yolov8n.pt")
        self._conf_threshold: float = float(vision_cfg.get("yolo_confidence", 0.35))
        self._model: Optional[Any] = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Load the YOLOv8 model. Downloads weights on first call (~6 MB for nano)."""
        if not _YOLO_AVAILABLE or YOLO is None:
            return
        try:
            self._model = YOLO(self._model_name)
            logger.info(f"[ObjectDetector] YOLOv8 loaded: {self._model_name}")
        except Exception as e:
            logger.error(f"[ObjectDetector] Load failed: {e}")

    async def load_async(self) -> None:
        """Async wrapper for load()."""
        await asyncio.to_thread(self.load)

    def detect(self, frame: Optional[np.ndarray]) -> list[dict]:
        """
        Run object detection on a single frame.

        Args:
            frame: BGR numpy array from camera.

        Returns:
            List of detection dicts (empty if no detections or model not loaded).
        """
        model = self._model
        if frame is None or model is None:
            return []

        try:
            results = model(frame, conf=self._conf_threshold, verbose=False)
            detections = []

            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names.get(cls_id, str(cls_id))
                    confidence = float(box.conf[0])
                    coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    detections.append(
                        {
                            "class":      cls_name,
                            "confidence": round(confidence, 3),
                            "box":        [round(c) for c in coords],
                            "label":      f"{cls_name} {confidence:.2f}",
                        }
                    )

            logger.debug(
                f"[ObjectDetector] {len(detections)} objects: "
                + ", ".join(d["class"] for d in detections[:5])
            )
            return detections

        except Exception as e:
            logger.warning(f"[ObjectDetector] Detection failed: {e}")
            return []

    async def detect_async(self, frame: Optional[np.ndarray]) -> list[dict]:
        """Async wrapper — runs inference in thread pool."""
        return await asyncio.to_thread(self.detect, frame)

    @staticmethod
    def summarize(detections: list[dict]) -> str:
        """
        Convert detection list to a brief text summary for use in LLM prompts.
        Groups by class and counts them.

        Example: "1 person, 2 cats, 1 laptop"
        """
        if not detections:
            return "nothing notable"

        counts: dict[str, int] = {}
        for d in detections:
            cls = d["class"]
            counts[cls] = counts.get(cls, 0) + 1

        # Prioritize notable classes first
        notable = {k: v for k, v in counts.items() if k in NOTABLE_CLASSES}
        other = {k: v for k, v in counts.items() if k not in NOTABLE_CLASSES}

        parts = []
        for cls, n in sorted(notable.items()):
            parts.append(f"{n} {cls}" if n == 1 else f"{n} {cls}s")
        for cls, n in sorted(other.items()):
            parts.append(f"{n} {cls}" if n == 1 else f"{n} {cls}s")

        return ", ".join(parts) if parts else "nothing notable"

    @staticmethod
    def has_person(detections: list[dict]) -> bool:
        """Return True if at least one person was detected."""
        return any(d["class"] == "person" for d in detections)
