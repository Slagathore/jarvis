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

# COCO classes that count as a "pet" — Jarvis surfaces these specifically so
# downstream code can react ("cat is on the keyboard again").
PET_CLASSES: set[str] = {"cat", "dog", "bird"}


class ObjectDetector:
    """
    YOLOv8n object detector for room scene understanding.

    Uses the nano model by default for maximum speed on CPU.
    Switches to GPU automatically if CUDA is available via ultralytics.
    """

    def __init__(self, config: dict) -> None:
        vision_cfg = config.get("vision", {})
        # YOLOv8m default: nano was missing partially-occluded animals
        # in dim Wyze frames (cat on patterned tablecloth, dog under
        # sheets). Medium is ~8x slower than nano (~12ms on a 4070 Ti);
        # still well within budget at 5fps. Conf threshold lowered to
        # 0.20 to catch low-confidence pet detections.
        self._model_name: str = vision_cfg.get("yolo_model", "yolov8m.pt")
        self._conf_threshold: float = float(vision_cfg.get("yolo_confidence", 0.20))
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
            # Single warmup predict on a black frame so ultralytics' lazy
            # fuse() runs once on a known-safe input. Without this, the
            # first real frame from cameras can race concurrent calls
            # mid-fuse and fail with "'Conv' object has no attribute 'bn'".
            try:
                import numpy as _np
                self._model.predict(
                    _np.zeros((640, 640, 3), dtype=_np.uint8),
                    conf=self._conf_threshold,
                    verbose=False,
                )
            except Exception as we:
                logger.debug(f"[ObjectDetector] warmup predict skipped: {we}")
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

        # Perf tracking — per-frame YOLO inference is the most common
        # culprit when the dashboard feels laggy, so we record every
        # call.
        from modules.context.perf_tracker import perf
        with perf().timeit(f"yolo.{self._model_name.replace('.pt','')}"):
            return self._detect_inner(frame, model)

    def _detect_inner(self, frame: np.ndarray, model: Any) -> list[dict]:
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

    @staticmethod
    def pets(detections: list[dict]) -> list[dict]:
        """Return only the detections that are pets (cat/dog/bird)."""
        return [d for d in detections if d.get("class") in PET_CLASSES]
