"""
JARVIS — World Model
====================
Mission: §23.4 — OWLv2 wrapper for low-frequency open-vocabulary
         detection. Used to find user-declared objects (wallet,
         keys, glasses) that aren't in YOLO's COCO classes.

         Same lazy-import + Null fallback pattern as CLIPEncoder:
         missing transformers / weights download failure / no CUDA →
         NullOpenVocabDetector. The rest of the system runs without
         it; you just lose the ability to track non-YOLO objects.

         Cost: ~120ms/frame at 640×640 on RTX 4070 Ti — too
         expensive for per-frame use. ObservationBuilder runs this
         on its own per-room loop at `detection_interval_seconds.
         open_vocabulary` (default 30s). Object motion is rare
         enough that 30s polling is fine for objects users actually
         want to find.

Modules: modules/vision/openvocab_detector.py
Classes: OpenVocabDetector, NullOpenVocabDetector
Spec:    new 2.md §23.1, §23.4.

#todo: Grounding DINO alternative — slightly faster + higher recall
       per spec, but the HuggingFace integration story is messier.
       Swap in if OWLv2 recall on Cole's specific household objects
       is unsatisfactory. Same `detect()` interface contract.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
from loguru import logger


class OpenVocabDetector:
    """OWLv2 (`google/owlv2-base-patch16-ensemble`) zero-shot detector.
    Construct once at orchestrator boot; call `detect(frame, queries)`
    on demand from the open-vocab loop."""

    DEFAULT_MODEL = "google/owlv2-base-patch16-ensemble"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda",
        score_threshold: float = 0.20,
    ) -> None:
        import torch
        # transformers stubs incorrectly type `from_pretrained` as
        # returning a `str` for some classes; narrow with cast + local
        # type-ignores. Runtime types are correct.
        from transformers import (
            Owlv2Processor,
            Owlv2ForObjectDetection,
        )
        self.device = device if torch.cuda.is_available() else "cpu"
        self._torch = torch
        self.processor: Any = Owlv2Processor.from_pretrained(model_name)  # type: ignore[no-untyped-call]
        self.model: Any = Owlv2ForObjectDetection.from_pretrained(
            model_name,
        ).to(self.device)  # type: ignore[attr-defined,union-attr]
        self.model.eval()
        self.score_threshold = float(score_threshold)
        logger.info(
            f"[OpenVocabDetector] {model_name} loaded on {self.device}"
        )

    def detect(
        self,
        image_bgr: Optional[np.ndarray],
        queries: list[str],
    ) -> list[dict]:
        """Run OWLv2 with a list of text queries on a frame. Returns
        `[{name, bbox: (x1,y1,x2,y2), score}, ...]`. Empty / None
        input or empty queries → []. Failures are logged + return
        []; the caller treats missing detections as the absence of
        the queried objects."""
        if image_bgr is None or image_bgr.size == 0 or not queries:
            return []
        from PIL import Image
        rgb = image_bgr[..., ::-1]
        try:
            pil = Image.fromarray(rgb)
        except Exception as e:
            logger.debug(f"[OpenVocabDetector] PIL conversion failed: {e}")
            return []
        try:
            with self._torch.no_grad():
                inputs = self.processor(  # type: ignore[operator]
                    text=[queries], images=pil, return_tensors="pt",
                ).to(self.device)
                outputs = self.model(**inputs)
                target_sizes = self._torch.tensor(
                    [pil.size[::-1]],
                ).to(self.device)
                results = self.processor.post_process_object_detection(  # type: ignore[attr-defined]
                    outputs=outputs,
                    threshold=self.score_threshold,
                    target_sizes=target_sizes,
                )[0]
        except Exception as e:
            logger.debug(f"[OpenVocabDetector] inference failed: {e}")
            return []
        out: list[dict] = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"],
        ):
            x1, y1, x2, y2 = (int(c) for c in box.tolist())
            out.append({
                "name": queries[int(label)],
                "bbox": (x1, y1, x2, y2),
                "score": float(score),
            })
        return out


class NullOpenVocabDetector:
    """No-op detector for environments without OWLv2 / transformers.
    detect() always returns []. Open-vocab tracking simply doesn't
    fire; YOLO continues at full FPS, world-model state machine is
    unaffected."""

    score_threshold: float = 1.0   # nothing matches

    def detect(
        self,
        image_bgr: Optional[np.ndarray],
        queries: list[str],
    ) -> list[dict]:
        return []


def try_load(
    model_name: str = OpenVocabDetector.DEFAULT_MODEL,
    device: str = "cuda",
    score_threshold: float = 0.20,
) -> object:
    """Construct a real OpenVocabDetector or fall back to Null on
    any failure (missing transformers, weights pull issues, CUDA OOM,
    etc.)."""
    try:
        return OpenVocabDetector(
            model_name=model_name,
            device=device,
            score_threshold=score_threshold,
        )
    except Exception as e:
        logger.warning(
            f"[OpenVocabDetector] could not load — open-vocab object "
            f"detection (wallet / keys / etc.) will not fire. Install "
            f"with `pip install transformers` (already present?) + "
            f"ensure the model can be downloaded. Error: {e}"
        )
        return NullOpenVocabDetector()
