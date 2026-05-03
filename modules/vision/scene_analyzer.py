"""
JARVIS — Ambient Home AI
========================
Mission: Generate natural-language descriptions of room scenes by sending
         camera frames to the configured Ollama vision model. These descriptions
         become the room baseline and provide context for the LLM to reference
         when answering questions like "what's the state of the kitchen?"

         A local frame-similarity check runs first — if the scene hasn't
         meaningfully changed since the last description, the LLM call is
         skipped entirely and the cached description is returned. This keeps
         vision-model load proportional to actual activity in the room.

Modules: modules/vision/scene_analyzer.py
Classes: SceneAnalyzer
Functions:
    SceneAnalyzer.__init__(config, llm)      — Initialize with config and LLM
    SceneAnalyzer.describe_async(frame, room, objects) — Generate room description
    SceneAnalyzer._build_prompt(room, objects) — Build the vision query prompt
    SceneAnalyzer.last_description(room)      — Get last description for a room
    SceneAnalyzer._frame_signature(frame)     — Downscaled grayscale signature
    SceneAnalyzer._scene_changed(room, sig)   — Compare new sig to reference

Variables:
    SceneAnalyzer._llm             — OllamaLLM reference for vision queries
    SceneAnalyzer._descriptions    — {room: str} cache of last descriptions
    SceneAnalyzer._references      — {room: np.ndarray} 32x32 grayscale baselines
    SceneAnalyzer._change_threshold — Mean absolute difference required to re-describe
    SceneAnalyzer._config          — Full config dict

#todo: Add detailed room inventory tracking (which objects are normally present)
#todo: Add "what changed" diff mode for anomaly detection vs baseline
#todo: Store full description history in database for temporal queries
"""

from typing import Any, Optional, TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from modules.brain.llm import OllamaLLM

# Signature size — small enough to be fast, large enough to capture meaningful
# scene changes (someone enters, lights flip, mess appears).
_SIG_SIZE = 32


class SceneAnalyzer:
    """
    Uses the configured Ollama vision model to generate natural-language room descriptions.

    Results are cached per room and updated at the vision scan interval.
    Skips the LLM call when the scene hasn't changed enough vs the last described frame.
    """

    def __init__(self, config: dict, llm: Optional["OllamaLLM"]) -> None:
        self._config = config
        self._llm = llm
        self._descriptions: dict[str, str] = {}
        self._references: dict[str, np.ndarray] = {}
        # MAD on 0-255 grayscale of a blurred 32x32 signature.
        # Cheap sensors (ESP32-CAM in low light) routinely produce 5-15 MAD from
        # noise alone, so the global default is conservative and noisy rooms get
        # per-room overrides via vision.scene_change_threshold_per_room.
        scene_cfg = config.get("vision", {}) if isinstance(config.get("vision"), dict) else {}
        self._change_threshold: float = float(
            scene_cfg.get("scene_change_threshold", 12.0)
        )
        per_room = scene_cfg.get("scene_change_threshold_per_room") or {}
        self._per_room_thresholds: dict[str, float] = {
            str(k): float(v) for k, v in per_room.items()
        }

    async def describe_async(
        self,
        frame: Optional[np.ndarray],
        room: str,
        objects: Optional[list[dict[str, Any]]] = None,
    ) -> Optional[str]:
        """
        Generate a natural-language description of a room from a camera frame.

        Skips the vision-model call when the frame is similar to the last frame
        that produced a description, returning the cached description instead.

        Args:
            frame:   BGR numpy array from camera (or None for text-only fallback).
            room:    Room identifier used for caching.
            objects: Optional list of YOLO detections for grounding the description.

        Returns:
            Description string or None if vision query fails.
        """
        if frame is None:
            return self._descriptions.get(room)
        if self._llm is None:
            logger.debug("[SceneAnalyzer] No LLM configured — returning cached description")
            return self._descriptions.get(room)

        # Local change detection — skip the LLM call when the scene is essentially
        # the same as the last frame we described.
        signature = self._frame_signature(frame)
        if not self._scene_changed(room, signature):
            logger.debug(f"[SceneAnalyzer] Skipping '{room}' — scene unchanged")
            return self._descriptions.get(room)

        prompt = self._build_prompt(room, objects)

        try:
            description = await self._llm.vision_query(frame, prompt)
            if description:
                stripped = description.strip()
                self._descriptions[room] = stripped
                # Only update the reference frame after a successful description,
                # so a transient failure doesn't lose our baseline.
                self._references[room] = signature
                logger.info(
                    f"[SceneAnalyzer] New description for '{room}': "
                    f"{stripped[:80]}..."
                )
                return stripped
        except Exception as e:
            logger.warning(f"[SceneAnalyzer] Vision query failed for '{room}': {e}")

        return self._descriptions.get(room)

    @staticmethod
    def _frame_signature(frame: np.ndarray) -> np.ndarray:
        """
        Downscale a BGR frame to a noise-suppressed 32x32 grayscale signature.
        Pure numpy — no cv2 dependency. Fast enough to run on every frame.

        Two-stage downscale: stride to ~64x64, then 2x2 average pool to 32x32.
        The pool step averages four neighbouring pixels, attenuating per-pixel
        sensor noise (which is uncorrelated) by ~2x while preserving real
        scene structure.
        """
        h, w = frame.shape[:2]
        target = _SIG_SIZE * 2
        step_h = max(1, h // target)
        step_w = max(1, w // target)
        small = frame[::step_h, ::step_w]
        if small.ndim == 3:
            small = small.mean(axis=2)
        small = small.astype(np.float32)
        # Trim to even dimensions before 2x2 pool
        h2 = small.shape[0] - (small.shape[0] % 2)
        w2 = small.shape[1] - (small.shape[1] % 2)
        small = small[:h2, :w2]
        return (
            small[0::2, 0::2]
            + small[1::2, 0::2]
            + small[0::2, 1::2]
            + small[1::2, 1::2]
        ) / 4.0

    def _threshold_for(self, room: str) -> float:
        """Per-room override beats the global threshold."""
        return self._per_room_thresholds.get(room, self._change_threshold)

    def _scene_changed(self, room: str, signature: np.ndarray) -> bool:
        """
        Mean absolute difference between current and reference signatures.
        Returns True (scene changed) when no reference exists yet — the first
        frame for a room always describes so we have a baseline to compare to.
        """
        reference = self._references.get(room)
        if reference is None:
            return True
        if reference.shape != signature.shape:
            # Resolution changed (camera reopened at a different size, etc.) —
            # treat as changed and rebuild the reference on this pass.
            return True
        threshold = self._threshold_for(room)
        mad = float(np.mean(np.abs(reference - signature)))
        changed = mad >= threshold
        logger.debug(
            f"[SceneAnalyzer] '{room}' frame MAD={mad:.2f} "
            f"(threshold={threshold:.2f}) → "
            f"{'changed' if changed else 'stable'}"
        )
        return changed

    def last_description(self, room: str) -> Optional[str]:
        """Return the most recent generated description for a room, or None."""
        return self._descriptions.get(room)

    def _build_prompt(
        self,
        room: str,
        objects: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        """
        Build the vision query prompt for the configured vision model.
        Includes room context and optional object detection pre-ground.
        """
        base = (
            f"Describe what you see in this {room} in 2-3 sentences. "
            "Focus on: what the person is doing, the state of the room (tidy/messy), "
            "any notable objects or activities. Be specific and factual."
        )

        if objects:
            from modules.vision.object_detector import ObjectDetector
            obj_summary = ObjectDetector.summarize(objects)
            if obj_summary != "nothing notable":
                base += f" Detected objects include: {obj_summary}."

        return base
