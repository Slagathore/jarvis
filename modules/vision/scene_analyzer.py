"""
JARVIS — Ambient Home AI
========================
Mission: Generate natural-language descriptions of room scenes by sending
         camera frames to the configured Ollama vision model. These descriptions
         become the room baseline and provide context for the LLM to reference
         when answering questions like "what's the state of the kitchen?"

         SceneAnalyzer is called infrequently (configurable interval, default
         5 minutes) to keep inference load low. It combines object detection
         results with the model's free-form description for richer context.

Modules: modules/vision/scene_analyzer.py
Classes: SceneAnalyzer
Functions:
    SceneAnalyzer.__init__(config, llm)      — Initialize with config and LLM
    SceneAnalyzer.describe_async(frame, room, objects) — Generate room description
    SceneAnalyzer._build_prompt(room, objects) — Build the vision query prompt
    SceneAnalyzer.last_description(room)      — Get last description for a room

Variables:
    SceneAnalyzer._llm             — OllamaLLM reference for vision queries
    SceneAnalyzer._descriptions    — {room: str} cache of last descriptions
    SceneAnalyzer._config          — Full config dict

#todo: Add scene change detection — compare embeddings to skip unchanged scenes
#todo: Add detailed room inventory tracking (which objects are normally present)
#todo: Add "what changed" diff mode for anomaly detection vs baseline
#todo: Store full description history in database for temporal queries
"""

import asyncio
from typing import Any, Optional, TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from modules.brain.llm import OllamaLLM


class SceneAnalyzer:
    """
    Uses the configured Ollama vision model to generate natural-language room descriptions.

    Results are cached per room and updated at the vision scan interval.
    """

    def __init__(self, config: dict, llm: Optional["OllamaLLM"]) -> None:
        self._config = config
        self._llm = llm
        self._descriptions: dict[str, str] = {}

    async def describe_async(
        self,
        frame: Optional[np.ndarray],
        room: str,
        objects: Optional[list[dict[str, Any]]] = None,
    ) -> Optional[str]:
        """
        Generate a natural-language description of a room from a camera frame.

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

        prompt = self._build_prompt(room, objects)

        try:
            description = await self._llm.vision_query(frame, prompt)
            if description:
                self._descriptions[room] = description.strip()
                logger.info(
                    f"[SceneAnalyzer] New description for '{room}': "
                    f"{description[:80]}..."
                )
                return description.strip()
        except Exception as e:
            logger.warning(f"[SceneAnalyzer] Vision query failed for '{room}': {e}")

        return self._descriptions.get(room)

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
