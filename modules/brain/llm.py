"""
JARVIS — Ambient Home AI
========================
Mission: Wrap the Ollama API for both text (LLM chat) and vision image
         description queries. All communication is async. Handles connection errors
         gracefully and exposes a simple availability check for health monitoring.

Modules: modules/brain/llm.py
Classes: OllamaLLM
Functions:
    OllamaLLM.__init__(config)           — Store config
    OllamaLLM.chat(messages)             — Send messages, return response string
    OllamaLLM.vision_query(frame, prompt)— Describe an image frame
    OllamaLLM.is_available()             — Sync health check (bool)
    OllamaLLM.is_available_async()       — Async health check (bool)

Variables:
    OllamaLLM._client   — ollama.AsyncClient instance
    OllamaLLM._model    — LLM model name string
    OllamaLLM._vision   — Vision model name string
    OllamaLLM._timeout  — Request timeout seconds

#todo: Add streaming response support for progressive TTS (speak as tokens arrive)
#todo: Cache repeated vision queries for the same frame hash to avoid redundant GPU calls
#todo: Add model switching at runtime (switch to larger model for complex requests)
#todo: Track token usage per conversation for cost estimation / quota management
#todo: Add retry logic with exponential backoff for transient Ollama errors
"""

import asyncio
import base64
from typing import Any

import httpx
import numpy as np
from loguru import logger

from core.exceptions import LLMError


class OllamaLLM:
    """
    Async Ollama LLM client for chat and vision queries.

    Config keys used (from config["ollama"]):
        model:           Text model to use (e.g., "gemini-3-flash-preview:cloud")
        vision_model:    Vision model. Defaults to `model` when omitted.
        base_url:        Ollama API URL (default "http://localhost:11434")
        timeout_seconds: Request timeout
        system_prompt:   Jarvis personality prompt

    Usage:
        llm = OllamaLLM(config)
        response = await llm.chat([
            {"role": "system", "content": "..."},
            {"role": "user", "content": "Hey, what time is it?"},
        ])
    """

    def __init__(self, config: dict) -> None:
        cfg = config["ollama"]
        self._model: str = cfg["model"]
        self._vision_model: str = cfg.get("vision_model", self._model)
        self._base_url: str = cfg.get("base_url", "http://localhost:11434")
        self._timeout: int = cfg.get("timeout_seconds", 30)
        self._system_prompt: str = cfg.get("system_prompt", "You are Jarvis.")

        try:
            import ollama
            self._client: Any = ollama.AsyncClient(host=self._base_url)
        except ImportError as e:
            raise LLMError("ollama package not installed. Run: pip install ollama") from e

    async def chat(self, messages: list[dict[str, Any]]) -> str:
        """
        Send a list of messages to the LLM and return the response text.

        Args:
            messages: OpenAI-format message list:
                      [{"role": "system"|"user"|"assistant", "content": str}, ...]

        Returns:
            The assistant's response string.

        Raises:
            LLMError: On connection failure, timeout, or invalid response.
        """
        try:
            response = await asyncio.wait_for(
                self._client.chat(
                    model=self._model,
                    messages=messages,
                ),
                timeout=self._timeout,
            )
            text = response["message"]["content"].strip()
            logger.debug(f"[LLM] Response ({len(text)} chars): {text[:100]}...")
            return text

        except asyncio.TimeoutError:
            raise LLMError(f"Ollama chat timed out after {self._timeout}s")
        except Exception as e:
            raise LLMError(f"Ollama chat failed: {e}") from e

    async def vision_query(
        self,
        frame: np.ndarray,
        prompt: str = "Describe what you see in this image. Be concise.",
    ) -> str:
        """
        Send a camera frame to the vision model and get a description.
        Encodes the frame as JPEG, then base64 for the Ollama API.

        Args:
            frame: OpenCV BGR or RGB uint8 numpy array (H, W, C).
            prompt: The question to ask about the image.

        Returns:
            Natural language description of the image.

        Raises:
            LLMError: If the vision model is unavailable or returns an error.
        """
        try:
            import cv2
        except ImportError as e:
            raise LLMError("opencv-python required for vision queries") from e

        # Encode frame as JPEG bytes, then base64
        success, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not success:
            raise LLMError("Failed to encode camera frame as JPEG")

        img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        try:
            response = await asyncio.wait_for(
                self._client.chat(
                    model=self._vision_model,
                    messages=[{
                        "role": "user",
                        "content": prompt,
                        "images": [img_b64],
                    }],
                ),
                timeout=self._timeout,
            )
            description = response["message"]["content"].strip()
            logger.debug(f"[LLM] Vision: {description[:120]}")
            return description

        except asyncio.TimeoutError:
            raise LLMError(f"Vision query timed out after {self._timeout}s")
        except Exception as e:
            raise LLMError(f"Vision query failed: {e}") from e

    def is_available(self) -> bool:
        """
        Synchronous health check. Returns True if Ollama is reachable.
        Suitable for calling during startup before the event loop runs.
        """
        try:
            r = httpx.get(f"{self._base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    async def is_available_async(self) -> bool:
        """Async health check. Returns True if Ollama is reachable."""
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(f"{self._base_url}/api/tags", timeout=3)
                return r.status_code == 200
        except Exception:
            return False

    @property
    def system_prompt(self) -> str:
        """The base personality system prompt for this Jarvis instance."""
        return self._system_prompt
