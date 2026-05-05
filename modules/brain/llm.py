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
from typing import Any, Optional

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

        # Lazy-init Gemini direct-API client. Only built when a model name with
        # the ':gapi' suffix gets selected — keeps the import + httpx pool out
        # of the boot path for users who never use the direct API.
        self._gemini_direct: Optional[Any] = None

    @staticmethod
    def _is_gemini_direct(model_name: str) -> bool:
        """True for models routed through Google's direct API (not via Ollama)."""
        return ":gapi" in (model_name or "")

    def _get_gemini_direct(self) -> Any:
        if self._gemini_direct is None:
            from modules.brain.gemini_direct import GeminiDirectClient
            from pathlib import Path
            # Load .env so GEMINI_API_KEY is visible. Idempotent — python-dotenv
            # is a no-op if already loaded or absent.
            try:
                from dotenv import load_dotenv
                load_dotenv(Path(__file__).resolve().parents[2] / ".env")
            except ImportError:
                pass
            self._gemini_direct = GeminiDirectClient(timeout=self._timeout)
        return self._gemini_direct

    @property
    def model(self) -> str:
        """Current text-chat model name. Read by health checks + dashboard."""
        return self._model

    @property
    def vision_model(self) -> str:
        return self._vision_model

    @property
    def client(self) -> Any:
        """Underlying ollama.AsyncClient — exposed so model-management code
        can call list/pull/delete on the same connection."""
        return self._client

    def set_active_model(self, name: str) -> None:
        """Hot-swap the chat model. Takes effect on the next chat() call.
        Vision model is untouched — set separately via set_vision_model()."""
        self._model = name
        logger.info(f"[LLM] Active chat model switched to '{name}'")

    def set_vision_model(self, name: str) -> None:
        self._vision_model = name
        logger.info(f"[LLM] Vision model switched to '{name}'")

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
        # Dispatch: ':gapi' models go through Google's direct API instead of Ollama.
        if self._is_gemini_direct(self._model):
            return await self._get_gemini_direct().chat(messages, model=self._model)

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

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_handlers: dict[str, Any],
        max_iterations: int = 20,
    ) -> str:
        """
        Chat with tool-calling support. Loops: if the LLM emits tool_calls,
        execute them via tool_handlers, append the results, call the LLM again.
        Returns the final text response once the LLM stops requesting tools.

        Args:
            messages:       Standard chat messages list.
            tools:          OpenAI/Ollama tool schema list. Each entry:
                            {"type": "function", "function": {"name", "description", "parameters"}}
            tool_handlers:  {function_name: async_callable(**args) → result}
                            Result is JSON-serialized and fed back to the LLM.
            max_iterations: Cap on tool-call loops so a misbehaving model can't
                            burn budget forever. 20 covers realistic
                            code-exploration AND computer-control sequences
                            (e.g. list_files → read_file → grep → edit_file →
                            restart_self, or screenshot → click → screenshot →
                            type → screenshot → click). The same cap applies
                            to all tools — calendar, memory, ask_claude,
                            computer, self-edit. A persistent 20-tool runaway
                            still gets the warning + empty-response fallback.

        Returns:
            The assistant's final text response.

        Raises:
            LLMError: On connection failure, timeout, or unrecognized tool call.
        """
        import json

        # Dispatch: ':gapi' models route to Gemini direct, with its own tool loop.
        if self._is_gemini_direct(self._model):
            return await self._get_gemini_direct().chat_with_tools(
                messages=messages,
                tools=tools,
                model=self._model,
                tool_dispatcher=tool_handlers,
                max_iterations=max_iterations,
            )

        working_messages = list(messages)

        for iteration in range(max_iterations):
            try:
                response = await asyncio.wait_for(
                    self._client.chat(
                        model=self._model,
                        messages=working_messages,
                        tools=tools,
                    ),
                    timeout=self._timeout,
                )
            except asyncio.TimeoutError:
                raise LLMError(f"Ollama chat (tools) timed out after {self._timeout}s")
            except Exception as e:
                raise LLMError(f"Ollama chat (tools) failed: {e}") from e

            message = response.get("message", {}) or {}
            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                text = (message.get("content") or "").strip()
                logger.debug(
                    f"[LLM] Tool-loop done in {iteration + 1} iter(s), "
                    f"final response ({len(text)} chars): {text[:100]}..."
                )
                return text

            # Append the assistant turn that requested the tools
            working_messages.append(dict(message))

            for call in tool_calls:
                fn = (call.get("function") or {})
                name = fn.get("name") or ""
                # Ollama may pass arguments as a dict OR a JSON string depending
                # on model. Normalize to dict.
                raw_args = fn.get("arguments") or {}
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        args = {}
                else:
                    args = raw_args

                handler = tool_handlers.get(name)
                if handler is None:
                    result = {"error": f"unknown tool: {name}"}
                    logger.warning(f"[LLM] Model requested unknown tool '{name}'")
                else:
                    try:
                        logger.debug(f"[LLM] Tool call: {name}({args})")
                        result = await handler(**args)
                    except Exception as e:
                        logger.warning(f"[LLM] Tool '{name}' raised: {e}")
                        result = {"error": str(e)}

                working_messages.append({
                    "role":    "tool",
                    "name":    name,
                    "content": json.dumps(result, default=str),
                })

        logger.warning(
            f"[LLM] Tool loop hit max_iterations={max_iterations} without final answer"
        )
        return ""

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

        # Dispatch: ':gapi' models go to Gemini direct API
        if self._is_gemini_direct(self._vision_model):
            return await self._get_gemini_direct().vision_query(
                buf.tobytes(), prompt, model=self._vision_model
            )

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
