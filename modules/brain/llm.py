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
import time
from typing import Any, Optional

import httpx
import numpy as np
from loguru import logger

from core.exceptions import LLMError
from modules.context.perf_tracker import perf


class OllamaLLM:
    """
    Async Ollama LLM client for chat and vision queries.

    Config keys used (from config["ollama"]):
        model:           Text model to use (e.g., "kimi-k2.7-code:cloud")
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
        # Pattern D: action model used mid-loop after any action tool call.
        # Empty / None disables the swap and the loop stays on self._model.
        self._action_model: Optional[str] = cfg.get("action_model") or None
        self._base_url: str = cfg.get("base_url", "http://localhost:11434")
        self._timeout: int = cfg.get("timeout_seconds", 30)
        self._system_prompt: str = cfg.get("system_prompt", "You are Jarvis.")
        # Global thinking default (config.ollama.think): False (default)
        # sends no think kwarg so every model keeps its own default; True or
        # "low"/"medium"/"high" enables the thinking pass on models that
        # support it (kimi-k2.7-code does). Per-model dashboard settings
        # (thinking_enabled) override this.
        self._default_think: Any = cfg.get("think", False)
        # Whether message.thinking gets logged when a response carries it.
        # Thinking is never mixed into returned content either way.
        self._show_thinking: bool = bool(cfg.get("show_thinking", False))

        try:
            import ollama
            self._client: Any = ollama.AsyncClient(host=self._base_url)
        except ImportError as e:
            raise LLMError("ollama package not installed. Run: pip install ollama") from e

        # Lazy-init Gemini direct-API client. Only built when a model name with
        # the ':gapi' suffix gets selected — keeps the import + httpx pool out
        # of the boot path for users who never use the direct API.
        self._gemini_direct: Optional[Any] = None
        # Shared httpx client for the async health check — avoids paying TLS
        # / connection setup on every probe of /api/tags. Sync is_available()
        # is one-shot at boot, so it stays plain httpx.get.
        self._health_client: Optional[httpx.AsyncClient] = None

        # Set of model names that need think=False forced for tool-call
        # round-trips. Pre-populated for Gemini-3 Ollama-cloud variants
        # because Ollama-cloud strips the thought_signature field on
        # round-trip — Gemini's API then 400s the next call complaining
        # the prior assistant tool_call lacks a signature, and there's no
        # way to recover *that conversation* once a tool call with thinking
        # has happened. Forcing think=False from the very first request
        # avoids ever poisoning the history. Plain chat() calls (no tools)
        # are unaffected — those work fine with thinking on. We also still
        # learn additional model names dynamically if a new Gemini-3 cloud
        # variant ships and trips the same 400.
        self._tools_force_no_think: set[str] = set()
        for candidate in {self._model, self._vision_model, self._action_model}:
            if candidate and self._needs_no_think_for_tools(candidate):
                self._tools_force_no_think.add(candidate)

    @staticmethod
    def _is_gemini_direct(model_name: str) -> bool:
        """True for models routed through Google's direct API (not via Ollama)."""
        return ":gapi" in (model_name or "")

    @staticmethod
    def _needs_no_think_for_tools(model_name: str) -> bool:
        """Legacy hook, kept only for the startup pre-population path.

        NOTE: forcing think=False does NOT prevent the Gemini-via-Ollama-cloud
        tool-call 400. A probe confirmed think=omitted / False / True all 400
        identically on the round-trip — the `thought_signature` requirement is
        on the functionCall part itself, independent of thinking mode. The 400
        is an Ollama-cloud limitation (it strips the signature); the real fix
        is routing Gemini through its direct API. See chat_with_tools.
        """
        n = (model_name or "").lower()
        return n.startswith("gemini-3") and ":cloud" in n

    @staticmethod
    def _is_cloud_model(model_name: str) -> bool:
        n = (model_name or "").lower()
        return ":cloud" in n or ":gapi" in n

    def _record_model_call(
        self,
        provider: str,
        model: str,
        started: float,
        *,
        ok: bool,
        timeout: bool = False,
        tool_iterations: int = 0,
    ) -> None:
        perf().record_model_call(
            provider,
            model,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            ok=ok,
            timeout=timeout,
            cloud=self._is_cloud_model(model),
            tool_iterations=tool_iterations,
        )

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

    @property
    def action_model(self) -> Optional[str]:
        """Local model used mid-loop after any action tool call (Pattern D).
        Returns None when Pattern D is disabled."""
        return self._action_model

    def set_action_model(self, name: Optional[str]) -> None:
        """Hot-swap the action model. Pass None to disable Pattern D
        entirely (loops stay on the chat model regardless of what tools
        get called)."""
        self._action_model = name or None
        logger.info(f"[LLM] Action model set to '{name or 'disabled'}'")

    def set_settings_provider(self, registry: Any) -> None:
        """Wire the ModelRegistry back into the LLM so chat()/vision_query()
        can pull per-model sampling overrides + thinking toggle on every call.
        Without this, defaults from the model's modelfile are used."""
        self._settings_provider = registry

    async def close(self) -> None:
        """Release long-lived provider clients."""
        if self._gemini_direct is not None:
            try:
                await self._gemini_direct.aclose()
            except Exception:
                pass
            self._gemini_direct = None

    async def _build_options_and_think(self, model_name: str) -> tuple[dict, Optional[bool]]:
        """Translate stored model_settings into Ollama's options dict + think
        kwarg. Skips fields the user hasn't overridden so the model's
        defaults remain in effect for everything not explicitly tuned.

        The think fallback is the global config.ollama.think knob: False
        means "send nothing" (models keep their own default), anything else
        (True / "low" / "medium" / "high") is passed through. A per-model
        thinking_enabled setting always wins over the global knob."""
        default_think = self._default_think if self._default_think is not False else None
        if not hasattr(self, "_settings_provider") or self._settings_provider is None:
            return {}, default_think
        try:
            s = await self._settings_provider.get_settings(model_name)
        except Exception as e:
            logger.debug(f"[LLM] settings lookup failed: {e}")
            return {}, default_think
        if not s:
            return {}, default_think
        options: dict = {}
        # Map our field names to Ollama's option keys (note: repeat_penalty,
        # not repetition_penalty).
        mapping = {
            "temperature":      "temperature",
            "top_p":            "top_p",
            "top_k":            "top_k",
            "min_p":            "min_p",
            "presence_penalty": "presence_penalty",
            "repetition_penalty": "repeat_penalty",
        }
        for ours, theirs in mapping.items():
            v = s.get(ours)
            if v is not None:
                options[theirs] = v
        think = s.get("thinking_enabled")
        if think is None:
            think = default_think
        return options, think

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

        options, think = await self._build_options_and_think(self._model)
        chat_kwargs: dict = {"model": self._model, "messages": messages}
        if options:
            chat_kwargs["options"] = options
        if think is not None:
            chat_kwargs["think"] = think
        started = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                self._client.chat(**chat_kwargs),
                timeout=self._timeout,
            )
            text = response["message"]["content"].strip()
            thinking = response["message"].get("thinking")
            if thinking and self._show_thinking:
                logger.info(f"[LLM] Thinking: {thinking}")
            logger.debug(f"[LLM] Response ({len(text)} chars): {text[:100]}...")
            self._record_model_call("ollama", self._model, started, ok=True)
            return text

        except asyncio.TimeoutError:
            self._record_model_call(
                "ollama", self._model, started, ok=False, timeout=True
            )
            raise LLMError(f"Ollama chat timed out after {self._timeout}s")
        except Exception as e:
            self._record_model_call("ollama", self._model, started, ok=False)
            raise LLMError(f"Ollama chat failed: {e}") from e

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_handlers: dict[str, Any],
        max_iterations: int = 200,
        action_tool_names: Optional[set] = None,
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

        # Pattern D needs per-iteration dispatch (so a Gemini chat model can
        # swap to local Qwen mid-loop). If Pattern D is disabled AND the
        # chat model is :gapi, fall back to GeminiDirect's one-shot loop —
        # it's faster and we don't need swapping anyway.
        if self._is_gemini_direct(self._model) and not self._action_model:
            return await self._get_gemini_direct().chat_with_tools(
                messages=messages,
                tools=tools,
                model=self._model,
                tool_dispatcher=tool_handlers,
                max_iterations=max_iterations,
            )

        working_messages = list(messages)
        # Pattern D state: start every loop on the chat model (best planner).
        # Once any tool call lands in `action_tool_names`, swap to the local
        # action_model for the rest of this loop — that's where the rate-limit-
        # burning, mechanical execution work happens (screenshot/click/type/
        # read_file etc.) and the cheap local model handles it fine.
        action_set = set(action_tool_names or ())
        active_model = self._model
        options, think = await self._build_options_and_think(active_model)
        loop_swapped = False

        for iteration in range(max_iterations):
            # Dispatch this single iteration based on active_model. The
            # OUTER loop (tool execution + swap decision) is unified;
            # only the model call differs.
            if self._is_gemini_direct(active_model):
                # Gemini direct returns normalized {text, function_calls}
                gem_result = await self._get_gemini_direct()._generate(
                    working_messages, active_model, tools=tools,
                )
                text = gem_result.get("text", "")
                gemini_calls = gem_result.get("function_calls") or []
                # Translate Gemini's call shape to Ollama's so the rest of
                # the loop body stays uniform.
                tool_calls = [
                    {"function": {"name": c.get("name"), "arguments": c.get("args") or {}}}
                    for c in gemini_calls
                ]
                message = {"role": "assistant", "content": text}
            else:
                chat_kwargs: dict = {
                    "model": active_model,
                    "messages": working_messages,
                    "tools": tools,
                }
                if options:
                    chat_kwargs["options"] = options
                # Effective think value: per-model setting unless we've
                # learned this model needs think=False for tool calls
                # (Gemini 3 thinking models via Ollama-cloud lose the
                # thought_signature on round-trip and 400 the next call).
                # Learned override is in-memory only — a restart re-learns
                # via one wasted round-trip, fine for a million-context
                # session that's already paid for.
                effective_think = think
                if active_model in self._tools_force_no_think:
                    effective_think = False
                if effective_think is not None:
                    chat_kwargs["think"] = effective_think
                started = time.perf_counter()
                try:
                    response = await asyncio.wait_for(
                        self._client.chat(**chat_kwargs),
                        timeout=self._timeout,
                    )
                except asyncio.TimeoutError:
                    self._record_model_call(
                        "ollama", active_model, started, ok=False, timeout=True
                    )
                    raise LLMError(f"Ollama chat (tools) timed out after {self._timeout}s")
                except Exception as e:
                    # Workaround: Gemini 3 thinking models via Ollama-cloud
                    # 400 with "thought_signature" on tool-call round-trips
                    # because Ollama-cloud doesn't preserve the signature
                    # field. Detect the specific error and retry once with
                    # the trailing assistant/tool messages stripped — the
                    # poisoned history is the actual root, not just whether
                    # think was on. Earlier code only retried when
                    # effective_think was still True; that was wrong, the
                    # error fires even on think=False because Ollama-cloud
                    # decorates assistant tool_calls with a signature
                    # requirement Gemini then complains about on the next
                    # round-trip. Always retry with stripped history; we
                    # also keep the model in _tools_force_no_think so a
                    # restart-re-learn doesn't burn a round-trip.
                    err_text = str(e).lower()
                    looks_like_signature_400 = (
                        "thought_signature" in err_text
                        and "400" in err_text
                    )
                    if looks_like_signature_400:
                        logger.warning(
                            f"[LLM] '{active_model}' tripped Gemini-3 "
                            "thought_signature 400 (effective_think="
                            f"{effective_think}) — stripping trailing "
                            "tool/assistant turns and retrying with "
                            "think=False"
                        )
                        self._tools_force_no_think.add(active_model)
                        chat_kwargs["think"] = False
                        # Setting think=False on the NEXT request isn't enough
                        # if the conversation already contains a prior
                        # assistant tool_call from a think=True turn —
                        # Gemini will 400 again complaining about the
                        # historical message lacking a thought_signature.
                        # Strip any trailing assistant + tool messages so
                        # the retry starts from the user turn intact and
                        # the model regenerates the tool call cleanly.
                        trimmed = list(working_messages)
                        while trimmed and trimmed[-1].get("role") in (
                            "assistant", "tool"
                        ):
                            trimmed.pop()
                        chat_kwargs["messages"] = trimmed
                        working_messages = trimmed
                        self._record_model_call(
                            "ollama", active_model, started, ok=False
                        )
                        started = time.perf_counter()
                        try:
                            response = await asyncio.wait_for(
                                self._client.chat(**chat_kwargs),
                                timeout=self._timeout,
                            )
                        except asyncio.TimeoutError:
                            self._record_model_call(
                                "ollama",
                                active_model,
                                started,
                                ok=False,
                                timeout=True,
                            )
                            raise LLMError(
                                f"Ollama chat (tools) timed out after {self._timeout}s "
                                "(after thought_signature retry)"
                            )
                        except Exception as e2:
                            self._record_model_call(
                                "ollama", active_model, started, ok=False
                            )
                            raise LLMError(
                                f"Ollama chat (tools) failed even after "
                                f"think=False retry: {e2}"
                            ) from e2
                    else:
                        self._record_model_call(
                            "ollama", active_model, started, ok=False
                        )
                        raise LLMError(f"Ollama chat (tools) failed: {e}") from e
                else:
                    self._record_model_call(
                        "ollama", active_model, started, ok=True
                    )

                message = response.get("message", {}) or {}
                thinking = message.get("thinking")
                if thinking and self._show_thinking:
                    logger.info(f"[LLM] Thinking: {thinking}")
                tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                text = (message.get("content") or "").strip()
                logger.debug(
                    f"[LLM] Tool-loop done in {iteration + 1} iter(s), "
                    f"final response ({len(text)} chars): {text[:100]}..."
                )
                perf().record_model_call(
                    "tool-loop",
                    active_model,
                    latency_ms=0.0,
                    ok=True,
                    cloud=self._is_cloud_model(active_model),
                    tool_iterations=iteration + 1,
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

                # Pattern D: as soon as the model uses an action tool, swap
                # to action_model for all subsequent iterations of THIS loop.
                # The first call (planning) stays on the smart model; every
                # follow-up (mechanical execution) hits the local model.
                if (
                    not loop_swapped
                    and self._action_model
                    and name in action_set
                    and self._action_model != active_model
                ):
                    logger.info(
                        f"[LLM/Pattern-D] action tool '{name}' triggered — "
                        f"swapping loop from '{active_model}' to '{self._action_model}'"
                    )
                    active_model = self._action_model
                    options, think = await self._build_options_and_think(active_model)
                    loop_swapped = True

        logger.warning(
            f"[LLM] Tool loop hit max_iterations={max_iterations} without final answer"
        )
        perf().record_model_call(
            "tool-loop",
            active_model,
            latency_ms=0.0,
            ok=False,
            cloud=self._is_cloud_model(active_model),
            tool_iterations=max_iterations,
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

        options, think = await self._build_options_and_think(self._vision_model)
        chat_kwargs: dict = {
            "model": self._vision_model,
            "messages": [{
                "role": "user",
                "content": prompt,
                "images": [img_b64],
            }],
        }
        if options:
            chat_kwargs["options"] = options
        if think is not None:
            chat_kwargs["think"] = think
        started = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                self._client.chat(**chat_kwargs),
                timeout=self._timeout,
            )
            description = response["message"]["content"].strip()
            thinking = response["message"].get("thinking")
            if thinking and self._show_thinking:
                logger.info(f"[LLM] Thinking: {thinking}")
            logger.debug(f"[LLM] Vision: {description[:120]}")
            self._record_model_call("ollama-vision", self._vision_model, started, ok=True)
            return description

        except asyncio.TimeoutError:
            self._record_model_call(
                "ollama-vision",
                self._vision_model,
                started,
                ok=False,
                timeout=True,
            )
            raise LLMError(f"Vision query timed out after {self._timeout}s")
        except Exception as e:
            self._record_model_call(
                "ollama-vision", self._vision_model, started, ok=False
            )
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
            if self._health_client is None or self._health_client.is_closed:
                self._health_client = httpx.AsyncClient(timeout=3)
            r = await self._health_client.get(f"{self._base_url}/api/tags")
            return r.status_code == 200
        except Exception:
            return False

    async def aclose(self) -> None:
        """Release the shared health client. Idempotent; called by the
        orchestrator on shutdown."""
        if self._health_client is not None and not self._health_client.is_closed:
            try:
                await self._health_client.aclose()
            except Exception as e:
                logger.debug(f"[LLM] health client aclose failed: {e}")
        self._health_client = None

    @property
    def system_prompt(self) -> str:
        """The base personality system prompt for this Jarvis instance."""
        return self._system_prompt
