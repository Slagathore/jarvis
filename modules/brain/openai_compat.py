"""
JARVIS — Ambient Home AI
========================
Mission: OpenAICompatClient — a side-route that talks to Ollama through its
         OpenAI-compatible `/v1` endpoint instead of the native API.

         WHY THIS EXISTS — a narrow, model-specific workaround:
         Ollama's native API is lossy with Gemini-3's `thought_signature`.
         Gemini-3 attaches that opaque token to every functionCall it emits
         and REQUIRES it back on the functionCall part when the conversation
         history is re-sent for a tool round-trip. Ollama's native
         deserialization drops it (ollama/ollama#14567), so the second
         request 400s "missing a thought_signature" — every multi-turn tool
         call with `gemini-3-flash-preview:cloud` fails.

         Ollama's `/v1` OpenAI-compatible surface does NOT drop it: the
         assistant tool-call message is round-tripped verbatim. Probed and
         confirmed working. So models known to hit the native-API bug are
         routed here instead — see config.ollama.openai_compat_models and
         OllamaLLM._uses_openai_compat.

         This is deliberately a SEPARATE pathway, not a replacement: it is a
         per-model opt-in, and the upstream Ollama fix (PR #14676) will make
         it unnecessary. When that lands, empty the config list and this
         module simply stops being reached.

Modules: modules/brain/openai_compat.py
Classes: OpenAICompatClient
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Any, Optional

import httpx
from loguru import logger

from core.exceptions import LLMError
from modules.context.perf_tracker import perf


class OpenAICompatClient:
    """Async client for an OpenAI-compatible /chat/completions endpoint —
    here, Ollama's `/v1`. Mirrors the slice of the OllamaLLM surface the
    dispatcher needs: chat / chat_with_tools / vision_query.

    The base_url is the Ollama base (e.g. http://localhost:11434); `/v1`
    is appended. Model names are passed through unchanged (the `:cloud`
    suffix is part of the real model id at the /v1 surface)."""

    def __init__(self, base_url: str = "http://localhost:11434",
                 timeout: float = 30.0) -> None:
        self._endpoint = base_url.rstrip("/") + "/v1/chat/completions"
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None

    # ── Core POST ────────────────────────────────────────────────────────────

    async def _post(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict]] = None,
    ) -> dict:
        """One /chat/completions call. Returns the parsed JSON response.
        Raises LLMError on transport / HTTP failure."""
        body: dict = {"model": model, "messages": messages}
        if tools:
            body["tools"] = tools
        client = await self._get_client()
        started = time.perf_counter()
        try:
            resp = await client.post(
                self._endpoint,
                headers={
                    # Ollama's /v1 ignores the key for local use and handles
                    # cloud-model auth itself; a token is still required to
                    # be present.
                    "Authorization": "Bearer ollama",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            self._record(model, started, ok=False)
            raise LLMError(
                f"openai-compat {e.response.status_code}: "
                f"{e.response.text[:300]}"
            ) from e
        except Exception as e:
            self._record(model, started, ok=False,
                         timeout=isinstance(e, httpx.TimeoutException))
            raise LLMError(f"openai-compat request failed: {e}") from e
        self._record(model, started, ok=True)
        return data

    def _record(self, model: str, started: float, *, ok: bool,
                timeout: bool = False) -> None:
        try:
            perf().record_model_call(
                "openai-compat", model,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                ok=ok, timeout=timeout, cloud=True,
            )
        except Exception:
            pass

    @staticmethod
    def _content_text(message: dict) -> str:
        c = message.get("content")
        if isinstance(c, str):
            return c.strip()
        # Some servers return content as a list of parts.
        if isinstance(c, list):
            return "".join(
                p.get("text", "") for p in c if isinstance(p, dict)
            ).strip()
        return ""

    # ── Public surface ───────────────────────────────────────────────────────

    async def chat(self, messages: list[dict[str, Any]], model: str) -> str:
        """Plain chat completion — returns the assistant's text."""
        data = await self._post(model, messages)
        try:
            return self._content_text(data["choices"][0]["message"])
        except (KeyError, IndexError) as e:
            raise LLMError(f"openai-compat: malformed response: {e}") from e

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict],
        model: str,
        tool_dispatcher: Any,
        max_iterations: int = 200,
    ) -> str:
        """Tool-loop chat. Mirrors OllamaLLM.chat_with_tools semantics.

        The assistant tool-call message is appended to the history exactly
        as the server returned it — that verbatim round-trip is what keeps
        Gemini-3's thought_signature intact (the whole point of this route).

        tool_dispatcher is the {name: callable} mapping; callables may be
        sync or async.
        """
        running = list(messages)
        text = ""
        for iteration in range(max_iterations):
            data = await self._post(model, running, tools=tools)
            try:
                msg = data["choices"][0]["message"]
            except (KeyError, IndexError) as e:
                raise LLMError(f"openai-compat: malformed response: {e}") from e
            text = self._content_text(msg)
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                logger.debug(
                    f"[OpenAICompat] tool-loop done in {iteration + 1} "
                    f"iter(s), final ({len(text)} chars)"
                )
                return text
            # Append the assistant turn VERBATIM — preserves any signature /
            # reasoning fields the server attached to the tool_calls.
            running.append(msg)
            for tc in tool_calls:
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                raw_args = fn.get("arguments")
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args or "{}")
                    except json.JSONDecodeError:
                        args = {}
                elif isinstance(raw_args, dict):
                    args = raw_args
                else:
                    args = {}
                handler = (
                    tool_dispatcher.get(name)
                    if isinstance(tool_dispatcher, dict) else None
                )
                if handler is None:
                    result: Any = {"error": f"unknown tool '{name}'"}
                else:
                    try:
                        result = (
                            await handler(**args)
                            if asyncio.iscoroutinefunction(handler)
                            else handler(**args)
                        )
                    except Exception as e:
                        result = {"error": f"{type(e).__name__}: {e}"}
                running.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id"),
                    "content": json.dumps(result, default=str),
                })
        logger.warning(
            f"[OpenAICompat] tool-loop hit max_iterations ({max_iterations})"
        )
        return text or "(tool loop exhausted)"

    async def vision_query(
        self,
        image_bytes: bytes,
        prompt: str,
        model: str,
        mime_type: str = "image/jpeg",
    ) -> str:
        """Single-image vision query via the OpenAI image_url content part."""
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
            ],
        }]
        data = await self._post(model, messages)
        try:
            return self._content_text(data["choices"][0]["message"])
        except (KeyError, IndexError) as e:
            raise LLMError(f"openai-compat: malformed response: {e}") from e
