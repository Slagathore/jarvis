"""
JARVIS — Ambient Home AI
========================
Mission: Direct client for Google's Gemini API (generativelanguage.googleapis.com).

         Bypasses Ollama for the lowest possible cloud latency. Used as a
         drop-in alternative to OllamaLLM.chat / chat_with_tools / vision_query
         when the active model name carries the ':gapi' suffix.

         API key: read from GEMINI_API_KEY env var or .env file. Rotate keys
         that have been pasted in chat or shared in screenshots.

Modules: modules/brain/gemini_direct.py
Classes: GeminiDirectClient
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from typing import Any, Optional

import httpx
from loguru import logger
from modules.context.perf_tracker import perf

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def _strip_gapi_suffix(model_name: str) -> str:
    """'gemini-3-flash-preview:gapi' -> 'gemini-3-flash-preview'."""
    return model_name.replace(":gapi", "").strip()


def _convert_messages_to_gemini(messages: list[dict[str, Any]]) -> tuple[list[dict], Optional[str]]:
    """
    Translate OpenAI-format chat messages to Gemini's contents+systemInstruction
    format.

    Returns (contents, system_instruction). Gemini requires alternating
    user/model roles; consecutive same-role messages get merged.
    """
    system_instr: Optional[str] = None
    contents: list[dict] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            # Gemini puts the system prompt in a separate field, not in contents
            if system_instr is None:
                system_instr = content
            else:
                system_instr = system_instr + "\n\n" + content
            continue
        gemini_role = "model" if role == "assistant" else "user"
        # Tool/function results land here too — Gemini wants role=function
        if role == "tool":
            contents.append({
                "role": "function",
                "parts": [{
                    "functionResponse": {
                        "name": msg.get("name", "tool"),
                        "response": {"content": content},
                    }
                }],
            })
            continue
        function_calls = msg.get("_gemini_function_calls") or []
        if role == "assistant" and function_calls:
            parts = []
            if content:
                parts.append({"text": content})
            for call in function_calls:
                part: dict = {
                    "functionCall": {
                        "name": call.get("name"),
                        "args": call.get("args") or {},
                    }
                }
                # Re-attach the thoughtSignature captured from the original
                # response (_extract_function_calls). Gemini-3 rejects a
                # functionCall part in history that lacks it.
                sig = call.get("thought_signature")
                if sig:
                    part["thoughtSignature"] = sig
                parts.append(part)
            contents.append({"role": "model", "parts": parts})
            continue
        # Merge consecutive same-role messages
        if contents and contents[-1].get("role") == gemini_role:
            contents[-1]["parts"].append({"text": content})
        else:
            contents.append({"role": gemini_role, "parts": [{"text": content}]})
    return contents, system_instr


def _convert_tools_to_gemini(ollama_tools: list[dict]) -> list[dict]:
    """Translate Ollama-style tool defs to Gemini's tools/functionDeclarations.

    Ollama tool format:
      {"type": "function", "function": {"name", "description", "parameters": {...JSON schema...}}}

    Gemini tool format:
      {"functionDeclarations": [{"name", "description", "parameters": {...JSON schema...}}]}
    """
    declarations = []
    for t in ollama_tools or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if not fn:
            continue
        declarations.append({
            "name": fn.get("name"),
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
        })
    if not declarations:
        return []
    return [{"functionDeclarations": declarations}]


class GeminiDirectClient:
    """Async client for Google's Gemini direct API.

    Exposes chat / chat_with_tools / vision_query — the same surface as
    OllamaLLM so the dispatcher can swap freely. All methods accept the
    target model name explicitly so a single client serves multiple models.
    """

    def __init__(self, api_key: Optional[str] = None, timeout: float = 30.0) -> None:
        self._api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        if not self._api_key:
            logger.warning("[Gemini] No GEMINI_API_KEY set — direct API calls will fail")

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

    @property
    def has_key(self) -> bool:
        return bool(self._api_key)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
    ) -> str:
        """Plain chat completion. Returns the model's text response."""
        return (await self._generate(messages, model, tools=None)).get("text", "")

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict],
        model: str,
        tool_dispatcher: Any,
        max_iterations: int = 200,
    ) -> str:
        """Tool-loop chat. Mirrors OllamaLLM.chat_with_tools semantics:
        passes tool defs, executes any tool calls the model emits via
        tool_dispatcher, feeds results back, loops until the model returns
        a plain text response or we hit max_iterations.

        tool_dispatcher is the same {name: async fn} mapping used by the
        Ollama path — see orchestrator._build_tool_dispatcher.
        """
        running_messages = list(messages)
        text = ""
        for iteration in range(max_iterations):
            result = await self._generate(running_messages, model, tools=tools)
            text = result.get("text", "")
            calls = result.get("function_calls") or []
            if not calls:
                logger.debug(
                    f"[Gemini] Tool-loop done in {iteration + 1} iter(s), "
                    f"final response ({len(text)} chars): {text[:120]}..."
                )
                perf().record_model_call(
                    "tool-loop",
                    _strip_gapi_suffix(model),
                    latency_ms=0.0,
                    ok=True,
                    cloud=True,
                    tool_iterations=iteration + 1,
                )
                return text
            # Append the model's tool-call turn so Gemini sees it on the next round
            running_messages.append({
                "role": "assistant",
                "content": text or "",
                "_gemini_function_calls": calls,
            })
            # Dispatch each call and append a tool message for each
            for call in calls:
                fn_name = call.get("name") or ""
                fn_args = call.get("args") or {}
                handler = (
                    tool_dispatcher.get(fn_name)
                    if isinstance(tool_dispatcher, dict)
                    else None
                )
                if handler is None:
                    tool_result = {"error": f"unknown tool '{fn_name}'"}
                else:
                    try:
                        tool_result = await handler(**fn_args) if asyncio.iscoroutinefunction(handler) else handler(**fn_args)
                    except Exception as e:
                        tool_result = {"error": f"{type(e).__name__}: {e}"}
                running_messages.append({
                    "role": "tool",
                    "name": fn_name,
                    "content": json.dumps(tool_result, default=str),
                })
        # Out of iterations — return whatever last text we have
        perf().record_model_call(
            "tool-loop",
            _strip_gapi_suffix(model),
            latency_ms=0.0,
            ok=False,
            cloud=True,
            tool_iterations=max_iterations,
        )
        return text or "(tool loop exhausted)"

    async def vision_query(
        self,
        image_bytes: bytes,
        prompt: str,
        model: str,
        mime_type: str = "image/jpeg",
    ) -> str:
        """Send a single image + prompt; return the model's description."""
        api_model = _strip_gapi_suffix(model)
        url = f"{GEMINI_BASE_URL}/models/{api_model}:generateContent"
        body = {
            "contents": [{
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {"inlineData": {
                        "mimeType": mime_type,
                        "data": base64.b64encode(image_bytes).decode("ascii"),
                    }},
                ],
            }],
        }
        client = await self._get_client()
        started = time.perf_counter()
        try:
            resp = await client.post(
                url,
                headers={"X-goog-api-key": self._api_key, "Content-Type": "application/json"},
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            perf().record_model_call(
                "gemini-vision",
                api_model,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                ok=False,
                timeout=isinstance(e, httpx.TimeoutException),
                cloud=True,
            )
            logger.warning(f"[Gemini] vision_query failed: {e}")
            return ""
        perf().record_model_call(
            "gemini-vision",
            api_model,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            ok=True,
            cloud=True,
        )
        return self._extract_text(data)

    async def _generate(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: Optional[list[dict]],
    ) -> dict:
        """Single call to generateContent. Returns {"text", "function_calls"}."""
        api_model = _strip_gapi_suffix(model)
        url = f"{GEMINI_BASE_URL}/models/{api_model}:generateContent"
        contents, system_instr = _convert_messages_to_gemini(messages)
        body: dict = {"contents": contents}
        if system_instr:
            body["systemInstruction"] = {"parts": [{"text": system_instr}]}
        if tools:
            converted = _convert_tools_to_gemini(tools)
            if converted:
                body["tools"] = converted
                # Auto mode lets the model decide; ANY would force a tool call
                body["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}

        client = await self._get_client()
        started = time.perf_counter()
        try:
            resp = await client.post(
                url,
                headers={"X-goog-api-key": self._api_key, "Content-Type": "application/json"},
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as e:
            perf().record_model_call(
                "gemini",
                api_model,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                ok=False,
                cloud=True,
            )
            logger.warning(f"[Gemini] generate {e.response.status_code}: {e.response.text[:300]}")
            return {"text": "", "function_calls": []}
        except Exception as e:
            perf().record_model_call(
                "gemini",
                api_model,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                ok=False,
                timeout=isinstance(e, httpx.TimeoutException),
                cloud=True,
            )
            logger.warning(f"[Gemini] generate failed: {e}")
            return {"text": "", "function_calls": []}

        perf().record_model_call(
            "gemini",
            api_model,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            ok=True,
            cloud=True,
        )
        return {
            "text": self._extract_text(data),
            "function_calls": self._extract_function_calls(data),
        }

    def _extract_text(self, data: dict) -> str:
        try:
            candidates = data.get("candidates") or []
            if not candidates:
                return ""
            parts = candidates[0].get("content", {}).get("parts", []) or []
            chunks = [p.get("text", "") for p in parts if "text" in p]
            return "".join(chunks).strip()
        except Exception:
            return ""

    def _extract_function_calls(self, data: dict) -> list[dict]:
        calls: list[dict] = []
        try:
            candidates = data.get("candidates") or []
            if not candidates:
                return calls
            parts = candidates[0].get("content", {}).get("parts", []) or []
            for p in parts:
                fc = p.get("functionCall")
                if fc:
                    call = {"name": fc.get("name"), "args": fc.get("args") or {}}
                    # Gemini-3 attaches a thoughtSignature to each functionCall
                    # part. It is REQUIRED back on that part when the history
                    # is re-sent for the tool round-trip — drop it and the
                    # next request 400s "missing a thought_signature". Capture
                    # it here; _convert_messages_to_gemini re-emits it.
                    sig = p.get("thoughtSignature")
                    if sig:
                        call["thought_signature"] = sig
                    calls.append(call)
        except Exception:
            pass
        return calls
