"""
JARVIS — Ambient Home AI
========================
Mission: TriageGate — Stage 4 of the voice cascade. Speech reached this
         stage because the VAD opened, no wake word fired, and no sound
         event matched — but it was still transcribed. Most such speech
         is ambient household chatter Jarvis should ignore. This stage
         asks a small, permanently-loaded LLM one binary question:
         "is this speech actually directed at Jarvis, or about something
         Jarvis should act on?"

         Only on a YES does the transcript escalate to the full
         assistant. This is the single place a tiny LLM earns its keep —
         a cheap pattern-match against escalation rules, not reasoning.

         MODEL: configurable; a ~0.6-3B instruction model kept resident
         in Ollama. Output is constrained with Ollama's `format: json`
         so the verdict parses reliably regardless of model.

         FAIL-CLOSED: any failure (model missing, timeout, bad JSON)
         yields escalate=False. Non-wake-word speech is ambient by
         default — a broken gate must not start spamming the assistant.

Modules: modules/brain/triage_gate.py
Classes: TriageGate
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

import httpx
from loguru import logger

_SYSTEM_PROMPT = """You are a triage filter for a home AI assistant named Jarvis.
You receive a transcript of speech that did NOT contain the wake word.
Decide if this speech should escalate to the full assistant.

Escalate (true) if:
- The speaker asks a question to no one in particular ("what time is it")
- The speaker references Jarvis or the assistant directly
- The speaker mentions a household event (timer, alarm, reminder, lights)
- The speaker seems distressed or asking for help

Do NOT escalate (false) if:
- It is conversation between people, not directed at the assistant
- It is background TV / media / music / song lyrics
- It is routine household chatter

Respond with JSON only: {"escalate": <true|false>, "reason": "<one short sentence>"}"""


class TriageGate:
    """Binary escalation classifier for non-wake-word speech.

    Config keys (from config["voice"]["triage"], all optional):
        enabled:     bool  — master switch (default True)
        model:       Ollama model tag (default "qwen3:0.6b")
        base_url:    Ollama base URL (default http://localhost:11434)
        timeout_s:   per-call timeout (default 6.0 — it is a fast gate)
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", True))
        self._model = str(cfg.get("model", "qwen3:0.6b"))
        self._base = str(cfg.get("base_url", "http://localhost:11434")).rstrip("/")
        self._timeout = float(cfg.get("timeout_s", 6.0))
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

    async def should_escalate(
        self, transcript: str, *, context: Optional[str] = None
    ) -> dict:
        """Classify one transcript. Returns {escalate: bool, reason: str,
        latency_ms: float}. Fails closed — any error yields escalate=False."""
        text = (transcript or "").strip()
        if not self.enabled or not text:
            return {"escalate": False, "reason": "gate disabled or empty",
                    "latency_ms": 0.0}

        user_msg = f'Transcript: "{text}"'
        if context:
            user_msg = f"Context: {context}\n{user_msg}"

        started = time.perf_counter()
        try:
            client = await self._get_client()
            resp = await client.post(
                f"{self._base}/api/chat",
                json={
                    "model": self._model,
                    "stream": False,
                    "format": "json",  # constrain output to valid JSON
                    "options": {"temperature": 0.0},
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                },
            )
            resp.raise_for_status()
            content = (resp.json().get("message") or {}).get("content", "")
            verdict = json.loads(content)
            latency = (time.perf_counter() - started) * 1000.0
            escalate = bool(verdict.get("escalate", False))
            reason = str(verdict.get("reason", "")).strip() or "(no reason)"
            logger.debug(
                f"[Triage] {'ESCALATE' if escalate else 'ignore'} "
                f"({latency:.0f}ms) — {reason}"
            )
            return {"escalate": escalate, "reason": reason,
                    "latency_ms": round(latency, 1)}
        except Exception as e:
            # Fail closed: a broken gate must not escalate ambient chatter.
            logger.debug(f"[Triage] gate failed ({type(e).__name__}: {e}) "
                         f"— defaulting to no-escalate")
            return {"escalate": False, "reason": f"triage unavailable: {e}",
                    "latency_ms": round((time.perf_counter() - started) * 1000.0, 1)}
