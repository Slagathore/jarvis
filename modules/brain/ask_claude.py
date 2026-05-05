"""
JARVIS — Ambient Home AI
========================
Mission: Thin async wrapper around Anthropic's Claude API. Used by the
         `ask_claude` LLM tool so Jarvis (running on a smaller local
         model) can escalate hard questions — debugging a stack trace,
         reasoning about a tricky design, getting a second opinion on
         code — to a stronger reasoning model.

         Reads ANTHROPIC_API_KEY from env (or .env). Default model is
         claude-sonnet-4-6 — fast enough for interactive use, strong on
         code/reasoning. Can be overridden per-call.

Modules: modules/brain/ask_claude.py
Classes: ClaudeClient
"""

from __future__ import annotations

import os
from typing import Optional

from loguru import logger

DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"


class ClaudeClient:
    """Lazy-init wrapper. Constructing this is cheap; the underlying
    AsyncAnthropic client is built on first call."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self._client = None
        if not self._api_key:
            logger.warning("[Claude] No ANTHROPIC_API_KEY set — ask_claude tool will fail")

    @property
    def has_key(self) -> bool:
        return bool(self._api_key)

    def _get_client(self):
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as e:
                raise RuntimeError(
                    "anthropic package not installed. Run: pip install anthropic"
                ) from e
            self._client = AsyncAnthropic(api_key=self._api_key)
        return self._client

    async def ask(
        self,
        question: str,
        context: Optional[str] = None,
        model: str = DEFAULT_CLAUDE_MODEL,
        max_tokens: int = 2048,
    ) -> str:
        """Single-turn ask. Returns Claude's text response.

        context is optional — usually a relevant code snippet, error message,
        or background. Kept separate from question so the system role can be
        used to frame Claude as 'a senior engineer Jarvis is consulting'.
        """
        client = self._get_client()
        system = (
            "You are a senior engineer being consulted by Jarvis (a local AI "
            "running a smaller model). Give direct, technically precise answers. "
            "If the question is ambiguous, state your assumption and answer. "
            "If code is involved, prefer concrete examples over high-level "
            "explanation. Keep responses under ~500 words unless the user "
            "explicitly asks for depth."
        )
        user_content = question if not context else f"{question}\n\nContext:\n{context}"
        try:
            resp = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_content}],
            )
            chunks = []
            for block in resp.content or []:
                # Newer SDKs return TextBlock objects
                text = getattr(block, "text", None)
                if text:
                    chunks.append(text)
                elif isinstance(block, dict) and block.get("type") == "text":
                    chunks.append(block.get("text", ""))
            return "".join(chunks).strip()
        except Exception as e:
            logger.warning(f"[Claude] ask failed: {e}")
            return f"(ask_claude failed: {e})"
