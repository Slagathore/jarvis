"""
JARVIS — Ambient Home AI
========================
Mission: Manage the catalog of LLM models available to Jarvis.

         Lists installed Ollama models, pulls new ones, removes unused ones,
         and tracks per-model capabilities (tool use, vision, context window,
         cloud vs local, thinking-mode toggles, custom user notes). The
         dashboard surfaces all of this so Cole can switch models live and
         see at a glance which ones support what.

         Capabilities for well-known model families ship as a hardcoded
         table (CAPABILITIES_TABLE below) so a freshly-pulled model already
         has correct metadata. Anything outside the table is shown as
         'unknown' and the dashboard exposes free-text notes Cole can fill in.

Modules: modules/brain/model_registry.py
Classes: ModelRegistry, ModelCapabilities (dataclass)
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, asdict
from typing import Any, AsyncIterator, Optional

from loguru import logger


@dataclass
class ModelCapabilities:
    name: str
    is_cloud: bool = False
    context_window: Optional[int] = None       # tokens, None = unknown
    tool_use: bool = False
    vision: bool = False
    thinking_mode: bool = False                # CoT / 'think' modes
    feature_toggles: list[str] = None          # type: ignore[assignment]
    notes: str = ""                            # short capability summary

    def __post_init__(self):
        if self.feature_toggles is None:
            self.feature_toggles = []


# Hardcoded capability table for well-known models. Match by prefix so e.g.
# 'qwen2.5-vl:7b' resolves to the qwen2.5-vl row. Order matters — first match
# wins, so put more-specific prefixes BEFORE less-specific ones (qwen2.5-vl
# before qwen2.5).
#
# Sources: model cards, Ollama library, vendor docs as of 2026-05.
# Cole can override any field via the dashboard's per-model notes.
CAPABILITIES_TABLE: list[ModelCapabilities] = [
    # ── Cloud (Ollama-hosted) ──────────────────────────────────────────────
    ModelCapabilities(
        name="gemini-3-flash-preview",
        is_cloud=True,
        context_window=1_000_000,
        tool_use=True,
        vision=True,
        thinking_mode=False,
        notes="Google Gemini 3 Flash via Ollama Cloud. 1M context, fast tool calling, native image understanding.",
    ),
    ModelCapabilities(
        name="gemini-3-pro",
        is_cloud=True,
        context_window=2_000_000,
        tool_use=True,
        vision=True,
        notes="Slower than flash; better reasoning. 2M context.",
    ),
    # ── Local: Qwen family (best balance for Cole's needs) ────────────────
    ModelCapabilities(
        name="qwen2.5-vl",
        is_cloud=False,
        context_window=128_000,
        tool_use=True,
        vision=True,
        thinking_mode=False,
        notes="Qwen 2.5 vision-language. Best local pick for Jarvis: tool use + vision + 128k. 7B fits 4070 Ti easily.",
    ),
    ModelCapabilities(
        name="qwen3-vl",
        is_cloud=False,
        context_window=256_000,
        tool_use=True,
        vision=True,
        thinking_mode=True,
        feature_toggles=["/think", "/no_think"],
        notes="Qwen 3 vision-language. Toggle CoT with /think + /no_think in the prompt. Strong tool calling.",
    ),
    ModelCapabilities(
        name="qwen3",
        is_cloud=False,
        context_window=128_000,
        tool_use=True,
        vision=False,
        thinking_mode=True,
        feature_toggles=["/think", "/no_think"],
        notes="Qwen 3 text-only. Use qwen3-vl variant if you want vision.",
    ),
    ModelCapabilities(
        name="qwen2.5",
        is_cloud=False,
        context_window=128_000,
        tool_use=True,
        notes="Qwen 2.5 text-only. Tool use yes; no vision (use qwen2.5-vl for that).",
    ),
    # ── Local: Llama family ────────────────────────────────────────────────
    ModelCapabilities(
        name="llama3.3",
        is_cloud=False,
        context_window=128_000,
        tool_use=True,
        notes="Llama 3.3 text-only. Strong tool calling, no native vision.",
    ),
    ModelCapabilities(
        name="llama3.2-vision",
        is_cloud=False,
        context_window=128_000,
        tool_use=False,
        vision=True,
        notes="Vision but limited tool use — not a great solo Jarvis brain. Pair as vision-only model.",
    ),
    ModelCapabilities(
        name="llama3.2",
        is_cloud=False,
        context_window=128_000,
        tool_use=True,
        notes="Llama 3.2 text-only. Smaller and faster than 3.3.",
    ),
    ModelCapabilities(
        name="llama3.1",
        is_cloud=False,
        context_window=128_000,
        tool_use=True,
        notes="Llama 3.1 — older but battle-tested tool calling.",
    ),
    # ── Local: Reasoning / CoT specialists ─────────────────────────────────
    ModelCapabilities(
        name="deepseek-r1",
        is_cloud=False,
        context_window=128_000,
        tool_use=False,
        thinking_mode=True,
        notes="Strong CoT, but limited native tool use — not ideal as Jarvis's primary. Useful as a 'think hard' fallback.",
    ),
    ModelCapabilities(
        name="gpt-oss",
        is_cloud=False,
        context_window=128_000,
        tool_use=True,
        notes="Open-source GPT family. Solid tool calling.",
    ),
    # ── Local: Vision specialists ──────────────────────────────────────────
    ModelCapabilities(
        name="llava",
        is_cloud=False,
        context_window=4_096,
        vision=True,
        tool_use=False,
        notes="Vision-only, weak tool calling. Use only as a paired vision model.",
    ),
    ModelCapabilities(
        name="bakllava",
        is_cloud=False,
        context_window=4_096,
        vision=True,
        notes="LLaVA variant. Vision-only.",
    ),
]


def _strip_tag(model_name: str) -> str:
    """'qwen2.5-vl:7b' -> 'qwen2.5-vl'. ':cloud' suffix preserved by checking
    the cloud flag separately upstream."""
    return model_name.split(":", 1)[0].lower()


def lookup_capabilities(model_name: str) -> ModelCapabilities:
    """Resolve a model name to capabilities. Falls back to an 'unknown' record
    if no prefix matches — dashboard surfaces this as 'capabilities not
    catalogued, edit notes if you know'."""
    base = _strip_tag(model_name)
    is_cloud = ":cloud" in model_name.lower()
    for entry in CAPABILITIES_TABLE:
        if base.startswith(entry.name):
            # Clone so callers can mutate without affecting the shared table
            return ModelCapabilities(
                name=model_name,
                is_cloud=entry.is_cloud or is_cloud,
                context_window=entry.context_window,
                tool_use=entry.tool_use,
                vision=entry.vision,
                thinking_mode=entry.thinking_mode,
                feature_toggles=list(entry.feature_toggles),
                notes=entry.notes,
            )
    return ModelCapabilities(
        name=model_name,
        is_cloud=is_cloud,
        notes="Capabilities not catalogued — edit notes if you know what this model supports.",
    )


class ModelRegistry:
    """Adapter over Ollama's HTTP API plus a SQLite-backed user-notes store.

    Read-only operations (list, capabilities) are fast. Mutations
    (pull, delete, set_active) are async and broadcast notifications so
    the dashboard can show progress / refresh on completion.
    """

    def __init__(
        self,
        db: Any,
        llm: Any,
        config: dict,
        notifier: Optional[Any] = None,
        broadcast: Optional[Any] = None,
    ) -> None:
        self._db = db
        self._llm = llm
        self._config = config
        self._notifier = notifier
        self._broadcast = broadcast

    async def init_schema(self) -> None:
        """Idempotent: create the user notes / overrides table if needed."""
        try:
            await self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS model_notes (
                    model_name TEXT PRIMARY KEY,
                    user_notes TEXT,
                    capability_overrides TEXT  -- JSON; merged onto the catalog row
                )
                """
            )
        except Exception as e:
            logger.warning(f"[ModelRegistry] schema init failed: {e}")

    # ── Listing ─────────────────────────────────────────────────────────────

    async def list_installed(self) -> list[dict]:
        """Return [{name, size_bytes, modified_at, capabilities, user_notes}, ...]."""
        try:
            client = self._llm.client
            resp = await client.list()
        except Exception as e:
            logger.warning(f"[ModelRegistry] ollama list failed: {e}")
            return []

        models = resp.get("models") if isinstance(resp, dict) else getattr(resp, "models", [])
        out: list[dict] = []
        notes_rows = {}
        try:
            for r in await self._db.fetchall("SELECT * FROM model_notes"):
                notes_rows[r["model_name"]] = {
                    "user_notes": r["user_notes"] or "",
                    "overrides": r["capability_overrides"] or "",
                }
        except Exception:
            pass

        for m in models or []:
            name = m.get("model") if isinstance(m, dict) else getattr(m, "model", "")
            if not name:
                # Some ollama versions return 'name' instead
                name = m.get("name") if isinstance(m, dict) else getattr(m, "name", "")
            size = m.get("size") if isinstance(m, dict) else getattr(m, "size", None)
            modified = (
                m.get("modified_at") if isinstance(m, dict) else getattr(m, "modified_at", None)
            )
            caps = lookup_capabilities(name)
            note_row = notes_rows.get(name) or {}
            out.append({
                "name": name,
                "size_bytes": size,
                "modified_at": str(modified) if modified else None,
                "capabilities": asdict(caps),
                "user_notes": note_row.get("user_notes", ""),
                "active_chat": (name == self._llm.model),
                "active_vision": (name == self._llm.vision_model),
            })
        return out

    def list_known_recommendations(self) -> list[dict]:
        """Return the catalog table for the 'pull a new model' UI."""
        return [asdict(c) for c in CAPABILITIES_TABLE]

    # ── Mutations ───────────────────────────────────────────────────────────

    async def pull(self, name: str) -> AsyncIterator[dict]:
        """Async iterator of pull-progress dicts. Caller forwards to a WS or
        progress endpoint. Each dict matches Ollama's stream format."""
        client = self._llm.client
        try:
            async for chunk in await client.pull(name, stream=True):
                # Normalize: ollama's chunk objects vary by version
                if isinstance(chunk, dict):
                    yield chunk
                else:
                    yield {
                        "status": getattr(chunk, "status", "pulling"),
                        "completed": getattr(chunk, "completed", None),
                        "total": getattr(chunk, "total", None),
                        "digest": getattr(chunk, "digest", None),
                    }
        except Exception as e:
            logger.warning(f"[ModelRegistry] pull '{name}' failed: {e}")
            yield {"status": "error", "error": str(e)}

    async def delete(self, name: str) -> bool:
        """Remove a model from the local Ollama install."""
        if name == self._llm.model:
            return False  # don't delete the model that's currently in use
        client = self._llm.client
        try:
            await client.delete(name)
            return True
        except Exception as e:
            logger.warning(f"[ModelRegistry] delete '{name}' failed: {e}")
            return False

    async def set_active(self, name: str, kind: str = "chat") -> bool:
        """Hot-swap the active model. kind ∈ {'chat', 'vision'}.
        Caller is responsible for any UI notification — we just flip the
        in-memory pointer and update config.yaml-on-disk so the new value
        survives restart."""
        if kind == "vision":
            self._llm.set_vision_model(name)
        else:
            self._llm.set_active_model(name)
        # Persist to config so a restart picks it up. Best-effort — log on failure.
        try:
            from pathlib import Path as _Path
            import yaml as _yaml
            cfg_path = _Path(__file__).resolve().parents[2] / "config.yaml"
            data = _yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "ollama" in data:
                key = "vision_model" if kind == "vision" else "model"
                data["ollama"][key] = name
                cfg_path.write_text(_yaml.dump(data, sort_keys=False), encoding="utf-8")
        except Exception as e:
            logger.warning(f"[ModelRegistry] config persist failed: {e}")
        if self._notifier is not None:
            try:
                await self._notifier.notify(
                    kind="model.switched",
                    title=f"Active {kind} model: {name}",
                    message="Hot-swapped — takes effect on the next request.",
                    severity="info",
                )
            except Exception:
                pass
        return True

    async def set_user_notes(self, name: str, notes: str) -> bool:
        try:
            await self._db.execute(
                "INSERT INTO model_notes (model_name, user_notes) VALUES (?, ?) "
                "ON CONFLICT(model_name) DO UPDATE SET user_notes = excluded.user_notes",
                (name, notes),
            )
            return True
        except Exception as e:
            logger.warning(f"[ModelRegistry] set_user_notes failed: {e}")
            return False
