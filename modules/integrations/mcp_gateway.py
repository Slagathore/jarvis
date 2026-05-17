"""
JARVIS — Integrations / MCP Gateway
===================================
Mission: MCPGateway — a Model Context Protocol client (audit roadmap B4).
         It connects to configured MCP servers, discovers the tools they
         expose, and surfaces them to Jarvis's LLM tool registry so the
         assistant can call them like any native tool.

         This is what the long-dead IntegrationRegistry scaffolding was
         always meant to become: a plug-in surface for external
         capability (Home Assistant, filesystem, web, GitHub, ...) that
         does NOT require bespoke per-service code in the orchestrator.

         Tools are namespaced `mcp__<server>__<tool>` so they never
         collide with native tools and the LLM can see which server a
         capability comes from.

         TRANSPORT: stdio (the gateway spawns each server as a
         subprocess). Sessions are held open for the gateway's lifetime
         via a single AsyncExitStack, closed in stop().

         GRACEFUL: the `mcp` package is an optional dependency. Absent —
         or `mcp.enabled: false` — the gateway loads disabled, exposes
         zero tools, and the rest of Jarvis is unaffected.

Modules: modules/integrations/mcp_gateway.py
Classes: MCPGateway
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import Any, Optional

from loguru import logger


def _tool_name(server: str, tool: str) -> str:
    return f"mcp__{server}__{tool}"


class MCPGateway:
    """Connects to configured MCP servers and bridges their tools into
    the orchestrator's LLM tool registry.

    Config (from config["mcp"], all optional):
        enabled: bool — master switch (default True)
        servers: list of {name, command, args?, env?} — each is an MCP
                 server launched over stdio.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", True))
        self._server_cfgs: list[dict] = list(cfg.get("servers", []) or [])
        self._stack: Optional[AsyncExitStack] = None
        # server name → live mcp.ClientSession
        self._sessions: dict[str, Any] = {}
        # tool name (mcp__server__tool) → (server, raw_tool_name)
        self._tools: dict[str, tuple[str, str]] = {}
        # tool name → OpenAI-format function schema
        self._schemas: dict[str, dict] = {}
        self.loaded = False

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Connect to every configured MCP server and discover its tools.
        Blocking-ish (does the stdio handshake). Never raises — a server
        that fails to connect is skipped; the others still come up."""
        if not self.enabled or not self._server_cfgs:
            logger.info("[MCP] disabled or no servers configured")
            return
        try:
            from mcp import (  # type: ignore[import-not-found]
                ClientSession, StdioServerParameters)
            from mcp.client.stdio import (  # type: ignore[import-not-found]
                stdio_client)
        except ImportError:
            logger.warning(
                "[MCP] 'mcp' package not installed — gateway disabled "
                "(pip install mcp)"
            )
            return

        self._stack = AsyncExitStack()
        for sc in self._server_cfgs:
            name = sc.get("name")
            command = sc.get("command")
            if not name or not command:
                logger.warning(f"[MCP] skipping server with no name/command: {sc}")
                continue
            try:
                params = StdioServerParameters(
                    command=command, args=list(sc.get("args", []) or []),
                    env=sc.get("env") or None,
                )
                read, write = await self._stack.enter_async_context(
                    stdio_client(params))
                session = await self._stack.enter_async_context(
                    ClientSession(read, write))
                await session.initialize()
                self._sessions[name] = session
                await self._discover_tools(name, session)
            except Exception as e:
                logger.warning(f"[MCP] server '{name}' failed to connect: {e}")
        self.loaded = bool(self._sessions)
        logger.info(
            f"[MCP] gateway ready — {len(self._sessions)} server(s), "
            f"{len(self._tools)} tool(s)"
        )

    async def _discover_tools(self, server: str, session: Any) -> None:
        """tools/list against one server; register each tool."""
        try:
            result = await session.list_tools()
        except Exception as e:
            logger.warning(f"[MCP] '{server}' tools/list failed: {e}")
            return
        for tool in getattr(result, "tools", []) or []:
            raw = getattr(tool, "name", None)
            if not raw:
                continue
            full = _tool_name(server, raw)
            self._tools[full] = (server, raw)
            schema = getattr(tool, "inputSchema", None) or {
                "type": "object", "properties": {}
            }
            self._schemas[full] = {
                "type": "function",
                "function": {
                    "name": full,
                    "description": (getattr(tool, "description", "") or
                                    f"{raw} (via MCP server '{server}')"),
                    "parameters": schema,
                },
            }
        logger.info(f"[MCP] '{server}' exposed "
                    f"{sum(1 for s, _ in self._tools.values() if s == server)} "
                    f"tool(s)")

    async def stop(self) -> None:
        """Close every MCP session + the server subprocesses."""
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except Exception as e:
                logger.debug(f"[MCP] stack close failed: {e}")
            self._stack = None
        self._sessions.clear()
        self._tools.clear()
        self._schemas.clear()
        self.loaded = False

    # ── Tool-registry bridge ─────────────────────────────────────────────────

    def tool_registry(self) -> tuple[list[dict], dict]:
        """Return (tools_schema, name->handler) for _build_tool_registry to
        merge. Each handler routes back through call() to the owning
        server. Empty when the gateway is not loaded."""
        if not self.loaded:
            return [], {}
        tools = list(self._schemas.values())
        handlers = {name: self._make_handler(name) for name in self._tools}
        return tools, handlers

    def _make_handler(self, full_name: str):
        async def _handler(**kwargs) -> Any:
            return await self.call(full_name, kwargs)
        return _handler

    async def call(self, full_name: str, arguments: dict) -> Any:
        """Invoke an MCP tool by its namespaced name. Returns the tool's
        text output, or an {"error": ...} dict on failure."""
        entry = self._tools.get(full_name)
        if entry is None:
            return {"error": f"unknown MCP tool '{full_name}'"}
        server, raw = entry
        session = self._sessions.get(server)
        if session is None:
            return {"error": f"MCP server '{server}' not connected"}
        try:
            result = await asyncio.wait_for(
                session.call_tool(raw, arguments or {}), timeout=30.0)
        except Exception as e:
            logger.debug(f"[MCP] call {full_name} failed: {e}")
            return {"error": f"MCP call failed: {e}"}
        return self._extract_content(result)

    @staticmethod
    def _extract_content(result: Any) -> Any:
        """Flatten an MCP CallToolResult to text — most tools return one
        or more TextContent items."""
        parts: list[str] = []
        for item in getattr(result, "content", []) or []:
            text = getattr(item, "text", None)
            if text is not None:
                parts.append(str(text))
        if parts:
            return "\n".join(parts)
        # Fall back to a structured result if there was no text content.
        sc = getattr(result, "structuredContent", None)
        return sc if sc is not None else "(no content)"

    def status(self) -> dict:
        """Dashboard view — which servers are up and their tool counts."""
        per_server: dict[str, int] = {}
        for server, _ in self._tools.values():
            per_server[server] = per_server.get(server, 0) + 1
        return {
            "enabled": self.enabled,
            "loaded": self.loaded,
            "servers": [
                {"name": s, "tools": per_server.get(s, 0)}
                for s in self._sessions
            ],
        }
