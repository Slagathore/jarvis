"""
JARVIS — Ambient Home AI
========================
Mission: Bridge Jarvis to external automation tools (IFTTT, Home Assistant,
         Zapier, custom scripts, etc.) via two HTTP webhook channels:

         INBOUND  — external services POST to /api/webhook/{name}; we publish
                    a "webhook.{name}" event onto the internal event bus that
                    other modules can subscribe to. Token-gated for security.

         OUTBOUND — when configured event types fire on the internal bus, we
                    POST the payload to each configured URL. e.g. fire a Home
                    Assistant scene when a person is recognized, send to IFTTT
                    when a reminder fires, etc.

         Both ends are config-driven so users can wire arbitrary automations
         without code changes.

Modules: modules/network/webhooks.py
Classes: WebhookManager
Functions:
    WebhookManager.__init__(config, event_bus)    — Wire deps
    WebhookManager.load()                          — Open HTTP client + subscribe outbound
    WebhookManager.close()                         — Release HTTP client
    WebhookManager.trigger_inbound(name, token, payload)
                                                   — Validate token + dispatch onto bus
    WebhookManager.list_outbound_routes()          — For dashboard / diagnostics
"""

import asyncio
from typing import Any, Optional

import httpx
from loguru import logger


class WebhookManager:
    """
    Two-way HTTP bridge between Jarvis's event bus and external services.

    Config (`webhooks` section):
        inbound_tokens:
          ifttt_lights: "abc123…"      # /api/webhook/ifttt_lights requires X-Webhook-Token: abc123…
          ha_doorbell:  "..."

        outbound:
          speech:                      # event type → list of URLs to POST
            - "https://maker.ifttt.com/trigger/jarvis_spoke/with/key/XYZ"
          person_recognized:
            - "http://homeassistant.local:8123/api/webhook/jarvis_person"
          reminder.due:
            - "https://hooks.zapier.com/hooks/catch/.../"
    """

    def __init__(self, config: dict, event_bus) -> None:
        cfg = config.get("webhooks", {}) if isinstance(config.get("webhooks"), dict) else {}
        self._bus = event_bus
        self._inbound_tokens: dict[str, str] = dict(cfg.get("inbound_tokens", {}) or {})
        self._outbound: dict[str, list[str]] = {
            str(k): [str(u) for u in (v or [])]
            for k, v in (cfg.get("outbound", {}) or {}).items()
        }
        self._timeout: float = float(cfg.get("timeout_seconds", 10))
        self._http: Optional[httpx.AsyncClient] = None

    async def load(self) -> None:
        """Open the shared HTTP client + register outbound subscriptions on the bus."""
        self._http = httpx.AsyncClient(timeout=self._timeout)
        for event_type, urls in self._outbound.items():
            if not urls:
                continue
            handler = self._make_outbound_handler(event_type, urls)
            self._bus.subscribe(event_type, handler)
            logger.info(
                f"[Webhooks] Outbound: {event_type} → {len(urls)} URL(s)"
            )
        in_count = len(self._inbound_tokens)
        out_count = sum(len(v) for v in self._outbound.values())
        logger.info(
            f"[Webhooks] Ready ({in_count} inbound token(s), {out_count} outbound URL(s))"
        )

    async def close(self) -> None:
        """Release the HTTP client."""
        if self._http is not None:
            try:
                await self._http.aclose()
            except Exception:
                pass
            self._http = None

    def _make_outbound_handler(self, event_type: str, urls: list[str]):
        """Build an async handler that POSTs each event payload to all URLs in parallel."""
        async def _handler(event: Any) -> None:
            payload = {"event_type": event_type, "event": event}
            await asyncio.gather(
                *(self._post_one(url, payload) for url in urls),
                return_exceptions=True,
            )
        return _handler

    async def _post_one(self, url: str, payload: dict) -> None:
        if self._http is None:
            return
        try:
            resp = await self._http.post(url, json=payload)
            if resp.status_code >= 400:
                logger.warning(
                    f"[Webhooks] Outbound {url} returned {resp.status_code}"
                )
        except Exception as e:
            logger.warning(f"[Webhooks] Outbound POST {url} failed: {e}")

    async def trigger_inbound(self, name: str, token: str, payload: Any) -> None:
        """
        Dispatch an inbound webhook onto the bus. Raises PermissionError if the
        token doesn't match the configured value for this name. Modules subscribe
        via `event_bus.subscribe('webhook.{name}', handler)`.
        """
        expected = self._inbound_tokens.get(name)
        if expected is None:
            # Treat unknown names as 404; caller maps to HTTPException
            raise KeyError(f"unknown webhook name: {name}")
        if token != expected:
            raise PermissionError(f"invalid token for webhook '{name}'")
        await self._bus.publish(f"webhook.{name}", payload)
        logger.info(f"[Webhooks] Inbound webhook '{name}' dispatched")

    def list_outbound_routes(self) -> dict[str, list[str]]:
        """For dashboard/diag display. Returns {event_type: [urls]}."""
        return dict(self._outbound)

    def list_inbound_names(self) -> list[str]:
        """Return inbound webhook names (NOT the tokens)."""
        return list(self._inbound_tokens.keys())
