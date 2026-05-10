"""
JARVIS — Notifications
======================
Mission: Channel implementations for the §31 notification dispatcher.
         Three channels in v4 — ntfy (recommended primary, self-hosted
         Docker), Telegram (good audit trail, less reliable for DND
         override), Home Assistant (depends on an HA instance, useful
         when one is already on the network for other reasons).

         Each channel implements `send(alert)` — async, idempotent in
         the "calling twice produces two notifications" sense, and is
         expected to raise on failure so the dispatcher logs the error.

Modules: modules/notifications/channels.py
Classes: NotificationChannel (Protocol), NtfyChannel, TelegramChannel,
         HAChannel
Spec:    new 2.md §31.2 (Channel implementations).

#todo: Add a fallback `LogChannel` that just prints to the loguru sink.
       Useful for local dev runs where Cole doesn't want phone alerts
       firing during testing — set `routing.fire: [log]` to see what
       WOULD have gone out.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional

import httpx
from loguru import logger

from modules.notifications.dispatcher import Alert, AlertPriority


class NotificationChannel(ABC):
    """Abstract base — every concrete channel ships a `name` class
    attribute and an async `send(alert)` method. Failure raises;
    dispatcher catches.
    """
    name: str = "base"

    @abstractmethod
    async def send(self, alert: Alert) -> None:
        ...


# ── ntfy ────────────────────────────────────────────────────────────────────

class NtfyChannel(NotificationChannel):
    """ntfy.sh-compatible channel. Default targets a self-hosted server
    (Docker on the same box as Jarvis); the public ntfy.sh works too
    if you want zero infrastructure but accept that your topic name
    is the only thing keeping strangers from reading your alerts.

    Urgent priority overrides Android's DND reliably; iOS is more
    restrictive but does honor the priority field for critical alerts.
    """

    name = "ntfy"

    def __init__(
        self,
        server: str = "http://localhost:8080",
        topic: str = "jarvis_alerts",
        timeout_s: float = 5.0,
    ) -> None:
        self._server = server.rstrip("/")
        self._topic = topic
        self._timeout = timeout_s

    async def send(self, alert: Alert) -> None:
        priority_map = {
            AlertPriority.URGENT: "urgent",
            AlertPriority.HIGH:   "high",
            AlertPriority.NORMAL: "default",
        }
        url = f"{self._server}/{self._topic}"
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                url,
                content=alert.body.encode("utf-8"),
                headers={
                    "Title":    alert.title,
                    "Priority": priority_map[alert.priority],
                    "Tags":     self._tags_for(alert.alarm_type),
                },
            )
            resp.raise_for_status()

    @staticmethod
    def _tags_for(alarm_type: str) -> str:
        # ntfy renders these as emoji shortcodes in the phone
        # notification — rotating_light = 🚨, fire = 🔥, etc.
        return {
            "fire":       "rotating_light,fire",
            "cat_escape": "rotating_light,cat",
            "door_open":  "door,warning",
        }.get(alarm_type, "warning")


# ── Telegram ────────────────────────────────────────────────────────────────

class TelegramChannel(NotificationChannel):
    """Telegram bot channel. Reuses the existing Mira-Telegram bot
    infrastructure — same token, separate chat for alerts so the
    companion-bot conversation doesn't get cluttered.

    Token + chat_id come from environment variables (per the doc) so
    secrets stay out of config.yaml.
    """

    name = "telegram"

    def __init__(
        self,
        bot_token_env: str = "TELEGRAM_BOT_TOKEN",
        alert_chat_id_env: str = "TELEGRAM_ALERT_CHAT_ID",
        timeout_s: float = 5.0,
    ) -> None:
        self._bot_token_env = bot_token_env
        self._chat_id_env = alert_chat_id_env
        self._timeout = timeout_s

    async def send(self, alert: Alert) -> None:
        token = os.environ.get(self._bot_token_env)
        chat_id = os.environ.get(self._chat_id_env)
        if not token or not chat_id:
            raise RuntimeError(
                f"Telegram channel requires {self._bot_token_env} and "
                f"{self._chat_id_env} env vars to be set"
            )
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text":    f"*{_md_escape(alert.title)}*\n{_md_escape(alert.body)}",
            "parse_mode": "Markdown",
            "disable_notification": alert.priority == AlertPriority.NORMAL,
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()


def _md_escape(text: str) -> str:
    """Escape Telegram Markdown's reserved chars so a `*` in the body
    doesn't accidentally bold half the message."""
    if not text:
        return ""
    for c in r"_*[]()~`>#+-=|{}.!":
        text = text.replace(c, "\\" + c)
    return text


# ── Home Assistant ──────────────────────────────────────────────────────────

class HAChannel(NotificationChannel):
    """Home Assistant `notify` service channel. Pushes to whatever HA
    notify service Cole has configured (mobile_app integration, custom
    automation, etc.). Token comes from an env var.

    Requires an HA instance reachable on the LAN. Worth running anyway
    for speaker integration; if HA isn't otherwise on the roadmap, the
    dependency cost is real — disable this channel via the routing
    config to avoid useless connection attempts.
    """

    name = "home_assistant"

    def __init__(
        self,
        base_url: str = "http://homeassistant.local:8123",
        service: str = "mobile_app_cole_phone",
        token_env: str = "HOME_ASSISTANT_TOKEN",
        timeout_s: float = 5.0,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._service = service
        self._token_env = token_env
        self._timeout = timeout_s

    async def send(self, alert: Alert) -> None:
        token = os.environ.get(self._token_env)
        if not token:
            raise RuntimeError(
                f"Home Assistant channel requires {self._token_env} env var"
            )
        url = f"{self._base}/api/services/notify/{self._service}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
        }
        payload = {
            "title":   alert.title,
            "message": alert.body,
            "data":    {
                "priority": alert.priority.value,
                **alert.metadata,
            },
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()


# ── Factory: build channels from a config dict ──────────────────────────────


def build_channels_from_config(
    cfg: Optional[dict],
) -> list[NotificationChannel]:
    """Inspect a `notifications:` block from config.yaml and return
    the enabled channel instances. Any disabled channel is skipped
    entirely so a missing env var doesn't fail boot.

    Expected shape (mirrors §31.3):
        notifications:
          ntfy: {enabled: bool, server: str, topic: str}
          telegram: {enabled: bool, bot_token_env: str, alert_chat_id_env: str}
          home_assistant: {enabled: bool, base_url: str, service: str, token_env: str}
    """
    cfg = cfg or {}
    out: list[NotificationChannel] = []

    ntfy_cfg = cfg.get("ntfy") or {}
    if ntfy_cfg.get("enabled", False):
        out.append(NtfyChannel(
            server=str(ntfy_cfg.get("server", "http://localhost:8080")),
            topic=str(ntfy_cfg.get("topic", "jarvis_alerts")),
        ))

    tg_cfg = cfg.get("telegram") or {}
    if tg_cfg.get("enabled", False):
        out.append(TelegramChannel(
            bot_token_env=str(tg_cfg.get("bot_token_env", "TELEGRAM_BOT_TOKEN")),
            alert_chat_id_env=str(tg_cfg.get("alert_chat_id_env", "TELEGRAM_ALERT_CHAT_ID")),
        ))

    ha_cfg = cfg.get("home_assistant") or {}
    if ha_cfg.get("enabled", False):
        out.append(HAChannel(
            base_url=str(ha_cfg.get("base_url", "http://homeassistant.local:8123")),
            service=str(ha_cfg.get("service", "mobile_app_cole_phone")),
            token_env=str(ha_cfg.get("token_env", "HOME_ASSISTANT_TOKEN")),
        ))

    if not out:
        logger.info(
            "[Notifier] No channels enabled in config.notifications — "
            "alarm alerts will only land in the local DB log"
        )
    return out
