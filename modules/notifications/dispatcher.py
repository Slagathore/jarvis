"""
JARVIS — Notifications
======================
Mission: Unified phone-alert dispatcher for the §29 alarm subsystem.
         Fans an `Alert` payload across N channels (ntfy / Telegram /
         Home Assistant in v4) in parallel; logs per-channel delivery
         success or failure into the persistent `notification_deliveries`
         table for the dashboard's audit view.

         Channels run concurrently — one channel's HTTP timeout doesn't
         block the others. Per-alarm-type routing lets Cole skip e.g.
         Telegram for door-open alerts (less urgent) while keeping it
         on for fire / cat-escape.

Modules: modules/notifications/dispatcher.py
Classes: AlertPriority, Alert, NotificationDispatcher
Spec:    new 2.md §31 (Notification Dispatcher).

#todo: Retry policy per channel — currently a 5s HTTP timeout fails
       hard. ntfy's server is on the same box, so retries are cheap; HA
       might be on a flaky LAN segment. A small exponential backoff
       inside the channel call would make delivery more reliable.
#todo: Rate-limit per alarm_type per minute so a flapping detector
       can't fire 200 phone alerts before being silenced.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from loguru import logger


class AlertPriority(str, Enum):
    """Three priority levels — channels translate these to their own
    severity scales (ntfy → urgent/high/default, Telegram →
    silent/loud, HA → metadata field for the user's automations).
    """
    URGENT = "urgent"   # life safety: fire alarm
    HIGH = "high"       # pet safety, perimeter integrity
    NORMAL = "normal"   # informational, deliveries-style


@dataclass
class Alert:
    """One notification payload. Built by the alarm subsystem (§29) or
    any caller that wants to push to the user's phone."""
    alarm_type: str                          # "fire" | "cat_escape" | "door_open"
    title: str                               # short headline
    body: str                                # one or two sentences
    priority: AlertPriority = AlertPriority.NORMAL
    metadata: dict[str, Any] = field(default_factory=dict)
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class NotificationDispatcher:
    """Fan-out dispatcher. Constructed once at orchestrator boot; lives
    as long as the orchestrator does. `send(alert)` is fire-and-forget
    from the caller's perspective — return is `None`, channels publish
    in parallel, delivery results land in the DB.
    """

    def __init__(
        self,
        channels: list[Any],            # NotificationChannel instances
        routing: dict[str, list[str]],  # alarm_type → [channel.name, ...]
        db_manager: Optional[Any] = None,
    ) -> None:
        # Index channels by name for O(1) routing lookup.
        self._channels_by_name: dict[str, Any] = {
            getattr(c, "name", c.__class__.__name__): c for c in channels
        }
        # Default route: every alarm_type goes to every channel. Routing
        # config overrides on a per-alarm-type basis.
        self._routing: dict[str, list[str]] = dict(routing or {})
        self._db = db_manager
        self._all_names = list(self._channels_by_name.keys())
        if not self._all_names:
            logger.warning(
                "[Notifier] No channels registered — alerts will be logged only"
            )

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def configured_channels(self) -> list[str]:
        return list(self._channels_by_name.keys())

    def routes_for(self, alarm_type: str) -> list[str]:
        """Channels that will receive `alarm_type` events."""
        return list(self._routing.get(alarm_type, self._all_names))

    async def send(self, alert: Alert) -> dict[str, Any]:
        """Dispatch `alert` to every channel routed to its alarm_type.
        Returns a per-channel result map for callers that want to
        surface per-channel success on the spot (the dashboard test
        button does); the persistent log is written either way.
        """
        target_names = self.routes_for(alert.alarm_type)
        targets = [
            (n, self._channels_by_name[n])
            for n in target_names
            if n in self._channels_by_name
        ]
        if not targets:
            logger.warning(
                f"[Notifier] No channels for alarm_type '{alert.alarm_type}'"
            )
            return {}

        # asyncio.gather with return_exceptions so one channel's failure
        # doesn't sink the others. Each channel returns whatever it wants
        # (httpx Response, dict, None) — we treat any exception as failure.
        coros = [c.send(alert) for _, c in targets]
        results = await asyncio.gather(*coros, return_exceptions=True)

        out: dict[str, Any] = {}
        for (name, _), res in zip(targets, results):
            ok = not isinstance(res, BaseException)
            err: Optional[str] = None
            if isinstance(res, BaseException):
                err = f"{res.__class__.__name__}: {res}"
                logger.warning(
                    f"[Notifier] '{name}' send failed for "
                    f"{alert.alarm_type}: {err}"
                )
            else:
                logger.debug(
                    f"[Notifier] '{name}' delivered {alert.alarm_type}"
                )
            out[name] = {"success": ok, "error": err}
            if self._db is not None:
                try:
                    await self._log_delivery(name, alert, ok, err)
                except Exception as e:
                    logger.debug(
                        f"[Notifier] delivery log write failed: {e}"
                    )
        return out

    # ── Persistence ────────────────────────────────────────────────────────

    async def _log_delivery(
        self,
        channel_name: str,
        alert: Alert,
        success: bool,
        error: Optional[str],
    ) -> None:
        """Append one row to `notification_deliveries`. Schema is
        defined in the §32 v4 migration set; we ensure-create it
        idempotently on first call so the dispatcher is usable even
        before the world-model schema has run."""
        assert self._db is not None  # gated by send()'s _db-not-None check
        await self._ensure_delivery_table()
        await self._db.execute(
            """
            INSERT INTO notification_deliveries (
                alert_ts, alarm_type, channel, success, error, payload
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                alert.ts.isoformat(),
                alert.alarm_type,
                channel_name,
                int(success),
                error,
                _safe_payload_text(alert),
            ),
        )

    async def _ensure_delivery_table(self) -> None:
        """Lighter variant of the §32 schema that doesn't require
        alarm_fires to exist yet (alarm_fires lands when §29 ships).
        Drops the FK; the column is kept for forward-compat so a future
        ALTER can re-introduce the constraint once §29 is wired."""
        assert self._db is not None
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS notification_deliveries (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                alarm_fire_id   INTEGER NULL,
                alert_ts        TEXT NOT NULL,
                alarm_type      TEXT NOT NULL,
                channel         TEXT NOT NULL,
                success         INTEGER NOT NULL,
                error           TEXT NULL,
                payload         TEXT NULL,
                delivered_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_notification_deliveries_ts "
            "ON notification_deliveries(alert_ts DESC)"
        )

    async def close(self) -> None:
        """Release every channel's shared httpx client. Idempotent —
        safe to call once at orchestrator shutdown."""
        for name, channel in self._channels_by_name.items():
            aclose = getattr(channel, "aclose", None)
            if aclose is None:
                continue
            try:
                await aclose()
            except Exception as e:
                logger.debug(f"[Notifier] '{name}' aclose failed: {e}")

    async def recent_deliveries(self, limit: int = 50) -> list[dict]:
        """Fetch the last `limit` delivery rows for the dashboard panel."""
        if self._db is None:
            return []
        await self._ensure_delivery_table()
        assert self._db is not None
        rows = await self._db.fetchall(
            """
            SELECT id, alert_ts, alarm_type, channel, success, error, payload
            FROM notification_deliveries
            ORDER BY id DESC LIMIT ?
            """,
            (int(limit),),
        )
        return [dict(r) for r in rows]


def _safe_payload_text(alert: Alert) -> str:
    """Compact text representation of an Alert for the audit log.
    Keeps full body so the dashboard can show what the user's phone
    received; truncates at 1KB to avoid blowing up the row."""
    s = f"[{alert.priority.value}] {alert.title} — {alert.body}"
    if alert.metadata:
        s += f" | {alert.metadata}"
    return s[:1024]
