"""
JARVIS — Integration plugin contracts
=====================================
Mission: Provide a narrow interface for future sensors and actuators so new
         device integrations can live in modules/integrations/* instead of
         adding more direct wiring to core/orchestrator.py.

This is intentionally small:
  - Sensor plugins publish observations/events onto EventBus.
  - Actuator plugins subscribe to EventBus topics or expose command handlers.
  - The registry owns lifecycle ordering and shutdown cleanup.

Existing integrations can migrate one at a time. The orchestrator only needs
to construct IntegrationContext and call registry.start_all()/stop_all().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

from core.event_bus import EventBus


@dataclass(slots=True)
class IntegrationContext:
    """Shared runtime handles exposed to integration plugins."""

    config: dict
    bus: EventBus
    dashboard: Optional[Any] = None
    db: Optional[Any] = None


@runtime_checkable
class IntegrationPlugin(Protocol):
    """Common lifecycle for any sensor/actuator integration."""

    name: str

    async def start(self, ctx: IntegrationContext) -> None:
        ...

    async def stop(self) -> None:
        ...

    def status(self) -> dict:
        ...


@runtime_checkable
class SensorPlugin(IntegrationPlugin, Protocol):
    """Marker protocol for integrations that primarily publish events."""


@runtime_checkable
class ActuatorPlugin(IntegrationPlugin, Protocol):
    """Marker protocol for integrations that primarily perform actions."""


class IntegrationRegistry:
    """Lifecycle container for integration plugins.

    The registry is deliberately dumb: plugins own their own config parsing,
    event subscriptions, background tasks, and cleanup. That keeps this layer
    stable even as new hardware arrives.
    """

    def __init__(self) -> None:
        self._plugins: dict[str, IntegrationPlugin] = {}

    def register(self, plugin: IntegrationPlugin) -> None:
        name = str(getattr(plugin, "name", "")).strip()
        if not name:
            raise ValueError("integration plugin must define a non-empty name")
        self._plugins[name] = plugin

    async def start_all(self, ctx: IntegrationContext) -> None:
        for plugin in list(self._plugins.values()):
            await plugin.start(ctx)

    async def stop_all(self) -> None:
        for plugin in reversed(list(self._plugins.values())):
            try:
                await plugin.stop()
            except Exception:
                pass

    def status(self) -> list[dict]:
        out: list[dict] = []
        for name, plugin in self._plugins.items():
            try:
                item = dict(plugin.status())
            except Exception as e:
                item = {"status": "error", "error": str(e)}
            item.setdefault("name", name)
            out.append(item)
        return out
