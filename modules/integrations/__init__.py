"""
Small plugin-style interface for adding sensors and actuators without growing
core/orchestrator.py for every new integration.
"""

from modules.integrations.base import (
    ActuatorPlugin,
    IntegrationContext,
    IntegrationPlugin,
    IntegrationRegistry,
    SensorPlugin,
)

__all__ = [
    "ActuatorPlugin",
    "IntegrationContext",
    "IntegrationPlugin",
    "IntegrationRegistry",
    "SensorPlugin",
]
