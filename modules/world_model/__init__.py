"""World Model — persistent entity tracking, bounded-house state machine.

See scripts/massive_new_integration/new 2.md for the full spec.
"""
from modules.world_model.types import (
    EntityState,
    EventType,
    WorldEntity,
    Observation,
    EntityEvent,
)
from modules.world_model.store import WorldStore
from modules.world_model.world_model import WorldModel
from modules.world_model.query_tools import WorldQueryTools
from modules.world_model.pets import (
    Affinity,
    BehavioralProfileBuilder,
    bootstrap_pets_from_config,
)

__all__ = [
    "EntityState",
    "EventType",
    "WorldEntity",
    "Observation",
    "EntityEvent",
    "WorldStore",
    "WorldModel",
    "WorldQueryTools",
    "Affinity",
    "BehavioralProfileBuilder",
    "bootstrap_pets_from_config",
]
