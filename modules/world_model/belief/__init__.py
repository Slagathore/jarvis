"""
JARVIS — World Model / Belief
=============================
The belief-state layer (audit roadmap D4). See resolver.py for the running
entry point and types.py for the data model.
"""

from modules.world_model.belief.resolver import BeliefResolver
from modules.world_model.belief.types import (
    BeliefHypothesis,
    BeliefState,
    EvidenceFrame,
)

__all__ = [
    "BeliefResolver",
    "BeliefHypothesis",
    "BeliefState",
    "EvidenceFrame",
]
