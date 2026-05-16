"""
JARVIS — World Model / Belief
=============================
Mission: Data model for the belief-state tracker (audit roadmap D4).

         The current WorldModel is a *single-projection* tracker: one
         room / bbox / confidence per entity. A belief tracker instead
         holds competing *hypotheses*, each with confidence split into
         four independent axes so they can decay at different rates:

           identity   — how sure we are this is entity X (slow decay)
           location   — how sure we are about WHERE X is (slow decay)
           visibility — how sure we are X is currently observable here
                        (fast decay — "not detected for 30s" hits this,
                        not location: the white-dog-on-white-blanket case)
           state      — derived from the other three

Modules: modules/world_model/belief/types.py
Classes: BeliefState, EvidenceFrame, BeliefHypothesis
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


class BeliefState:
    """Belief states. Strings (not an Enum) so they round-trip to SQLite
    text columns and JSON without conversion."""

    PRESENT_CONFIRMED = "present_confirmed"      # seen recently, strong evidence
    PRESENT_UNSEEN = "present_unseen"            # believed here, currently not detected
    SUSPECTED_ELSEWHERE = "suspected_elsewhere"  # evidence elsewhere, not enough to move
    TRANSITIONING = "transitioning"              # movement supported by a path/door
    IN_HOUSE_UNMONITORED = "in_house_unmonitored"  # probably home, outside camera coverage
    DEPARTED = "departed"                        # left
    UNKNOWN = "unknown"                          # no reliable belief

    ALL = (
        PRESENT_CONFIRMED, PRESENT_UNSEEN, SUSPECTED_ELSEWHERE,
        TRANSITIONING, IN_HOUSE_UNMONITORED, DEPARTED, UNKNOWN,
    )


@dataclass
class EvidenceFrame:
    """One piece of evidence about an entity. Every observation becomes
    evidence — never immediate truth. An empty observation batch produces
    an `absence` EvidenceFrame, which is *weak* negative evidence."""

    ts: datetime
    entity_key: str                  # 'person:3', 'cat:summer', ...
    entity_type: str                 # person | cat | dog | object
    source: str                      # 'vision.observation' | 'manual' | ...
    evidence_type: str               # 'sighting' | 'absence' | 'manual_tag'
    room: Optional[str] = None
    camera: Optional[str] = None
    # 0..1 strength of this evidence. For a sighting this is detector ×
    # identity confidence; for an absence it is the detectability-scaled
    # strength of "we should have seen it and didn't".
    score: float = 0.0
    bbox: Optional[tuple] = None
    payload: dict = field(default_factory=dict)

    def payload_json(self) -> str:
        try:
            return json.dumps(self.payload, default=str)
        except Exception:
            return "{}"


@dataclass
class BeliefHypothesis:
    """One hypothesis about where/what an entity is. An entity may carry
    several (one primary + competitors); the projection layer derives the
    single "where is X" answer from the primary."""

    hypothesis_id: str
    entity_key: str
    entity_type: str
    state: str = BeliefState.UNKNOWN
    room: Optional[str] = None
    camera: Optional[str] = None
    confidence_identity: float = 0.0
    confidence_location: float = 0.0
    confidence_visibility: float = 0.0
    confidence_state: float = 0.0
    is_primary: bool = True
    last_confirmed_ts: Optional[datetime] = None
    last_evidence_ts: Optional[datetime] = None
    evidence_breakdown: dict = field(default_factory=dict)

    def recompute_state_confidence(self) -> None:
        """state confidence is derived — the weakest-link of identity and
        location, lifted a little by visibility when we can currently see
        the entity."""
        base = min(self.confidence_identity, self.confidence_location)
        self.confidence_state = round(
            min(1.0, base * 0.7 + self.confidence_visibility * 0.3), 4
        )

    def evidence_breakdown_json(self) -> str:
        try:
            return json.dumps(self.evidence_breakdown, default=str)
        except Exception:
            return "{}"
