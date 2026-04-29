"""
JARVIS — Ambient Home AI
========================
Mission: Define the ActivityState dataclass — the single shared object that
         represents Jarvis's current understanding of what Cole is doing,
         where he is, and whether it's a good time to interrupt. All context
         modules read and write this structure; the orchestrator broadcasts it
         to the dashboard on every change.

Modules: modules/context/state.py
Classes: ActivityState
Functions:
    ActivityState.__init__(...)          — Create a state (all fields optional)
    ActivityState.is_same_activity(other)— Compare activities ignoring metadata
    ActivityState.to_dict()             — Serialize to dict for JSON/dashboard

Variables:
    ActivityState.activity       — str, current activity label
    ActivityState.location       — str, current room id
    ActivityState.interruptibility — float 0-1, how interruptible Cole is right now
    ActivityState.confidence     — float 0-1, fusion confidence
    ActivityState.signals        — list[str], signal names that contributed
    ActivityState.context        — dict, extra context (game name, window, project)
    ActivityState.updated_at     — datetime of last update
    UNKNOWN_STATE                — Default state before any signals arrive

#todo: Add mood field derived from voice tone analysis
#todo: Add focus_depth derived from how long the current activity has been continuous
#todo: Add predicted_duration for how long the current activity likely continues
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class ActivityState:
    """
    A snapshot of what Jarvis believes Cole is currently doing.

    This is the central shared context object that flows from detection modules
    (PCMonitor, AudioClassifier, PostureAnalyzer) through StateFusion to the
    orchestrator, interruptibility gate, curiosity engine, and dashboard.

    Fields:
        activity:         One of the activity labels defined in config.yaml
                          (gaming, coding, sleeping, idle, etc.)
        location:         Room identifier string ("office", "bedroom", etc.)
        interruptibility: Normalized 0–1 score of how open Cole is to being
                          interrupted. 0 = do not interrupt, 1 = always ok.
        confidence:       0–1 fusion confidence in this state assessment.
        signals:          List of signal source names that contributed
                          (e.g., ["pc:gaming", "audio:music", "vision:sitting"]).
        context:          Freeform extra context. Keys used:
                          - "game": active game name
                          - "window_title": active window title string
                          - "project": detected project/file name
                          - "posture": current posture label
        updated_at:       When this state was last computed.
    """

    activity: str = "unknown"
    location: str = "office"
    interruptibility: float = 0.5
    confidence: float = 0.0
    signals: list[str] = field(default_factory=list)
    context: dict = field(default_factory=dict)
    updated_at: datetime = field(default_factory=datetime.now)

    def is_same_activity(self, other: "ActivityState") -> bool:
        """
        Return True if other has the same activity + location as self.
        Ignores confidence, signals, context, and timestamp.
        Used to detect meaningful state changes worth broadcasting.
        """
        return self.activity == other.activity and self.location == other.location

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for JSON encoding."""
        return {
            "activity":         self.activity,
            "location":         self.location,
            "interruptibility": round(self.interruptibility, 3),
            "confidence":       round(self.confidence, 3),
            "signals":          list(self.signals),
            "context":          dict(self.context),
            "updated_at":       self.updated_at.isoformat(),
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ActivityState):
            return False
        return (
            self.activity == other.activity
            and self.location == other.location
            and abs(self.interruptibility - other.interruptibility) < 0.05
        )


# Default state used before any detection data arrives
UNKNOWN_STATE = ActivityState(
    activity="unknown",
    location="office",
    interruptibility=0.5,
    confidence=0.0,
)
