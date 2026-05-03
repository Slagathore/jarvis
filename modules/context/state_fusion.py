"""
JARVIS — Ambient Home AI
========================
Mission: Fuse signals from multiple detection sources (PC monitor, audio classifier,
         posture analyzer) into a single authoritative ActivityState. Uses a
         weighted voting system where more reliable signals override weaker ones.
         The fused state is what gets broadcast to the dashboard and used by
         the interruptibility gate.

Modules: modules/context/state_fusion.py
Classes: StateFusion
Functions:
    StateFusion.__init__(config)     — Initialize with config
    StateFusion.fuse(signals)        — Combine signal dict → ActivityState
    StateFusion._resolve_activity(votes) — Pick winner from weighted votes
    StateFusion._compute_confidence(votes, winner) — Score fusion confidence

Variables:
    StateFusion._config          — Full config dict
    StateFusion._activity_scores — interruptibility score map from config
    SIGNAL_WEIGHTS               — How much each signal source is trusted

Signal input format:
    signals = {
        "pc":      {"activity": "gaming", "context": {"game": "Elden Ring"}, "confidence": 0.9},
        "audio":   {"activity": "gaming", "confidence": 0.6},
        "posture": {"activity": "sitting", "confidence": 0.8},
        "vision":  {"activity": "unknown", "confidence": 0.3},
    }

#todo: Add Bayesian update model — weight signals by historical accuracy
#todo: Add temporal smoothing — require N consecutive identical signals before changing
#todo: Add location inference from which camera has a person present
#todo: Add anomaly detection — flag contradictory signals for debugging
"""

from datetime import datetime
from typing import Any, Mapping, Optional

from loguru import logger

from modules.context.state import ActivityState

# How much each signal source's vote is trusted (0–1).
# PC monitor is most reliable since it's exact; audio is weakest.
SIGNAL_WEIGHTS: dict[str, float] = {
    "pc":      0.85,   # Window/process exact match — very reliable
    "audio":   0.50,   # YAMNet audio classification — moderate
    "posture": 0.70,   # MediaPipe body pose — reliable when person present
    "vision":  0.60,   # Ollama scene description — general
    "sleep":   0.90,   # Sleep tracker override — high priority
}


class StateFusion:
    """
    Combines multiple signals into a single ActivityState.

    The fusion algorithm:
    1. Each signal votes for an activity with a weight × confidence score.
    2. The activity with the highest weighted score wins.
    3. Context dicts are merged (more specific sources override less specific).
    4. Interruptibility is looked up from config using the winning activity.
    5. Confidence reflects how unanimous the vote was.
    """

    def __init__(self, config: dict, activity_history=None) -> None:
        self._config = config
        self._activity_scores: dict[str, float] = (
            config.get("interruptibility", {}).get("activity_scores", {})
        )
        self._current_room: str = "office"
        self._pending_vision: dict[str, dict[str, Any]] = {}
        # Optional ActivityHistory — when provided, time-of-day priors derived
        # from past activity_log rows get blended into the vote scores. With
        # zero history this contributes nothing; as the table fills, the prior
        # gradually pulls votes toward whatever Cole typically does at this
        # time slot. Effectively a "Bayesian-lite" fusion: posterior ∝ prior × likelihood.
        self._activity_history = activity_history
        # Strength of the time-of-day prior. 0.0 = pure weighted vote (legacy).
        # 0.5 = a 5-prior-occurrences bump multiplies the vote by 1.5×. Tuned
        # conservatively so signals still drive the call.
        fusion_cfg = (
            config.get("fusion", {}) if isinstance(config.get("fusion"), dict) else {}
        )
        self._prior_weight: float = float(fusion_cfg.get("prior_weight", 0.1))

    async def fuse(
        self,
        signals: Mapping[str, Mapping[str, Any] | None],
        room: Optional[str] = None,
    ) -> ActivityState:
        """
        Fuse signal dict into a new ActivityState.

        Args:
            signals: Dict mapping source name → signal dict (or None if no data).
                     Each signal dict should have "activity" and optionally "confidence"
                     and "context".
            room:    Override room. If None, uses internal tracking.

        Returns:
            A new ActivityState reflecting the fused understanding.
        """
        if room:
            self._current_room = room

        # Collect weighted votes from each signal source
        vote_scores: dict[str, float] = {}
        merged_context: dict[str, Any] = {}
        contributing_signals: list[str] = []

        for source, signal in signals.items():
            if not signal:
                continue

            # Always merge context, even from signals that don't vote on
            # activity. Posture, vision-presence, etc. carry context (sitting/
            # lying, person count) that the LLM and dashboard want even when
            # the signal isn't strong enough to drive activity classification.
            context = signal.get("context")
            if isinstance(context, Mapping):
                merged_context.update(context)

            activity = signal.get("activity")
            if not activity or activity == "unknown":
                continue

            source_confidence = float(signal.get("confidence", 0.5))
            source_weight = SIGNAL_WEIGHTS.get(source, 0.5)
            vote_score = source_weight * source_confidence

            vote_scores[activity] = vote_scores.get(activity, 0.0) + vote_score
            contributing_signals.append(f"{source}:{activity}")

        if not vote_scores:
            return ActivityState(
                activity="unknown",
                location=self._current_room,
                interruptibility=self._activity_scores.get("unknown", 0.5),
                confidence=0.0,
                signals=[],
                context={},
                updated_at=datetime.now(),
            )

        # Bayesian-lite: if we have ActivityHistory, look up typical activities
        # for the current day-of-week / hour and bump the matching vote scores.
        # vote × (1 + prior_weight × occurrences). Activities with no history
        # entry pass through unchanged. With prior_weight=0 this is a no-op.
        if self._activity_history is not None and self._prior_weight > 0:
            try:
                now = datetime.now()
                priors = await self._activity_history.typical_activity(
                    day_of_week=int(now.strftime("%w")),
                    hour=now.hour,
                    top_n=10,
                )
                for activity, count in priors:
                    if activity in vote_scores:
                        vote_scores[activity] *= (1.0 + self._prior_weight * count)
            except Exception as e:
                logger.debug(f"[StateFusion] prior lookup failed: {e}")

        # Winner = highest weighted vote score
        winner_activity = max(vote_scores, key=lambda a: vote_scores[a])
        confidence = self._compute_confidence(vote_scores, winner_activity)
        interruptibility = self._activity_scores.get(winner_activity, 0.5)

        state = ActivityState(
            activity=winner_activity,
            location=self._current_room,
            interruptibility=float(interruptibility),
            confidence=float(confidence),
            signals=contributing_signals,
            context=merged_context,
            updated_at=datetime.now(),
        )

        logger.debug(
            f"[StateFusion] → {winner_activity} "
            f"(conf={confidence:.2f}, interrupt={interruptibility:.2f}) "
            f"| signals: {contributing_signals}"
        )
        return state

    def inject_vision(self, room_id: str, data: Mapping[str, Any]) -> None:
        """
        Store a vision signal to be included in the next fuse() call.

        Called by the orchestrator's _vision_loop() after analyzing a camera
        frame. The stored signal is consumed on the next fuse() invocation.

        Args:
            room_id: Room this vision data belongs to.
            data:    Dict with keys: activity, confidence, context (optional).
        """
        self._pending_vision[room_id] = dict(data)
        logger.debug(f"[StateFusion] Vision signal staged for '{room_id}': {data}")

    def pop_vision(self, room_id: str) -> Optional[dict[str, Any]]:
        """
        Consume and return the staged vision signal for a room.
        Returns None if no signal was staged.
        """
        return self._pending_vision.pop(room_id, None)

    def _compute_confidence(
        self,
        vote_scores: dict[str, float],
        winner: str,
    ) -> float:
        """
        Compute confidence as the winner's score proportion of the total.
        A unanimous vote (only one activity) gives 1.0.
        A split vote with multiple activities gives a lower score.
        """
        total = sum(vote_scores.values())
        if total == 0:
            return 0.0
        return vote_scores[winner] / total
