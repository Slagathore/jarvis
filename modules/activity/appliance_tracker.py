"""
JARVIS — Ambient Home AI
========================
Mission: Track the state of home appliances (washer, dryer, dishwasher) by
         watching audio classifier output. Each appliance has a simple state
         machine (idle → running → done) driven by the presence or absence of
         its characteristic sound. When an appliance transitions to "done",
         the tracker publishes an `appliance.state_changed` event so Jarvis
         can announce it.

         This is non-invasive and requires no smart-home hardware — just a
         microphone near the laundry room or kitchen.

Modules: modules/activity/appliance_tracker.py
Classes: ApplianceTracker, ApplianceState (dataclass)
Functions:
    ApplianceTracker.__init__(config, event_bus)  — Init with config
    ApplianceTracker.update(classifications)       — Feed audio classifier output
    ApplianceTracker.get_states()                  — Return current state dict
    ApplianceTracker._transition(appliance, seen)  — State machine logic
    ApplianceTracker._publish_done(appliance)      — Emit event

Variables:
    ApplianceTracker._states        — {appliance_name: ApplianceState}
    ApplianceTracker._event_bus     — Reference to EventBus for publishing
    ApplianceTracker._running_since — {appliance_name: datetime} tracking start time
    ApplianceState.name             — Appliance identifier
    ApplianceState.status           — "idle" | "running" | "done"
    ApplianceState.started_at       — datetime when running started
    ApplianceState.done_at          — datetime when done was detected

#todo: Add minimum run time validation (washer can't finish in 30 seconds)
#todo: Add runtime statistics logging for average cycle times
#todo: Add smart reminder scheduling — "dryer done in ~10 min based on typical cycle"
#todo: Add per-room appliance assignment (which room's mic hears which appliance)
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from loguru import logger

# Minimum seconds of detected sound before marking appliance as "running"
MIN_RUNNING_SECONDS: float = 15.0

# Seconds of silence before an appliance is considered done
DONE_SILENCE_SECONDS: float = 120.0

KNOWN_APPLIANCES: list[str] = ["washer", "dryer", "dishwasher", "microwave", "vacuum"]


@dataclass
class ApplianceState:
    """Snapshot of a single appliance's current state."""
    name: str
    status: str = "idle"         # "idle" | "running" | "done"
    started_at: Optional[datetime] = None
    done_at: Optional[datetime] = None
    last_heard_at: Optional[datetime] = None  # Last time audio confirmed running


class ApplianceTracker:
    """
    Appliance state machine driven by AudioClassifier output.

    Transitions:
        idle    → running  : Appliance sound detected for MIN_RUNNING_SECONDS
        running → done     : No appliance sound for DONE_SILENCE_SECONDS
        done    → idle     : After publishing notification (resets for next cycle)

    The `appliance.state_changed` event is published on the "done" transition
    with payload: {appliance, status, runtime_minutes, urgency}.
    """

    def __init__(self, config: dict, event_bus) -> None:
        self._config = config
        self._event_bus = event_bus

        appliance_cfg = config.get("appliances", {})
        self._silence_done_threshold: float = float(
            appliance_cfg.get("silence_done_threshold_seconds", DONE_SILENCE_SECONDS)
        )
        # Urgency levels per appliance (how urgently to announce)
        self._urgency_map: dict[str, float] = {
            "washer":     0.6,
            "dryer":      float(appliance_cfg.get("dryer_done_urgency", 0.8)),
            "dishwasher": float(appliance_cfg.get("dishwasher_done_urgency", 0.4)),
            "microwave":  0.5,
            "vacuum":     0.3,
        }

        # Initialize state machine for each known appliance
        self._states: dict[str, ApplianceState] = {
            name: ApplianceState(name=name) for name in KNOWN_APPLIANCES
        }

    def update(self, classifications: list[dict]) -> None:
        """
        Feed AudioClassifier output into the state machine.

        Args:
            classifications: List of {label, yamnet_class, score} dicts from
                             AudioClassifier.classify() or classify_async().
        """
        # Build set of currently detected appliance labels
        detected: set[str] = {
            r["label"]
            for r in classifications
            if r["label"] in KNOWN_APPLIANCES and r["score"] > 0.15
        }

        for appliance_name in KNOWN_APPLIANCES:
            self._transition(appliance_name, appliance_name in detected)

    async def update_async(self, classifications: list[dict]) -> None:
        """Async wrapper that runs update then dispatches any events."""
        self.update(classifications)

    def get_states(self) -> dict[str, dict]:
        """Return current state of all tracked appliances as serializable dicts."""
        return {
            name: {
                "status":     state.status,
                "started_at": state.started_at.isoformat() if state.started_at else None,
                "done_at":    state.done_at.isoformat() if state.done_at else None,
                "runtime_minutes": (
                    round(
                        (datetime.now() - state.started_at).total_seconds() / 60, 1
                    )
                    if state.started_at and state.status == "running"
                    else None
                ),
            }
            for name, state in self._states.items()
        }

    def _transition(self, appliance: str, currently_heard: bool) -> None:
        """
        Run the state machine for one appliance based on whether its
        sound is currently detected.
        """
        state = self._states[appliance]
        now = datetime.now()

        if currently_heard:
            state.last_heard_at = now

            if state.status == "idle":
                state.status = "running"
                state.started_at = now
                state.done_at = None
                logger.info(f"[Appliance] {appliance} started running")

        else:
            # Not currently heard — check how long since last detection
            if state.status == "running":
                silence_seconds = (
                    (now - state.last_heard_at).total_seconds()
                    if state.last_heard_at
                    else self._silence_done_threshold + 1
                )

                if silence_seconds >= self._silence_done_threshold:
                    runtime_minutes = (
                        (now - state.started_at).total_seconds() / 60
                        if state.started_at
                        else 0
                    )
                    state.status = "done"
                    state.done_at = now
                    logger.info(
                        f"[Appliance] {appliance} done "
                        f"(ran {runtime_minutes:.1f}min)"
                    )
                    asyncio.create_task(
                        self._publish_done(appliance, runtime_minutes)
                    )

            elif state.status == "done":
                # Reset to idle after publishing (next cycle can be detected)
                # We leave done state until a new running cycle is detected
                pass

    async def _publish_done(self, appliance: str, runtime_minutes: float) -> None:
        """Publish appliance.state_changed event to the event bus."""
        urgency = self._urgency_map.get(appliance, 0.5)
        await self._event_bus.publish(
            "appliance.state_changed",
            {
                "appliance":        appliance,
                "status":           "done",
                "runtime_minutes":  round(runtime_minutes, 1),
                "urgency":          urgency,
            },
        )
        logger.info(
            f"[Appliance] Published done event for {appliance} "
            f"(urgency={urgency}, runtime={runtime_minutes:.1f}min)"
        )
