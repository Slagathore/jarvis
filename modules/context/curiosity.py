"""
JARVIS — Ambient Home AI
========================
Mission: Generate proactive ambient speech when Jarvis notices something
         interesting or useful to mention. Rather than always waiting to be
         asked, the CuriosityEngine checks for patterns worth commenting on:
         gaming sessions running too long, morning greetings, nap check-ins,
         ComfyUI renders finishing, coding sprints running long, etc.

         Each observation topic has a cooldown so Jarvis doesn't repeat itself.
         An observation is only generated when interruptibility allows it.

Modules: modules/context/curiosity.py
Classes: CuriosityEngine
Functions:
    CuriosityEngine.__init__(config, llm)  — Initialize with config and LLM ref
    CuriosityEngine.check_async(state)     — Check if any topic is ready; return prompt or None
    CuriosityEngine._generate(topic, state) — Build and run the LLM prompt for a topic
    CuriosityEngine._is_on_cooldown(topic) — True if this topic was recently used
    CuriosityEngine._mark_used(topic)      — Record that this topic was triggered

Variables:
    CuriosityEngine._topic_cooldowns   — {topic: last_triggered datetime}
    CuriosityEngine._cooldown_hours    — {topic: float} from config
    CuriosityEngine._llm               — OllamaLLM reference for observation generation
    CuriosityEngine._config            — Full config dict

Topic list (checked in priority order):
    "morning_greeting"     — First interaction of the day, 7-11 AM
    "gaming_too_long"      — Playing same game for > 2 hours
    "coding_too_long"      — Coding for > 3 hours without break
    "nap_checkin"          — Lying down 15+ minutes, not yet confirmed sleeping
    "comfyui_running"      — ComfyUI process detected (image gen session)
    "appliance_done"       — Handled separately by appliance_tracker
    "idle_too_long"        — No activity for > 1 hour during waking hours

#todo: Add weather-based greeting ("gonna be hot today, heads up")
#todo: Add session summary at end of day ("You gamed for 4 hours today")
#todo: Add gentle hydration reminder if no audio/movement for 2+ hours
#todo: Add "nice save" message when coding context after long gaming session
#todo: Track which observations Cole responds positively to and weight those higher
"""

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from loguru import logger

from modules.context.state import ActivityState

if TYPE_CHECKING:
    from modules.brain.llm import OllamaLLM


# Ordered list of topics to check. Earlier = higher priority.
TOPIC_ORDER = [
    "morning_greeting",
    "gaming_too_long",
    "coding_too_long",
    "nap_checkin",
    "comfyui_running",
    "idle_too_long",
]

# Default cooldown hours if not in config
DEFAULT_COOLDOWNS: dict[str, float] = {
    "morning_greeting":  12.0,   # Once per day
    "gaming_too_long":   2.0,    # Every 2 hours of gaming
    "coding_too_long":   3.0,    # Every 3 hours of coding
    "nap_checkin":       1.0,    # Once per nap attempt
    "comfyui_running":   4.0,    # Once per ComfyUI session
    "idle_too_long":     1.5,    # Every 1.5 hours of idle
}


class CuriosityEngine:
    """
    Generates proactive ambient observations for Jarvis to speak.

    Usage:
        observation = await engine.check_async(state)
        if observation:
            await tts.speak_async(observation)
            interruptibility.record_interruption()
    """

    def __init__(self, config: dict, llm: Optional["OllamaLLM"]) -> None:
        self._config = config
        self._llm = llm
        curiosity_cfg = config.get("curiosity", {})
        self._min_interruptibility: float = float(
            curiosity_cfg.get("min_interruptibility", 0.35)
        )
        # Per-topic cooldown hours, with config overrides
        self._cooldown_hours: dict[str, float] = dict(DEFAULT_COOLDOWNS)
        self._cooldown_hours.update(
            curiosity_cfg.get("topic_cooldowns_hours", {})
        )
        # Track last trigger time per topic
        self._topic_cooldowns: dict[str, Optional[datetime]] = {
            t: None for t in TOPIC_ORDER
        }
        # Track activity start times to detect "too long" situations
        self._activity_started: dict[str, Optional[datetime]] = {}
        self._greeted_today: Optional[datetime] = None

    async def check_async(
        self, state: ActivityState
    ) -> Optional[str]:
        """
        Check all topics in priority order. Return the first observation text
        that is ready (cooldown expired, conditions met, LLM generates something),
        or None if nothing is ready.
        """
        if state.interruptibility < self._min_interruptibility:
            return None

        # Update activity timing tracking
        self._track_activity_timing(state)

        for topic in TOPIC_ORDER:
            if self._is_on_cooldown(topic):
                continue
            observation = await self._evaluate_topic(topic, state)
            if observation:
                self._mark_used(topic)
                logger.info(f"[Curiosity] Firing topic '{topic}'")
                return observation

        return None

    async def _evaluate_topic(
        self, topic: str, state: ActivityState
    ) -> Optional[str]:
        """Evaluate whether a specific topic is ready and generate its text."""
        now = datetime.now()

        if topic == "morning_greeting":
            if 7 <= now.hour < 11:
                if self._greeted_today is None or (
                    now - self._greeted_today
                ).total_seconds() > 3600 * 12:
                    self._greeted_today = now
                    return await self._generate(topic, state)

        elif topic == "gaming_too_long":
            if state.activity == "gaming":
                started = self._activity_started.get("gaming")
                if started:
                    hours = (now - started).total_seconds() / 3600
                    if hours >= 2.0:
                        return await self._generate(topic, state, hours=hours)

        elif topic == "coding_too_long":
            if state.activity in ("coding", "programming"):
                started = self._activity_started.get(state.activity)
                if started:
                    hours = (now - started).total_seconds() / 3600
                    if hours >= 3.0:
                        return await self._generate(topic, state, hours=hours)

        elif topic == "nap_checkin":
            if state.activity == "napping":
                started = self._activity_started.get("napping")
                if started:
                    mins = (now - started).total_seconds() / 60
                    if mins >= 15:
                        return await self._generate(topic, state, minutes=mins)

        elif topic == "comfyui_running":
            if state.context.get("process_name", "").lower() in (
                "comfyui", "comfyui.exe", "python"
            ) or "comfyui" in state.context.get("window_title", "").lower():
                return await self._generate(topic, state)

        elif topic == "idle_too_long":
            if state.activity == "idle":
                started = self._activity_started.get("idle")
                if started:
                    # Only during waking hours
                    if 9 <= now.hour < 22:
                        hours = (now - started).total_seconds() / 3600
                        if hours >= 1.0:
                            return await self._generate(topic, state, hours=hours)

        return None

    async def _generate(
        self,
        topic: str,
        state: ActivityState,
        **kwargs,
    ) -> Optional[str]:
        """
        Use the LLM to generate a natural, brief observation for this topic.
        Returns None if LLM fails or returns empty.
        """
        prompts = {
            "morning_greeting": (
                "Generate a brief, warm good morning greeting from Jarvis. "
                "One or two sentences. Mention the time of day. Be natural, not robotic."
            ),
            "gaming_too_long": (
                f"Cole has been gaming for {kwargs.get('hours', 2):.1f} hours. "
                "Generate a short, casual, non-preachy check-in from Jarvis. "
                "Maybe suggest a stretch. One sentence, friendly tone."
            ),
            "coding_too_long": (
                f"Cole has been coding for {kwargs.get('hours', 3):.1f} hours. "
                "Generate a brief, encouraging check-in from Jarvis. "
                "Suggest a short break. One sentence."
            ),
            "nap_checkin": (
                f"Cole appears to be napping ({kwargs.get('minutes', 15):.0f} minutes). "
                "Generate a gentle, soft-spoken check-in from Jarvis. "
                "Ask if he wants a wake alarm. Very brief."
            ),
            "comfyui_running": (
                "Cole appears to be using ComfyUI for AI image generation. "
                "Generate a brief, enthusiastic observation from Jarvis about creative work. "
                "One sentence."
            ),
            "idle_too_long": (
                f"Cole has been idle for {kwargs.get('hours', 1):.1f} hours. "
                "Generate a gentle, curious check-in from Jarvis. "
                "Wonder what he's up to. One sentence, light tone."
            ),
        }

        prompt_text = prompts.get(topic)
        if not prompt_text:
            return None
        if self._llm is None:
            logger.debug(f"[Curiosity] No LLM available for topic '{topic}'")
            return None

        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are Jarvis, Cole's ambient AI assistant. "
                        "Generate exactly what you would say aloud — no meta-commentary, "
                        "no quotes, just the speech. Keep it concise and natural."
                    ),
                },
                {"role": "user", "content": prompt_text},
            ]
            response = await self._llm.chat(messages)
            return response.strip() if response else None
        except Exception as e:
            logger.warning(f"[Curiosity] LLM generation failed for '{topic}': {e}")
            return None

    def _is_on_cooldown(self, topic: str) -> bool:
        """Return True if this topic has been triggered recently."""
        last = self._topic_cooldowns.get(topic)
        if last is None:
            return False
        hours_since = (datetime.now() - last).total_seconds() / 3600
        cooldown = self._cooldown_hours.get(topic, 2.0)
        return hours_since < cooldown

    def _mark_used(self, topic: str) -> None:
        """Record this topic as just triggered."""
        self._topic_cooldowns[topic] = datetime.now()

    def _track_activity_timing(self, state: ActivityState) -> None:
        """
        Track when each activity started so duration can be measured.
        Called on every check_async invocation.
        """
        now = datetime.now()
        activity = state.activity
        if activity not in self._activity_started or self._activity_started[activity] is None:
            self._activity_started[activity] = now
        # Reset timer if activity changed from a different one
        for other_activity in list(self._activity_started.keys()):
            if other_activity != activity:
                self._activity_started[other_activity] = None
        if activity not in self._activity_started:
            self._activity_started[activity] = now
