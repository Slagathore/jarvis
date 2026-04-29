"""
JARVIS — Ambient Home AI
========================
Mission: Build complete message lists for LLM API calls by combining the base
         system prompt with context about what the user is currently doing,
         the room they're in, the time of day, relevant memories, and the
         conversation history. The LLM should always have full situational
         awareness without the orchestrator needing to manage prompt assembly.

Modules: modules/brain/prompt_builder.py
Classes: PromptBuilder
Functions:
    PromptBuilder.__init__(config)          — Initialize with config
    PromptBuilder.build(text, state, session, room, extras)
                                            — Assemble full message list for LLM

Variables:
    PromptBuilder._system_prompt  — Base personality prompt string

#todo: Add memory retrieval — pull relevant past events from the DB into context
#todo: Add room baseline description to give Jarvis visual context
#todo: Add time-of-day awareness to system prompt (morning/evening/night)
#todo: Add user preference context (Cole's known projects, current game, etc.)
#todo: Support dynamic injection of reminder context when a reminder is due
"""

from datetime import datetime
from typing import Optional

from loguru import logger


class PromptBuilder:
    """
    Assembles the full message list sent to the LLM for each interaction.

    The structure is:
        1. System prompt (personality + current context)
        2. Conversation history (from session)
        3. The user's current message

    The system prompt is dynamic — it is rebuilt for every call to include
    the current activity state, room, and time so Jarvis is always aware.
    """

    def __init__(self, config: dict) -> None:
        self._base_system = config["ollama"].get(
            "system_prompt",
            "You are Jarvis, an ambient home AI assistant.",
        ).strip()

    def build(
        self,
        user_text: str,
        state=None,  # modules.context.state.ActivityState — optional
        session=None,  # modules.brain.session.ConversationSession — optional
        room: str = "office",
        extras: Optional[dict] = None,
    ) -> list[dict]:
        """
        Assemble the complete message list for an LLM call.

        Args:
            user_text: The user's current utterance.
            state:     Current ActivityState (for context injection). Can be None.
            session:   ConversationSession for the room. History appended if provided.
            room:      Room identifier string.
            extras:    Optional dict of additional context strings to inject.

        Returns:
            List of {"role": ..., "content": ...} dicts ready for ollama.chat().
        """
        system_content = self._build_system(state, room, extras)

        messages: list[dict] = [
            {"role": "system", "content": system_content}
        ]

        # Inject conversation history
        if session is not None:
            messages.extend(session.get_messages())

        # The user's current message
        messages.append({"role": "user", "content": user_text})

        logger.debug(f"[Prompt] Built {len(messages)} messages for room '{room}'")
        return messages

    def _build_system(
        self,
        state=None,
        room: str = "office",
        extras: Optional[dict] = None,
    ) -> str:
        """
        Construct the dynamic system prompt by appending situational context
        to the base personality prompt.
        """
        now = datetime.now()
        time_str = now.strftime("%I:%M %p")
        day_str = now.strftime("%A, %B %-d")  # e.g., "Saturday, April 19"

        lines = [self._base_system, ""]
        lines.append(f"Current time: {time_str} on {day_str}.")
        lines.append(f"Active room: {room.replace('_', ' ').title()}.")

        # Activity state context
        if state is not None:
            activity = getattr(state, "activity", "unknown")
            confidence = getattr(state, "confidence", 0.0)
            context = getattr(state, "context", {})

            lines.append(
                f"What Cole is doing right now: {activity.replace('_', ' ')} "
                f"(confidence {confidence:.0%})."
            )

            if context.get("game"):
                lines.append(f"Active game: {context['game']}.")
            if context.get("project"):
                lines.append(f"Active project/file: {context['project']}.")
            if context.get("window_title"):
                lines.append(f"Active window: {context['window_title']}.")

        # Any caller-injected extras (room baseline, reminders, etc.)
        if extras:
            for key, value in extras.items():
                if value:
                    lines.append(f"{key.replace('_', ' ').title()}: {value}.")

        return "\n".join(lines)
