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
from typing import Any, Optional

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
        # How many recent non-speech events to surface as memory context
        self._memory_event_limit: int = int(
            config.get("memory", {}).get("prompt_memory_events", 8)
        )
        # Optional persona manager — injected post-construction by the
        # orchestrator. When set, _build_system uses
        # persona_manager.composed_system_prompt() (overlay + active
        # persona) as the personality base instead of self._base_system.
        # Kept Optional so PromptBuilder still works in unit tests + on
        # legacy configs that don't have a personas section.
        self._persona_manager = None  # set via attach_persona_manager()

    def attach_persona_manager(self, persona_manager) -> None:
        """Wire the PersonaManager so build() uses the active persona's
        composed system prompt (overlay + persona-specific text) on
        every call. Called by the orchestrator at boot, after both
        managers exist.
        """
        self._persona_manager = persona_manager
        logger.info("[Prompt] PersonaManager attached")

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

    async def build_with_memory(
        self,
        user_text: str,
        state=None,
        session=None,
        room: str = "office",
        db=None,
        extras: Optional[dict] = None,
    ) -> list[dict]:
        """
        Like build(), but also pulls a brief summary of recent non-speech events
        from the database (vision descriptions, appliance state changes, etc.)
        and injects it into the system prompt. Lets the LLM reference what's
        been happening in the room without needing to re-derive context.
        """
        enriched: dict[str, Any] = dict(extras or {})
        if db is not None:
            memory_block = await self._fetch_recent_events(db, room)
            if memory_block:
                enriched["recent_events"] = memory_block
        return self.build(
            user_text=user_text,
            state=state,
            session=session,
            room=room,
            extras=enriched,
        )

    async def _fetch_recent_events(self, db, room: str) -> Optional[str]:
        """
        Pull last N non-speech events for the room — vision descriptions,
        appliance changes, jarvis observations, etc. Excludes user_speech /
        jarvis_speech because those are already in the conversation history.
        """
        try:
            rows = await db.fetchall(
                "SELECT timestamp, type, content FROM events "
                "WHERE room = ? AND type NOT IN ('user_speech', 'jarvis_speech') "
                "ORDER BY timestamp DESC LIMIT ?",
                (room, self._memory_event_limit),
            )
        except Exception as e:
            logger.debug(f"[Prompt] memory fetch failed: {e}")
            return None
        if not rows:
            return None
        lines = ["Recent room events (newest first, for context only — don't recite):"]
        for r in rows:
            content = (r["content"] or "")[:140]
            ts = (r["timestamp"] or "")[:19]
            lines.append(f"  [{ts}] {r['type']}: {content}")
        return "\n".join(lines)

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
        time_str = now.strftime("%I:%M %p").lstrip("0")
        day_str = f"{now.strftime('%A, %B')} {now.day}, {now.year}"
        iso_str = now.strftime("%Y-%m-%dT%H:%M:%S")

        # Persona system: when wired, the PersonaManager composes the
        # full personality prompt as overlay + active-persona prompt.
        # Falls back to the legacy ollama.system_prompt for tests / configs
        # without a personas section. The overlay is always present —
        # that's what keeps default-Jarvis discreet about hidden personas.
        base = self._base_system
        if self._persona_manager is not None:
            try:
                base = self._persona_manager.composed_system_prompt()
            except Exception as e:
                logger.warning(
                    f"[Prompt] persona compose failed, falling back to "
                    f"ollama.system_prompt: {e}"
                )

        lines = [base, ""]
        lines.append(f"Current time: {time_str} on {day_str} (ISO: {iso_str}).")
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

        # Any caller-injected extras (room baseline, calendar, reminders, etc.)
        # Multi-line values get their own section so structured context (lists
        # of events, etc.) doesn't get squashed onto one line with a label.
        if extras:
            for key, value in extras.items():
                if not value:
                    continue
                str_value = str(value)
                if "\n" in str_value:
                    lines.append("")
                    lines.append(str_value)
                else:
                    lines.append(f"{key.replace('_', ' ').title()}: {str_value}.")

        return "\n".join(lines)
