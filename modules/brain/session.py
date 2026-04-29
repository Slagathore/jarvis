"""
JARVIS — Ambient Home AI
========================
Mission: Maintain per-room conversation sessions. Each room has its own rolling
         context window so Jarvis can refer back to earlier parts of a conversation
         naturally. Sessions expire after a configurable idle period so stale
         context doesn't bleed into new conversations.

Modules: modules/brain/session.py
Classes: ConversationSession, SessionManager
Functions:
    ConversationSession.__init__(room, max_turns) — Create session
    ConversationSession.add_turn(role, content)   — Append a conversation turn
    ConversationSession.get_messages()            — Return full history as message dicts
    ConversationSession.clear()                   — Reset the session
    ConversationSession.is_expired(ttl_seconds)   — True if idle too long

    SessionManager.__init__(config)               — Initialize with config
    SessionManager.get_session(room)              — Get or create session for room
    SessionManager.clear_room(room)               — Manually expire a room's session
    SessionManager.cleanup_expired()              — Async background cleanup loop

Variables:
    ConversationSession.room         — Room identifier string
    ConversationSession.turns        — list of {"role": str, "content": str}
    ConversationSession.last_active  — datetime of last activity
    SessionManager._sessions         — dict mapping room → ConversationSession
    DEFAULT_SESSION_TTL_SECONDS      — Default idle expiry (30 minutes)

#todo: Persist sessions to SQLite so they survive restarts
#todo: Add session summary compression — summarize old turns to save context tokens
#todo: Support multi-user sessions (distinguish Anna, Sophie, Cole by voice)
#todo: Add session metadata (topic tracking, sentiment trend)
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional

from loguru import logger

# Sessions expire after 30 minutes of inactivity
DEFAULT_SESSION_TTL_SECONDS = 30 * 60


class ConversationSession:
    """
    A single room's conversation context. Stores the last N turns and
    tracks when the session was last active.

    Turns are stored as OpenAI-format message dicts for direct use in LLM calls:
        {"role": "user" | "assistant", "content": "..."}
    """

    def __init__(self, room: str, max_turns: int = 20) -> None:
        self.room: str = room
        self.max_turns: int = max_turns
        self.turns: list[dict] = []
        self.last_active: datetime = datetime.now()

    def add_turn(self, role: str, content: str) -> None:
        """
        Append a message to the session. Role must be "user" or "assistant".
        Trims the oldest turn pairs when max_turns is exceeded to keep
        context within the LLM's token budget.
        """
        if role not in ("user", "assistant"):
            raise ValueError(f"Invalid role '{role}' — must be 'user' or 'assistant'")

        self.turns.append({"role": role, "content": content.strip()})
        self.last_active = datetime.now()

        # Keep within budget — remove oldest pairs (user+assistant) together
        # to avoid leaving an orphaned user turn at the start
        while len(self.turns) > self.max_turns * 2:
            # Remove the oldest pair
            self.turns.pop(0)
            if self.turns and self.turns[0]["role"] == "assistant":
                self.turns.pop(0)

    def get_messages(self) -> list[dict]:
        """Return all turns as a flat list of message dicts, ready for the LLM."""
        return list(self.turns)

    def clear(self) -> None:
        """Wipe the conversation history. Called on explicit reset or long idle."""
        self.turns.clear()
        logger.debug(f"[Session] Cleared session for room '{self.room}'")

    def is_expired(self, ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS) -> bool:
        """Return True if the session has been idle longer than ttl_seconds."""
        idle = (datetime.now() - self.last_active).total_seconds()
        return idle > ttl_seconds

    @property
    def turn_count(self) -> int:
        """Number of individual messages (not pairs) in the session."""
        return len(self.turns)

    @property
    def last_user_message(self) -> Optional[str]:
        """The most recent user turn content, or None if no turns exist."""
        for t in reversed(self.turns):
            if t["role"] == "user":
                return t["content"]
        return None


class SessionManager:
    """
    Manages ConversationSession instances for all rooms.
    Creates sessions on demand and expires idle ones in the background.

    Config keys used (from config["memory"]):
        max_conversation_turns: int — max turns per session (default 20)
    """

    def __init__(self, config: dict) -> None:
        self._max_turns: int = config["memory"].get("max_conversation_turns", 20)
        self._sessions: dict[str, ConversationSession] = {}

    def get_session(self, room: str) -> ConversationSession:
        """
        Return the active session for a room, creating a new one if needed.
        This is the primary method called by the orchestrator.
        """
        if room not in self._sessions:
            self._sessions[room] = ConversationSession(room, self._max_turns)
            logger.debug(f"[Session] New session for room '{room}'")
        return self._sessions[room]

    def clear_room(self, room: str) -> None:
        """Force-clear a room's session. The next get_session() will start fresh."""
        if room in self._sessions:
            self._sessions[room].clear()
            del self._sessions[room]
            logger.info(f"[Session] Room '{room}' session cleared")

    async def cleanup_expired(
        self,
        ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS,
        check_interval_seconds: int = 300,
    ) -> None:
        """
        Background task that expires idle sessions.
        Runs forever — start as an asyncio Task.

        Args:
            ttl_seconds:            Idle time before expiry.
            check_interval_seconds: How often to scan for expired sessions.
        """
        while True:
            await asyncio.sleep(check_interval_seconds)
            expired = [
                room for room, sess in self._sessions.items()
                if sess.is_expired(ttl_seconds)
            ]
            for room in expired:
                logger.info(f"[Session] Expiring idle session for room '{room}'")
                del self._sessions[room]
