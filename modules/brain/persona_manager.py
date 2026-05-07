"""
JARVIS — Ambient Home AI
========================
Mission: Own the persona system — Jarvis's switchable system-prompt bundles.
         A persona changes Jarvis's tone and behavior; it does NOT change the
         LLM model or TTS voice (both come from global config). Switching
         is a prompt-and-state-only mutation.

         Persona transitions are gated by a privacy check for
         requires_privacy=True personas (Cole must be alone in his current
         room) and have hard auto-revert floors that cannot be bypassed by
         the user-facing lock:

         - Person-entry: any non-Cole face in the active room → unconditional
           revert. Even with the lock set. The lock is a convenience flag for
           time/phone reverts, not a safety override.
         - Away timeout: idle/away >N seconds → revert (lock-respecting).
         - Phone call start: revert + remember the previous persona so we can
           offer to resume after the call ends if Cole is still alone.

         The persona overlay is a global prefix prepended to every persona's
         system prompt — including default. This keeps default-Jarvis discreet
         about hidden personas (otherwise asking "what other modes do you
         have?" would leak them).

         Architecture note: the bootstrap doc proposed bus-event subscriptions
         for the auto-revert triggers, but the existing codebase publishes
         vision/identity events through the dashboard `_broadcast` channel
         rather than the EventBus. To avoid building a parallel event layer,
         this manager exposes notify_*() methods that the orchestrator calls
         from its existing vision_loop / context_loop / pc_monitor hook
         points. Same outcomes, fits the current architecture.

Modules: modules/brain/persona_manager.py
Classes: PersonaState, PersonaManager

#todo: Persistence — serialize PersonaState to disk so restarts don't drop
       the active persona. v1 punts; safer to require re-activation on restart.
#todo: Memory tagging — tag memories created during a private persona so the
       prompt builder can filter them out when default is active. v1 trusts
       the discretion overlay.
#todo: Voice triggers — wire wake-word follow-up phrases ("private mode",
       "back to normal") to call set()/revert() once the dashboard surface
       is settled.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from loguru import logger

from core.config import PersonaConfig, PersonaRevertCfg
from core.exceptions import PersonaError


@dataclass
class PersonaState:
    """In-memory state for the persona system. Not persisted in v1 — a
    restart drops the active persona back to default, which is the safe
    behavior for a feature that's gated on Cole being alone.
    """
    active: str = "default"
    # When True, time-based and phone-based auto-revert are skipped.
    # Person-entry auto-revert ALWAYS fires regardless of lock. The lock
    # is for "I'm staying in this mode for the next hour" intent, not for
    # "ignore safety floors."
    locked: bool = False
    # Set when a phone call interrupted a private persona. After the call
    # ends and Cole is still alone, the manager offers to resume into this.
    pending_resume: Optional[str] = None
    # Wall-clock timestamp of the most recent persona transition. Useful
    # for cooldowns ("don't auto-offer the same persona twice within 60s")
    # and for "you've been in <persona> for 12 minutes" status display.
    last_change_ts: float = 0.0
    # Per-room "Cole is alone here right now" flags, updated by the
    # orchestrator's vision loop. Read by _is_alone() during activation
    # and after phone-call resume offers.
    room_occupancy: dict[str, "RoomOccupancy"] = field(default_factory=dict)


@dataclass
class RoomOccupancy:
    """What we believe about who's in a given room. Updated whenever the
    orchestrator finishes a vision pass on that room.
    """
    person_count: int = 0
    cole_present: bool = False
    # When True, vision didn't see anyone reliably (e.g., low light, no
    # frame). PersonaManager treats unknown as "not safe" for activation
    # but as "no signal to revert on" — fail closed on activation,
    # fail open on revert.
    unknown: bool = True
    updated_at: float = 0.0


# Identity strings the orchestrator passes when face recognition has a
# match. We accept either lowercase 'cole' or any case to be tolerant of
# upstream changes.
_COLE_ALIASES = {"cole"}


class PersonaManager:
    """Owns the active persona, the auto-revert state machine, and the
    output-leak filter that scrubs persona name mentions when discretion
    is required.
    """

    def __init__(
        self,
        personas: dict[str, PersonaConfig],
        overlay: str,
        revert_cfg: PersonaRevertCfg,
        broadcast: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> None:
        if "default" not in personas:
            raise PersonaError("personas dict missing required 'default' key")
        self._personas = personas
        self._overlay = (overlay or "").strip()
        self._revert_cfg = revert_cfg
        # Optional dashboard broadcast hook — orchestrator wires this so
        # persona transitions push to all open dashboard tabs.
        self._broadcast = broadcast
        self._state = PersonaState(last_change_ts=time.time())
        # Tracks the in-flight away-revert timer task so we can cancel
        # if the user comes back / changes persona before it fires.
        self._away_timer_task: Optional[asyncio.Task] = None
        # Phone-call state. The orchestrator's PC monitor pings
        # notify_phone_call_(started|ended); we track the boolean here
        # to dedup repeat notifications.
        self._call_active: bool = False
        # Compiled regex of hidden-persona names — built once for the
        # output filter. Includes both the persona key ("uwu") and the
        # display name if set. Word-boundary anchored so we don't
        # accidentally scrub "uwucat" or similar.
        self._leak_pattern = self._compile_leak_pattern()
        logger.info(
            f"[PersonaManager] Ready. Personas: {sorted(personas.keys())} "
            f"(hidden: {sorted([n for n, p in personas.items() if not p.visible_in_ui])})"
        )

    # ── Public read API ──────────────────────────────────────────────────

    def state(self) -> PersonaState:
        """Return the live state (mutable — for diagnostics, not for
        external mutation). Callers should treat as read-only."""
        return self._state

    def current(self) -> PersonaConfig:
        return self._personas[self._state.active]

    def current_name(self) -> str:
        return self._state.active

    def is_locked(self) -> bool:
        return self._state.locked

    def list_visible(self) -> list[dict]:
        """Return only personas with visible_in_ui=True. Used by the
        dashboard's dropdown — hidden personas (uwu) MUST NOT appear here.
        Activation of hidden personas requires the user to type the name
        into the command box, which means knowing it exists.
        """
        return [
            {
                "name": name,
                "display": p.display_name or name,
                "requires_privacy": p.requires_privacy,
            }
            for name, p in self._personas.items()
            if p.visible_in_ui
        ]

    def composed_system_prompt(self) -> str:
        """Return the full system prompt: overlay + active persona prompt.
        This is what the prompt builder injects as the system message.
        Overlay is unconditional — it makes default-Jarvis discreet too.
        """
        persona_prompt = self.current().system_prompt.strip()
        if self._overlay:
            return f"{self._overlay}\n\n{persona_prompt}"
        return persona_prompt

    # ── Public mutation API ──────────────────────────────────────────────

    async def set(self, name: str, *, lock: bool = False, force: bool = False) -> None:
        """Switch to `name`. Raises PersonaError on unknown name or when a
        requires_privacy persona is requested without privacy AND without
        force=True. force=True is the user-facing override; it does NOT
        bypass the person-entry auto-revert that fires after activation.
        """
        if name not in self._personas:
            raise PersonaError(f"Unknown persona: '{name}'")
        target = self._personas[name]
        if target.requires_privacy and not force:
            if not self._is_alone_for_active_room():
                raise PersonaError(
                    f"Cannot activate '{name}' — Cole is not confirmed alone"
                )
        old = self._state.active
        self._state.active = name
        self._state.locked = bool(lock)
        self._state.last_change_ts = time.time()
        # Activating a fresh persona clears any pending phone-resume.
        # Otherwise a manual switch right after a call would step on the
        # offer the manager was about to make.
        self._state.pending_resume = None
        # If we just activated a persona that doesn't need privacy, kill
        # any pending away-revert timer (it was for the previous private
        # persona and is now meaningless).
        self._cancel_away_timer()
        await self._announce("persona.changed", {
            "from": old, "to": name, "locked": self._state.locked,
        })
        logger.info(
            f"[PersonaManager] {old} → {name} (locked={self._state.locked})"
        )

    async def revert(self, *, reason: str, save_resume: bool = False) -> None:
        """Snap to default. save_resume=True stores the previous persona
        in pending_resume so a later notify_phone_call_ended can offer to
        come back. No-op when already on default — keeps logs from
        spamming on duplicate revert triggers.
        """
        old = self._state.active
        if old == "default":
            return
        if save_resume:
            self._state.pending_resume = old
        else:
            self._state.pending_resume = None
        self._state.active = "default"
        self._state.locked = False
        self._state.last_change_ts = time.time()
        self._cancel_away_timer()
        await self._announce("persona.reverted", {"from": old, "reason": reason})
        logger.info(f"[PersonaManager] revert: {old} → default ({reason})")

    def set_lock(self, locked: bool) -> None:
        """Toggle the lock without changing the active persona. Lock only
        affects time/phone reverts; person-entry revert is unconditional.
        """
        self._state.locked = bool(locked)
        logger.info(f"[PersonaManager] lock = {self._state.locked}")

    # ── Notifications from the orchestrator ──────────────────────────────

    def notify_room_occupancy(
        self,
        room: str,
        person_count: int,
        cole_present: bool,
    ) -> None:
        """Called by the orchestrator's vision loop after each pass.
        Updates the manager's view of who's in each room. Pure record-
        update — the actual revert decision happens in
        notify_face_identified, which fires on the specific
        non-Cole-detected case.
        """
        self._state.room_occupancy[room] = RoomOccupancy(
            person_count=int(person_count),
            cole_present=bool(cole_present),
            unknown=False,
            updated_at=time.time(),
        )

    async def notify_face_identified(self, room: str, identity: Optional[str]) -> None:
        """Orchestrator's vision loop calls this after each face check.
        identity is the recognizer's verdict ('cole', some other person's
        name, or None for "face seen but no match"). Triggers the
        unconditional person-entry revert when:
          - Active persona requires privacy
          - The identified face is not Cole
          - The room being reported is Cole's currently-active room
        """
        cur = self.current()
        if not cur.requires_privacy:
            return
        if identity and identity.strip().lower() in _COLE_ALIASES:
            return  # Cole's face — fine
        # Non-Cole face (named OR unknown) in the active room is a hard
        # revert. Even when locked. The lock is not a person-entry override.
        await self.revert(reason=f"foreign_person_in_{room}")

    async def notify_state_changed(self, new_state: str) -> None:
        """Orchestrator hooks the activity-state machine here. We watch
        for transitions INTO 'away' to start the away-revert timer, and
        transitions OUT to cancel it.
        """
        if new_state == "away":
            await self._maybe_start_away_timer()
        else:
            self._cancel_away_timer()

    async def notify_phone_call_started(self) -> None:
        """Called when PCMonitor (or any other detector) sees a process /
        window matching call_processes + call_window_keywords. Reverts a
        private persona and remembers it so we can offer to resume after
        the call ends.
        """
        if self._call_active:
            return  # already mid-call, no-op
        self._call_active = True
        if self._state.locked:
            return
        cur = self.current()
        if not cur.requires_privacy:
            return
        await self.revert(reason="phone_call_started", save_resume=True)

    async def notify_phone_call_ended(self) -> None:
        """Pair to notify_phone_call_started. If we have a pending resume
        AND Cole is still alone, broadcast a `persona.resume_offered`
        event the dashboard renders as a prompt. Voice-side resume wiring
        comes later — for v1, the user clicks 'resume' in the command box.
        """
        if not self._call_active:
            return
        self._call_active = False
        pending = self._state.pending_resume
        if not pending:
            return
        if not self._is_alone_for_active_room():
            self._state.pending_resume = None
            return
        await self._announce("persona.resume_offered", {
            "persona": pending,
            "window_s": self._revert_cfg.phone_resume_window_s,
        })
        logger.info(f"[PersonaManager] resume offered for '{pending}'")

    async def accept_pending_resume(self) -> bool:
        """User-facing 'resume' command. Reactivates the pending persona
        (subject to privacy gate) and clears the pending state. Returns
        True on success, False when there's nothing to resume.
        """
        pending = self._state.pending_resume
        if not pending:
            return False
        self._state.pending_resume = None
        try:
            await self.set(pending)
            return True
        except PersonaError as e:
            logger.warning(f"[PersonaManager] resume failed: {e}")
            return False

    # ── Output leak filter ───────────────────────────────────────────────

    def filter_output(self, text: str) -> str:
        """Defense in depth: scrub hidden-persona name mentions from text
        being sent to TTS or the dashboard, but only when the active
        persona is `default` AND someone other than Cole is present in
        his active room. The prompt overlay is supposed to handle this
        upstream; the filter catches LLM slips.

        We deliberately don't scrub when the active persona IS the hidden
        one (Cole's clearly already in on it) and don't scrub when the
        room is reliably alone.
        """
        if not text or not self._leak_pattern:
            return text
        # Only scrub on default — hidden persona conversations are
        # private-by-context, no leak possible.
        if self._state.active != "default":
            return text
        # Only scrub when not alone (or when occupancy unknown — fail
        # closed; better to over-redact than out the user).
        if self._is_alone_for_active_room():
            return text
        scrubbed, n = self._leak_pattern.subn("[redacted]", text)
        if n > 0:
            logger.warning(
                f"[PersonaManager] Output filter scrubbed {n} hidden-persona "
                "mention(s) — investigate the prompt path"
            )
        return scrubbed

    # ── Internals ────────────────────────────────────────────────────────

    def _compile_leak_pattern(self) -> Optional[re.Pattern[str]]:
        """Build a regex matching every hidden persona's name + display
        name. Word-boundary anchored to avoid scrubbing legitimate
        substrings. Returns None when no hidden personas exist.
        """
        names: list[str] = []
        for name, p in self._personas.items():
            if p.visible_in_ui:
                continue
            names.append(re.escape(name))
            if p.display_name and p.display_name != name:
                names.append(re.escape(p.display_name))
        if not names:
            return None
        # Word-boundary on both sides — \b doesn't match between two
        # non-word chars, but persona names are alphanumeric so this is fine.
        pattern = r"\b(" + "|".join(names) + r")\b"
        return re.compile(pattern, flags=re.IGNORECASE)

    def _is_alone_for_active_room(self) -> bool:
        """Conservative 'alone' check — the active room is determined by
        whichever orchestrator-side hook fed us notify_face_identified
        most recently. We treat 'unknown' as 'not alone' for activation
        (fail closed) so a stale or absent vision pass can't be exploited
        to activate a private persona.

        For revert purposes, the orchestrator does its own check via
        notify_face_identified directly — this function is for the
        proactive activation gate.
        """
        if not self._state.room_occupancy:
            return False  # no vision data at all → fail closed
        # Pick the most recently-updated room as a proxy for "Cole's
        # current room." The orchestrator updates only the room that
        # most recently passed a vision scan AND showed Cole's face.
        latest = max(
            self._state.room_occupancy.values(),
            key=lambda o: o.updated_at,
            default=None,
        )
        if latest is None or latest.unknown:
            return False
        # Alone = Cole present + no other person in frame, OR no people
        # detected at all (but Cole could be off-camera). For v1 we
        # treat both as "alone." If face recog stays shaky, tighten to
        # require explicit Cole-detected.
        return latest.person_count <= 1

    async def _maybe_start_away_timer(self) -> None:
        """Schedule an away-timeout revert if conditions warrant. The
        timer is cancellable so a user-return-from-away cancels cleanly.
        """
        cur = self.current()
        if not cur.requires_privacy:
            return  # not in a private persona; no revert to do
        if self._state.locked:
            return
        self._cancel_away_timer()
        self._away_timer_task = asyncio.create_task(self._away_timer())

    async def _away_timer(self) -> None:
        try:
            await asyncio.sleep(self._revert_cfg.away_timeout_s)
            # Re-check at fire time — state may have changed
            if self._state.locked:
                return
            if not self.current().requires_privacy:
                return
            await self.revert(reason="user_away_timeout")
        except asyncio.CancelledError:
            return

    def _cancel_away_timer(self) -> None:
        if self._away_timer_task is not None and not self._away_timer_task.done():
            self._away_timer_task.cancel()
        self._away_timer_task = None

    async def _announce(self, event_type: str, payload: dict) -> None:
        """Push a persona-state change to the dashboard via _broadcast.
        No-op if no broadcast hook was wired (e.g., during unit tests).
        """
        if self._broadcast is None:
            return
        try:
            await self._broadcast({"type": event_type, **payload})
        except Exception as e:
            logger.debug(f"[PersonaManager] broadcast failed: {e}")
