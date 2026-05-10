"""
JARVIS — Safety
===============
Mission: Base `Alarm` class. Owns the per-instance state machine
         (INACTIVE → FIRING_AUDIO → MUTED → RESOLVED → retrigger) and
         the wiring into the dispatcher. Concrete alarms (CatEscape,
         DoorOpen, Fire) subclass and implement:

           - the trigger predicate (when to fire)
           - the condition-clearance predicate (when condition is false)
           - the audio announcement template
           - any alarm-specific suppression / rearm / override rules

         Condition-clearance is the PRIMARY auto-resolve path. Voice-
         silence and visual confirmation are secondary, with rearm
         timers. See §29.1 for the why.

Modules: modules/safety/alarms/alarm.py
Classes: Alarm
Spec:    new 2.md §29.1, §29.7.

#todo: alarm_fires persistent log integration. Today every transition
       just logs to loguru; the §32 v4 schema has alarm_fires for
       reviewable history. Wire when the schema migration runs.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from loguru import logger

from modules.safety.alarms.state import AlarmState


class Alarm:
    """Base alarm. Subclasses customize trigger / condition / audio.

    Lifecycle:
        - register on a dispatcher → `attach(dispatcher)`
        - subscribe to whatever bus topic carries your trigger signal
          in `start()` (subclass)
        - call `await self.fire(...)` when condition becomes true
        - call `await self.condition_cleared(...)` when condition
          becomes false (primary auto-resolve)
        - voice silence handler `voice_silence()` is invoked by the
          dispatcher on user voice command
    """

    PRIORITY: int = 100   # lower = higher priority. fire=10, cat=20, door=30.
    ALARM_TYPE: str = "base"

    def __init__(
        self,
        bus: Any,
        notifier: Optional[Any] = None,
        store: Optional[Any] = None,
        mute_seconds: float = 300.0,        # 5 min default
    ) -> None:
        self.bus = bus
        # NotificationDispatcher (§31). Optional — alarms still fire
        # locally without it, just no phone alerts.
        self.notifier = notifier
        # AlarmStore (§32 alarm_fires + alarm_state persistence).
        # Optional — None falls back to NullAlarmStore semantics.
        from modules.safety.alarms.store import NullAlarmStore
        self.store = store if store is not None else NullAlarmStore()
        self._mute_seconds = float(mute_seconds)
        # Public, dispatcher reads this to compute audio_owner.
        self.state: AlarmState = AlarmState.INACTIVE
        # Most-recent fire id, for correlating audio + phone alerts +
        # alarm_fires log rows.
        self.fire_id: Optional[str] = None
        self.fired_at: Optional[datetime] = None
        # Hooks the dispatcher injects.
        self._dispatcher: Optional[Any] = None
        self._mute_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    # ── Wiring ──────────────────────────────────────────────────────────────

    def attach(self, dispatcher: Any) -> None:
        """Called by AlarmDispatcher.register so the alarm can call
        back into dispatcher.on_alarm_state_change on transitions."""
        self._dispatcher = dispatcher

    async def start(self) -> None:
        """Subclasses override to subscribe to bus topics that carry
        their trigger signal. Base class is a no-op."""
        return

    async def stop(self) -> None:
        """Cancel any in-flight mute timers. Subclasses override to
        unsubscribe from bus topics if needed."""
        await self._cancel_mute_timer()

    # ── State machine entrypoints ───────────────────────────────────────────

    async def fire(self, context: Optional[dict] = None) -> None:
        """Trigger condition became true. Move to FIRING_AUDIO unless
        we're suppressed. Phone alert fires once per fire (debounced
        by fire_id)."""
        async with self._lock:
            if self.state == AlarmState.SUPPRESSED:
                logger.info(
                    f"[Alarm:{self.ALARM_TYPE}] suppressed, ignoring fire"
                )
                return
            old = self.state
            # New fire identity each time we transition INACTIVE/RESOLVED
            # → FIRING_AUDIO. MUTED → FIRING_AUDIO (rearm) reuses the
            # same fire_id so the audit log groups the rearm with its
            # original fire.
            if old in (AlarmState.INACTIVE, AlarmState.RESOLVED):
                self.fire_id = uuid.uuid4().hex
                self.fired_at = datetime.now(timezone.utc)
            self.state = AlarmState.FIRING_AUDIO
            await self._cancel_mute_timer()
            logger.warning(
                f"[Alarm:{self.ALARM_TYPE}] FIRING (fire_id={self.fire_id}, "
                f"context={context or {}})"
            )

        # Notify outside the lock so notifier / store failure doesn't
        # block state transitions.
        if old in (AlarmState.INACTIVE, AlarmState.RESOLVED):
            await self._send_phone_alert(context)
            # New fire → audit row. Rearms (MUTED → FIRING_AUDIO) reuse
            # the same fire_id so record_fire's dedup keeps audit clean.
            if self.fire_id is not None and self.fired_at is not None:
                await self.store.record_fire(
                    fire_id=self.fire_id,
                    alarm_type=self.ALARM_TYPE,
                    fired_at=self.fired_at,
                    metadata=context or {},
                )
        await self.store.record_state(
            self.ALARM_TYPE, AlarmState.FIRING_AUDIO.value,
            metadata={"fire_id": self.fire_id},
        )
        await self._notify_dispatcher(old, AlarmState.FIRING_AUDIO)

    async def condition_cleared(
        self, context: Optional[dict] = None
    ) -> None:
        """Trigger condition became false (cat back inside, door
        closed, fire signal gone). Primary auto-resolve."""
        async with self._lock:
            if self.state in (AlarmState.INACTIVE, AlarmState.RESOLVED):
                return
            old = self.state
            self.state = AlarmState.RESOLVED
            resolved_fire_id = self.fire_id
            await self._cancel_mute_timer()
            logger.info(
                f"[Alarm:{self.ALARM_TYPE}] RESOLVED — condition cleared "
                f"(fire_id={self.fire_id})"
            )
        # Stamp the resolution on the alarm_fires row + flip alarm_state.
        # Both record_* paths swallow failures; persistence shouldn't
        # block dispatcher notification.
        if resolved_fire_id is not None:
            resolution_kind = (
                (context or {}).get("reason") or "condition_clear"
            )
            await self.store.record_resolution(
                fire_id=resolved_fire_id,
                alarm_type=self.ALARM_TYPE,
                resolution=resolution_kind,
                metadata=context or {},
            )
        await self.store.record_state(
            self.ALARM_TYPE, AlarmState.RESOLVED.value,
            metadata={"fire_id": resolved_fire_id},
        )
        await self._notify_dispatcher(old, AlarmState.RESOLVED)

    async def voice_silence(self) -> None:
        """User said 'Jarvis, stop alarm'. Move to MUTED with rearm
        timer; condition is still active so the rearm fires
        when the timer expires (unless condition_cleared first)."""
        async with self._lock:
            if self.state != AlarmState.FIRING_AUDIO:
                return
            old = self.state
            self.state = AlarmState.MUTED
            self._mute_task = asyncio.create_task(
                self._mute_timer(), name=f"alarm_mute:{self.ALARM_TYPE}"
            )
            logger.info(
                f"[Alarm:{self.ALARM_TYPE}] MUTED for "
                f"{self._mute_seconds:.0f}s (fire_id={self.fire_id})"
            )
        await self.store.record_state(
            self.ALARM_TYPE, AlarmState.MUTED.value,
            metadata={
                "fire_id": self.fire_id,
                "mute_seconds": self._mute_seconds,
            },
        )
        await self._notify_dispatcher(old, AlarmState.MUTED)

    async def suppress(self) -> None:
        """Higher-layer suppression (global disarm, per-cat exclusion).
        Won't fire while in this state regardless of condition."""
        async with self._lock:
            old = self.state
            self.state = AlarmState.SUPPRESSED
            await self._cancel_mute_timer()
        await self._notify_dispatcher(old, AlarmState.SUPPRESSED)

    async def unsuppress(self) -> None:
        async with self._lock:
            if self.state != AlarmState.SUPPRESSED:
                return
            old = self.state
            self.state = AlarmState.INACTIVE
        await self._notify_dispatcher(old, AlarmState.INACTIVE)

    # ── Internals ───────────────────────────────────────────────────────────

    async def _mute_timer(self) -> None:
        """If the condition is still active when the mute window
        expires, re-fire. Subclasses override `_condition_still_true`
        if they need a richer check."""
        try:
            await asyncio.sleep(self._mute_seconds)
            if await self._condition_still_true():
                logger.info(
                    f"[Alarm:{self.ALARM_TYPE}] mute window expired, "
                    "rearming"
                )
                await self.fire({"reason": "mute_timer_expired"})
            else:
                # Mute expired but condition has cleared — natural resolve.
                await self.condition_cleared({"reason": "mute_expired_clear"})
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.exception(f"[Alarm:{self.ALARM_TYPE}] mute timer crashed: {e}")

    async def _cancel_mute_timer(self) -> None:
        if self._mute_task is None:
            return
        self._mute_task.cancel()
        try:
            await self._mute_task
        except (asyncio.CancelledError, Exception):
            pass
        self._mute_task = None

    async def _condition_still_true(self) -> bool:
        """Subclasses override. Default: assume false so the mute
        timer doesn't ping-pong forever in subclasses that forgot
        to implement it."""
        return False

    async def _send_phone_alert(self, context: Optional[dict]) -> None:
        """Build the §31 Alert payload and send. No-op when notifier
        wasn't injected (tests, smoke runs)."""
        if self.notifier is None:
            return
        try:
            from modules.notifications import Alert, AlertPriority
            title, body = self._announcement(context or {})
            priority = self._priority()
            await self.notifier.send(Alert(
                alarm_type=self.ALARM_TYPE,
                title=title,
                body=body,
                priority=priority,
                metadata={"fire_id": self.fire_id, **(context or {})},
            ))
        except Exception as e:
            logger.debug(
                f"[Alarm:{self.ALARM_TYPE}] phone-alert build/send failed: {e}"
            )

    async def _notify_dispatcher(
        self, old: AlarmState, new: AlarmState
    ) -> None:
        if self._dispatcher is None:
            return
        try:
            await self._dispatcher.on_alarm_state_change(
                self.ALARM_TYPE, old, new
            )
        except Exception as e:
            logger.exception(
                f"[Alarm:{self.ALARM_TYPE}] dispatcher notify failed: {e}"
            )

    # ── Subclass extension points ───────────────────────────────────────────

    def _announcement(self, context: dict) -> tuple[str, str]:
        """Return (title, body) for both phone alerts and TTS. Subclasses
        override to inject room / cat name / door name."""
        return (
            f"{self.ALARM_TYPE.replace('_', ' ').title()} alarm",
            f"Alarm fired at {self.fired_at}",
        )

    def _priority(self) -> Any:
        """Notification-dispatcher priority. Defaults derive from PRIORITY:
        fire=URGENT, cat=HIGH, door=NORMAL. Override in subclasses for
        finer control."""
        from modules.notifications import AlertPriority
        if self.PRIORITY <= 10:
            return AlertPriority.URGENT
        if self.PRIORITY <= 20:
            return AlertPriority.HIGH
        return AlertPriority.NORMAL
