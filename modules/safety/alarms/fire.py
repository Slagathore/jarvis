"""
JARVIS — Safety
===============
Mission: Fire-detected alarm. Fires on any `fire.signal` event with
         `active=True`; resolves on continuous absence for
         `fire_signal_clearance_seconds` (default 60s).

         §29.4's resident-trust model is the core design choice: this
         household has competent adults and a commercial-grade
         extinguisher. Sustained klaxon while you're holding the
         extinguisher creates more risk than it mitigates, so the
         alarm accepts more silencing risk in exchange for better
         ergonomics — voice silence + visual-confirmation-with-non-
         increasing-signal both put it in a 3-min MUTED state.

         The hard-rearms (signal-increase override + unattended
         rearm + mute-timer expiration) catch the failure modes
         where silencing was wrong:
            • signal grew after silence → fire is winning, audio comes
              back regardless of human acknowledgment
            • 5min of silence with signal still active and nobody in
              the fire room → distracted resident, audio comes back
            • mute expired and signal still active → standard rearm

         Broadcast-on-silence: when ANY silencing path engages, every
         household member's phone gets a notification ("Cole
         acknowledged fire alarm in kitchen at 14:14"). The audio
         silence is local-room behavior; the broadcast keeps the
         distributed record intact.

         Source-agnostic: smoke / thermal / vision-fire-detector are
         interchangeable publishers of `fire.signal` events. Same
         schema, same handler. When the vision-fire-detector lands,
         it just publishes to the same topic.

Modules: modules/safety/alarms/fire.py
Classes: FireAlarm
Spec:    new 2.md §29.4.

#todo: Vision-fire-detector publisher (separate module). Today
       fire.signal is consumed only — no module currently emits
       it. The dashboard test button + a future ELM-vision detector
       will both publish to this same topic.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from loguru import logger

from modules.safety.alarms.alarm import Alarm
from modules.safety.alarms.state import AlarmState, AlarmType


class FireAlarm(Alarm):
    """Subscribes to `fire.signal` (active/signal level/room) and
    `world.entity_event` (visual-confirmation paths)."""

    PRIORITY = 10   # fire = highest. cat=20, door=30.
    ALARM_TYPE = AlarmType.FIRE

    def __init__(
        self,
        bus: Any,
        notifier: Optional[Any] = None,
        store: Optional[Any] = None,
        # Tunables — match §28.5d defaults.
        signal_clearance_seconds: float = 60.0,
        visual_confirmation_dwell_seconds: float = 3.0,
        unattended_rearm_seconds: float = 300.0,
        signal_increase_override_seconds: float = 30.0,
        signal_increase_threshold: float = 0.05,
        mute_seconds: float = 180.0,    # 3 min — shorter than default 5
    ) -> None:
        super().__init__(
            bus=bus, notifier=notifier, store=store,
            mute_seconds=mute_seconds,
        )
        self._clearance_s = float(signal_clearance_seconds)
        self._visual_dwell_s = float(visual_confirmation_dwell_seconds)
        self._unattended_rearm_s = float(unattended_rearm_seconds)
        self._override_s = float(signal_increase_override_seconds)
        self._sig_increase_threshold = float(signal_increase_threshold)

        # Per-room signal level. fire.signal payload shape:
        #   {room: str, active: bool, level: float (0-1), source: str, ts: ...}
        self._signal_level: dict[str, float] = {}
        self._signal_active: dict[str, bool] = {}
        self._signal_clear_since: dict[str, float] = {}
        # Per-room human-presence dwell: {room: monotonic_first_seen}.
        self._human_in_room_since: dict[str, float] = {}
        # Active fire context — which room we're firing for.
        self._active_room: Optional[str] = None
        # Hard-override timer: if set + future, voice_silence is rejected.
        self._override_until: float = 0.0
        # Last signal level observed for the active room — used by the
        # "signal increased after mute" override.
        self._signal_level_at_mute: Optional[float] = None
        # Tasks owned by this instance (clearance watcher, unattended
        # rearm watcher); cancelled in stop().
        self._watch_tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        await self.bus.subscribe("fire.signal", self._on_fire_signal)
        await self.bus.subscribe("world.entity_event", self._on_entity_event)
        logger.info("[FireAlarm] watching fire.signal + world.entity_event")
        # No module in the codebase publishes fire.signal yet (see the
        # #todo in this file's header) — until a smoke/thermal/vision
        # detector lands, this alarm is armed but can never trigger.
        # Shout at boot so nobody assumes fire coverage exists.
        # DELETE this warning in the same change that ships a publisher.
        logger.warning(
            "[FireAlarm] armed but INERT: nothing publishes fire.signal — "
            "there is NO fire detection until a detector module ships"
        )

    async def stop(self) -> None:
        for t in list(self._watch_tasks):
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._watch_tasks.clear()
        await super().stop()

    # ── fire.signal subscriber ─────────────────────────────────────────────

    async def _on_fire_signal(self, payload: dict) -> None:
        """{room, active, level, source}. Tracks per-room signal level
        and triggers the state machine."""
        room = payload.get("room")
        if not room:
            return
        active = bool(payload.get("active", False))
        level = float(payload.get("level", 1.0 if active else 0.0))
        prev_level = self._signal_level.get(room, 0.0)
        prev_active = self._signal_active.get(room, False)
        self._signal_level[room] = level
        self._signal_active[room] = active

        if active:
            # Reset clearance timer on any fresh active reading.
            self._signal_clear_since.pop(room, None)
            if not prev_active or self.state in (
                AlarmState.INACTIVE, AlarmState.RESOLVED,
            ):
                # Fresh fire signal in this room — fire if we're not
                # already audio-active for this room.
                if (
                    self._active_room != room
                    or self.state == AlarmState.RESOLVED
                ):
                    self._active_room = room
                    await self.fire({
                        "room": room,
                        "level": level,
                        "source": payload.get("source", "unknown"),
                    })
            # Signal-increase override: if we're MUTED and the level
            # rose meaningfully past the level at mute time, the alarm
            # re-fires regardless of human acknowledgment, and a
            # 30s no-silence window opens.
            if self.state == AlarmState.MUTED and self._active_room == room:
                baseline = self._signal_level_at_mute or 0.0
                if (level - baseline) >= self._sig_increase_threshold:
                    logger.warning(
                        f"[FireAlarm] signal increase override fired in "
                        f"'{room}' (Δ={level - baseline:.2f}); "
                        f"voice silence locked for {self._override_s:.0f}s"
                    )
                    self._override_until = (
                        time.monotonic() + self._override_s
                    )
                    await self.fire({
                        "room": room, "level": level,
                        "reason": "signal_increase_override",
                    })
        else:
            # Signal cleared (or never was active). Track first-clear
            # timestamp for the clearance watcher.
            if room not in self._signal_clear_since:
                self._signal_clear_since[room] = time.monotonic()
            if (
                prev_active
                and self._active_room == room
                and self.state in (AlarmState.FIRING_AUDIO, AlarmState.MUTED)
            ):
                # Spawn a watcher; the actual condition_cleared call
                # waits for `clearance_seconds` of continuous absence.
                self._spawn_clearance_watcher(room)

    async def _on_entity_event(self, payload: dict) -> None:
        """Visual-confirmation path: a human dwelling in the fire room
        for ≥`visual_dwell_s` AND signal not increasing → MUTED."""
        if self._active_room is None:
            return
        if payload.get("entity_type") != "person":
            return
        room_raw = payload.get("room")
        if not isinstance(room_raw, str) or room_raw != self._active_room:
            return
        # Type-narrow `room` to str so the subsequent dict APIs see
        # a concrete key type instead of `Any | None`.
        room: str = room_raw
        # Track entries; first time a human shows up in the active fire
        # room, stamp the dwell start. Once the dwell exceeds threshold
        # AND signal is non-increasing, mute via visual confirmation.
        event_type = payload.get("event_type")
        if event_type in ("first_seen", "reappeared", "moved_to"):
            self._human_in_room_since.setdefault(
                room, time.monotonic(),
            )
        # On every entity event in the room, check whether the dwell
        # condition is satisfied.
        first_seen_at = self._human_in_room_since.get(room)
        if first_seen_at is None:
            return
        dwell = time.monotonic() - first_seen_at
        if dwell < self._visual_dwell_s:
            return
        # Signal-not-increasing check: compare current level to a
        # rolling baseline (last seen 5s ago via `_signal_level`).
        # Approximation: the level we got with the latest fire.signal
        # is "not increasing" if active=False or level <= baseline.
        # Strict implementation needs windowed history; the simple
        # rule is adequate for v4.
        current_level = self._signal_level.get(room, 0.0)
        baseline = self._signal_level_at_mute or current_level
        if current_level > baseline + self._sig_increase_threshold:
            return
        if self.state != AlarmState.FIRING_AUDIO:
            return
        logger.info(
            f"[FireAlarm] visual confirmation in '{room}' (dwell="
            f"{dwell:.1f}s, level={current_level:.2f}); muting"
        )
        await self._mute_via("visual_confirm")

    # ── Voice silence with override gate + broadcast ───────────────────────

    async def voice_silence(self) -> None:
        """Override §29.4: refuse to silence if the signal-increase
        override is in effect (hardware-level priority for fires
        that are growing). Otherwise mute + broadcast."""
        if time.monotonic() < self._override_until:
            logger.warning(
                "[FireAlarm] voice_silence rejected — signal-increase "
                "override active"
            )
            return
        await self._mute_via("voice_silence")

    async def _mute_via(self, kind: str) -> None:
        """Shared mute path used by voice_silence + visual_confirm.
        Stamps the level at mute (for the increase-override comparison)
        and triggers the broadcast-on-silence."""
        self._signal_level_at_mute = (
            self._signal_level.get(self._active_room or "")
            if self._active_room else None
        )
        # Spawn the unattended-rearm watcher BEFORE the parent's
        # voice_silence transitions to MUTED, so the watcher sees a
        # consistent state when it wakes.
        self._spawn_unattended_rearm_watcher()
        await super().voice_silence()
        await self._broadcast_on_silence(kind)

    async def _broadcast_on_silence(self, kind: str) -> None:
        """Per §29.4: a silence event triggers a phone notification to
        every resident. Uses the same dispatcher as the fire-firing
        notification but with a different title."""
        if self.notifier is None:
            return
        try:
            from modules.notifications import Alert, AlertPriority
            room = self._active_room or "unknown room"
            await self.notifier.send(Alert(
                alarm_type=self.ALARM_TYPE,
                title=f"FIRE alarm muted ({kind}) in {room}",
                body=(
                    f"Fire alarm in {room} was silenced ({kind}). "
                    "Audio will rearm if signal increases or stays "
                    f"active for {int(self._unattended_rearm_s)}s."
                ),
                priority=AlertPriority.HIGH,
                metadata={
                    "fire_id": self.fire_id,
                    "silence_kind": kind,
                    "room": room,
                },
            ))
        except Exception as e:
            logger.debug(f"[FireAlarm] silence broadcast failed: {e}")

    # ── Background watchers ────────────────────────────────────────────────

    def _spawn_clearance_watcher(self, room: str) -> None:
        """After clearance_seconds of continuous signal-absence in this
        room, call condition_cleared. Single-shot: cancelled if signal
        goes active again."""
        async def _watch():
            try:
                await asyncio.sleep(self._clearance_s)
                if self._signal_active.get(room, False):
                    return  # signal came back
                if self._active_room != room:
                    return
                await self.condition_cleared({
                    "reason": "fire_signal_clearance",
                    "room": room,
                })
                self._active_room = None
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.exception(f"[FireAlarm] clearance watcher crashed: {e}")
        # Cancel any existing watcher for this room.
        self._cancel_room_watchers(room)
        task = asyncio.create_task(_watch(), name=f"fire_clearance:{room}")
        self._watch_tasks.append(task)

    def _spawn_unattended_rearm_watcher(self) -> None:
        """If after `_unattended_rearm_s` of silence the signal is
        still active AND no human is in the fire room, refire."""
        if self._active_room is None:
            return
        room = self._active_room

        async def _watch():
            try:
                await asyncio.sleep(self._unattended_rearm_s)
                if not self._signal_active.get(room, False):
                    return
                if self.state != AlarmState.MUTED:
                    return
                # Crude "no human in fire room" — we last saw a human
                # in this room more than 30s ago, treat as unattended.
                last_seen = self._human_in_room_since.get(room)
                if (
                    last_seen is not None
                    and (time.monotonic() - last_seen) < 30.0
                ):
                    return  # human still there
                logger.warning(
                    f"[FireAlarm] unattended rearm in '{room}' — "
                    "signal still active and human absent"
                )
                await self.fire({
                    "room": room,
                    "reason": "unattended_rearm",
                })
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.exception(f"[FireAlarm] unattended watcher crashed: {e}")
        task = asyncio.create_task(_watch(), name="fire_unattended_rearm")
        self._watch_tasks.append(task)

    def _cancel_room_watchers(self, room: str) -> None:
        """Cancel any clearance / rearm watchers tagged for this room.
        Lazy cleanup — finished tasks come out of the list naturally
        on next iteration; we just cancel the live ones."""
        survivors: list[asyncio.Task] = []
        for t in self._watch_tasks:
            if t.done():
                continue
            name = getattr(t, "get_name", lambda: "")()
            if name.endswith(f":{room}") or "unattended" in name:
                t.cancel()
                continue
            survivors.append(t)
        self._watch_tasks = survivors

    # ── Alarm hooks ────────────────────────────────────────────────────────

    def _announcement(self, context: dict) -> tuple[str, str]:
        room = context.get("room") or self._active_room or "unknown room"
        title = f"FIRE DETECTED IN {room.upper().replace('_', ' ')}."
        reason = context.get("reason", "")
        body_extra = (
            " Signal still rising." if reason == "signal_increase_override"
            else " Audio will stop when the signal clears."
        )
        body = (
            f"Fire signal active in {room}.{body_extra} "
            "Voice silence: 'Jarvis, I see it' / 'Jarvis, stop alarm'."
        )
        return title, body

    async def _condition_still_true(self) -> bool:
        """For mute-rearm. The fire signal is still active in our
        active_room → condition holds."""
        if self._active_room is None:
            return False
        return bool(self._signal_active.get(self._active_room, False))
