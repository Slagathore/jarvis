"""
JARVIS — Safety
===============
Mission: Door-open-without-human alarm. Fires when an exterior door is
         observed open AND no human has been within
         `door_supervision_radius_m` (default ~3m proxy via "any human
         in the door's room") for `door_unsupervised_grace_seconds`
         (default 15s).

         The publisher of `door.state` events is intentionally
         pluggable — vision today, reed-switch retrofit later. This
         module subscribes to the topic without caring about source.

         §29.3 trajectory awareness: if a recent EXITED_VIA_DOOR
         event matches this door, the countdown does NOT start (a
         human is on the porch / just stepped out). When that human's
         entity transitions to DEPARTED past T_door_return_seconds AND
         the door is still open, the countdown engages — that's the
         "walked away and forgot to close it" path.

Modules: modules/safety/alarms/door_open.py
Classes: DoorOpenAlarm
Spec:    new 2.md §29.3.

#todo: Real "within 3m of door" geometry. Today the proxy is "any
       human in the same room as the door"; that's coarser than 3m
       but biases toward NOT firing (room ⊃ 3m radius), which is
       the wrong direction for a safety alarm and is flagged in the
       doc. When room→world coordinates land, replace _human_near_door
       with a true distance check.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Optional

from loguru import logger

from modules.safety.alarms.alarm import Alarm
from modules.safety.alarms.state import AlarmType


class DoorOpenAlarm(Alarm):
    """Watches `door.state` + `world.entity_event` for the
    'door open and nobody supervising it' condition. Each known door
    has its own per-door countdown task; only one alarm instance is
    needed to cover all doors in the house."""

    PRIORITY = 30   # door = lowest; fire=10, cat=20.
    ALARM_TYPE = AlarmType.DOOR_OPEN

    def __init__(
        self,
        bus: Any,
        rooms_config: list[dict],
        notifier: Optional[Any] = None,
        store: Optional[Any] = None,
        # Tunables — match §28.5c defaults.
        unsupervised_grace_seconds: float = 15.0,
        door_return_seconds: float = 60.0,
        recent_exit_window_seconds: float = 60.0,
        mute_seconds: float = 300.0,
    ) -> None:
        super().__init__(
            bus=bus, notifier=notifier, store=store,
            mute_seconds=mute_seconds,
        )
        # door_id → {room, display_name}. The door_id is whatever the
        # publisher uses — typically the exterior_exit polygon's name
        # (e.g. "back_door"). _door_state[door_id] = "open"|"closed".
        self._doors: dict[str, dict] = self._index_doors(rooms_config)
        self._door_state: dict[str, str] = {
            d: "unknown" for d in self._doors
        }
        self._door_state_since: dict[str, float] = {}
        # Per-door countdown task.
        self._countdown_tasks: dict[str, asyncio.Task] = {}
        # Recent EXITED_VIA_DOOR (door_id → monotonic ts).
        self._recent_exits: dict[str, float] = {}
        # Last human presence per room (monotonic). Used by the
        # "no human within 3m" proxy.
        self._last_human_in_room: dict[str, float] = {}
        # Active fire context — door_id we last fired for.
        self._active_door: Optional[str] = None

        self._grace_s = float(unsupervised_grace_seconds)
        self._door_return_s = float(door_return_seconds)
        self._recent_exit_window_s = float(recent_exit_window_seconds)

    @staticmethod
    def _index_doors(rooms_config: list[dict]) -> dict[str, dict]:
        """Pull every exterior_exit polygon out of the rooms config and
        treat its `name` as the door_id. Indexed by door_id so a
        door.state event can route to the right room."""
        out: dict[str, dict] = {}
        for r in rooms_config:
            wm = r.get("world_model") or {}
            if not wm.get("enabled", True):
                continue
            for ex in wm.get("exits", []):
                if ex.get("kind") != "exterior_exit":
                    continue
                door_id = ex.get("name") or f"{r['id']}_exterior"
                out[door_id] = {
                    "room": r["id"],
                    "display_name": (
                        ex.get("display_name") or door_id.replace("_", " ")
                    ),
                }
        return out

    async def start(self) -> None:
        await self.bus.subscribe("door.state", self._on_door_state)
        await self.bus.subscribe("vision.observation", self._on_observation)
        await self.bus.subscribe("world.entity_event", self._on_entity_event)
        logger.info(
            f"[DoorOpen] watching {len(self._doors)} exterior door(s): "
            f"{sorted(self._doors.keys())}"
        )

    async def stop(self) -> None:
        for task in list(self._countdown_tasks.values()):
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._countdown_tasks.clear()
        await super().stop()

    # ── Bus subscribers ────────────────────────────────────────────────────

    async def _on_door_state(self, payload: dict) -> None:
        """{door_id, state, source, ts}. State ∈ {open, closed, unknown}.
        We act on edge transitions only — repeated 'open' events for an
        already-open door are no-ops."""
        door_id = payload.get("door_id")
        new_state = payload.get("state")
        if not door_id or new_state not in ("open", "closed", "unknown"):
            return
        if door_id not in self._doors:
            logger.debug(
                f"[DoorOpen] door_state for unknown door '{door_id}'; "
                "skipping (configure as exterior_exit to track)"
            )
            return
        old_state = self._door_state.get(door_id)
        if old_state == new_state:
            return
        self._door_state[door_id] = new_state
        self._door_state_since[door_id] = time.monotonic()
        if new_state == "open":
            self._maybe_start_countdown(door_id, payload)
        elif new_state == "closed":
            self._cancel_countdown(door_id)
            # Auto-resolve if we were firing for this door.
            if self._active_door == door_id:
                self._active_door = None
                await self.condition_cleared({
                    "reason": "door_closed",
                    "door_id": door_id,
                })

    async def _on_observation(self, payload: dict) -> None:
        """Update last-human-in-room timestamps from every observation
        batch. Used by the supervision check; cheap to always update."""
        room = payload.get("room")
        if not room:
            return
        observations = payload.get("observations") or []
        for obs in observations:
            obj_class = getattr(obs, "obj_class", None) or (
                obs.get("obj_class") if isinstance(obs, dict) else None
            )
            if obj_class == "person":
                self._last_human_in_room[room] = time.monotonic()
                # If we were counting down on a door in this room, the
                # countdown should cancel — supervision returned.
                for door_id, info in self._doors.items():
                    if info["room"] == room and door_id in self._countdown_tasks:
                        logger.info(
                            f"[DoorOpen] '{door_id}' countdown cancelled — "
                            f"human observed in '{room}'"
                        )
                        self._cancel_countdown(door_id)
                break  # one human is enough to update the room

    async def _on_entity_event(self, payload: dict) -> None:
        """Watch for EXITED_VIA_DOOR / DEPARTED to drive trajectory-aware
        countdown logic."""
        event_type = payload.get("event_type")
        meta = payload.get("metadata") or {}
        # Approximate EXITED_VIA_DOOR via the existing DEPARTED event,
        # which already carries `via_exit` (= door_id) when the
        # WorldModel sees a person cross an exterior_exit polygon.
        if event_type == "departed":
            via = meta.get("via_exit")
            if via and via in self._doors:
                self._recent_exits[via] = time.monotonic()
                # If the door is open and we'd otherwise start a
                # countdown, the recent-exit window will block it.
                logger.debug(
                    f"[DoorOpen] recent exit via '{via}' — countdown "
                    f"suppressed for {self._recent_exit_window_s:.0f}s"
                )

    # ── Countdown ──────────────────────────────────────────────────────────

    def _maybe_start_countdown(
        self, door_id: str, payload: dict,
    ) -> None:
        """Door went open. If a human is in the room OR there was a
        recent exit through this door, no countdown. Otherwise,
        schedule one for `unsupervised_grace_seconds`."""
        room = self._doors[door_id]["room"]
        now = time.monotonic()
        # Recent-exit suppression.
        last_exit = self._recent_exits.get(door_id)
        if last_exit is not None and (now - last_exit) <= self._recent_exit_window_s:
            logger.info(
                f"[DoorOpen] '{door_id}' opened but recent EXITED_VIA_DOOR "
                f"({now - last_exit:.0f}s ago); not starting countdown"
            )
            return
        # Active human supervision.
        if self._human_near_door(door_id):
            logger.info(
                f"[DoorOpen] '{door_id}' opened with human in '{room}'; "
                "supervision implied, countdown deferred"
            )
            return
        # Begin the countdown.
        if door_id in self._countdown_tasks:
            return  # already counting
        self._countdown_tasks[door_id] = asyncio.create_task(
            self._countdown(door_id, payload),
            name=f"door_open_countdown:{door_id}",
        )

    def _cancel_countdown(self, door_id: str) -> None:
        task = self._countdown_tasks.pop(door_id, None)
        if task is not None:
            task.cancel()

    async def _countdown(self, door_id: str, payload: dict) -> None:
        """Wait `_grace_s`. If the door is still open AND no human
        has appeared in the room, fire."""
        try:
            await asyncio.sleep(self._grace_s)
            if self._door_state.get(door_id) != "open":
                return
            if self._human_near_door(door_id):
                logger.info(
                    f"[DoorOpen] '{door_id}' grace expired but human "
                    "appeared in time; suppressing fire"
                )
                return
            self._active_door = door_id
            info = self._doors[door_id]
            await self.fire({
                "door_id": door_id,
                "door_name": info["display_name"],
                "room": info["room"],
                "source": payload.get("source", "vision"),
            })
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.exception(f"[DoorOpen] countdown crashed: {e}")
        finally:
            # Drop the task reference so a future re-open can schedule
            # a fresh one (e.g. door closed before grace, then re-opened).
            self._countdown_tasks.pop(door_id, None)

    def _human_near_door(self, door_id: str) -> bool:
        """Proxy: a human was observed in this door's room within the
        last `_grace_s`. The doc's intent is "within 3m of the door";
        this is room-level, which is broader. Biases toward NOT firing,
        which the §29.3 doc flags as the wrong asymmetry — once the
        room→world coordinate transform exists, swap this for a real
        distance check."""
        room = self._doors[door_id]["room"]
        last = self._last_human_in_room.get(room)
        if last is None:
            return False
        return (time.monotonic() - last) <= self._grace_s

    # ── State machine hooks ────────────────────────────────────────────────

    def _announcement(self, context: dict) -> tuple[str, str]:
        door = (
            context.get("door_name")
            or (
                self._doors.get(self._active_door, {}).get("display_name")
                if self._active_door else None
            )
            or "exterior door"
        )
        title = f"{door.upper()} OPEN."
        body = (
            f"{door} has been open with no one nearby for "
            f"{int(self._grace_s)}s. Audio will stop when the door is "
            "closed."
        )
        return title, body

    async def _condition_still_true(self) -> bool:
        """For mute-rearm. The door is still 'open without supervision'
        if our tracked state is open AND no human has shown up nearby
        since last we checked."""
        if self._active_door is None:
            return False
        if self._door_state.get(self._active_door) != "open":
            return False
        return not self._human_near_door(self._active_door)
