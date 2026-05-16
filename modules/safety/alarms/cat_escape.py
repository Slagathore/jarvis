"""
JARVIS — Safety
===============
Mission: Cat-escape alarm. Concrete subclass of Alarm that subscribes
         to `vision.observation` and `door.state`, then fires when an
         `entity_type='cat'` observation has its bbox center inside an
         `exterior_exit` polygon whose matching exterior door is open.

         Auto-resolve (primary): the same cat is observed in any
         monitored interior room. Implementation: subscribe to
         `world.entity_event` and watch for REAPPEARED with the
         tracked cat's entity_id back inside the house. For the
         framework smoke test we accept either signal — entity_event
         or a fresh vision.observation in an interior room.

         Suppression mechanisms (per §29.2):
            - Global disarm (armed=False)
            - Per-cat suppression (planned outing)
            - Post-fire mute window (default 5 min)

Modules: modules/safety/alarms/cat_escape.py
Classes: CatEscapeAlarm
Spec:    new 2.md §29.2.

#todo: Intentional-retrieval window. If a human exits through the
       same door within 30s of the cat alarm firing, mute for 5 min.
       Needs the §13.x EXITED_VIA_DOOR signal — wire when that lands.
#todo: Pre-arm check at noon daily — synthesize a test observation per
       exterior_exit polygon, verify the subscriber path lights up, play
       a soft chime confirmation. Schedule via the orchestrator's
       existing daily-tasks loop.
"""
from __future__ import annotations

from typing import Any, Optional

from loguru import logger

from modules.safety.alarms.alarm import Alarm
from modules.safety.alarms.state import AlarmType
from modules.world_model.geometry import bbox_center, point_in_polygon


class CatEscapeAlarm(Alarm):
    """Watches every camera-equipped room for a `cat` observation
    inside an `exterior_exit` polygon. Fires per-fire-id; auto-
    resolves when the cat is seen in any monitored interior room.
    """

    PRIORITY = 20      # cat = high (fire=10, door=30)
    ALARM_TYPE = AlarmType.CAT_ESCAPE

    def __init__(
        self,
        bus: Any,
        rooms_config: list[dict],
        notifier: Optional[Any] = None,
        store: Optional[Any] = None,
        armed: bool = True,
        require_door_open: bool = True,
        mute_seconds: float = 300.0,           # 5 min default
    ) -> None:
        super().__init__(
            bus=bus, notifier=notifier, store=store,
            mute_seconds=mute_seconds,
        )
        # Per-room exterior_exit polygons. Indexed by room id; each
        # entry is a list of {name, polygon} dicts.
        self._exits_by_room = self._index_exits(rooms_config)
        self._door_state: dict[str, str] = {
            ex.get("name", "exterior door"): "unknown"
            for exits in self._exits_by_room.values()
            for ex in exits
        }
        self._require_door_open = bool(require_door_open)
        self._armed = bool(armed)
        # Per-cat suppression window: {cat_name: until_ts_monotonic}.
        # Used for "I'm taking Sneaky out for 20 minutes."
        self._cat_suppressions: dict[str, float] = {}
        # Last-cat-name attribution for the active fire — used by the
        # auto-resolve subscriber to know which cat to watch for inside.
        self._active_cat_name: Optional[str] = None

    @staticmethod
    def _index_exits(rooms_config: list[dict]) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for r in rooms_config:
            wm = r.get("world_model") or {}
            if not wm.get("enabled", True):
                continue
            exterior = [
                ex for ex in wm.get("exits", [])
                if ex.get("kind") == "exterior_exit"
            ]
            if exterior:
                out[r["id"]] = exterior
        return out

    @property
    def armed(self) -> bool:
        return self._armed

    async def disarm(self) -> None:
        """Global disarm. Per §29.2, the dashboard / voice can disarm
        for a duration; the orchestrator schedules an `unsuppress`
        when the duration elapses. This method just flips the flag."""
        self._armed = False
        await self.suppress()
        logger.info("[CatEscape] disarmed (global)")

    async def rearm(self) -> None:
        self._armed = True
        await self.unsuppress()
        logger.info("[CatEscape] rearmed (global)")

    def suppress_cat(self, cat_name: str, seconds: float = 1200.0) -> None:
        """Per-cat suppression — 'I'm taking Sneaky out for 20 minutes.'
        Doesn't suppress the alarm globally; only this one cat is
        excluded from triggering it for the window."""
        import time as _time
        until = _time.monotonic() + max(0.0, float(seconds))
        self._cat_suppressions[cat_name.lower()] = until
        logger.info(
            f"[CatEscape] '{cat_name}' suppressed for {seconds:.0f}s"
        )

    def _is_cat_suppressed(self, cat_name: Optional[str]) -> bool:
        if not cat_name:
            return False
        import time as _time
        until = self._cat_suppressions.get(cat_name.lower())
        if until is None:
            return False
        if _time.monotonic() < until:
            return True
        # Expired — clean up
        self._cat_suppressions.pop(cat_name.lower(), None)
        return False

    # ── Bus subscription ────────────────────────────────────────────────────

    async def start(self) -> None:
        await self.bus.subscribe("vision.observation", self._on_observation_batch)
        await self.bus.subscribe("door.state", self._on_door_state)
        await self.bus.subscribe("world.entity_event", self._on_entity_event)
        logger.info(
            f"[CatEscape] watching {len(self._exits_by_room)} room(s) "
            f"with exterior_exit polygons"
            + ("; requiring matching door.state=open"
               if self._require_door_open else "")
        )

    async def _on_door_state(self, payload: dict) -> None:
        """Track {door_id, state} from the same publisher DoorOpenAlarm
        uses. Unknown/closed doors do not allow a cat escape fire when
        `_require_door_open` is enabled."""
        door_id = payload.get("door_id")
        state = payload.get("state")
        if not door_id or state not in ("open", "closed", "unknown"):
            return
        if door_id not in self._door_state:
            return
        self._door_state[str(door_id)] = str(state)

    async def _on_observation_batch(self, payload: dict) -> None:
        """Per-frame trigger check. Walks the observation list; if any
        cat-observation has its bbox center inside an exterior_exit
        polygon for this room, fire."""
        if not self._armed:
            return
        room = payload.get("room")
        if room is None or room not in self._exits_by_room:
            return
        exits = self._exits_by_room[room]
        observations = payload.get("observations") or []
        for obs in observations:
            obj_class = getattr(obs, "obj_class", None) or (
                obs.get("obj_class") if isinstance(obs, dict) else None
            )
            if obj_class != "cat":
                continue
            bbox = getattr(obs, "bbox", None) or (
                obs.get("bbox") if isinstance(obs, dict) else None
            )
            if not bbox:
                continue
            cx, cy = bbox_center(bbox)
            for ex in exits:
                if point_in_polygon(cx, cy, ex.get("polygon") or []):
                    if not self._door_allows_fire(ex):
                        continue
                    cat_name = self._cat_name_from_obs(obs)
                    if self._is_cat_suppressed(cat_name):
                        logger.info(
                            f"[CatEscape] '{cat_name}' suppressed — skip fire"
                        )
                        continue
                    await self._fire_for_cat(cat_name, ex, room)
                    return

    def _door_allows_fire(self, exit_def: dict) -> bool:
        if not self._require_door_open:
            return True
        door_id = exit_def.get("name", "exterior door")
        state = self._door_state.get(door_id, "unknown")
        if state == "open":
            return True
        logger.debug(
            f"[CatEscape] cat in exterior_exit '{door_id}' but door "
            f"state is {state}; suppressing"
        )
        return False

    async def _on_entity_event(self, payload: dict) -> None:
        """Auto-resolve listener: a cat reappearing in an interior
        room means the condition has cleared. The world model's
        REAPPEARED event is the cleanest signal — fired exactly when
        the cat transitions back into PRESENT in any indoor camera."""
        if self.fire_id is None:
            return
        if payload.get("entity_type") != "cat":
            return
        if payload.get("event_type") != "reappeared":
            return
        # Match by name — the world model emits entity_name on every
        # event payload. Fall back to "any cat" if we never captured
        # a name during the fire (unidentified cat case).
        active = (self._active_cat_name or "").lower()
        observed = (payload.get("entity_name") or "").lower()
        if active and observed and active != observed:
            return
        await self.condition_cleared({"reappeared_in": payload.get("room")})
        self._active_cat_name = None

    # ── Per-fire bookkeeping ────────────────────────────────────────────────

    async def _fire_for_cat(
        self, cat_name: Optional[str], exit_def: dict, room: str
    ) -> None:
        """Stamp the active-cat name + exit context onto self, then
        delegate to the base class's fire(). Re-entry while already
        FIRING_AUDIO is a no-op (Alarm.fire is idempotent on state)."""
        self._active_cat_name = cat_name
        await self.fire({
            "cat_name": cat_name or "unidentified cat",
            "exit_name": exit_def.get("name", "exterior door"),
            "exit_room": room,
        })

    @staticmethod
    def _cat_name_from_obs(obs: Any) -> Optional[str]:
        """Pull the named cat from an Observation (Phase 4) or fall
        back to None for an unidentified cat (Phase 1 / unknown)."""
        # Phase 4 Observation will carry person_name-equivalent for cats —
        # for now metadata.display_name is the convention used by the
        # WorldModel when an entity is named. Fall back to None.
        meta = getattr(obs, "metadata", None) or (
            obs.get("metadata") if isinstance(obs, dict) else None
        ) or {}
        return meta.get("display_name") or meta.get("entity_name")

    # ── Announcement / phone alert template ─────────────────────────────────

    def _announcement(self, context: dict) -> tuple[str, str]:
        cat = (context.get("cat_name")
               or self._active_cat_name
               or "unidentified cat")
        exit_name = context.get("exit_name", "exterior door")
        title = f"{exit_name.upper()}. {cat.upper()} IS OUTSIDE."
        body = (
            f"{cat} was observed exiting via {exit_name}. "
            "Audio will stop automatically when the cat is seen back inside."
        )
        return title, body

    async def _condition_still_true(self) -> bool:
        """For the rearm-after-mute decision. The cat is still outside
        if we never received a REAPPEARED event during the mute
        window — i.e., self.fire_id is still set and we never called
        condition_cleared. The base class flips state to RESOLVED on
        condition_cleared, so reaching here means it's still active."""
        return self.fire_id is not None
