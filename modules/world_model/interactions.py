"""
JARVIS — World Model
====================
Mission: §24.3 InteractionMonitor — temporal-correlation engine that
         detects PICKED_UP / PLACED_DOWN / HANDED_OFF events by
         watching the entity-event bus and joining INTERACTED_WITH
         (hand × object overlap, fired by WorldModel) with the same
         object's LOST_VISIBILITY (pickup) or FIRST_SEEN / REAPPEARED
         (placement) events.

         Lives in modules/world_model/ rather than vision/ because
         it's a *world-state* derivation — it consumes events the
         WorldModel emitted, then emits its own higher-level events.
         No frame access, no detector calls.

         Bounded ring buffers per kind (default 500 entries each):
            recent_interactions       — INTERACTED_WITH events
            recent_object_losses      — object LOST_VISIBILITY events
            recent_object_appearances — object FIRST_SEEN / REAPPEARED
                                        events
         Stale entries fall off the deque end naturally.

Modules: modules/world_model/interactions.py
Classes: InteractionMonitor
Spec:    new 2.md §24.3, §24.4.

#todo: HANDED_OFF detection per §24.4. Same-object INTERACTED_WITH
       events with different person_ids in quick succession → fire
       HANDED_OFF(from=A, to=B, object=...). Wire when the persona
       phrasing actually wants it ("Cole handed Anna the wallet").
#todo: Drop-without-contact: PLACED_DOWN can fire on a fresh
       FIRST_SEEN near recent hand activity even without a prior
       INTERACTED_WITH. The current impl already accepts that path
       via _check_for_placedown's same-room interaction hunt — the
       window-search just finds an interaction, not necessarily one
       that touched THIS object. Promotes false positives slightly;
       acceptable per §24.4.
"""
from __future__ import annotations

import asyncio
import uuid
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Optional

from loguru import logger

from modules.world_model.types import EventType


class InteractionMonitor:
    """Subscribes to world.entity_event, correlates the three patterns
    above into PICKED_UP / PLACED_DOWN events, persists + republishes
    them. Construct after WorldModel is started; call start() to
    subscribe."""

    def __init__(
        self,
        bus: Any,
        world: Any,                       # WorldModel — for store + entities
        config: Optional[dict] = None,
    ) -> None:
        self.bus = bus
        self.world = world
        self.cfg = config or {}
        # Ring buffers — bounded; older entries fall off naturally.
        # (ts, event_dict) tuples so we can answer "what was within
        # ±N seconds" without re-parsing ISO strings every time.
        self.recent_interactions: deque = deque(maxlen=500)
        self.recent_object_losses: deque = deque(maxlen=500)
        self.recent_object_appearances: deque = deque(maxlen=500)
        # In-flight dedup: object_ids for which a PICKED_UP / PLACED_DOWN
        # has been emitted within the last (pickup_settle + 1) seconds.
        # Without this, multiple INTERACTED_WITH events spawn concurrent
        # _check_for_pickup tasks that all see the same loss and each
        # emit a duplicate PICKED_UP.
        self._recent_pickup_obj_ids: dict[str, datetime] = {}
        self._recent_placedown_obj_ids: dict[str, datetime] = {}
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        await self.bus.subscribe("world.entity_event", self._on_event)
        logger.info(
            "[InteractionMonitor] started; pickup_settle="
            f"{self.cfg.get('pickup_settle_seconds', 3)}s, "
            f"place_window={self.cfg.get('place_window_seconds', 4)}s"
        )

    async def stop(self) -> None:
        for t in list(self._tasks):
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks.clear()

    # ── Event router ──────────────────────────────────────────────────────

    async def _on_event(self, event: dict) -> None:
        et = event.get("event_type")
        ts = self._parse_ts(event.get("ts"))
        if ts is None:
            return
        entity_type = event.get("entity_type")
        if et == EventType.INTERACTED_WITH.value:
            self.recent_interactions.append((ts, event))
            # Pickup detection runs as a deferred check so we have time
            # to see the LOST_VISIBILITY for this object.
            self._spawn(self._check_for_pickup(event, ts))
        elif et == EventType.LOST_VISIBILITY.value and entity_type == "object":
            self.recent_object_losses.append((ts, event))
        elif (
            et in (
                EventType.FIRST_SEEN.value,
                EventType.REAPPEARED.value,
            )
            and entity_type == "object"
        ):
            self.recent_object_appearances.append((ts, event))
            # Placedown detection runs synchronously — we already have
            # the recent_interactions buffer to check.
            self._spawn(self._check_for_placedown(event, ts))

    def _spawn(self, coro: Any) -> None:
        """Track the asyncio task so we can cancel them in stop()."""
        task = asyncio.create_task(coro)
        self._tasks.append(task)
        task.add_done_callback(self._tasks.remove)

    # ── Pickup correlator ─────────────────────────────────────────────────

    async def _check_for_pickup(
        self, interaction_event: dict, ts: datetime,
    ) -> None:
        """After an INTERACTED_WITH fires, wait T_pickup_seconds. If the
        object referenced in the metadata loses visibility within that
        window (we look both backwards and forwards in the loss buffer),
        emit a PICKED_UP event attributing the action to the person
        from the interaction."""
        wait_s = float(self.cfg.get("pickup_settle_seconds", 3.0))
        try:
            await asyncio.sleep(wait_s)
        except asyncio.CancelledError:
            return
        meta = interaction_event.get("metadata") or {}
        obj_id = meta.get("object_id")
        if not obj_id:
            return
        # In-flight dedup: claim the object_id BEFORE searching the loss
        # buffer so concurrent checks for the same object short-circuit.
        # Stale claims (older than wait_s+1) get pruned on each call so
        # a legitimate later pickup of the same object can still fire.
        cutoff = ts - timedelta(seconds=wait_s + 1.0)
        self._recent_pickup_obj_ids = {
            k: v for k, v in self._recent_pickup_obj_ids.items()
            if v >= cutoff
        }
        if obj_id in self._recent_pickup_obj_ids:
            return
        # Look for a LOST_VISIBILITY for that object within ±wait_s+1.
        for loss_ts, loss in list(self.recent_object_losses):
            if loss.get("entity_id") != obj_id:
                continue
            if abs((loss_ts - ts).total_seconds()) > wait_s + 1.0:
                continue
            # Claim BEFORE the await so racing checks see it.
            self._recent_pickup_obj_ids[obj_id] = ts
            payload = self._build_pickup_payload(interaction_event, loss, loss_ts)
            await self._emit(payload)
            return

    def _build_pickup_payload(
        self,
        interaction_event: dict,
        loss_event: dict,
        loss_ts: datetime,
    ) -> dict:
        meta = interaction_event.get("metadata") or {}
        return {
            "id": str(uuid.uuid4()),
            "ts": loss_ts.isoformat(),
            "entity_id": loss_event.get("entity_id"),
            "entity_name": (
                loss_event.get("entity_name")
                or meta.get("object_name")
                or "object"
            ),
            "entity_type": "object",
            "person_id": interaction_event.get("person_id"),
            "event_type": EventType.PICKED_UP.value,
            "room": interaction_event.get("room"),
            "camera": interaction_event.get("camera"),
            "bbox": loss_event.get("bbox"),
            "landmark": interaction_event.get("landmark"),
            "state": loss_event.get("state"),
            "confidence": interaction_event.get("confidence", 0.0),
            "snapshot_path": interaction_event.get("snapshot_path"),
            "related_entity_id": interaction_event.get("entity_id"),
            "metadata": {
                **meta,
                "object_id": loss_event.get("entity_id"),
                "object_name": (
                    loss_event.get("entity_name")
                    or meta.get("object_name")
                ),
                "object_lost_at": loss_ts.isoformat(),
                "source_room": interaction_event.get("room"),
                "person_name": interaction_event.get("entity_name"),
            },
        }

    # ── Placedown correlator ──────────────────────────────────────────────

    async def _check_for_placedown(
        self, appearance_event: dict, ts: datetime,
    ) -> None:
        """When an object FIRST_SEEN / REAPPEARED fires, look back in
        the recent_interactions buffer for any same-room hand activity
        within `place_window_seconds`. If found, attribute the
        placement to that person."""
        wait_s = float(self.cfg.get("place_window_seconds", 4.0))
        room = appearance_event.get("room")
        obj_id = appearance_event.get("entity_id")
        if not room or not obj_id:
            return
        # Same in-flight dedup pattern as pickup.
        dedup_cutoff = ts - timedelta(seconds=wait_s + 1.0)
        self._recent_placedown_obj_ids = {
            k: v for k, v in self._recent_placedown_obj_ids.items()
            if v >= dedup_cutoff
        }
        if obj_id in self._recent_placedown_obj_ids:
            return
        cutoff = ts - timedelta(seconds=wait_s)
        for inter_ts, inter in reversed(list(self.recent_interactions)):
            if inter_ts < cutoff:
                # Buffer is in chronological order; older entries
                # can't qualify either, so we can stop.
                break
            if inter.get("room") != room:
                continue
            self._recent_placedown_obj_ids[obj_id] = ts
            payload = self._build_placedown_payload(
                appearance_event, inter, ts,
            )
            await self._emit(payload)
            return

    def _build_placedown_payload(
        self,
        appearance_event: dict,
        interaction_event: dict,
        ts: datetime,
    ) -> dict:
        a_meta = appearance_event.get("metadata") or {}
        i_meta = interaction_event.get("metadata") or {}
        return {
            "id": str(uuid.uuid4()),
            "ts": ts.isoformat(),
            "entity_id": appearance_event.get("entity_id"),
            "entity_name": (
                appearance_event.get("entity_name")
                or a_meta.get("detected_class")
                or "object"
            ),
            "entity_type": "object",
            "person_id": interaction_event.get("person_id"),
            "event_type": EventType.PLACED_DOWN.value,
            "room": appearance_event.get("room"),
            "camera": appearance_event.get("camera"),
            "bbox": appearance_event.get("bbox"),
            "landmark": appearance_event.get("landmark"),
            "state": appearance_event.get("state"),
            "confidence": appearance_event.get("confidence", 0.0),
            "snapshot_path": appearance_event.get("snapshot_path"),
            "related_entity_id": interaction_event.get("entity_id"),
            "metadata": {
                **a_meta,
                "person_id": interaction_event.get("person_id"),
                "person_name": interaction_event.get("entity_name"),
                "dest_room": appearance_event.get("room"),
                "object_id": appearance_event.get("entity_id"),
                "object_name": (
                    appearance_event.get("entity_name")
                    or a_meta.get("detected_class")
                ),
                "hand_bbox": (i_meta.get("hand_bbox")),
            },
        }

    # ── Emit ──────────────────────────────────────────────────────────────

    async def _emit(self, payload: dict) -> None:
        """Persist via WorldStore + republish on the bus. Failures are
        non-fatal — losing one PICKED_UP event isn't worth crashing the
        loop."""
        try:
            await self.world.store.append_event(payload)
        except Exception as e:
            logger.debug(
                f"[InteractionMonitor] append_event failed for "
                f"{payload.get('event_type')}: {e}"
            )
        try:
            await self.bus.publish("world.entity_event", payload)
        except Exception as e:
            logger.debug(
                f"[InteractionMonitor] publish failed for "
                f"{payload.get('event_type')}: {e}"
            )

    @staticmethod
    def _parse_ts(raw: Any) -> Optional[datetime]:
        if isinstance(raw, datetime):
            return raw
        if isinstance(raw, str):
            try:
                return datetime.fromisoformat(raw)
            except ValueError:
                return None
        return None
