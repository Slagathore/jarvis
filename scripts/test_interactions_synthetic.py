"""
JARVIS — World Model
====================
Phase 5 §24 verification: synthetic test for the interaction
correlator. Drives the InteractionMonitor with hand-crafted
INTERACTED_WITH × LOST_VISIBILITY × FIRST_SEEN events and asserts
that PICKED_UP / PLACED_DOWN events get emitted with the right
attribution.

Spec: new 2.md §24.3, §24.5.
Runs in <2s, no real cameras, no MediaPipe.
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.world_model.interactions import InteractionMonitor
from modules.world_model.types import EventType


# ── Stubs ───────────────────────────────────────────────────────────────────


class StubBus:
    """Drives the monitor synchronously — handlers are invoked the same
    way as the real EventBus."""

    def __init__(self) -> None:
        self.handlers: dict[str, list] = {}
        self.published: list[tuple[str, dict]] = []

    async def subscribe(self, topic: str, handler: Any) -> None:
        self.handlers.setdefault(topic, []).append(handler)

    async def publish(self, topic: str, payload: dict) -> None:
        self.published.append((topic, payload))
        for h in self.handlers.get(topic, []):
            await h(payload)


class StubStore:
    """Append-only in-memory event log."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    async def append_event(self, payload: dict) -> None:
        self.events.append(dict(payload))

    async def search_events(self, **kw) -> list[dict]:
        # Filter using the same shape the real WorldStore exposes.
        events = list(self.events)
        if "entity_id" in kw and kw["entity_id"]:
            events = [e for e in events if e.get("entity_id") == kw["entity_id"]]
        if "event_types" in kw and kw["event_types"]:
            events = [e for e in events if e.get("event_type") in kw["event_types"]]
        if "person_id" in kw and kw["person_id"] is not None:
            events = [
                e for e in events
                if e.get("person_id") == kw["person_id"]
            ]
        events.sort(key=lambda e: e.get("ts", ""), reverse=True)
        limit = kw.get("limit") or len(events)
        return events[:limit]


class StubWorld:
    """Minimal duck for InteractionMonitor.world."""

    def __init__(self) -> None:
        self.store = StubStore()


# ── Helpers ─────────────────────────────────────────────────────────────────


def _interacted_with(
    *, ts: datetime, person_name: str, person_id: int, room: str,
    object_id: str, object_name: str, hand_bbox=(100, 100, 200, 200),
) -> dict:
    return {
        "id": f"itx-{ts.isoformat()}",
        "ts": ts.isoformat(),
        "entity_id": f"person-{person_id}",
        "entity_name": person_name,
        "entity_type": "person",
        "person_id": person_id,
        "event_type": EventType.INTERACTED_WITH.value,
        "room": room,
        "camera": room,
        "bbox": [50, 50, 250, 350],
        "metadata": {
            "object_id": object_id,
            "object_name": object_name,
            "hand_bbox": list(hand_bbox),
        },
    }


def _object_lost(
    *, ts: datetime, object_id: str, object_name: str, room: str,
) -> dict:
    return {
        "id": f"lost-{ts.isoformat()}",
        "ts": ts.isoformat(),
        "entity_id": object_id,
        "entity_name": object_name,
        "entity_type": "object",
        "person_id": None,
        "event_type": EventType.LOST_VISIBILITY.value,
        "room": room,
        "camera": room,
        "bbox": [110, 110, 150, 140],
        "metadata": {"reason": "in_frame_disappearance"},
    }


def _object_first_seen(
    *, ts: datetime, object_id: str, object_name: str, room: str,
) -> dict:
    return {
        "id": f"first-{ts.isoformat()}",
        "ts": ts.isoformat(),
        "entity_id": object_id,
        "entity_name": object_name,
        "entity_type": "object",
        "person_id": None,
        "event_type": EventType.FIRST_SEEN.value,
        "room": room,
        "camera": room,
        "bbox": [400, 400, 440, 440],
        "metadata": {"detected_class": "cell phone"},
    }


# ── Tests ───────────────────────────────────────────────────────────────────


async def test_pickup_event_emitted_after_loss() -> None:
    """INTERACTED_WITH on a phone in office, then phone LOST_VISIBILITY
    within wait_s → PICKED_UP event with metadata.source_room=office,
    person_name=Cole."""
    bus = StubBus()
    world = StubWorld()
    monitor = InteractionMonitor(
        bus=bus, world=world,
        config={"pickup_settle_seconds": 0.05},
    )
    await monitor.start()

    t0 = datetime.now(timezone.utc)
    obj_id = "phone-1"
    await bus.publish(
        "world.entity_event",
        _interacted_with(
            ts=t0, person_name="Cole", person_id=42, room="office",
            object_id=obj_id, object_name="cell phone",
        ),
    )
    # Object lost shortly after — within the pickup window.
    await bus.publish(
        "world.entity_event",
        _object_lost(
            ts=t0 + timedelta(seconds=0.02),
            object_id=obj_id, object_name="cell phone", room="office",
        ),
    )
    # Wait for the deferred pickup check to settle (50ms wait).
    await asyncio.sleep(0.15)
    await monitor.stop()

    pickups = [
        e for e in world.store.events
        if e.get("event_type") == EventType.PICKED_UP.value
    ]
    assert len(pickups) == 1, (
        f"expected 1 pickup, got {len(pickups)}: {pickups}"
    )
    p = pickups[0]
    assert p.get("entity_id") == obj_id, p
    assert p["metadata"]["person_name"] == "Cole"
    assert p["metadata"]["source_room"] == "office"
    assert p["metadata"]["object_lost_at"] is not None
    print("PASS: PICKED_UP event emitted with correct attribution")


async def test_placedown_event_emitted_after_appearance() -> None:
    """INTERACTED_WITH in kitchen (Cole moves around), then a phone
    FIRST_SEEN in kitchen within place_window_seconds → PLACED_DOWN
    event with metadata.dest_room=kitchen, person_name=Cole."""
    bus = StubBus()
    world = StubWorld()
    monitor = InteractionMonitor(
        bus=bus, world=world,
        config={
            "pickup_settle_seconds": 0.05,
            "place_window_seconds": 5.0,
        },
    )
    await monitor.start()

    t0 = datetime.now(timezone.utc)
    obj_id = "phone-2"
    # 1. Cole's hand visible in kitchen (no object yet — synthetic
    #    interaction event without a corresponding loss).
    await bus.publish(
        "world.entity_event",
        _interacted_with(
            ts=t0, person_name="Cole", person_id=42, room="kitchen",
            object_id="some-other-object", object_name="cup",
        ),
    )
    # 2. Object FIRST_SEEN in kitchen 1.5s later.
    await bus.publish(
        "world.entity_event",
        _object_first_seen(
            ts=t0 + timedelta(seconds=1.5),
            object_id=obj_id, object_name="cell phone", room="kitchen",
        ),
    )
    # placedown is sync (no asyncio.sleep), but spawn() defers the
    # task — let it run.
    await asyncio.sleep(0.05)
    await monitor.stop()

    placedowns = [
        e for e in world.store.events
        if e.get("event_type") == EventType.PLACED_DOWN.value
    ]
    assert len(placedowns) == 1, (
        f"expected 1 placedown, got {len(placedowns)}: {placedowns}"
    )
    p = placedowns[0]
    assert p["metadata"]["person_name"] == "Cole"
    assert p["metadata"]["dest_room"] == "kitchen"
    assert p["metadata"]["object_id"] == obj_id
    print("PASS: PLACED_DOWN event emitted with correct attribution")


async def test_no_pickup_when_object_loss_in_different_room() -> None:
    """Pickup must NOT fire if the LOST_VISIBILITY is for a different
    object than the one in the interaction event."""
    bus = StubBus()
    world = StubWorld()
    monitor = InteractionMonitor(
        bus=bus, world=world,
        config={"pickup_settle_seconds": 0.05},
    )
    await monitor.start()

    t0 = datetime.now(timezone.utc)
    await bus.publish(
        "world.entity_event",
        _interacted_with(
            ts=t0, person_name="Cole", person_id=42, room="office",
            object_id="phone-A", object_name="cell phone",
        ),
    )
    await bus.publish(
        "world.entity_event",
        _object_lost(
            ts=t0 + timedelta(seconds=0.02),
            object_id="cup-B", object_name="cup", room="office",
        ),
    )
    await asyncio.sleep(0.15)
    await monitor.stop()

    pickups = [
        e for e in world.store.events
        if e.get("event_type") == EventType.PICKED_UP.value
    ]
    assert len(pickups) == 0, (
        f"unexpected pickup events: {pickups}"
    )
    print("PASS: no PICKED_UP when interaction object != lost object")


async def test_pickup_dedup() -> None:
    """Multiple INTERACTED_WITH events for the same object + one loss
    must result in exactly one PICKED_UP event (dedup by object_id)."""
    bus = StubBus()
    world = StubWorld()
    monitor = InteractionMonitor(
        bus=bus, world=world,
        config={"pickup_settle_seconds": 0.05},
    )
    await monitor.start()

    t0 = datetime.now(timezone.utc)
    obj_id = "wallet-1"
    for i in range(3):
        await bus.publish(
            "world.entity_event",
            _interacted_with(
                ts=t0 + timedelta(seconds=i * 0.01),
                person_name="Cole", person_id=42, room="office",
                object_id=obj_id, object_name="wallet",
            ),
        )
    await bus.publish(
        "world.entity_event",
        _object_lost(
            ts=t0 + timedelta(seconds=0.05),
            object_id=obj_id, object_name="wallet", room="office",
        ),
    )
    await asyncio.sleep(0.2)
    await monitor.stop()

    pickups = [
        e for e in world.store.events
        if e.get("event_type") == EventType.PICKED_UP.value
    ]
    assert len(pickups) == 1, (
        f"expected exactly 1 pickup after dedup, got {len(pickups)}"
    )
    print("PASS: pickup dedup — multiple interactions yield one PICKED_UP")


async def test_handoff_event_emitted_for_different_persons() -> None:
    """§24.4 — INTERACTED_WITH for the same object by person A then
    person B within the handoff window emits HANDED_OFF(from=A, to=B,
    object). entity_id is the object so search_events(entity_id=wallet)
    returns the row; metadata captures both names + rooms."""
    bus = StubBus()
    world = StubWorld()
    monitor = InteractionMonitor(
        bus=bus, world=world,
        config={
            "pickup_settle_seconds": 0.05,
            "place_window_seconds": 4.0,
            "handoff_window_seconds": 5.0,
        },
    )
    await monitor.start()

    t0 = datetime.now(timezone.utc)
    obj_id = "wallet-handoff"
    # Cole touches it first.
    await bus.publish(
        "world.entity_event",
        _interacted_with(
            ts=t0, person_name="Cole", person_id=42, room="office",
            object_id=obj_id, object_name="wallet",
        ),
    )
    # Anna touches the same object 1s later → handoff.
    await bus.publish(
        "world.entity_event",
        _interacted_with(
            ts=t0 + timedelta(seconds=1.0),
            person_name="Anna", person_id=43, room="office",
            object_id=obj_id, object_name="wallet",
        ),
    )
    await asyncio.sleep(0.15)
    await monitor.stop()

    handoffs = [
        e for e in world.store.events
        if e.get("event_type") == EventType.HANDED_OFF.value
    ]
    assert len(handoffs) == 1, (
        f"expected 1 handoff, got {len(handoffs)}: {handoffs}"
    )
    h = handoffs[0]
    assert h.get("entity_id") == obj_id, h
    assert h["metadata"]["from_person_name"] == "Cole"
    assert h["metadata"]["to_person_name"] == "Anna"
    assert h["metadata"]["from_person_id"] == 42
    assert h["metadata"]["to_person_id"] == 43
    print("PASS: HANDED_OFF event emitted with from + to attribution")


async def test_no_handoff_when_same_person_re_touches() -> None:
    """If person A touches an object then touches it AGAIN (no other
    person between), no HANDED_OFF fires — that's just a re-grasp,
    not a transfer."""
    bus = StubBus()
    world = StubWorld()
    monitor = InteractionMonitor(
        bus=bus, world=world,
        config={
            "pickup_settle_seconds": 0.05,
            "handoff_window_seconds": 5.0,
        },
    )
    await monitor.start()

    t0 = datetime.now(timezone.utc)
    obj_id = "phone-self"
    for i in range(3):
        await bus.publish(
            "world.entity_event",
            _interacted_with(
                ts=t0 + timedelta(seconds=i * 0.4),
                person_name="Cole", person_id=42, room="office",
                object_id=obj_id, object_name="phone",
            ),
        )
    await asyncio.sleep(0.15)
    await monitor.stop()

    handoffs = [
        e for e in world.store.events
        if e.get("event_type") == EventType.HANDED_OFF.value
    ]
    assert handoffs == [], (
        f"unexpected handoff event for same-person re-touch: {handoffs}"
    )
    print("PASS: no HANDED_OFF when same person re-touches")


async def test_handoff_dedup() -> None:
    """A flicker of multiple INTERACTED_WITH events from person B for
    the same object after person A's grip should produce exactly ONE
    HANDED_OFF (deduped on (object_id, from, to) within the window)."""
    bus = StubBus()
    world = StubWorld()
    monitor = InteractionMonitor(
        bus=bus, world=world,
        config={
            "pickup_settle_seconds": 0.05,
            "handoff_window_seconds": 5.0,
        },
    )
    await monitor.start()

    t0 = datetime.now(timezone.utc)
    obj_id = "wallet-flicker"
    # Cole touch.
    await bus.publish(
        "world.entity_event",
        _interacted_with(
            ts=t0, person_name="Cole", person_id=42, room="office",
            object_id=obj_id, object_name="wallet",
        ),
    )
    # Anna touch repeatedly — three frames in quick succession.
    for i in range(3):
        await bus.publish(
            "world.entity_event",
            _interacted_with(
                ts=t0 + timedelta(seconds=0.5 + i * 0.05),
                person_name="Anna", person_id=43, room="office",
                object_id=obj_id, object_name="wallet",
            ),
        )
    await asyncio.sleep(0.15)
    await monitor.stop()

    handoffs = [
        e for e in world.store.events
        if e.get("event_type") == EventType.HANDED_OFF.value
    ]
    assert len(handoffs) == 1, (
        f"expected 1 handoff after dedup, got {len(handoffs)}"
    )
    print("PASS: HANDED_OFF dedup — multi-frame flicker yields one event")


async def main() -> None:
    await test_pickup_event_emitted_after_loss()
    await test_placedown_event_emitted_after_appearance()
    await test_no_pickup_when_object_loss_in_different_room()
    await test_pickup_dedup()
    await test_handoff_event_emitted_for_different_persons()
    await test_no_handoff_when_same_person_re_touches()
    await test_handoff_dedup()
    print("\nAll §24 interaction tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
