"""
JARVIS — World Model
====================
Phase 4 (§22) verification: synthetic test for the pet bootstrap +
behavioral profile builder + animal cost-function smoke test.

Runs in <2s. Uses an in-memory aiosqlite DB so we exercise the real
WorldStore (schema migration + affinity roundtrip) without hitting
data/jarvis.db.

Spec: new 2.md §22.0a / §22.4 / §22.6 / §22.7 / §22.5.
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import aiosqlite

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.world_model.cluster_builder import (
    AnimalClusterBuilder,
    apply_cluster_labels,
)
from modules.world_model.pets import (
    BehavioralProfileBuilder,
    bootstrap_pets_from_config,
    resolve_resident_ids,
)
from modules.world_model.store import WorldStore
from modules.world_model.types import EntityState, Observation, WorldEntity
from modules.world_model.world_model import WorldModel


# ── In-memory db facade ─────────────────────────────────────────────────────


class InMemoryDB:
    """Tiny duck of DatabaseManager. Backed by aiosqlite :memory:.
    Exposes the three coros WorldStore + pets bootstrap need."""

    def __init__(self) -> None:
        self.conn: Optional[aiosqlite.Connection] = None

    async def init(self) -> None:
        self.conn = await aiosqlite.connect(":memory:")
        self.conn.row_factory = aiosqlite.Row
        await self.conn.execute("PRAGMA foreign_keys=ON")
        # Minimal `persons` table — just enough for bootstrap to resolve
        # household_owner names → ids. Real DB has more columns.
        await self.conn.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                created_at TEXT
            )
        """)
        await self.conn.commit()

    async def close(self) -> None:
        if self.conn is not None:
            await self.conn.close()

    async def execute(self, sql: str, params: tuple = ()) -> int:
        assert self.conn is not None
        cursor = await self.conn.execute(sql, params)
        await self.conn.commit()
        return int(cursor.lastrowid or 0)

    async def fetchall(self, sql: str, params: tuple = ()) -> list:
        assert self.conn is not None
        cursor = await self.conn.execute(sql, params)
        return list(await cursor.fetchall())

    async def fetchone(self, sql: str, params: tuple = ()):
        assert self.conn is not None
        cursor = await self.conn.execute(sql, params)
        return await cursor.fetchone()


# ── Stubs ───────────────────────────────────────────────────────────────────


class StubBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def publish(self, topic: str, payload: dict) -> None:
        self.events.append((topic, payload))

    async def subscribe(self, topic: str, handler: Any) -> None:
        pass


class StubIdentityManager:
    async def identify_from_embedding_async(self, emb: Any, modality: str = "face"):
        return None


# ── Test config — abbreviated household ─────────────────────────────────────


HOUSEHOLD_CONFIG = {
    "residents": [
        {"id": "cole", "display_name": "Cole", "primary_room": "office"},
        {"id": "anna", "display_name": "Anna", "primary_room": "bedroom"},
        {"id": "jeff", "display_name": "Jeff", "primary_room": "jeff_room"},
    ],
    "pets": {
        "cats": [
            {
                "name": "Spooky", "household_owner": "cole",
                "color_class": "black", "coat_texture": "fluffy_curly",
                "expected_size": "medium", "home_room": "bedroom",
                "personality": "calm",
                "affinities": [
                    {"person": "cole", "strength": "high",
                     "contexts": ["sleeping"]},
                ],
            },
            {
                "name": "Velcro", "household_owner": "jeff",
                "unmonitored_home": "jeff_room",
                "color_class": "black", "coat_texture": "straight_sleek",
                "expected_size": "medium-large", "home_room": "jeff_room",
                "personality": "spirited",
                "conflicts_with": ["Summer"],
                "affinities": [
                    {"person": "jeff", "strength": "high",
                     "contexts": ["proximity_general", "sleeping"]},
                    {"person": "anna", "strength": "medium",
                     "contexts": ["physical_contact", "rubbing"]},
                ],
            },
        ],
        "dogs": [
            {
                "name": "Summer", "household_owner": "cole",
                "breed_class": "medium-longhair",
                "color_class": "cream", "expected_size": "medium",
                "home_rooms": ["bedroom", "living_room", "outdoor"],
                "personality": "smart_demanding_excitable",
                "affinities": [
                    {"person": "anna", "strength": "high",
                     "contexts": ["proximity_general"]},
                ],
            },
        ],
    },
}


ROOMS_CONFIG = [
    {"id": "office", "world_model": {"enabled": True,
        "frame_width": 640, "frame_height": 480,
        "exits": [], "landmarks": []}},
    {"id": "bedroom", "world_model": {"enabled": True,
        "frame_width": 1920, "frame_height": 1080,
        "exits": [], "landmarks": []}},
    {"id": "living_room", "world_model": {"enabled": True,
        "frame_width": 1920, "frame_height": 1080,
        "exits": [], "landmarks": []}},
]


# ── Tests ───────────────────────────────────────────────────────────────────


async def test_resolve_residents() -> None:
    db = InMemoryDB(); await db.init()
    try:
        ids = await resolve_resident_ids(db, HOUSEHOLD_CONFIG["residents"])
        assert set(ids.keys()) == {"cole", "anna", "jeff"}, ids
        # Must persist — second call returns same IDs.
        ids2 = await resolve_resident_ids(db, HOUSEHOLD_CONFIG["residents"])
        assert ids == ids2
        # And persons table actually got rows.
        rows = await db.fetchall("SELECT name FROM persons ORDER BY name")
        assert sorted(r["name"] for r in rows) == ["Anna", "Cole", "Jeff"]
    finally:
        await db.close()
    print("PASS: resolve_resident_ids")


async def test_bootstrap_idempotent_and_affinities() -> None:
    db = InMemoryDB(); await db.init()
    try:
        store = WorldStore(db)
        await store.ensure_schema()

        pets1 = await bootstrap_pets_from_config(store, HOUSEHOLD_CONFIG)
        names1 = sorted(p.display_name for p in pets1 if p.display_name)
        assert names1 == ["Spooky", "Summer", "Velcro"], names1

        # All pets should have household_owner_id resolved.
        for p in pets1:
            assert p.household_owner_id is not None, p.display_name
            assert p.is_resident is True

        # Velcro's unmonitored_home_room set; others None.
        velcro = next(p for p in pets1 if p.display_name == "Velcro")
        assert velcro.unmonitored_home_room == "jeff_room"

        # Idempotent re-run: same entity ids.
        pets2 = await bootstrap_pets_from_config(store, HOUSEHOLD_CONFIG)
        ids1 = sorted(p.id for p in pets1)
        ids2 = sorted(p.id for p in pets2)
        assert ids1 == ids2, "bootstrap re-ran created duplicate entities"

        # Affinity rows persisted + reloadable via load_entities.
        loaded = await store.load_entities()
        spooky = next(e for e in loaded if e.display_name == "Spooky")
        affs = spooky.metadata.get("affinities", [])
        assert len(affs) == 1
        assert affs[0]["strength"] == "high"
        assert affs[0]["contexts"] == ["sleeping"]

        velcro_loaded = next(e for e in loaded if e.display_name == "Velcro")
        velcro_affs = velcro_loaded.metadata.get("affinities", [])
        assert len(velcro_affs) == 2
    finally:
        await db.close()
    print("PASS: bootstrap idempotent + affinities roundtrip")


async def test_bootstrap_archives_dropped_pet() -> None:
    db = InMemoryDB(); await db.init()
    try:
        store = WorldStore(db)
        await store.ensure_schema()
        # Initial config has Spooky, Velcro, Summer.
        await bootstrap_pets_from_config(store, HOUSEHOLD_CONFIG)
        # Now drop Velcro and re-run.
        trimmed = {
            "residents": HOUSEHOLD_CONFIG["residents"],
            "pets": {
                "cats": [HOUSEHOLD_CONFIG["pets"]["cats"][0]],  # Spooky only
                "dogs": HOUSEHOLD_CONFIG["pets"]["dogs"],
            },
        }
        await bootstrap_pets_from_config(store, trimmed)
        loaded = {e.display_name: e for e in await store.load_entities()}
        # Velcro's row stays for history but archived_at is set.
        assert loaded["Velcro"].archived_at is not None
        # Spooky still active.
        assert loaded["Spooky"].archived_at is None
        # And restoring config un-archives.
        await bootstrap_pets_from_config(store, HOUSEHOLD_CONFIG)
        loaded2 = {e.display_name: e for e in await store.load_entities()}
        assert loaded2["Velcro"].archived_at is None
    finally:
        await db.close()
    print("PASS: bootstrap soft-archive on config removal")


async def test_animal_cost_color_filter() -> None:
    """Color-class mismatch must hard-reject — black observation can't
    match cream entity, etc."""
    db = InMemoryDB(); await db.init()
    try:
        store = WorldStore(db); await store.ensure_schema()
        await bootstrap_pets_from_config(store, HOUSEHOLD_CONFIG)
        wm = WorldModel(
            bus=StubBus(), store=store,
            rooms_config=ROOMS_CONFIG,
            identity_manager=StubIdentityManager(),
            config={},
        )
        await wm.start()

        spooky = wm.find_entity_by_name("Spooky")
        velcro = wm.find_entity_by_name("Velcro")
        summer = wm.find_entity_by_name("Summer")
        assert spooky and velcro and summer

        ts = datetime.utcnow()
        # Black-cat observation vs Spooky (black) → low cost.
        # Black-cat observation vs Summer (cream dog) — wrong species would
        # already fail. Use a striped cat observation vs Spooky to test
        # color filter.
        black_obs = Observation(
            camera="bedroom", room="bedroom", obj_class="cat",
            bbox=(100, 100, 200, 200), confidence=0.9, ts=ts,
            metadata={"color_class": "black", "size_normalized": 0.04},
        )
        striped_obs = Observation(
            camera="bedroom", room="bedroom", obj_class="cat",
            bbox=(100, 100, 200, 200), confidence=0.9, ts=ts,
            metadata={"color_class": "striped", "size_normalized": 0.04},
        )

        c_black_spooky = wm._animal_pair_cost(black_obs, spooky, species="cat")
        c_striped_spooky = wm._animal_pair_cost(
            striped_obs, spooky, species="cat"
        )

        # Black on Spooky should be far cheaper than striped on Spooky.
        assert c_black_spooky < c_striped_spooky, (
            f"black->Spooky={c_black_spooky}, striped->Spooky={c_striped_spooky}"
        )
        # Striped on Spooky must be the cost_reject*2 rejection.
        assert c_striped_spooky >= 1.0 * 2  # cost_reject default 1.0

        # Spooky vs Velcro on a generic black observation: with empty
        # profiles and no continuity, they should be close (the doc says
        # day-1 attribution is essentially a coin flip — that's OK as
        # long as the cost function returns finite costs without crashing).
        c_velcro = wm._animal_pair_cost(black_obs, velcro, species="cat")
        assert 0.0 <= c_black_spooky < float("inf")
        assert 0.0 <= c_velcro < float("inf")
    finally:
        await db.close()
    print("PASS: animal cost-function color filter + finite costs")


async def test_archived_pet_never_matches() -> None:
    """An archived entity is reject-2x even on a perfect-color-class match."""
    db = InMemoryDB(); await db.init()
    try:
        store = WorldStore(db); await store.ensure_schema()
        await bootstrap_pets_from_config(store, HOUSEHOLD_CONFIG)
        # Manually archive Spooky.
        loaded = await store.load_entities()
        spooky = next(e for e in loaded if e.display_name == "Spooky")
        spooky.archived_at = datetime.utcnow()
        await store.upsert_entity(spooky)

        wm = WorldModel(
            bus=StubBus(), store=store,
            rooms_config=ROOMS_CONFIG,
            identity_manager=StubIdentityManager(),
            config={"cost_reject": 1.0},
        )
        await wm.start()

        ts = datetime.utcnow()
        obs = Observation(
            camera="bedroom", room="bedroom", obj_class="cat",
            bbox=(100, 100, 200, 200), confidence=0.9, ts=ts,
            metadata={"color_class": "black", "size_normalized": 0.04},
        )
        # _pair_cost wraps the species dispatch + archive guard.
        loaded_spooky = wm.find_entity_by_name("Spooky")
        assert loaded_spooky is not None
        cost = wm._pair_cost(obs, loaded_spooky)
        assert cost >= 2.0  # cost_reject*2 = 2.0
    finally:
        await db.close()
    print("PASS: archived pet rejects unconditionally")


async def test_behavioral_profile_builder() -> None:
    """Seed events for Spooky + Velcro and confirm the profile builds
    a usable room_distribution + co_occurrence_partners."""
    db = InMemoryDB(); await db.init()
    try:
        store = WorldStore(db); await store.ensure_schema()
        await bootstrap_pets_from_config(store, HOUSEHOLD_CONFIG)
        wm = WorldModel(
            bus=StubBus(), store=store,
            rooms_config=ROOMS_CONFIG,
            identity_manager=StubIdentityManager(),
            config={},
        )
        await wm.start()

        spooky = wm.find_entity_by_name("Spooky")
        velcro = wm.find_entity_by_name("Velcro")
        assert spooky and velcro

        # Seed 30 first_seen events for Spooky in the bedroom over the
        # last 5 days, plus 5 in the living_room.
        base = datetime.utcnow() - timedelta(days=5)
        for i in range(30):
            await store.append_event({
                "id": f"sp-bd-{i}", "ts": (base + timedelta(hours=i)).isoformat(),
                "entity_id": spooky.id, "entity_name": "Spooky",
                "entity_type": "cat", "person_id": None,
                "event_type": "first_seen", "room": "bedroom",
                "camera": "bedroom", "bbox": None, "landmark": None,
                "state": "present", "confidence": 0.9,
                "metadata": {"size_normalized": 0.05, "color_class": "black"},
            })
        for i in range(5):
            await store.append_event({
                "id": f"sp-lr-{i}", "ts": (base + timedelta(days=2, hours=i)).isoformat(),
                "entity_id": spooky.id, "entity_name": "Spooky",
                "entity_type": "cat", "person_id": None,
                "event_type": "moved_to", "room": "living_room",
                "camera": "living_room", "bbox": None, "landmark": None,
                "state": "present", "confidence": 0.85,
                "metadata": {"size_normalized": 0.05, "color_class": "black"},
            })

        builder = BehavioralProfileBuilder()
        profile = await builder.rebuild_for(wm, spooky, days_back=10)

        assert profile["n_observations"] == 35
        assert profile["room_distribution"]["bedroom"] > 0.5
        # bbox_size_per_room only writes rooms with >= 5 samples; both
        # rooms qualify (30 and 5).
        assert "bedroom" in profile["bbox_size_per_room"]
        assert profile["bbox_size_per_room"]["bedroom"]["n"] == 30
    finally:
        await db.close()
    print("PASS: BehavioralProfileBuilder room_distribution + bbox_size_per_room")


async def test_cluster_builder_below_threshold() -> None:
    """Cluster builder returns {} when fewer than threshold events exist.
    Below-threshold means clustering doesn't run — but the path still
    needs to short-circuit cleanly."""
    db = InMemoryDB(); await db.init()
    try:
        store = WorldStore(db); await store.ensure_schema()
        await bootstrap_pets_from_config(store, HOUSEHOLD_CONFIG)
        # Reuse spooky's id so FK passes; treat the events as unattributed
        # by clearing entity_name (the cluster builder filters on
        # entity_name being None or starting with 'unknown_').
        spooky = (await store.load_entities())[0]  # pick any pet entity
        for i in range(5):
            await store.append_event({
                "id": f"ev-{i}",
                "ts": datetime.utcnow().isoformat(),
                "entity_id": spooky.id,
                "entity_name": f"unknown_cat_{i}",
                "entity_type": "cat", "person_id": None,
                "event_type": "first_seen", "room": "kitchen",
                "camera": "kitchen", "bbox": None, "landmark": None,
                "state": "present", "confidence": 0.8,
                "metadata": {"color_class": "black"},
            })
        cb = AnimalClusterBuilder(store, {"cluster_min_observations": 200})
        clusters = await cb.cluster_unattributed(species="cat")
        assert clusters == {}, "should return empty below threshold"
    finally:
        await db.close()
    print("PASS: AnimalClusterBuilder respects threshold")


async def main() -> None:
    await test_resolve_residents()
    await test_bootstrap_idempotent_and_affinities()
    await test_bootstrap_archives_dropped_pet()
    await test_animal_cost_color_filter()
    await test_archived_pet_never_matches()
    await test_behavioral_profile_builder()
    await test_cluster_builder_below_threshold()
    print("\nAll Phase 4 (§22) synthetic tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
