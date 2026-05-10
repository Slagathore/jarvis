"""
JARVIS — World Model
====================
Mission: Phase 4 (§22) — pets-by-name. Two responsibilities:

  1. `bootstrap_pets_from_config(...)` — idempotent first-run /
     hot-reload. Reads `pets.cats` and `pets.dogs` from config.yaml,
     creates one WorldEntity per declared pet (or refreshes seed
     metadata on an existing one), resolves household_owner names
     against the `persons` table, and syncs the per-pet affinity
     rows in `pet_affinities`. Safe to call on every boot.

  2. `BehavioralProfileBuilder` — async class. For each pet, queries
     the last 30 days of world_entity_events and writes a
     `behavioral_profile` dict back into entity.metadata. The profile
     is what makes Spooky/Velcro and Sparta/Serval cost-function
     tie-breaking work after enough data has accumulated. Runs
     nightly off the orchestrator's daily-task loop.

The bootstrap phase resolves resident names → persons.id by hitting
the existing `persons` table (created by IdentityManager). Pets are
NEVER inserted into `persons` — `household_owner_id` is FK-linked, and
the pet itself lives only in `world_entities`.

Modules: modules/world_model/pets.py
Classes: Affinity, BehavioralProfileBuilder
Funcs:   bootstrap_pets_from_config, resolve_resident_ids
Spec:    new 2.md §22.0a, §22.4, §22.6.
"""
from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np
from loguru import logger

from modules.world_model.store import WorldStore
from modules.world_model.types import EntityState, WorldEntity


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ── Affinity ────────────────────────────────────────────────────────────────


@dataclass
class Affinity:
    """One pet's preference toward one person, in one or more contexts.
    Mirrors the schema of `pet_affinities` but in-memory + typed."""
    person: str          # resident id token (e.g. 'cole')
    strength: str        # 'low' | 'medium' | 'high'
    contexts: list[str] = field(default_factory=list)
    # Filled in by bootstrap after resolving `person` → persons.id.
    person_id: Optional[int] = None


# ── Resident name → persons.id resolver ─────────────────────────────────────


async def resolve_resident_ids(
    db: Any, residents: list[dict],
) -> dict[str, int]:
    """
    Map resident-id tokens (cole/anna/jeff) → persons.id, creating any
    missing rows. Returns {'cole': 4, 'anna': 5, 'jeff': 6} (example).
    No-op for an empty residents list.
    """
    out: dict[str, int] = {}
    for r in residents or []:
        rid = str(r.get("id") or "").strip()
        name = str(r.get("display_name") or rid).strip()
        if not rid or not name:
            continue
        row = await db.fetchone(
            "SELECT id FROM persons WHERE name = ? COLLATE NOCASE", (name,)
        )
        if row is not None:
            out[rid] = int(row["id"])
            continue
        try:
            pid = await db.execute(
                "INSERT INTO persons (name, created_at) VALUES (?, ?)",
                (name, _utcnow().isoformat()),
            )
            if pid is not None:
                out[rid] = int(pid)
        except Exception as e:
            logger.warning(
                f"[pets] could not seed person '{name}' for resident '{rid}': {e}"
            )
    return out


# ── Bootstrap ───────────────────────────────────────────────────────────────


async def bootstrap_pets_from_config(
    store: WorldStore,
    config: dict,
) -> list[WorldEntity]:
    """
    Read config, materialize WorldEntity rows for every declared pet,
    sync `pet_affinities`. Idempotent: matches existing rows by
    (entity_type, display_name) and refreshes `seed` metadata + ownership
    + affinity rows so config edits propagate without restart of the DB.

    Returns the list of pet entities (cats + dogs) currently active.
    """
    residents = config.get("residents", []) or []
    resident_id_map = await resolve_resident_ids(store.db, residents)

    pets_block = config.get("pets") or {}
    cats = pets_block.get("cats") or []
    dogs = pets_block.get("dogs") or []

    existing = await store.load_entities()
    by_type_name: dict[tuple[str, str], WorldEntity] = {
        (e.entity_type, (e.display_name or "").lower()): e
        for e in existing
    }

    out: list[WorldEntity] = []
    declared_names: set[tuple[str, str]] = set()

    for cat in cats:
        ent = await _materialize_pet(
            store, cat, "cat", resident_id_map, by_type_name,
        )
        if ent is not None:
            out.append(ent)
            declared_names.add((ent.entity_type, (ent.display_name or "").lower()))
    for dog in dogs:
        ent = await _materialize_pet(
            store, dog, "dog", resident_id_map, by_type_name,
        )
        if ent is not None:
            out.append(ent)
            declared_names.add((ent.entity_type, (ent.display_name or "").lower()))

    # Soft-archive: any resident pet entity that's no longer declared in
    # config gets `archived_at` stamped — its history stays for queries
    # but it won't match new observations. New declarations clear it.
    now = _utcnow()
    for (etype, name), ent in by_type_name.items():
        if etype not in ("cat", "dog"):
            continue
        if not ent.is_resident:
            continue
        if (etype, name) in declared_names:
            continue
        if ent.archived_at is None:
            ent.archived_at = now
            await store.upsert_entity(ent)
            logger.info(
                f"[pets] archived '{ent.display_name}' ({etype}) — "
                "no longer in config"
            )

    logger.info(
        f"[pets] bootstrap complete: "
        f"{sum(1 for e in out if e.entity_type == 'cat')} cat(s), "
        f"{sum(1 for e in out if e.entity_type == 'dog')} dog(s)"
    )
    return out


async def _materialize_pet(
    store: WorldStore,
    cfg: dict,
    species: str,
    resident_ids: dict[str, int],
    existing_by_type_name: dict[tuple[str, str], WorldEntity],
) -> Optional[WorldEntity]:
    """Create or refresh one pet entity. Writes `seed` metadata,
    household_owner_id, unmonitored_home_room, and replaces affinity rows."""
    name = (cfg.get("name") or "").strip()
    if not name:
        return None

    owner_token = cfg.get("household_owner")
    owner_id = resident_ids.get(owner_token) if owner_token else None

    # Seed metadata — descriptors + behavioral hints. The cost function
    # reads from `seed` until the behavioral_profile has enough data.
    seed: dict[str, Any] = {
        "color_class": cfg.get("color_class"),
        "expected_size": cfg.get("expected_size"),
        "size_basis": cfg.get("size_basis", "static"),
        "personality": cfg.get("personality"),
    }
    if species == "cat":
        seed.update({
            "coat_length": cfg.get("coat_length"),
            "coat_texture": cfg.get("coat_texture"),
            "home_room": cfg.get("home_room"),
            "cyclic_home_rooms": cfg.get("cyclic_home_rooms"),
            "preferred_perches": cfg.get("preferred_perches", []),
            "preferred_landmarks": cfg.get("preferred_landmarks", []),
            "conflicts_with": cfg.get("conflicts_with", []),
            "hyper_alert_to": cfg.get("hyper_alert_to", []),
            "distinctive_features": cfg.get("distinctive_features", []),
            "sees_ghosts": bool(cfg.get("sees_ghosts", False)),
            "age_state": cfg.get("age_state"),
            "notes": cfg.get("notes"),
        })
    elif species == "dog":
        seed.update({
            "breed_class": cfg.get("breed_class"),
            "home_rooms": cfg.get("home_rooms", []),
            "feeding_room": cfg.get("feeding_room"),
            "anxiety_triggers": cfg.get("anxiety_triggers", []),
            "nicknames": cfg.get("nicknames", []),
        })

    # `home_room` for the entity's `last_seen_room` seed — pick the
    # first plausible room for cyclic / multi-home pets.
    home_room: Optional[str] = None
    if isinstance(cfg.get("home_rooms"), list) and cfg["home_rooms"]:
        home_room = next(
            (r for r in cfg["home_rooms"] if r != "outdoor"), cfg["home_rooms"][0]
        )
    elif isinstance(cfg.get("cyclic_home_rooms"), list) and cfg["cyclic_home_rooms"]:
        home_room = next(
            (r for r in cfg["cyclic_home_rooms"] if r != "any_closet"),
            cfg["cyclic_home_rooms"][0],
        )
    else:
        hr = cfg.get("home_room")
        if hr and hr != "cyclic":
            home_room = hr

    key = (species, name.lower())
    ent = existing_by_type_name.get(key)
    if ent is not None:
        # Refresh seed and ownership; preserve runtime state and
        # behavioral_profile that was learned from observations.
        ent.metadata.setdefault("seed", {})
        ent.metadata["seed"] = seed
        ent.metadata.setdefault("behavioral_profile", {})
        ent.household_owner_id = owner_id
        ent.unmonitored_home_room = cfg.get("unmonitored_home")
        ent.is_resident = True
        if cfg.get("archived"):
            ent.archived_at = ent.archived_at or _utcnow()
        else:
            ent.archived_at = None
    else:
        ent = WorldEntity(
            id=str(uuid.uuid4()),
            entity_type=species,
            person_id=None,
            display_name=name,
            state=EntityState.IN_ROOM_UNSEEN,
            last_seen_room=home_room,
            last_state_change_ts=_utcnow(),
            is_resident=True,
            household_owner_id=owner_id,
            unmonitored_home_room=cfg.get("unmonitored_home"),
            metadata={
                "seed": seed,
                "behavioral_profile": {},
            },
        )
        if cfg.get("archived"):
            ent.archived_at = _utcnow()

    await store.upsert_entity(ent)

    # Sync affinities: parse config block, resolve to person_ids, replace
    # the join-table rows. Affinities with unknown resident tokens are
    # dropped with a warning (config edits ahead of resident declarations
    # are caller error — typed validation will eventually enforce this).
    raw_affs = cfg.get("affinities") or []
    rows: list[dict] = []
    for aff in raw_affs:
        person_token = aff.get("person")
        pid = resident_ids.get(person_token)
        if pid is None:
            logger.warning(
                f"[pets] '{name}' affinity references unknown resident "
                f"'{person_token}'; skipping"
            )
            continue
        rows.append({
            "person_id": pid,
            "strength": aff.get("strength", "medium"),
            "contexts": list(aff.get("contexts", [])),
        })
    await store.replace_affinities(ent.id, rows)
    # Reflect in metadata for hot-path reads (matches load_entities() shape).
    ent.metadata["affinities"] = rows

    return ent


# ── BehavioralProfileBuilder ────────────────────────────────────────────────


class BehavioralProfileBuilder:
    """
    Nightly profile rebuild for resident pets. For each pet entity the
    orchestrator hands us, query the last `days_back` days of events
    and compute the six profile components from §22.6.

    Stateless — one instance can rebuild any pet. The orchestrator loop
    handles scheduling.
    """

    DEFAULT_WINDOW_DAYS = 30
    MIN_BBOX_SAMPLES_PER_ROOM = 5

    async def rebuild_for(
        self, world: Any, ent: WorldEntity, days_back: int = DEFAULT_WINDOW_DAYS,
    ) -> dict:
        """
        `world` only needs `.store` (for search_events) and `.entities`
        (for cat-vs-cat co_occurrence lookups). The full WorldModel is
        the natural caller; tests can pass a duck-typed namespace.

        Returns the freshly built profile dict and writes it to
        `ent.metadata['behavioral_profile']` + persists via upsert.
        """
        since = _utcnow() - timedelta(days=days_back)
        events = await world.store.search_events(
            entity_id=ent.id, since=since, limit=50000,
        )
        if not events:
            return ent.metadata.get("behavioral_profile") or {}

        profile = {
            "room_distribution": self._room_distribution(events),
            "room_distribution_by_hour": self._room_distribution_by_hour(events),
            "bbox_size_per_room": self._bbox_size_per_room(events),
            "stationary_fraction": self._stationary_fraction(events),
            "human_avoidance_score": await self._human_avoidance(world, ent, since),
            "co_occurrence_partners": await self._co_occurrence(world, ent, since),
            "n_observations": len(events),
            "window_start": since.isoformat(),
            "window_end": _utcnow().isoformat(),
        }
        ent.metadata["behavioral_profile"] = profile
        await world.store.upsert_entity(ent)
        return profile

    # ── Component calculators ──────────────────────────────────────────────

    @staticmethod
    def _room_distribution(events: list[dict]) -> dict[str, float]:
        counts: dict[str, int] = defaultdict(int)
        for e in events:
            r = e.get("room")
            if r:
                counts[r] += 1
        total = sum(counts.values()) or 1
        return {r: c / total for r, c in counts.items()}

    @staticmethod
    def _room_distribution_by_hour(
        events: list[dict],
    ) -> dict[int, dict[str, float]]:
        per_hour: dict[int, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for e in events:
            r = e.get("room")
            ts_raw = e.get("ts")
            if not r or not ts_raw:
                continue
            try:
                h = datetime.fromisoformat(ts_raw).hour
            except (TypeError, ValueError):
                continue
            per_hour[h][r] += 1
        out: dict[int, dict[str, float]] = {}
        for h, counts in per_hour.items():
            total = sum(counts.values()) or 1
            out[h] = {r: c / total for r, c in counts.items()}
        return out

    def _bbox_size_per_room(
        self, events: list[dict],
    ) -> dict[str, dict[str, float]]:
        per_room: dict[str, list[float]] = defaultdict(list)
        for e in events:
            r = e.get("room")
            meta = _decode_metadata(e.get("metadata"))
            sz = meta.get("size_normalized")
            if r and sz is not None:
                try:
                    per_room[r].append(float(sz))
                except (TypeError, ValueError):
                    continue
        return {
            r: {
                "mean": float(np.mean(sizes)),
                "std": float(np.std(sizes)),
                "n": len(sizes),
            }
            for r, sizes in per_room.items()
            if len(sizes) >= self.MIN_BBOX_SAMPLES_PER_ROOM
        }

    @staticmethod
    def _stationary_fraction(events: list[dict]) -> float:
        movements = sum(
            1 for e in events if e.get("event_type") == "moved_within_room"
        )
        appearances = sum(
            1 for e in events
            if e.get("event_type") in ("first_seen", "reappeared", "moved_to")
        )
        if appearances == 0:
            return 0.5
        ratio = movements / max(appearances, 1)
        return float(max(0.0, min(1.0, 1.0 - ratio / 5.0)))

    async def _human_avoidance(
        self, world: Any, ent: WorldEntity, since: datetime,
    ) -> float:
        """For each pet PRESENT-event, check whether a person was also in
        the same room within ±60s. Avoidance = 1 - cohabitation rate."""
        appear_types = ["reappeared", "moved_to", "first_seen"]
        pet_events = await world.store.search_events(
            entity_id=ent.id, event_types=appear_types,
            since=since, limit=10000,
        )
        if not pet_events:
            return 0.5
        cohab = 0
        for ce in pet_events:
            ts = _parse_event_ts(ce)
            if ts is None:
                continue
            window = await world.store.search_events(
                room=ce.get("room"),
                event_types=appear_types,
                since=ts - timedelta(seconds=60),
                until=ts + timedelta(seconds=60),
                limit=50,
            )
            if any(w.get("entity_type") == "person" for w in window):
                cohab += 1
        return float(1.0 - (cohab / max(len(pet_events), 1)))

    async def _co_occurrence(
        self, world: Any, ent: WorldEntity, since: datetime,
    ) -> dict[str, float]:
        """Same window logic — find which other pets show up alongside us."""
        appear_types = ["reappeared", "moved_to", "first_seen"]
        pet_events = await world.store.search_events(
            entity_id=ent.id, event_types=appear_types,
            since=since, limit=10000,
        )
        if not pet_events:
            return {}
        partner_counts: dict[str, int] = defaultdict(int)
        for ce in pet_events:
            ts = _parse_event_ts(ce)
            if ts is None:
                continue
            window = await world.store.search_events(
                room=ce.get("room"),
                event_types=appear_types,
                since=ts - timedelta(seconds=60),
                until=ts + timedelta(seconds=60),
                limit=50,
            )
            for w in window:
                if (w.get("entity_type") in ("cat", "dog")
                        and w.get("entity_id") != ent.id
                        and w.get("entity_name")):
                    partner_counts[w["entity_name"]] += 1
        n = max(len(pet_events), 1)
        return {name: c / n for name, c in partner_counts.items()}


# ── helpers ─────────────────────────────────────────────────────────────────


def _decode_metadata(raw: Any) -> dict:
    """Events store metadata as JSON; handle either decoded dict or string."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        import json
        try:
            return json.loads(raw)
        except Exception:
            return {}
    return {}


def _parse_event_ts(event: dict) -> Optional[datetime]:
    raw = event.get("ts")
    if isinstance(raw, datetime):
        ts = raw
    elif isinstance(raw, str):
        try:
            ts = datetime.fromisoformat(raw)
        except ValueError:
            return None
    else:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)
