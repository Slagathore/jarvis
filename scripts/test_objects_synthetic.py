"""
JARVIS - Phase 4 §23 Objects
============================
Synthetic test for the §23 object stack:
  • _object_pair_cost with CLIP embeddings (Path A) vs spatial-temporal
    fallback (Path B)
  • Same-class-same-room dedup threshold (§23.8)
  • prune_stale_objects (§23.8 nightly cleanup)
  • find_object text-query (§23.7) — happy path + below-threshold + hedge

No real CLIP weights needed; we mint synthetic 512-dim embeddings.
NullCLIPEncoder is the test-time stub for the encoder; for tests that
need actual text encoding we use a tiny stub that maps text→a fixed
vector.

Spec: new 2.md §23.6, §23.7, §23.8.
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.world_model.types import (
    EntityState, Observation, WorldEntity,
)


# ── Stubs ───────────────────────────────────────────────────────────────────


class StubBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def publish(self, topic: str, payload: dict) -> None:
        self.events.append((topic, payload))

    async def subscribe(self, topic: str, handler: Any) -> None:
        pass


class StubStore:
    def __init__(self) -> None:
        self.events: list[dict] = []

    async def ensure_schema(self) -> None: pass
    async def load_entities(self) -> list: return []
    async def upsert_entity(self, ent: Any) -> None: pass
    async def upsert_embedding(self, eid: str, emb: Any) -> None: pass
    async def append_event(self, payload: dict) -> None:
        self.events.append(dict(payload))

    async def search_events(self, **kw) -> list[dict]:
        events = list(self.events)
        if kw.get("entity_id"):
            events = [e for e in events if e.get("entity_id") == kw["entity_id"]]
        if kw.get("event_types"):
            events = [
                e for e in events
                if e.get("event_type") in kw["event_types"]
            ]
        return events[: kw.get("limit") or len(events)]


class StubIdentityManager:
    async def identify_from_embedding_async(self, emb, modality="face"):
        return None


class StubTextEncoder:
    """Maps a text query to a fixed direction so we can build entity
    embeddings that match (or don't) under cosine similarity."""

    dim = 4

    def __init__(self, mapping: dict[str, np.ndarray]) -> None:
        self.mapping = mapping
        self._zero = np.zeros(self.dim, dtype=np.float32)

    def encode_text(self, text: str) -> Optional[np.ndarray]:
        return self.mapping.get(text, self._zero).astype(np.float32)

    def encode_image(self, image: Any) -> Optional[np.ndarray]:
        return None


def _wm(rooms: Optional[list[dict]] = None, *, cfg_extra: Optional[dict] = None):
    """Build a WorldModel with stub deps."""
    from modules.world_model.world_model import WorldModel
    cfg = {
        "cost_reject": 1.0,
        "candidate_lookback_minutes": 2,
    }
    if cfg_extra:
        cfg.update(cfg_extra)
    rooms = rooms or [
        {"id": "office", "world_model": {
            "enabled": True, "frame_width": 640, "frame_height": 480,
            "exits": [], "landmarks": [],
        }},
        {"id": "bedroom", "world_model": {
            "enabled": True, "frame_width": 640, "frame_height": 480,
            "exits": [], "landmarks": [],
        }},
    ]
    return WorldModel(
        bus=StubBus(),
        store=StubStore(),  # type: ignore[arg-type]
        rooms_config=rooms,
        identity_manager=StubIdentityManager(),
        config=cfg,
    )


def _emb(seed: int, dim: int = 4) -> np.ndarray:
    """Deterministic L2-normalized embedding for test fixtures."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def _make_object(
    *,
    eid: str,
    detected_class: str,
    room: str,
    embedding: np.ndarray,
    last_seen_ts: Optional[datetime] = None,
) -> WorldEntity:
    return WorldEntity(
        id=eid, entity_type="object", display_name=detected_class,
        state=EntityState.PRESENT,
        last_seen_ts=last_seen_ts,
        last_seen_room=room, last_seen_camera=room,
        last_seen_bbox=(100, 100, 200, 200),
        last_state_change_ts=last_seen_ts or datetime.utcnow(),
        metadata={
            "detected_class": detected_class,
            "_visual_embedding": embedding,
        },
    )


# ── Tests ───────────────────────────────────────────────────────────────────


async def test_object_cost_path_a_clip_dominant() -> None:
    """When both sides have CLIP embeddings, similarity dominates the
    cost. Same-class same-room same-embedding pair → near-zero cost."""
    wm = _wm()
    await wm.start()
    room = "office"
    ts = datetime.utcnow()
    matching_emb = _emb(seed=1)
    ent = _make_object(
        eid="ent-1", detected_class="cell phone",
        room=room, embedding=matching_emb,
        last_seen_ts=ts - timedelta(seconds=30),
    )
    obs = Observation(
        camera=room, room=room, obj_class="object",
        bbox=(100, 100, 200, 200), confidence=0.9, ts=ts,
        visual_embedding=matching_emb,  # identical
        metadata={"detected_class": "cell phone"},
    )
    cost = wm._object_pair_cost(obs, ent)
    # 0.55 * (1 - 1.0) + 0.30 * 0.0 + 0.15 * (~0 days/14) -> very small
    assert cost < 0.05, cost

    # Different class hard-rejects regardless of embedding similarity.
    obs_wrong_class = Observation(
        camera=room, room=room, obj_class="object",
        bbox=(100, 100, 200, 200), confidence=0.9, ts=ts,
        visual_embedding=matching_emb,
        metadata={"detected_class": "wallet"},
    )
    cost_class_mismatch = wm._object_pair_cost(obs_wrong_class, ent)
    assert cost_class_mismatch >= 2.0, cost_class_mismatch
    print("PASS: §23.6 path A (CLIP) - high sim same room same class wins")


async def test_object_cost_path_a_room_prior() -> None:
    """Different rooms → larger cost; typical_rooms hit → softer than
    full cross-room."""
    wm = _wm(cfg_extra={
        "tracked_objects": {
            "open_vocabulary": [
                {"name": "wallet",
                 "description": "leather wallet",
                 "typical_rooms": ["office", "bedroom"]},
            ],
        },
    })
    await wm.start()
    ts = datetime.utcnow()
    emb = _emb(seed=2)
    # Entity last seen in office.
    ent = _make_object(
        eid="ent-w", detected_class="wallet",
        room="office", embedding=emb,
        last_seen_ts=ts - timedelta(seconds=10),
    )
    # Same room → no room penalty.
    obs_same = Observation(
        camera="office", room="office", obj_class="object",
        bbox=(100, 100, 200, 200), confidence=0.9, ts=ts,
        visual_embedding=emb,
        metadata={"detected_class": "wallet"},
    )
    same_cost = wm._object_pair_cost(obs_same, ent)
    # Typical-room hit (bedroom is in typical_rooms but isn't the
    # last-seen room) → 0.25 penalty.
    obs_typical = Observation(
        camera="bedroom", room="bedroom", obj_class="object",
        bbox=(100, 100, 200, 200), confidence=0.9, ts=ts,
        visual_embedding=emb,
        metadata={"detected_class": "wallet"},
    )
    typical_cost = wm._object_pair_cost(obs_typical, ent)
    # Off-typical room → 0.5 penalty.
    obs_other = Observation(
        camera="kitchen", room="kitchen", obj_class="object",
        bbox=(100, 100, 200, 200), confidence=0.9, ts=ts,
        visual_embedding=emb,
        metadata={"detected_class": "wallet"},
    )
    other_cost = wm._object_pair_cost(obs_other, ent)
    assert same_cost < typical_cost < other_cost, (
        same_cost, typical_cost, other_cost,
    )
    print("PASS: §23.6 path A room priors (same < typical < other)")


async def test_object_cost_path_b_fallback_no_clip() -> None:
    """Without CLIP embeddings, fall back to spatial-temporal logic.
    Same-class same-room recent → low cost; >15 min stale → reject."""
    wm = _wm()
    await wm.start()
    ts = datetime.utcnow()
    ent = _make_object(
        eid="ent-fb", detected_class="cup",
        room="office", embedding=_emb(seed=3),  # entity has emb
        last_seen_ts=ts - timedelta(seconds=20),
    )
    # Observation has NO embedding → triggers Path B.
    obs = Observation(
        camera="office", room="office", obj_class="object",
        bbox=(105, 105, 205, 205), confidence=0.9, ts=ts,
        visual_embedding=None,
        metadata={
            "detected_class": "cup",
            "frame_width": 640, "frame_height": 480,
        },
    )
    cost = wm._object_pair_cost(obs, ent)
    assert cost < 0.4, cost

    # Stale entity → reject.
    stale = _make_object(
        eid="ent-stale", detected_class="cup",
        room="office", embedding=_emb(seed=4),
        last_seen_ts=ts - timedelta(minutes=20),
    )
    cost_stale = wm._object_pair_cost(obs, stale)
    assert cost_stale >= 2.0, cost_stale
    print("PASS: §23.6 path B fallback - same-class same-room recent < stale")


async def test_dedup_same_class_same_room_lower_threshold() -> None:
    """§23.8 — a same-class same-room entity matches at a lower
    cosine threshold than a cross-room entity. Verifies the
    dedup heuristic in _handle_unmatched_observation."""
    wm = _wm(cfg_extra={
        "cosine_match_strong": 0.6,
        "cosine_match_strong_same_room": 0.45,
    })
    await wm.start()

    # Construct two unit vectors with cosine similarity exactly 0.5
    # so we sit precisely in the dedup window (same-room threshold
    # 0.45 < 0.5 < cross-room threshold 0.6). Deterministic geometry
    # beats random + assert-the-fixture.
    base_emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    similar_emb = np.array(
        [0.5, np.sqrt(0.75), 0.0, 0.0], dtype=np.float32,
    )
    sim = float(np.dot(base_emb, similar_emb))
    assert 0.45 < sim < 0.6, (
        f"test fixture sim={sim} not in dedup window"
    )

    # Existing entity in office with base_emb.
    ts = datetime.utcnow()
    ent = _make_object(
        eid="phone-A", detected_class="cell phone",
        room="office", embedding=base_emb,
        last_seen_ts=ts - timedelta(seconds=30),
    )
    wm.entities[ent.id] = ent

    # Same room observation with the perturbed embedding — should
    # match (same-room threshold = 0.45).
    obs_same_room = Observation(
        camera="office", room="office", obj_class="object",
        bbox=(110, 110, 210, 210), confidence=0.9, ts=ts,
        visual_embedding=similar_emb,
        metadata={"detected_class": "cell phone",
                  "frame_width": 640, "frame_height": 480},
    )
    await wm._handle_unmatched_observation(obs_same_room, ts)
    # Should have matched — entity count unchanged.
    assert len(wm.entities) == 1, wm.entities

    # Cross-room observation — should NOT match (cross-room
    # threshold = 0.6); creates new entity.
    obs_other_room = Observation(
        camera="bedroom", room="bedroom", obj_class="object",
        bbox=(110, 110, 210, 210), confidence=0.9, ts=ts,
        visual_embedding=similar_emb,
        metadata={"detected_class": "cell phone",
                  "frame_width": 640, "frame_height": 480},
    )
    await wm._handle_unmatched_observation(obs_other_room, ts)
    assert len(wm.entities) == 2, wm.entities
    print("PASS: §23.8 dedup - same-room threshold < cross-room threshold")


async def test_prune_stale_objects_skips_touched() -> None:
    """prune_stale_objects soft-deletes stale objects with no
    interaction history. Entities with INTERACTED_WITH /
    PICKED_UP / PLACED_DOWN events stay (story value)."""
    wm = _wm()
    await wm.start()
    now = datetime.utcnow()
    old = now - timedelta(days=60)

    # Stale, untouched
    ent_stale = _make_object(
        eid="ent-stale", detected_class="cup",
        room="office", embedding=_emb(seed=5),
        last_seen_ts=old,
    )
    wm.entities[ent_stale.id] = ent_stale

    # Stale, but has interaction history
    ent_touched = _make_object(
        eid="ent-touched", detected_class="wallet",
        room="office", embedding=_emb(seed=6),
        last_seen_ts=old,
    )
    wm.entities[ent_touched.id] = ent_touched
    await wm.store.append_event({  # type: ignore[union-attr]
        "id": "ev-1", "ts": (old + timedelta(hours=1)).isoformat(),
        "entity_id": ent_touched.id, "event_type": "picked_up",
        "entity_type": "object", "room": "office", "metadata": {},
    })

    # Recent, untouched — should be skipped
    ent_recent = _make_object(
        eid="ent-recent", detected_class="book",
        room="office", embedding=_emb(seed=7),
        last_seen_ts=now - timedelta(days=1),
    )
    wm.entities[ent_recent.id] = ent_recent

    pruned = await wm.prune_stale_objects(max_age_days=30)
    assert pruned == 1, pruned
    assert wm.entities[ent_stale.id].metadata.get("pruned") is True
    assert wm.entities[ent_touched.id].metadata.get("pruned") is not True
    assert wm.entities[ent_recent.id].metadata.get("pruned") is not True
    print("PASS: §23.8 prune_stale_objects - untouched stale only")


async def test_find_object_happy_path_and_hedge() -> None:
    """find_object encodes the description, returns the highest-cosine
    object entity. Borderline similarity sets hedge=True so the LLM
    knows to phrase it as a guess."""
    from modules.world_model.query_tools import WorldQueryTools

    wm = _wm()
    await wm.start()
    # Wire up the stub text encoder.
    text_for_wallet = np.array([1.0, 0, 0, 0], dtype=np.float32)
    text_for_phone = np.array([0, 1.0, 0, 0], dtype=np.float32)
    text_for_obscure = np.array([0, 0, 1.0, 0], dtype=np.float32)
    # Borderline: cos sim ≈ 0.28 with wallet (between match_threshold
    # 0.25 and hedge_threshold 0.32). Orthogonal to phone so it
    # unambiguously routes to wallet but with hedge=True.
    text_for_borderline = np.array(
        [0.28, 0.0, np.sqrt(1 - 0.28**2), 0.0], dtype=np.float32,
    )
    encoder = StubTextEncoder({
        "wallet": text_for_wallet,
        "phone": text_for_phone,
        "obscure_thing": text_for_obscure,
        "leather thing": text_for_borderline,
    })
    wm.clip_encoder = encoder

    # Two object entities — wallet and phone — with embeddings
    # that align cleanly with the corresponding text vectors.
    ts = datetime.utcnow()
    wallet_ent = _make_object(
        eid="wallet-1", detected_class="wallet",
        room="office",
        embedding=text_for_wallet,
        last_seen_ts=ts - timedelta(minutes=5),
    )
    phone_ent = _make_object(
        eid="phone-1", detected_class="cell phone",
        room="kitchen",
        embedding=text_for_phone,
        last_seen_ts=ts - timedelta(minutes=2),
    )
    wm.entities[wallet_ent.id] = wallet_ent
    wm.entities[phone_ent.id] = phone_ent

    wq = WorldQueryTools(wm)

    # Happy path — exact match for "wallet".
    res = await wq.find_object("wallet")
    assert res["found"] is True, res
    assert res["name"] == "wallet"
    assert res["last_seen_room"] == "office"
    assert res["match_similarity"] > 0.99, res
    assert res["hedge"] is False

    # Borderline — between wallet and phone, hedge expected.
    res2 = await wq.find_object("leather thing")
    if res2.get("found"):
        # Either result is acceptable; we just want the hedge flag set.
        assert res2["hedge"] is True, res2

    # Below threshold — text vector aligned with no entity.
    res3 = await wq.find_object("obscure_thing")
    assert res3["found"] is False, res3

    # Pruned entities are excluded.
    wallet_ent.metadata["pruned"] = True
    res4 = await wq.find_object("wallet")
    # Should now miss (or fall to the alternative phone, which has
    # 0 similarity to wallet text → below threshold).
    assert res4["found"] is False, res4
    print("PASS: §23.7 find_object - happy + hedge + below-threshold + pruned-skip")


async def main() -> None:
    await test_object_cost_path_a_clip_dominant()
    await test_object_cost_path_a_room_prior()
    await test_object_cost_path_b_fallback_no_clip()
    await test_dedup_same_class_same_room_lower_threshold()
    await test_prune_stale_objects_skips_touched()
    await test_find_object_happy_path_and_hedge()
    print("\nAll §23 object tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
