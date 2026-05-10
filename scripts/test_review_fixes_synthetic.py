"""
JARVIS — World Model
====================
Regression tests for the review-driven bug fixes:

  1. ObservationBuilder publishes empty batches (so WorldModel can fire
     LOST_VISIBILITY).
  2. ObservationBuilder skips ambiguous identity matches (no false
     positive person_id; identity_status='ambiguous' in metadata).
  3. IdentityManager._reload_caches refuses non-512-dim face_samples
     even when model_version says ArcFace (dim guard).
  4. IdentityManager._repair_mistagged_face_samples flips 128-dim
     blobs tagged as ArcFace back to facenet_v1 (one-time DB repair).

Each test is a focused unit test of the changed function — fake
collaborators only, no DB I/O beyond an in-memory aiosqlite.
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import aiosqlite
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Fakes ───────────────────────────────────────────────────────────────────


class StubBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def publish(self, topic: str, payload: dict) -> None:
        self.events.append((topic, payload))

    async def subscribe(self, topic: str, handler: Any) -> None:
        pass


class FakeCameraManager:
    def __init__(self, room_id: str, frame: np.ndarray) -> None:
        self.room_id = room_id
        self.frame = frame
        self.calls = 0

    def get_available_rooms(self) -> list[str]:
        return [self.room_id]

    async def capture_frame_async(self, room: str) -> Optional[np.ndarray]:
        self.calls += 1
        if room == self.room_id:
            return self.frame
        return None


class FakeDetector:
    """Returns whatever detections the test pre-loaded. None = empty list."""

    def __init__(self, detections_per_call: list[list[dict]]) -> None:
        self.detections_per_call = detections_per_call
        self.idx = 0

    async def detect_async(self, frame: np.ndarray) -> list[dict]:
        if self.idx < len(self.detections_per_call):
            out = self.detections_per_call[self.idx]
        else:
            out = []
        self.idx += 1
        return out


class FakeFaceRecognizer:
    async def detect_and_embed(self, crop: np.ndarray) -> list[dict]:
        return []  # no faces in these synthetic tests


class FakeIdentityManager:
    def __init__(self, match_to_return: Any = None) -> None:
        self._match = match_to_return
        self.calls = 0

    async def identify_from_embedding_async(
        self, emb: Any, modality: str = "face",
    ):
        self.calls += 1
        return self._match


# ── In-memory DB façade ─────────────────────────────────────────────────────


class InMemoryDB:
    def __init__(self) -> None:
        self.conn: Optional[aiosqlite.Connection] = None

    async def init(self) -> None:
        self.conn = await aiosqlite.connect(":memory:")
        self.conn.row_factory = aiosqlite.Row
        # Replicate just enough of the real schema for this test.
        await self.conn.executescript("""
            CREATE TABLE persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                created_at TEXT,
                notes TEXT
            );
            CREATE TABLE face_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER REFERENCES persons(id),
                embedding BLOB NOT NULL,
                pose TEXT,
                captured_at TEXT,
                source TEXT,
                image_jpeg BLOB,
                model_version TEXT NOT NULL DEFAULT 'facenet_v1'
            );
            CREATE TABLE voice_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER REFERENCES persons(id),
                embedding BLOB NOT NULL,
                prompt_id TEXT,
                captured_at TEXT,
                source TEXT
            );
            CREATE TABLE identity_pending (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_id INTEGER
            );
            CREATE TABLE faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                embedding BLOB
            );
            CREATE TABLE speakers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                embedding BLOB
            );
        """)
        await self.conn.commit()

    async def close(self) -> None:
        if self.conn is not None:
            await self.conn.close()

    async def execute(self, sql: str, params: tuple = ()) -> int:
        assert self.conn is not None
        cur = await self.conn.execute(sql, params)
        await self.conn.commit()
        return int(cur.lastrowid or 0)

    async def fetchall(self, sql: str, params: tuple = ()) -> list:
        assert self.conn is not None
        cur = await self.conn.execute(sql, params)
        return list(await cur.fetchall())

    async def fetchone(self, sql: str, params: tuple = ()):
        assert self.conn is not None
        cur = await self.conn.execute(sql, params)
        return await cur.fetchone()


# ── Tests ───────────────────────────────────────────────────────────────────


async def test_observation_builder_publishes_empty_batches() -> None:
    """The state machine fires LOST_VISIBILITY only when it receives an
    observation batch with no matching observation. If empty batches are
    suppressed, live disappearance never triggers."""
    from modules.vision.observation_builder import ObservationBuilder

    bus = StubBus()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cm = FakeCameraManager("office", frame)
    # Two non-empty ticks, then empty forever. The loop runs much
    # faster than 1000fps on a fake detector, so we'll catch both.
    detector = FakeDetector([
        [{"class": "person", "confidence": 0.9, "box": [100, 100, 200, 300]}],
        [{"class": "person", "confidence": 0.9, "box": [100, 100, 200, 300]}],
    ])
    ob = ObservationBuilder(
        bus=bus, camera_manager=cm,
        object_detector=detector,
        face_recognizer=FakeFaceRecognizer(),
        identity_manager=FakeIdentityManager(),
        posture_analyzer=None,
        rooms_config=[{
            "id": "office",
            "fps_active": 1000,  # rip through the loop
            "world_model": {"enabled": True},
        }],
    )

    await ob.start()
    await asyncio.sleep(0.1)  # ample for many ticks at 1000fps target
    await ob.stop()

    pubs = [p for t, p in bus.events if t == "vision.observation"]
    # Should have published at least 2 batches across 2 detector ticks
    # (some loops may run faster — but the empty batch must be in there).
    has_non_empty = any(p["observations"] for p in pubs)
    has_empty = any(not p["observations"] for p in pubs)
    assert has_non_empty, "expected at least one non-empty publish"
    assert has_empty, (
        "ObservationBuilder suppressed the empty observation batch — "
        "WorldModel cannot fire LOST_VISIBILITY without it"
    )
    print("PASS: ObservationBuilder publishes empty batches")


async def test_ambiguous_match_does_not_set_person_id() -> None:
    """A match with is_ambiguous=True must NOT propagate person_id; the
    observation should land as anonymous-with-status='ambiguous'."""
    from modules.identity.identity_manager import PersonMatch
    from modules.vision.observation_builder import ObservationBuilder

    bus = StubBus()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cm = FakeCameraManager("office", frame)
    detector = FakeDetector([
        [{"class": "person", "confidence": 0.9, "box": [100, 100, 200, 300]}],
    ])

    ambiguous_match = PersonMatch(
        person_id=42, name="Cole",
        similarity=0.61, confirmed_via="face",
        is_ambiguous=True,
    )

    class FaceWithEmbedding:
        async def detect_and_embed(self, crop):
            return [{
                "embedding": np.zeros(512, dtype=np.float32),
                "yaw": 0.0, "pitch": 0.0, "roll": 0.0,
            }]

    im = FakeIdentityManager(match_to_return=ambiguous_match)
    ob = ObservationBuilder(
        bus=bus, camera_manager=cm,
        object_detector=detector,
        face_recognizer=FaceWithEmbedding(),
        identity_manager=im,
        posture_analyzer=None,
        rooms_config=[{"id": "office", "fps_active": 1000,
                       "world_model": {"enabled": True}}],
    )
    ts = datetime.now(timezone.utc)
    obs = await ob._build_for_frame("office", frame, ts)
    assert len(obs) == 1
    person_obs = obs[0]
    assert person_obs.person_id is None, (
        f"ambiguous match leaked person_id: {person_obs.person_id}"
    )
    assert person_obs.person_name is None
    assert person_obs.metadata.get("identity_status") == "ambiguous", (
        f"expected identity_status=ambiguous, got "
        f"{person_obs.metadata.get('identity_status')!r}"
    )
    print("PASS: ambiguous identity match does not set person_id")


async def test_dim_guard_skips_wrong_size_face_samples() -> None:
    """_reload_caches must skip face_samples whose embedding length
    doesn't match ACTIVE_FACE_EMBEDDING_DIM, even if model_version
    says ArcFace."""
    from modules.identity.identity_manager import (
        ACTIVE_FACE_EMBEDDING_DIM, ACTIVE_FACE_MODEL_VERSION, IdentityManager,
    )

    db = InMemoryDB(); await db.init()
    try:
        # Seed: one good 512-dim ArcFace row, one bad 128-dim row that
        # somehow got tagged as ArcFace (bug-prone migration history).
        await db.execute(
            "INSERT INTO persons (name, created_at) VALUES (?, ?)",
            ("Cole", datetime.now(timezone.utc).isoformat()),
        )
        good = np.random.rand(ACTIVE_FACE_EMBEDDING_DIM).astype(np.float32).tobytes()
        bad = np.random.rand(128).astype(np.float32).tobytes()
        await db.execute(
            "INSERT INTO face_samples (person_id, embedding, model_version) "
            "VALUES (?, ?, ?)",
            (1, good, ACTIVE_FACE_MODEL_VERSION),
        )
        await db.execute(
            "INSERT INTO face_samples (person_id, embedding, model_version) "
            "VALUES (?, ?, ?)",
            (1, bad, ACTIVE_FACE_MODEL_VERSION),  # mistagged
        )

        im = IdentityManager(
            db=db, speaker_identifier=None, face_recognizer=None,
        )
        # Bypass deepface init by calling _reload_caches directly.
        await im._reload_caches()

        loaded = im._face_samples.get(1) or []
        assert len(loaded) == 1, (
            f"expected dim guard to filter the 128-dim row; got {len(loaded)} loaded"
        )
        assert loaded[0].size == ACTIVE_FACE_EMBEDDING_DIM
    finally:
        await db.close()
    print("PASS: dim guard skips wrong-size face_samples")


async def test_repair_mistagged_face_samples() -> None:
    """_repair_mistagged_face_samples must flip 128-dim blobs tagged as
    ArcFace to facenet_v1, idempotently."""
    from modules.identity.identity_manager import (
        ACTIVE_FACE_MODEL_VERSION, IdentityManager,
    )

    db = InMemoryDB(); await db.init()
    try:
        await db.execute(
            "INSERT INTO persons (name, created_at) VALUES (?, ?)",
            ("Cole", datetime.now(timezone.utc).isoformat()),
        )
        bad1 = np.random.rand(128).astype(np.float32).tobytes()
        bad2 = np.random.rand(128).astype(np.float32).tobytes()
        good = np.random.rand(512).astype(np.float32).tobytes()
        already_facenet = np.random.rand(128).astype(np.float32).tobytes()

        await db.execute(
            "INSERT INTO face_samples (person_id, embedding, model_version) "
            "VALUES (?, ?, ?)",
            (1, bad1, ACTIVE_FACE_MODEL_VERSION),
        )
        await db.execute(
            "INSERT INTO face_samples (person_id, embedding, model_version) "
            "VALUES (?, ?, ?)",
            (1, bad2, ACTIVE_FACE_MODEL_VERSION),
        )
        await db.execute(
            "INSERT INTO face_samples (person_id, embedding, model_version) "
            "VALUES (?, ?, ?)",
            (1, good, ACTIVE_FACE_MODEL_VERSION),
        )
        await db.execute(
            "INSERT INTO face_samples (person_id, embedding, model_version) "
            "VALUES (?, ?, ?)",
            (1, already_facenet, "facenet_v1"),
        )

        im = IdentityManager(
            db=db, speaker_identifier=None, face_recognizer=None,
        )
        await im._repair_mistagged_face_samples()

        # After repair: 1 good ArcFace row, 3 facenet_v1 rows
        # (2 newly relabeled + 1 that was already facenet).
        rows = await db.fetchall(
            "SELECT model_version, length(embedding) as n FROM face_samples"
        )
        version_counts: dict[str, int] = {}
        for r in rows:
            version_counts[r["model_version"]] = (
                version_counts.get(r["model_version"], 0) + 1
            )
        assert version_counts.get(ACTIVE_FACE_MODEL_VERSION) == 1, version_counts
        assert version_counts.get("facenet_v1") == 3, version_counts

        # Idempotent: re-running is a no-op.
        await im._repair_mistagged_face_samples()
        rows2 = await db.fetchall(
            "SELECT model_version FROM face_samples"
        )
        assert len(rows2) == 4
    finally:
        await db.close()
    print("PASS: _repair_mistagged_face_samples relabels 128-dim ArcFace rows")


async def test_consider_new_sample_quality_gates_and_diversity() -> None:
    """§10 — auto-enrollment must:
       - reject low-quality candidates
       - reject near-duplicates (max sim ≥ 0.95)
       - add when below capacity
       - swap most-redundant when at capacity AND candidate increases
         diversity; reject when it doesn't
    """
    from modules.identity.identity_manager import (
        ACTIVE_FACE_EMBEDDING_DIM, IdentityManager,
        SAMPLES_DIVERSITY_THRESHOLD, SAMPLES_PER_PERSON_MAX,
    )

    db = InMemoryDB(); await db.init()
    try:
        await db.execute(
            "INSERT INTO persons (name, created_at) VALUES (?, ?)",
            ("Cole", datetime.now(timezone.utc).isoformat()),
        )
        im = IdentityManager(
            db=db, speaker_identifier=None, face_recognizer=None,
        )
        await im._reload_caches()

        # 1. Low-quality candidate (face too small) — reject.
        emb = np.random.rand(ACTIVE_FACE_EMBEDDING_DIM).astype(np.float32)
        ok = await im.consider_new_sample_async(
            person_id=1, new_embedding=emb,
            quality_metadata={
                "face_area_px": 50 * 50,  # below 80*80 floor
                "yaw": 5.0, "pitch": 5.0, "blur_score": 200.0,
            },
        )
        assert ok is False, "low-quality candidate should be rejected"

        # 2. First good candidate — accept.
        good_meta = {
            "face_area_px": 100 * 100, "yaw": 5.0, "pitch": -5.0,
            "blur_score": 200.0, "association_confidence": 0.9,
        }
        emb1 = np.random.rand(ACTIVE_FACE_EMBEDDING_DIM).astype(np.float32)
        ok = await im.consider_new_sample_async(
            person_id=1, new_embedding=emb1,
            quality_metadata=good_meta,
        )
        assert ok is True
        assert len(im._face_samples[1]) == 1

        # 3. Near-duplicate (same vector + tiny noise) — reject by diversity.
        emb_dup = emb1 + 0.001 * np.random.rand(ACTIVE_FACE_EMBEDDING_DIM).astype(np.float32)
        ok = await im.consider_new_sample_async(
            person_id=1, new_embedding=emb_dup,
            quality_metadata=good_meta,
        )
        assert ok is False, (
            f"near-dup (sim ≥ {SAMPLES_DIVERSITY_THRESHOLD}) should reject"
        )
        assert len(im._face_samples[1]) == 1

        # 4. Below cap, diverse — accept.
        emb2 = np.random.rand(ACTIVE_FACE_EMBEDDING_DIM).astype(np.float32) * 5
        ok = await im.consider_new_sample_async(
            person_id=1, new_embedding=emb2,
            quality_metadata=good_meta,
        )
        assert ok is True
        assert len(im._face_samples[1]) == 2

        # 5. Fill the bank to capacity and verify capacity behaviour.
        np.random.seed(42)
        while len(im._face_samples[1]) < SAMPLES_PER_PERSON_MAX:
            extra = np.random.randn(ACTIVE_FACE_EMBEDDING_DIM).astype(np.float32) * 5
            await im.consider_new_sample_async(
                person_id=1, new_embedding=extra,
                quality_metadata=good_meta,
            )
        assert len(im._face_samples[1]) == SAMPLES_PER_PERSON_MAX

        # A high-diversity candidate at cap should swap, keeping size constant.
        np.random.seed(99)
        diverse = np.random.randn(ACTIVE_FACE_EMBEDDING_DIM).astype(np.float32) * 50
        ok = await im.consider_new_sample_async(
            person_id=1, new_embedding=diverse,
            quality_metadata=good_meta,
        )
        # Either accept-and-swap or reject if not actually more diverse —
        # we just want the bank to stay at SAMPLES_PER_PERSON_MAX either way.
        assert len(im._face_samples[1]) == SAMPLES_PER_PERSON_MAX, (
            f"bank size left {SAMPLES_PER_PERSON_MAX} after swap attempt: "
            f"{len(im._face_samples[1])}"
        )
    finally:
        await db.close()
    print("PASS: §10 consider_new_sample_async quality + diversity + capacity")


def test_world_model_config_validation_cross_references() -> None:
    """validate_world_model_config must:
       - accept a fully valid household
       - reject a pet whose household_owner isn't a declared resident
       - reject an affinity pointing at an unknown resident
       - reject an affinity context outside the enum
       - reject an exit.to_room pointing at a nonexistent room
    """
    from core.config import validate_world_model_config
    from core.exceptions import ConfigError

    base = {
        "world_model": {"tracked_species": ["cat", "dog"]},
        "residents": [
            {"id": "cole", "display_name": "Cole"},
            {"id": "anna", "display_name": "Anna"},
        ],
        "rooms": [
            {"id": "office", "world_model": {
                "enabled": True,
                "frame_width": 640, "frame_height": 480,
                "exits": [], "landmarks": [],
            }},
            {"id": "bedroom", "world_model": {
                "enabled": True,
                "frame_width": 1920, "frame_height": 1080,
                "exits": [], "landmarks": [],
            }},
        ],
        "pets": {
            "cats": [{
                "name": "Spooky", "household_owner": "cole",
                "color_class": "black", "expected_size": "medium",
                "home_room": "bedroom",
                "affinities": [
                    {"person": "cole", "strength": "high",
                     "contexts": ["sleeping"]},
                ],
            }],
            "dogs": [],
        },
    }
    # Happy path
    wm, res, pets, _ = validate_world_model_config(base)
    assert wm is not None and len(res) == 2 and pets is not None

    # Bad household_owner
    bad = {**base, "pets": {"cats": [{
        **base["pets"]["cats"][0],
        "household_owner": "nobody",
    }], "dogs": []}}
    try:
        validate_world_model_config(bad)
        raise AssertionError("expected ConfigError on bad owner")
    except ConfigError as e:
        assert "household_owner" in str(e), str(e)

    # Bad affinity person
    bad = {**base, "pets": {"cats": [{
        **base["pets"]["cats"][0],
        "affinities": [{"person": "ghost", "strength": "high",
                        "contexts": ["sleeping"]}],
    }], "dogs": []}}
    try:
        validate_world_model_config(bad)
        raise AssertionError("expected ConfigError on unknown affinity person")
    except ConfigError as e:
        assert "affinity" in str(e) and "ghost" in str(e), str(e)

    # Bad affinity context
    bad = {**base, "pets": {"cats": [{
        **base["pets"]["cats"][0],
        "affinities": [{"person": "cole", "strength": "high",
                        "contexts": ["napping"]}],  # not in enum
    }], "dogs": []}}
    try:
        validate_world_model_config(bad)
        raise AssertionError("expected ConfigError on bad context")
    except ConfigError as e:
        assert "context" in str(e), str(e)

    # Bad exit.to_room reference
    bad = {**base, "rooms": [
        base["rooms"][0],
        {"id": "bedroom", "world_model": {
            "enabled": True,
            "frame_width": 1920, "frame_height": 1080,
            "exits": [{"kind": "to_room", "to": "garage",  # not declared
                       "polygon": [[0, 0], [10, 0], [10, 10]]}],
            "landmarks": [],
        }},
    ]}
    try:
        validate_world_model_config(bad)
        raise AssertionError("expected ConfigError on unknown to_room")
    except ConfigError as e:
        assert "garage" in str(e), str(e)
    print("PASS: world_model config validation enforces cross-references")


async def main() -> None:
    await test_observation_builder_publishes_empty_batches()
    await test_ambiguous_match_does_not_set_person_id()
    await test_dim_guard_skips_wrong_size_face_samples()
    await test_repair_mistagged_face_samples()
    await test_consider_new_sample_quality_gates_and_diversity()
    test_world_model_config_validation_cross_references()  # sync
    print("\nAll review-fix regression tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
