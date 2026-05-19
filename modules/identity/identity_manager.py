"""
JARVIS — Ambient Home AI
========================
Mission: Cross-modal unified identity. A single `Person` is matched by EITHER
         voice OR face — whichever modality presents — and drift in one
         modality is corrected opportunistically using the other as an anchor.
         Multiple samples per person per modality are kept forever (never
         deleted), so a cold voice or new haircut can't lose recognition.

         Match rule per modality (hybrid):
           best_score >= T_match
           AND (one person enrolled OR best - second_best >= margin)
         Below T_stranger → "unknown" → captured into pending clusters that
         the dashboard surfaces for naming.

         Drift capture (passive, anchored):
           When voice strongly matches person X, the orchestrator calls
           verify_face(X, frame) at the next vision tick:
             - face matches X loosely (>= T_stranger_face) → auto-save as
               face_sample (source='drift_capture'). Cheap drift refresh.
             - face matches a different enrolled person above their match
               threshold → conflict → write to identity_pending for review.
             - face matches no one (< T_stranger_face) → write to
               identity_pending with anchored_via='voice' so the user can
               confirm "yes that's still cole" in the dash.
           Same logic in reverse for verify_voice.

Modules: modules/identity/identity_manager.py
Classes: IdentityManager, PersonMatch (dataclass)
"""

import asyncio
import io
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np
from loguru import logger


# ── Threshold defaults ──────────────────────────────────────────────────────
# T_match:    above this is a confirmed identification of an enrolled person.
# T_stranger: below this means "I do not recognize this at all" → goes into
#             pending clusters as a possible new persona. Between the two
#             sits the "loose match" zone used for drift auto-capture.
# margin:     when 2+ persons enrolled, best score must beat second-best by
#             at least this much, otherwise the match is flagged ambiguous.
# Face thresholds are ArcFace-tuned (cosines on InsightFace `buffalo_l`).
# match=0.5 / margin=0.10 are the values §11 of the World Model spec calls
# out — both are looser than Facenet would have wanted but appropriate
# for ArcFace's L2-normalized 512-dim space, and the margin gate is the
# real safety net for resident-vs-resident misidentification.
DEFAULTS = {
    "voice": {"match": 0.75, "stranger": 0.55, "margin": 0.05},
    "face":  {"match": 0.50, "stranger": 0.35, "margin": 0.10},
}

# Active face encoder. Stamped on every face_samples row written by this
# module so the centroid bank can filter to the live model. Kept in sync
# with FaceRecognizer.MODEL_VERSION — when ArcFace is replaced (e.g. by a
# future buffalo_xl), bump both in lockstep.
ACTIVE_FACE_MODEL_VERSION = "arcface_buffalo_l_v1"
# Embedding dimension that ACTIVE_FACE_MODEL_VERSION produces. The
# centroid bank loader uses this as a hard guard — a row tagged ArcFace
# but with wrong dims is corrupt and must not enter the live cache.
# Bump alongside ACTIVE_FACE_MODEL_VERSION when changing the encoder.
ACTIVE_FACE_EMBEDDING_DIM = 512

# §10 auto-enrollment thresholds. The diversity-replacement coreset
# algorithm caps samples per person and rejects near-duplicates,
# preventing both bank bloat and the silent-overfit failure where a
# person's centroid drifts toward whatever pose was most recently
# captured. Override per-person via config.identity.face.* if needed.
SAMPLES_PER_PERSON_MAX = 60        # capacity cap (bumped 30→60: with
                                   # multiple residents who look anything
                                   # alike, 30 wasn't enough headroom for
                                   # the margin gate to settle)
SAMPLES_DIVERSITY_THRESHOLD = 0.95  # reject candidate if max sim ≥ this
ENROLLMENT_QUALITY_GATES = {
    "min_face_area_px":      80 * 80,   # face crop must be ≥ 80×80
    "max_abs_yaw_deg":       45.0,
    "max_abs_pitch_deg":     35.0,
    "min_blur_score":        100.0,     # Laplacian variance
    "min_assoc_confidence":  0.85,      # WorldModel attribution conf
}

# §10 voice auto-enrollment. Same coreset algorithm, looser diversity
# threshold + smaller cap because voice clips have less effective
# entropy than face crops. Quality gates are duration / SNR / VAD /
# music — gated by the YAMNet pass per §10.
VOICE_SAMPLES_PER_PERSON_MAX = 40
VOICE_SAMPLES_DIVERSITY_THRESHOLD = 0.92

# Pairwise cosine above which a face/voice sample is considered
# redundant with another in the same person's bank — the harm-based
# prune pass keeps only one representative from each near-duplicate
# pair. Tighter than SAMPLES_DIVERSITY_THRESHOLD (which gates fresh
# enrollment) because removing a real-but-tight sample is more
# expensive than rejecting a fresh near-duplicate would have been.
PRUNE_REDUNDANCY_THRESHOLD = {
    "face": 0.97,
    "voice": 0.95,
}

# Pending-cluster merge threshold — if a new unknown sample's cosine to an
# existing pending-cluster centroid is at least this, fold into that cluster
# rather than starting a new one.
PENDING_MERGE_THRESHOLD = {
    # The cosine threshold for folding a new pending sample into an
    # existing pending cluster (i.e. "this is the same unknown person
    # we've been collecting"). Started at 0.55 for face but that was too
    # strict — the kind of low-quality face that hits the pending queue
    # in the first place rarely re-clears 0.55 against a centroid built
    # from other low-quality embeddings of the same person. Result: every
    # new capture spawned a fresh cluster and the queue grew unbounded
    # (Cole hit 2355 unresolved rows). 0.35 is loose enough that borderline
    # same-person captures collapse, strict enough that genuinely different
    # people stay in different clusters.
    "voice": 0.65,
    "face":  0.35,
}

# Hard cap on unresolved pending rows. Past this, _write_pending drops
# the oldest unresolved row before inserting the new one so the queue
# stays finite even if the merge fails to consolidate them all.
MAX_UNRESOLVED_PENDING = 200

# ── Passive-capture quality (recognition self-improvement) ──────────────────
# Coherence gate: a drift / live_question face sample is kept only if it
# AGREES with the person's existing bank — cosine to the bank mean must
# clear _COHERENCE_FLOOR. Stops misattributed / low-quality captures from
# poisoning a bank. Skipped for deliberate enroll and for cold-start (a
# person with < _COHERENCE_MIN_CORE samples has no core to judge against).
_COHERENCE_FLOOR = 0.20
_COHERENCE_MIN_CORE = 8

# Nightly bank gardener: per person, the lowest-centrality face samples
# (least like the rest of that person's bank — drift, misattribution) are
# quarantined. Conservative + gradual — at most _GARDENER_MAX_EVICT per
# person per run, never below _GARDENER_MIN_KEEP, only samples below
# _GARDENER_OUTLIER_FLOOR. Quarantine re-tags model_version (the loader
# filters to the exact active tag); clean_face_bank.py --undo reverses it.
_GARDENER_OUTLIER_FLOOR = 0.15
_GARDENER_MIN_KEEP = 20
_GARDENER_MAX_EVICT = 5
QUARANTINE_MODEL_VERSION = ACTIVE_FACE_MODEL_VERSION + "_quarantined"


@dataclass
class PersonMatch:
    """Result of an identify_*() call when something passed the match threshold."""
    person_id: int
    name: str
    similarity: float
    confirmed_via: str                  # "voice" or "face"
    is_ambiguous: bool = False          # margin too small — flag for review
    pending_verification: Optional[str] = None  # other modality to verify next


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Minimum face-bbox area we'll accept when adding a pending-review sample
# to the centroid bank. Distant / profile / partially-cropped faces
# embed poorly and drag centroids toward bad poses, which is exactly
# why Cole's match scores stopped rising even after 300 pending reviews.
# 80x80 = 6400 is conservative; tweak via /api/tunables if needed.
_PENDING_MIN_FACE_AREA_PX = 6400


def _passes_pending_area_gate(bbox: Optional[list]) -> bool:
    """Cheap area-only quality gate for pending-review samples. The
    full _passes_face_quality_gates needs blur/yaw/pitch which we don't
    persist on pending rows. Area alone catches the worst offenders
    (tiny / distant faces). Missing bbox → pass-through (legacy rows
    don't have face_bbox stored yet)."""
    if not bbox or len(bbox) != 4:
        return True
    try:
        x1, y1, x2, y2 = (float(c) for c in bbox)
    except (TypeError, ValueError):
        return True
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return area >= _PENDING_MIN_FACE_AREA_PX


# ── §10 auto-enrollment helpers ─────────────────────────────────────────────


def _passes_face_quality_gates(meta: dict) -> bool:
    """Reject candidate face samples that wouldn't help the centroid bank.
    Missing keys are treated as worst-case (rejection) so a stripped-down
    upstream — e.g. a fast-path observation that didn't compute pose —
    can't sneak through. The blur and pose fields are produced by
    ObservationBuilder._build_person_obs; assoc_conf is from the
    World Model."""
    g = ENROLLMENT_QUALITY_GATES
    # Face area: derive from bbox if metadata didn't pre-compute it.
    area = meta.get("face_area_px")
    if area is None:
        bbox = meta.get("face_bbox")
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            area = max(0, int(x2) - int(x1)) * max(0, int(y2) - int(y1))
    if area is None or area < g["min_face_area_px"]:
        return False
    yaw = meta.get("yaw")
    if yaw is None or abs(float(yaw)) > g["max_abs_yaw_deg"]:
        return False
    pitch = meta.get("pitch")
    if pitch is None or abs(float(pitch)) > g["max_abs_pitch_deg"]:
        return False
    blur = meta.get("blur_score")
    if blur is None or float(blur) < g["min_blur_score"]:
        return False
    # association confidence is the WorldModel's per-match attribution
    # confidence — only available after entity association ran. Default
    # to "passing" when the upstream skipped it (worldmodel-less
    # enrollment, e.g. dashboard manual flow).
    assoc = meta.get("association_confidence")
    if assoc is not None and float(assoc) < g["min_assoc_confidence"]:
        return False
    return True


def _most_redundant_index(embeddings: list[np.ndarray]) -> int:
    """Index of the embedding whose mean cosine similarity to the rest of
    the bank is highest — i.e. the sample that contributes the least
    diversity. Used to pick an eviction target when at capacity."""
    n = len(embeddings)
    if n < 2:
        return 0
    M = np.stack([
        e.astype(np.float32) / (np.linalg.norm(e) + 1e-9)
        for e in embeddings
    ])
    S = M @ M.T
    np.fill_diagonal(S, 0.0)
    return int(np.argmax(S.sum(axis=1) / (n - 1)))


def _avg_pairwise_sim(embeddings: list[np.ndarray]) -> float:
    """Mean off-diagonal cosine similarity over the bank. The diversity-
    replacement swap is only allowed if the *replaced* bank has lower
    avg pairwise sim than the original — otherwise we'd be lowering
    diversity, not raising it."""
    n = len(embeddings)
    if n < 2:
        return 0.0
    M = np.stack([
        e.astype(np.float32) / (np.linalg.norm(e) + 1e-9)
        for e in embeddings
    ])
    S = M @ M.T
    np.fill_diagonal(S, 0.0)
    return float(S.sum() / (n * (n - 1)))


def _would_increase_diversity(
    existing: list[np.ndarray],
    evict_idx: int,
    candidate: np.ndarray,
) -> bool:
    """Replacing existing[evict_idx] with `candidate` decreases avg sim?"""
    replaced = list(existing)
    replaced[evict_idx] = candidate
    return _avg_pairwise_sim(replaced) < _avg_pairwise_sim(existing)


class IdentityManager:
    """
    Unified cross-modal identity store.

    Holds one entry per person; matches across voice or face samples.
    Old samples are never deleted — drift capture appends new samples.
    """

    def __init__(
        self,
        db: Any,
        speaker_identifier: Any,
        face_recognizer: Any,
        config: Optional[dict] = None,
        notifier: Optional[Any] = None,
        broadcast: Optional[Any] = None,
    ) -> None:
        self._db = db
        self._spk = speaker_identifier
        self._face = face_recognizer
        self._notifier = notifier
        # Async callback for pushing live events to the dashboard. Fires
        # 'identity_pending_added' from _write_pending so the pending list
        # updates in real time without waiting for a manual reload.
        self._broadcast = broadcast
        cfg = (config or {}).get("identity", {}) if config else {}
        self._th = {
            "voice": {**DEFAULTS["voice"], **(cfg.get("voice") or {})},
            "face":  {**DEFAULTS["face"],  **(cfg.get("face")  or {})},
        }
        # Caches: person_id → name; person_id → list[np.ndarray] of all
        # samples for that modality. Rebuilt on enroll / migration.
        self._persons: dict[int, str] = {}
        self._voice_samples: dict[int, list[np.ndarray]] = {}
        self._face_samples: dict[int, list[np.ndarray]] = {}
        # Pending-cluster centroids in memory (faster than re-fetching every
        # capture). Rebuilt from DB on init.
        # cluster_id → (modality, centroid, count)
        self._pending_clusters: dict[int, tuple[str, np.ndarray, int]] = {}
        self._next_cluster_id: int = 1
        # Drift-verify queue: person_id → modality_to_verify ("face"|"voice")
        # When set, the next time the orchestrator captures the corresponding
        # modality it should call verify_*() to refresh that side.
        self._verify_pending: dict[int, str] = {}

    # ── lifecycle ───────────────────────────────────────────────────────────

    async def init(self) -> None:
        """Run schema-migration from legacy tables, then load caches."""
        await self._migrate_legacy_if_needed()
        await self._repair_mistagged_face_samples()
        await self._reload_caches()
        logger.info(
            f"[Identity] Ready ({len(self._persons)} persons, "
            f"{sum(len(v) for v in self._voice_samples.values())} voice samples, "
            f"{sum(len(v) for v in self._face_samples.values())} face samples)"
        )

    async def _migrate_legacy_if_needed(self) -> None:
        """One-time migration: legacy speakers/faces rows → persons + samples.

        We only migrate names that don't already exist as a person, and we tag
        the resulting samples with source='migration' so the dashboard can show
        which centroids came from the v1 system.
        """
        try:
            existing = {
                r["name"] for r in await self._db.fetchall("SELECT name FROM persons")
            }
        except Exception:
            existing = set()

        async def _ensure_person(name: str) -> Optional[int]:
            row = await self._db.fetchone(
                "SELECT id FROM persons WHERE name = ? COLLATE NOCASE", (name,)
            )
            if row is not None:
                return int(row["id"])
            try:
                return await self._db.execute(
                    "INSERT INTO persons (name, created_at) VALUES (?, ?)",
                    (name, _now_iso()),
                )
            except Exception as e:
                logger.warning(f"[Identity] could not create person '{name}': {e}")
                return None

        # Voice migration
        try:
            voice_rows = await self._db.fetchall(
                "SELECT name, embedding FROM speakers"
            )
        except Exception:
            voice_rows = []
        for row in voice_rows:
            name = row["name"]
            emb_blob = row["embedding"]
            if name in existing:
                # Person exists — but we still want their legacy centroid
                # represented as one of their samples if they don't have one yet.
                pass
            pid = await _ensure_person(name)
            if pid is None:
                continue
            # Skip if a migration sample already exists for this person
            seen = await self._db.fetchone(
                "SELECT 1 FROM voice_samples WHERE person_id = ? AND source = 'migration'",
                (pid,),
            )
            if seen is not None:
                continue
            await self._db.execute(
                "INSERT INTO voice_samples (person_id, embedding, prompt_id, captured_at, source) "
                "VALUES (?, ?, ?, ?, 'migration')",
                (pid, emb_blob, "legacy_centroid", _now_iso()),
            )

        # Face migration
        try:
            face_rows = await self._db.fetchall(
                "SELECT name, embedding FROM faces"
            )
        except Exception:
            face_rows = []
        for row in face_rows:
            name = row["name"]
            emb_blob = row["embedding"]
            pid = await _ensure_person(name)
            if pid is None:
                continue
            seen = await self._db.fetchone(
                "SELECT 1 FROM face_samples WHERE person_id = ? AND source = 'migration'",
                (pid,),
            )
            if seen is not None:
                continue
            # Legacy `faces` rows are 128-dim DeepFace/Facenet vectors.
            # Tag them as 'facenet_v1' so _reload_caches's
            # model_version filter excludes them from the live ArcFace
            # centroid bank — mixing 128-dim and 512-dim blobs in the
            # same cosine bucket corrupts identification. Re-enrolling
            # under ArcFace produces fresh 512-dim samples; the legacy
            # rows stay only for audit trail / dashboard history.
            await self._db.execute(
                "INSERT INTO face_samples (person_id, embedding, pose, captured_at, source, model_version) "
                "VALUES (?, ?, ?, ?, 'migration', 'facenet_v1')",
                (pid, emb_blob, "candid", _now_iso()),
            )

    async def _repair_mistagged_face_samples(self) -> None:
        """One-time DB repair for face_samples rows mistagged as ArcFace
        but holding 128-dim Facenet blobs. Idempotent — safe to run on
        every boot. Fixes the bug introduced by an earlier version of
        `_migrate_legacy_if_needed` that wrote ACTIVE_FACE_MODEL_VERSION
        for legacy migration rows.
        """
        try:
            row = await self._db.fetchone(
                "SELECT COUNT(*) AS n FROM face_samples "
                "WHERE model_version = ? AND length(embedding) = ?",
                (ACTIVE_FACE_MODEL_VERSION, 128 * 4),
            )
        except Exception as e:
            logger.debug(f"[Identity] mistagged-face check failed: {e}")
            return
        n = int(row["n"]) if row and row["n"] is not None else 0
        if n == 0:
            return
        await self._db.execute(
            "UPDATE face_samples SET model_version = 'facenet_v1' "
            "WHERE model_version = ? AND length(embedding) = ?",
            (ACTIVE_FACE_MODEL_VERSION, 128 * 4),
        )
        logger.warning(
            f"[Identity] repaired {n} face_samples row(s) mistagged as "
            f"{ACTIVE_FACE_MODEL_VERSION} (held 128-dim Facenet blobs). "
            "Re-enroll affected residents under ArcFace to restore live identification."
        )

    async def _reload_caches(self) -> None:
        self._persons.clear()
        self._voice_samples.clear()
        self._face_samples.clear()
        for r in await self._db.fetchall("SELECT id, name FROM persons"):
            pid = int(r["id"])
            self._persons[pid] = r["name"]
            self._voice_samples[pid] = []
            self._face_samples[pid] = []
        for r in await self._db.fetchall(
            "SELECT person_id, embedding FROM voice_samples"
        ):
            pid = int(r["person_id"])
            try:
                emb = np.frombuffer(r["embedding"], dtype=np.float32).copy()
                if emb.size and pid in self._voice_samples:
                    self._voice_samples[pid].append(emb)
            except Exception:
                continue
        # Filter to the active face model — old facenet_v1 (128-dim) rows
        # stay in the DB for history but don't enter the live centroid
        # bank since they're incomparable with current ArcFace embeddings.
        # Defense-in-depth: even when model_version says ArcFace, refuse
        # to load anything that isn't 512-dim. A bad row would otherwise
        # poison cosine similarity for the whole person.
        for r in await self._db.fetchall(
            "SELECT person_id, embedding FROM face_samples WHERE model_version = ?",
            (ACTIVE_FACE_MODEL_VERSION,),
        ):
            pid = int(r["person_id"])
            try:
                emb = np.frombuffer(r["embedding"], dtype=np.float32).copy()
                if emb.size != ACTIVE_FACE_EMBEDDING_DIM:
                    logger.warning(
                        f"[Identity] skipping face_sample for person {pid}: "
                        f"got {emb.size}-dim embedding, expected "
                        f"{ACTIVE_FACE_EMBEDDING_DIM}"
                    )
                    continue
                if pid in self._face_samples:
                    self._face_samples[pid].append(emb)
            except Exception:
                continue
        # Recompute next_cluster_id from existing pending rows
        try:
            row = await self._db.fetchone(
                "SELECT MAX(cluster_id) AS m FROM identity_pending"
            )
            m = (row["m"] if row and row["m"] is not None else 0)
            self._next_cluster_id = int(m) + 1
        except Exception:
            self._next_cluster_id = 1

    # ── identification ──────────────────────────────────────────────────────

    async def identify_voice(self, audio: np.ndarray) -> Optional[PersonMatch]:
        emb = await self._spk.embed_audio(audio) if self._spk is not None else None
        if emb is None:
            return None
        match = self._match_against("voice", emb)
        if match is None:
            await self._add_to_pending_cluster("voice", emb, audio_pcm16=_pcm16(audio))
        else:
            # Schedule face verify if person has any face samples (otherwise no
            # point) or none yet (we want to acquire a first face sample).
            self._verify_pending[match.person_id] = "face"
            match.pending_verification = "face"
        return match

    async def identify_face(self, frame: np.ndarray) -> Optional[PersonMatch]:
        # Use the bbox-aware variant so we can persist *which* face the
        # embedding came from. Lets the dashboard pending-review modal
        # draw a highlight around the targeted face when the frame
        # contains multiple people.
        full = (
            await self._face.embed_largest_face_full(frame)
            if self._face is not None else None
        )
        if full is None:
            return None
        emb = full.get("embedding")
        if emb is None:
            return None
        face_bbox = full.get("bbox")
        match = self._match_against("face", emb)
        if match is None:
            await self._add_to_pending_cluster(
                "face", emb, image_jpeg=_jpeg(frame), face_bbox=face_bbox,
            )
        else:
            self._verify_pending[match.person_id] = "voice"
            match.pending_verification = "voice"
        return match

    async def identify_from_embedding_async(
        self,
        embedding: np.ndarray,
        modality: str = "face",
        image_jpeg: Optional[bytes] = None,
        audio_pcm16: Optional[bytes] = None,
    ) -> Optional[PersonMatch]:
        """
        Match an already-extracted embedding against the centroid bank.

        Used by ObservationBuilder when the caller has already cropped the
        face out of the room frame and embedded it via FaceRecognizer.
        Skipping the embed step here means we don't do face detection
        twice (once for the person crop, once inside identify_face).

        Behavior mirrors `identify_face`: pending-cluster write on miss,
        voice-verification scheduling on hit. Callers may pass preview media
        for dashboard review when they already own the crop/audio.
        """
        if embedding is None:
            return None
        match = self._match_against(modality, embedding)
        if match is None:
            await self._add_to_pending_cluster(
                modality,
                embedding,
                image_jpeg=image_jpeg,
                audio_pcm16=audio_pcm16,
            )
        else:
            other = "voice" if modality == "face" else "face"
            self._verify_pending[match.person_id] = other
            match.pending_verification = other
        return match

    def _top_match_unconstrained(
        self, modality: str, emb: np.ndarray
    ) -> Optional[tuple[int, float]]:
        """Return (best person_id, similarity) ignoring match/margin
        thresholds. Used by the pending-review dashboard to suggest a
        likely person on a row that didn't pass the strict match
        threshold — better than the user picking blindly. None if no
        enrolled samples in the chosen modality."""
        samples = self._voice_samples if modality == "voice" else self._face_samples
        if not samples or emb is None:
            return None
        best: Optional[tuple[int, float]] = None
        for pid, embs in samples.items():
            if not embs:
                continue
            sim = max(_cosine(emb, e) for e in embs)
            if best is None or sim > best[1]:
                best = (pid, sim)
        return best

    def _match_against(
        self, modality: str, emb: np.ndarray
    ) -> Optional[PersonMatch]:
        """Return PersonMatch if best score passes hybrid threshold, else None."""
        samples = self._voice_samples if modality == "voice" else self._face_samples
        if not samples:
            return None
        # Best cosine across ALL samples per person, then argmax-by-person
        scored: list[tuple[int, float]] = []
        for pid, embs in samples.items():
            if not embs:
                continue
            best = max(_cosine(emb, e) for e in embs)
            scored.append((pid, best))
        if not scored:
            return None
        scored.sort(key=lambda t: t[1], reverse=True)
        best_pid, best_sim = scored[0]
        second_sim = scored[1][1] if len(scored) > 1 else -1.0

        thr = self._th[modality]
        if best_sim < thr["match"]:
            return None  # caller decides pending vs ignore via stranger floor
        # Margin check only if more than one person enrolled
        ambiguous = False
        if len(scored) > 1 and (best_sim - second_sim) < thr["margin"]:
            ambiguous = True
        return PersonMatch(
            person_id=best_pid,
            name=self._persons.get(best_pid, f"person_{best_pid}"),
            similarity=best_sim,
            confirmed_via=modality,
            is_ambiguous=ambiguous,
        )

    # ── enrollment (active, multi-sample) ───────────────────────────────────

    async def ensure_person(self, name: str) -> int:
        """Return person_id for `name`, creating the row if needed.

        Lookup is case-insensitive — 'Cole', 'cole', and 'COLE' resolve to the
        same person. The first-written casing is preserved as the display name.
        Two genuinely different people with the same name need distinct display
        labels (e.g. 'Cole' vs 'Cole S').
        """
        row = await self._db.fetchone(
            "SELECT id FROM persons WHERE name = ? COLLATE NOCASE", (name,)
        )
        if row is not None:
            return int(row["id"])
        pid = await self._db.execute(
            "INSERT INTO persons (name, created_at) VALUES (?, ?)",
            (name, _now_iso()),
        )
        self._persons[pid] = name
        self._voice_samples[pid] = []
        self._face_samples[pid] = []
        return pid

    async def enroll_face(
        self, name: str, frame: np.ndarray, pose: str = "candid"
    ) -> Optional[int]:
        emb = await self._face.embed_frame(frame) if self._face is not None else None
        if emb is None:
            logger.warning(f"[Identity] enroll_face '{name}' — no face detected")
            return None
        pid = await self.ensure_person(name)
        # Route through _save_sample so the cap+evict path is shared
        # with drift-capture / live_question / pending-resolve inserts.
        await self._save_sample(
            modality="face",
            person_id=pid,
            emb=emb,
            source="enroll",
            pose=pose,
            image_jpeg=_jpeg(frame),
        )
        logger.info(f"[Identity] Enrolled face for '{name}' (pose={pose})")
        # Caller used the return value only for logging in some paths;
        # the new row's id isn't easily fetchable without a SELECT.
        # Return person_id so callers that branched on truthiness still
        # work (they all check `is None`).
        return pid

    async def enroll_voice(
        self, name: str, audio: np.ndarray, prompt_id: str = "wake"
    ) -> Optional[int]:
        emb = await self._spk.embed_audio(audio) if self._spk is not None else None
        if emb is None:
            logger.warning(f"[Identity] enroll_voice '{name}' — embedding failed")
            return None
        pid = await self.ensure_person(name)
        await self._save_sample(
            modality="voice",
            person_id=pid,
            emb=emb,
            source="enroll",
            prompt_id=prompt_id,
        )
        logger.info(f"[Identity] Enrolled voice for '{name}' (prompt={prompt_id})")
        return pid

    # ── §10 auto-enrollment (diversity-replacement coreset) ─────────────────

    async def consider_new_sample_async(
        self,
        person_id: Optional[int],
        new_embedding: Optional[np.ndarray],
        crop_path: Optional[str] = None,
        quality_metadata: Optional[dict] = None,
    ) -> bool:
        """
        Auto-enrollment entry point. Called by WorldModel after every
        confident person observation that's been linked to a non-anonymous
        entity. Returns True if the sample was added to the bank, False
        if rejected (quality gate, ambiguity pause, near-duplicate, or
        not-more-diverse-than-most-redundant).

        Fire-and-forget: WorldModel's hot path doesn't await the result;
        a rejection here doesn't surface anywhere except the debug log.
        Identity ownership stays with this module — the World Model has
        nothing to do with sample storage or the centroid bank.

        Spec: new 2.md §10. Algorithm:
          1. Quality gates (face area, yaw, pitch, blur, assoc conf).
          2. Pause if person is in an active merge-candidate flag.
          3. Diversity gate: max cos sim to existing < 0.95.
          4. Below capacity → add directly.
          5. At capacity → swap the most-redundant existing sample only
             if doing so DECREASES the bank's average pairwise similarity
             (i.e. the candidate is genuinely more diverse).
        """
        if person_id is None or new_embedding is None:
            return False

        # 1. Quality gates.
        if not _passes_face_quality_gates(quality_metadata or {}):
            logger.debug(
                f"[Identity] auto-enroll rejected (quality) for person {person_id}"
            )
            return False

        # 2. Pause during merge ambiguity. The merge-candidate set is
        # populated when two persons' centroids cross the 0.7–0.85 band.
        # This list is intentionally small and short-lived — it gets
        # cleared on manual confirm/reject in the dashboard or auto-merge
        # at >0.85. Empty until that subsystem lands; the lookup is cheap.
        if person_id in self._merge_candidate_persons:
            logger.debug(
                f"[Identity] auto-enroll paused for person {person_id} "
                "(merge ambiguity)"
            )
            return False

        # Sanity: incoming embedding must match active model's dim. Defense
        # against an upstream regression that pipes a Facenet 128-dim blob
        # into the ArcFace bank.
        try:
            new_emb = np.asarray(new_embedding, dtype=np.float32).copy()
        except Exception:
            return False
        if new_emb.size != ACTIVE_FACE_EMBEDDING_DIM:
            logger.debug(
                f"[Identity] auto-enroll dim mismatch for person {person_id}: "
                f"{new_emb.size} != {ACTIVE_FACE_EMBEDDING_DIM}"
            )
            return False

        existing = list(self._face_samples.get(person_id, []))

        # 3. Diversity gate: reject if too similar to any existing sample.
        if existing:
            max_sim = max(_cosine(new_emb, e) for e in existing)
            if max_sim >= SAMPLES_DIVERSITY_THRESHOLD:
                logger.debug(
                    f"[Identity] auto-enroll rejected (near-dup, max_sim="
                    f"{max_sim:.3f}) for person {person_id}"
                )
                return False

        # 4. Below capacity → just add.
        cap = int(SAMPLES_PER_PERSON_MAX)
        if len(existing) < cap:
            await self._persist_face_sample(
                person_id, new_emb,
                pose=(quality_metadata or {}).get("pose", "candid"),
                source="auto",
            )
            self._face_samples.setdefault(person_id, []).append(new_emb)
            logger.info(
                f"[Identity] auto-enrolled face for person {person_id} "
                f"(now {len(existing) + 1}/{cap})"
            )
            return True

        # 5. At capacity → diversity-replacement swap.
        redundant_idx = _most_redundant_index(existing)
        if not _would_increase_diversity(existing, redundant_idx, new_emb):
            logger.debug(
                f"[Identity] auto-enroll rejected (not more diverse than "
                f"most-redundant existing) for person {person_id}"
            )
            return False

        # Find the DB row for the most-redundant in-memory embedding so we
        # can drop it from disk too. Order in self._face_samples mirrors
        # the SELECT order in _reload_caches (no explicit ORDER BY → SQLite
        # default is rowid ASC), so we can map redundant_idx → row id by
        # re-querying. Safer than tracking indices in-band.
        rows = await self._db.fetchall(
            "SELECT id, embedding FROM face_samples "
            "WHERE person_id = ? AND model_version = ? "
            "ORDER BY id ASC",
            (person_id, ACTIVE_FACE_MODEL_VERSION),
        )
        evict_id: Optional[int] = None
        for i, r in enumerate(rows):
            if i == redundant_idx:
                evict_id = int(r["id"])
                break
        if evict_id is not None:
            await self._db.execute(
                "DELETE FROM face_samples WHERE id = ?", (evict_id,)
            )
        # Update in-memory bank.
        replaced = list(existing)
        replaced[redundant_idx] = new_emb
        self._face_samples[person_id] = replaced
        await self._persist_face_sample(
            person_id, new_emb,
            pose=(quality_metadata or {}).get("pose", "candid"),
            source="auto",
        )
        logger.info(
            f"[Identity] auto-enrolled face for person {person_id}: "
            f"swapped most-redundant sample (id={evict_id})"
        )
        return True

    async def _persist_face_sample(
        self,
        person_id: int,
        embedding: np.ndarray,
        pose: str = "candid",
        source: str = "auto",
    ) -> Optional[int]:
        """Single insert path so quality + dim guards live in one place."""
        try:
            return await self._db.execute(
                "INSERT INTO face_samples (person_id, embedding, pose, "
                "captured_at, source, model_version) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    person_id,
                    embedding.astype(np.float32).tobytes(),
                    pose,
                    _now_iso(),
                    source,
                    ACTIVE_FACE_MODEL_VERSION,
                ),
            )
        except Exception as e:
            logger.warning(f"[Identity] face sample persist failed: {e}")
            return None

    @property
    def _merge_candidate_persons(self) -> set[int]:
        """Persons currently flagged as merge candidates (centroid sim
        0.7–0.85). Empty until the merge-candidate subsystem lands;
        consider_new_sample_async pauses enrollment when populated."""
        return getattr(self, "_merge_candidates", set()) or set()

    # ── drift verify (passive, anchored) ────────────────────────────────────

    def take_pending_verify(self, person_id: int) -> Optional[str]:
        """Pop and return the modality the orchestrator should next verify
        for this person, or None if nothing pending."""
        return self._verify_pending.pop(person_id, None)

    async def verify_face(
        self,
        person_id: int,
        frame: np.ndarray,
        single_face_required: bool = True,
    ) -> str:
        """Drift-verify: voice already anchored to person_id, now check the face.

        Returns one of: 'auto_saved', 'pending_drift', 'pending_conflict',
        'no_face', 'multiple_faces'.
        """
        if self._face is None:
            return "no_face"
        # Multi-face guard so we don't attribute the wrong person's face
        if single_face_required:
            n_faces = await asyncio.to_thread(_count_faces, frame)
            if n_faces == 0:
                return "no_face"
            if n_faces > 1:
                return "multiple_faces"
        emb = await self._face.embed_frame(frame)
        if emb is None:
            return "no_face"
        outcome = await self._classify_drift_sample(
            modality="face",
            emb=emb,
            anchor_person_id=person_id,
            anchored_via="voice",
            image_jpeg=_jpeg(frame),
        )
        return outcome

    async def verify_voice(
        self,
        person_id: int,
        audio: np.ndarray,
    ) -> str:
        """Drift-verify: face already anchored to person_id, now check the voice."""
        if self._spk is None:
            return "no_audio"
        emb = await self._spk.embed_audio(audio)
        if emb is None:
            return "no_audio"
        return await self._classify_drift_sample(
            modality="voice",
            emb=emb,
            anchor_person_id=person_id,
            anchored_via="face",
            audio_pcm16=_pcm16(audio),
        )

    async def _classify_drift_sample(
        self,
        modality: str,
        emb: np.ndarray,
        anchor_person_id: int,
        anchored_via: str,
        image_jpeg: Optional[bytes] = None,
        audio_pcm16: Optional[bytes] = None,
    ) -> str:
        """Anchor says this sample is from `anchor_person_id`. Decide whether
        we trust it enough to auto-save, or send it to the dashboard queue."""
        thr = self._th[modality]
        samples = self._voice_samples if modality == "voice" else self._face_samples
        # Check the sample's similarity to the anchored person
        anchor_embs = samples.get(anchor_person_id) or []
        anchor_sim = max((_cosine(emb, e) for e in anchor_embs), default=0.0)

        # Find best alternate match (in case we're actually a different person)
        best_other_pid: Optional[int] = None
        best_other_sim: float = -1.0
        for pid, embs in samples.items():
            if pid == anchor_person_id or not embs:
                continue
            s = max(_cosine(emb, e) for e in embs)
            if s > best_other_sim:
                best_other_sim, best_other_pid = s, pid

        # Conflict: another person matches above THEIR threshold strongly
        if best_other_sim >= thr["match"] and best_other_sim > anchor_sim + 0.05:
            await self._write_pending(
                kind=f"{modality}_drift",
                person_id=anchor_person_id,
                embedding=emb,
                similarity=anchor_sim,
                anchored_via=anchored_via,
                image_jpeg=image_jpeg,
                audio_pcm16=audio_pcm16,
                cluster_id=None,
            )
            logger.info(
                f"[Identity] {modality} drift conflict — anchor '{self._persons.get(anchor_person_id)}' "
                f"sim={anchor_sim:.2f}, other person sim={best_other_sim:.2f} → pending review"
            )
            return "pending_conflict"

        # Loose match — auto-save as a drift_capture sample
        if anchor_sim >= thr["stranger"]:
            await self._save_sample(
                modality=modality,
                person_id=anchor_person_id,
                emb=emb,
                source="drift_capture",
                image_jpeg=image_jpeg,
            )
            logger.info(
                f"[Identity] {modality} drift auto-saved for "
                f"'{self._persons.get(anchor_person_id)}' (sim={anchor_sim:.2f})"
            )
            return "auto_saved"

        # No loose match — anchor still says it's this person, but the sample
        # is far enough that we want a human to confirm before learning it.
        await self._write_pending(
            kind=f"{modality}_drift",
            person_id=anchor_person_id,
            embedding=emb,
            similarity=anchor_sim,
            anchored_via=anchored_via,
            image_jpeg=image_jpeg,
            audio_pcm16=audio_pcm16,
            cluster_id=None,
        )
        logger.info(
            f"[Identity] {modality} drift far from anchor "
            f"'{self._persons.get(anchor_person_id)}' (sim={anchor_sim:.2f}) → pending review"
        )
        return "pending_drift"

    def _passes_coherence_gate(
        self, person_id: int, emb: np.ndarray, source: str
    ) -> bool:
        """A passively-captured face sample is kept only if it agrees with
        the person's existing bank — cosine to the bank mean must clear
        _COHERENCE_FLOOR. Skipped for deliberate enroll/migration and for
        cold-start (< _COHERENCE_MIN_CORE samples — no core to judge yet)."""
        if source in ("enroll", "migration"):
            return True
        existing = self._face_samples.get(person_id, [])
        if len(existing) < _COHERENCE_MIN_CORE:
            return True  # cold start — no core to judge against yet
        mean = np.mean(np.stack(existing), axis=0)
        ea = emb.astype(np.float32)
        na = float(np.linalg.norm(ea))
        nm = float(np.linalg.norm(mean))
        if na <= 0.0 or nm <= 0.0:
            return True
        sim = float(np.dot(ea, mean) / (na * nm))
        if sim < _COHERENCE_FLOOR:
            logger.info(
                f"[Identity] coherence gate rejected a '{source}' face "
                f"sample for person {person_id} "
                f"(sim {sim:.2f} < {_COHERENCE_FLOOR})"
            )
            return False
        return True

    async def _save_sample(
        self,
        modality: str,
        person_id: int,
        emb: np.ndarray,
        source: str,
        pose: Optional[str] = None,
        prompt_id: Optional[str] = None,
        image_jpeg: Optional[bytes] = None,
    ) -> None:
        """Single insert path for face + voice samples. Enforces the
        per-person cap by evicting the most-redundant existing sample
        when at capacity — the user explicitly assigned this row
        (live_question / drift_capture / enroll), so we trust it and
        don't reject; we just make room.

        Pre-bump (cap=30) inserts ignored the cap entirely and the
        bank grew unbounded — large pending-cluster resolutions could
        add 50+ samples in one go and bias the centroid toward
        whatever pose was dominant in that cluster."""
        emb_f32 = emb.astype(np.float32)
        if modality == "voice":
            cap = int(VOICE_SAMPLES_PER_PERSON_MAX)
            existing = list(self._voice_samples.get(person_id, []))
            if len(existing) >= cap and len(existing) > 0:
                await self._evict_most_redundant(
                    modality="voice", person_id=person_id, existing=existing,
                )
            await self._db.execute(
                "INSERT INTO voice_samples (person_id, embedding, prompt_id, captured_at, source) "
                "VALUES (?, ?, ?, ?, ?)",
                (person_id, emb_f32.tobytes(), prompt_id or "candid", _now_iso(), source),
            )
            self._voice_samples.setdefault(person_id, []).append(emb_f32)
        else:
            # Coherence gate — a passive capture that doesn't look like
            # the person it's attributed to never enters the bank.
            if not self._passes_coherence_gate(person_id, emb_f32, source):
                return
            cap = int(SAMPLES_PER_PERSON_MAX)
            existing = list(self._face_samples.get(person_id, []))
            if len(existing) >= cap and len(existing) > 0:
                await self._evict_most_redundant(
                    modality="face", person_id=person_id, existing=existing,
                )
            await self._db.execute(
                "INSERT INTO face_samples (person_id, embedding, pose, captured_at, source, image_jpeg, model_version) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    person_id, emb_f32.tobytes(), pose or "candid", _now_iso(),
                    source, image_jpeg, ACTIVE_FACE_MODEL_VERSION,
                ),
            )
            self._face_samples.setdefault(person_id, []).append(emb_f32)

    async def _evict_most_redundant(
        self, modality: str, person_id: int, existing: list[np.ndarray],
    ) -> None:
        """Drop the in-memory + DB row for the highest-redundancy
        sample in this person's bank. Order in self._{face,voice}_samples
        mirrors the SELECT order in _reload_caches (rowid ASC), so we
        re-query by id to find the row matching the redundant index."""
        if not existing:
            return
        idx = _most_redundant_index(existing)
        if modality == "voice":
            rows = await self._db.fetchall(
                "SELECT id FROM voice_samples WHERE person_id = ? "
                "ORDER BY id ASC",
                (person_id,),
            )
            table = "voice_samples"
            cache = self._voice_samples
        else:
            rows = await self._db.fetchall(
                "SELECT id FROM face_samples WHERE person_id = ? "
                "AND model_version = ? ORDER BY id ASC",
                (person_id, ACTIVE_FACE_MODEL_VERSION),
            )
            table = "face_samples"
            cache = self._face_samples
        if idx >= len(rows):
            return
        evict_id = int(rows[idx]["id"])
        await self._db.execute(
            f"DELETE FROM {table} WHERE id = ?", (evict_id,),
        )
        # Drop from in-memory cache too.
        bank = cache.get(person_id, [])
        if idx < len(bank):
            del bank[idx]
        logger.debug(
            f"[Identity] {modality} cap-evict: dropped sample id={evict_id} "
            f"for person {person_id} (most redundant of {len(rows)})"
        )

    async def prune_resolved_pending(self, *, retain_days: int = 14) -> int:
        """Delete identity_pending rows that have already been reviewed
        (resolved IN (1,2)) and are older than `retain_days`.

        Each pending row carries an image_jpeg + audio_pcm16 BLOB; once
        applied or rejected they only bloat the DB (3k+ rows at audit
        time). Unresolved rows (resolved=0) are never touched — they're
        the live review queue. Called by the nightly maintenance pass.

        Returns the number of rows deleted.
        """
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=int(retain_days))
        ).isoformat()
        row = await self._db.fetchone(
            "SELECT COUNT(*) AS n FROM identity_pending "
            "WHERE resolved IN (1, 2) AND captured_at < ?",
            (cutoff,),
        )
        n = int(row["n"]) if row else 0
        if n:
            await self._db.execute(
                "DELETE FROM identity_pending "
                "WHERE resolved IN (1, 2) AND captured_at < ?",
                (cutoff,),
            )
        return n

    async def prune_bank_incoherent(self) -> dict:
        """Nightly bank gardener. Per person, quarantine the lowest-
        centrality face samples — the ones least like the rest of that
        person's bank (drift, misattribution). Conservative + gradual:
        at most _GARDENER_MAX_EVICT per person per run, never below
        _GARDENER_MIN_KEEP, only samples below _GARDENER_OUTLIER_FLOOR.
        On a healthy bank this is a no-op; on a degraded one it lifts
        cohesion a little each night. Quarantine re-tags model_version
        (clean_face_bank.py --undo reverses). Returns {quarantined,
        scanned_persons}."""
        quarantined = 0
        persons = list(self._face_samples.keys())
        for pid in persons:
            embs = self._face_samples.get(pid, [])
            if len(embs) <= _GARDENER_MIN_KEEP:
                continue
            mat = np.stack([
                e / (float(np.linalg.norm(e)) or 1.0) for e in embs
            ])
            n = mat.shape[0]
            sims = mat @ mat.T
            centrality = (sims.sum(axis=1) - 1.0) / (n - 1)
            evict_idx: list[int] = []
            for i in sorted(range(n), key=lambda j: centrality[j]):
                if centrality[i] >= _GARDENER_OUTLIER_FLOOR:
                    break
                if n - len(evict_idx) <= _GARDENER_MIN_KEEP:
                    break
                if len(evict_idx) >= _GARDENER_MAX_EVICT:
                    break
                evict_idx.append(i)
            if not evict_idx:
                continue
            # Cache order mirrors face_samples.id ASC (see _evict_most_
            # redundant). Re-query to map index -> row id.
            rows = await self._db.fetchall(
                "SELECT id FROM face_samples WHERE person_id=? "
                "AND model_version=? ORDER BY id ASC",
                (pid, ACTIVE_FACE_MODEL_VERSION),
            )
            if len(rows) != n:
                continue  # cache/DB out of sync — skip this person
            for i in sorted(evict_idx, reverse=True):
                await self._db.execute(
                    "UPDATE face_samples SET model_version=? WHERE id=?",
                    (QUARANTINE_MODEL_VERSION, int(rows[i]["id"])),
                )
                del embs[i]
                quarantined += 1
        if quarantined:
            logger.info(
                f"[Identity] bank gardener: quarantined {quarantined} "
                f"incoherent face sample(s)"
            )
        return {"quarantined": quarantined, "scanned_persons": len(persons)}

    async def prune_bank_redundancy(
        self,
        person_id: Optional[int] = None,
        modality: str = "face",
        threshold: Optional[float] = None,
    ) -> dict:
        """Harm-based eviction pass: scan a person's (or every
        person's) sample bank and remove near-duplicate rows whose
        pairwise cosine ≥ threshold. Each near-duplicate pair keeps
        the older / first-inserted representative — younger ones are
        more likely to come from drift-capture or live_question paths
        that didn't pass the full quality gates.

        This is the maintenance pass Cole asked for: don't evict when
        not at capacity unless a sample is doing more harm than good.
        High redundancy is the harm signal — duplicates pull the
        centroid toward whatever pose was over-sampled, which is
        exactly the failure mode where margin gates start rejecting
        legitimate matches.
        """
        thr = float(
            threshold
            if threshold is not None
            else PRUNE_REDUNDANCY_THRESHOLD.get(modality, 0.97)
        )
        if modality == "voice":
            cache = self._voice_samples
            table = "voice_samples"
            where_extra = ""
            params_extra: tuple = ()
        else:
            cache = self._face_samples
            table = "face_samples"
            where_extra = " AND model_version = ?"
            params_extra = (ACTIVE_FACE_MODEL_VERSION,)

        targets = [person_id] if person_id is not None else list(cache.keys())
        total_dropped = 0
        per_person: dict[int, int] = {}
        for pid in targets:
            embs = list(cache.get(pid, []))
            if len(embs) < 2:
                continue
            # Find the actual DB ids in the same order as the cache.
            rows = await self._db.fetchall(
                f"SELECT id FROM {table} WHERE person_id = ?{where_extra} "
                "ORDER BY id ASC",
                (pid, *params_extra),
            )
            if len(rows) != len(embs):
                # Cache and DB drifted (rare; rebuild bank in-memory).
                logger.debug(
                    f"[Identity] prune skipped person {pid}: cache/DB "
                    f"row count mismatch ({len(embs)} vs {len(rows)})"
                )
                continue
            row_ids = [int(r["id"]) for r in rows]
            # Greedy near-duplicate pass: walk samples by age (oldest
            # first), drop any later sample whose cosine to a kept
            # one is ≥ threshold. The oldest representative stays.
            kept_indices: list[int] = []
            dropped_ids: list[int] = []
            for i, e in enumerate(embs):
                is_dup = False
                for k in kept_indices:
                    if _cosine(e, embs[k]) >= thr:
                        is_dup = True
                        break
                if is_dup:
                    dropped_ids.append(row_ids[i])
                else:
                    kept_indices.append(i)
            if dropped_ids:
                placeholders = ",".join("?" for _ in dropped_ids)
                await self._db.execute(
                    f"DELETE FROM {table} WHERE id IN ({placeholders})",
                    tuple(dropped_ids),
                )
                cache[pid] = [embs[k] for k in kept_indices]
                per_person[pid] = len(dropped_ids)
                total_dropped += len(dropped_ids)
        if total_dropped:
            logger.info(
                f"[Identity] {modality} bank prune: dropped {total_dropped} "
                f"redundant sample(s) across {len(per_person)} person(s) "
                f"(threshold={thr:.2f})"
            )
        return {
            "modality": modality,
            "threshold": thr,
            "total_dropped": total_dropped,
            "per_person": per_person,
        }

    # ── pending clusters (for unknown samples) ──────────────────────────────

    async def _add_to_pending_cluster(
        self,
        modality: str,
        emb: np.ndarray,
        image_jpeg: Optional[bytes] = None,
        audio_pcm16: Optional[bytes] = None,
        face_bbox: Optional[tuple] = None,
    ) -> None:
        """An unknown sample (below match threshold but possibly a new person)."""
        # Don't pollute pending with garbage — drop samples below stranger floor
        # AND with no plausible alt match to anyone enrolled.
        thr = self._th[modality]
        samples = self._voice_samples if modality == "voice" else self._face_samples
        best_to_enrolled = 0.0
        for embs in samples.values():
            if not embs:
                continue
            s = max(_cosine(emb, e) for e in embs)
            if s > best_to_enrolled:
                best_to_enrolled = s
        # If it's somewhere in the middle (between stranger and match), the
        # match path returned None already; we want to keep these because they
        # look like a real person. If it's WAY below stranger, still capture
        # — we want gradual persona-building. So no early-exit.

        # Try to merge into an existing cluster of the same modality
        merge_thr = PENDING_MERGE_THRESHOLD[modality]
        target_cluster: Optional[int] = None
        best_cluster_sim = -1.0
        for cid, (mod, centroid, _count) in self._pending_clusters.items():
            if mod != modality:
                continue
            s = _cosine(emb, centroid)
            if s > best_cluster_sim:
                best_cluster_sim, target_cluster = s, cid
        if target_cluster is not None and best_cluster_sim >= merge_thr:
            mod, centroid, count = self._pending_clusters[target_cluster]
            new_count = count + 1
            new_centroid = (centroid * count + emb.astype(np.float32)) / new_count
            self._pending_clusters[target_cluster] = (mod, new_centroid, new_count)
            cluster_id = target_cluster
            await self._attach_pending_preview(
                kind=f"pending_cluster_{modality}",
                cluster_id=cluster_id,
                image_jpeg=image_jpeg,
                audio_pcm16=audio_pcm16,
                face_bbox=face_bbox,
            )
            return
        else:
            cluster_id = self._next_cluster_id
            self._next_cluster_id += 1
            self._pending_clusters[cluster_id] = (
                modality,
                emb.astype(np.float32).copy(),
                1,
            )

        await self._write_pending(
            kind=f"pending_cluster_{modality}",
            person_id=None,
            embedding=emb,
            similarity=best_to_enrolled,
            anchored_via=None,
            image_jpeg=image_jpeg,
            audio_pcm16=audio_pcm16,
            cluster_id=cluster_id,
            face_bbox=face_bbox,
        )

    async def _attach_pending_preview(
        self,
        kind: str,
        cluster_id: int,
        image_jpeg: Optional[bytes],
        audio_pcm16: Optional[bytes],
        face_bbox: Optional[tuple] = None,
    ) -> None:
        """Fill a missing preview on an existing unresolved cluster row."""
        changed = False
        if image_jpeg is not None:
            await self._db.execute(
                "UPDATE identity_pending SET image_jpeg = ? "
                "WHERE kind = ? AND cluster_id = ? AND resolved = 0 "
                "AND image_jpeg IS NULL",
                (image_jpeg, kind, cluster_id),
            )
            changed = True
        if face_bbox is not None:
            await self._db.execute(
                "UPDATE identity_pending SET face_bbox = ? "
                "WHERE kind = ? AND cluster_id = ? AND resolved = 0 "
                "AND face_bbox IS NULL",
                (json.dumps(list(face_bbox)), kind, cluster_id),
            )
            changed = True
        if audio_pcm16 is not None:
            await self._db.execute(
                "UPDATE identity_pending SET audio_pcm16 = ? "
                "WHERE kind = ? AND cluster_id = ? AND resolved = 0 "
                "AND audio_pcm16 IS NULL",
                (audio_pcm16, kind, cluster_id),
            )
            changed = True
        if changed and self._broadcast is not None:
            try:
                await self._broadcast({
                    "type": "identity_pending_added",
                    "kind": kind,
                    "cluster_id": cluster_id,
                })
            except Exception as e:
                logger.debug(f"[Identity] pending preview broadcast failed: {e}")

    async def _write_pending(
        self,
        kind: str,
        person_id: Optional[int],
        embedding: np.ndarray,
        similarity: float,
        anchored_via: Optional[str],
        image_jpeg: Optional[bytes],
        audio_pcm16: Optional[bytes],
        cluster_id: Optional[int],
        face_bbox: Optional[tuple] = None,
    ) -> int:
        # Enforce a hard cap on unresolved pending rows. Past the limit
        # we drop the oldest unresolved rows before writing the new one.
        # Prevents the queue from growing unbounded when face captures
        # keep landing just below the merge threshold (Cole's 2355-row
        # incident). Auto-rejects with resolved=2 so the cap doesn't
        # silently lose evidence of what was happening.
        try:
            row = await self._db.fetchone(
                "SELECT COUNT(*) AS n FROM identity_pending WHERE resolved = 0"
            )
            current = int(row["n"]) if row else 0
            if current >= MAX_UNRESOLVED_PENDING:
                drop = max(1, current - MAX_UNRESOLVED_PENDING + 1)
                await self._db.execute(
                    "UPDATE identity_pending SET resolved = 2 "
                    "WHERE id IN ("
                    "  SELECT id FROM identity_pending "
                    "  WHERE resolved = 0 ORDER BY captured_at ASC LIMIT ?"
                    ")",
                    (drop,),
                )
                logger.info(
                    f"[Identity] pending cap hit ({current}/"
                    f"{MAX_UNRESOLVED_PENDING}); auto-rejected {drop} "
                    "oldest unresolved row(s)"
                )
        except Exception as e:
            logger.debug(f"[Identity] pending cap enforcement failed: {e}")
        bbox_json = json.dumps(list(face_bbox)) if face_bbox else None
        pending_id = await self._db.execute(
            "INSERT INTO identity_pending "
            "(kind, person_id, cluster_id, embedding, image_jpeg, audio_pcm16, "
            " similarity, anchored_via, captured_at, resolved, face_bbox) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)",
            (
                kind,
                person_id,
                cluster_id,
                embedding.astype(np.float32).tobytes(),
                image_jpeg,
                audio_pcm16,
                float(similarity),
                anchored_via,
                _now_iso(),
                bbox_json,
            ),
        )
        # Live-refresh the dashboard's pending-review list. Without this the
        # list only repaints on manual reload.
        if self._broadcast is not None:
            try:
                await self._broadcast({
                    "type": "identity_pending_added",
                    "pending_id": int(pending_id),
                    "kind": kind,
                    "person_id": person_id,
                    "cluster_id": cluster_id,
                })
            except Exception as e:
                logger.debug(f"[Identity] pending broadcast failed: {e}")

        # Surface in the dashboard bell. Drift conflicts get a warning severity
        # because a person we know hasn't been recognized cleanly; cold
        # clusters are info-level (just "someone new keeps showing up").
        if self._notifier is not None:
            try:
                modality = "voice" if "voice" in kind else "face"
                if kind.startswith("pending_cluster_"):
                    title = f"New unknown {modality} captured"
                    message = (
                        f"Cluster #{cluster_id} — best match against enrolled people "
                        f"is sim {similarity:.2f}. Open pending review to assign or reject."
                    )
                    severity = "info"
                else:
                    person_name = (
                        self._persons.get(int(person_id))
                        if person_id is not None else "unknown"
                    )
                    title = f"Drift on {person_name} ({modality})"
                    message = (
                        f"{modality.capitalize()} sample didn't loosely match "
                        f"{person_name} (sim {similarity:.2f}, anchored via "
                        f"{anchored_via}). Confirm or reassign in pending review."
                    )
                    severity = "warning"
                await self._notifier.notify(
                    kind=f"identity.{'cluster' if 'cluster' in kind else 'drift'}",
                    title=title,
                    message=message,
                    target_type="pending",
                    target_id=int(pending_id),
                    action="open_pending",
                    severity=severity,
                )
            except Exception as e:
                logger.debug(f"[Identity] notification dispatch failed: {e}")
        return pending_id

    # ── public read API for dashboard / orchestrator ────────────────────────

    async def list_persons(self) -> list[dict]:
        rows = await self._db.fetchall(
            "SELECT id, name, created_at, notes FROM persons ORDER BY name"
        )
        # Pre-fetch which persons have a thumbnail-capable image so the
        # dashboard can decide whether to render an <img> or a placeholder.
        thumb_rows = await self._db.fetchall(
            "SELECT person_id FROM face_samples WHERE image_jpeg IS NOT NULL "
            "GROUP BY person_id"
        )
        has_thumb = {int(r["person_id"]) for r in thumb_rows}
        out = []
        for r in rows:
            pid = int(r["id"])
            out.append({
                "id": pid,
                "name": r["name"],
                "created_at": r["created_at"],
                "notes": r["notes"],
                "face_sample_count": len(self._face_samples.get(pid, [])),
                "voice_sample_count": len(self._voice_samples.get(pid, [])),
                "has_thumbnail": pid in has_thumb,
            })
        return out

    async def list_face_samples(self, person_id: int) -> list[dict]:
        """Per-sample metadata for a person's face_samples, newest first."""
        rows = await self._db.fetchall(
            "SELECT id, pose, captured_at, source, image_jpeg IS NOT NULL AS has_image "
            "FROM face_samples WHERE person_id = ? ORDER BY captured_at DESC",
            (person_id,),
        )
        return [
            {
                "id": int(r["id"]),
                "pose": r["pose"],
                "captured_at": r["captured_at"],
                "source": r["source"],
                "has_image": bool(r["has_image"]),
            }
            for r in rows
        ]

    async def list_voice_samples(self, person_id: int) -> list[dict]:
        """Per-sample metadata for a person's voice_samples, newest first."""
        rows = await self._db.fetchall(
            "SELECT id, prompt_id, captured_at, source FROM voice_samples "
            "WHERE person_id = ? ORDER BY captured_at DESC",
            (person_id,),
        )
        return [
            {
                "id": int(r["id"]),
                "prompt_id": r["prompt_id"],
                "captured_at": r["captured_at"],
                "source": r["source"],
            }
            for r in rows
        ]

    async def get_face_sample_image(self, sample_id: int) -> Optional[bytes]:
        row = await self._db.fetchone(
            "SELECT image_jpeg FROM face_samples WHERE id = ?", (sample_id,)
        )
        return row["image_jpeg"] if row and row["image_jpeg"] else None

    async def get_person_thumbnail(self, person_id: int) -> Optional[bytes]:
        """Return the most-recent face_sample JPEG for a person, or None."""
        row = await self._db.fetchone(
            "SELECT image_jpeg FROM face_samples WHERE person_id = ? "
            "AND image_jpeg IS NOT NULL ORDER BY captured_at DESC LIMIT 1",
            (person_id,),
        )
        return row["image_jpeg"] if row and row["image_jpeg"] else None

    async def delete_face_sample(self, sample_id: int) -> bool:
        row = await self._db.fetchone(
            "SELECT person_id, embedding FROM face_samples WHERE id = ?", (sample_id,)
        )
        if row is None:
            return False
        pid = int(row["person_id"])
        await self._db.execute("DELETE FROM face_samples WHERE id = ?", (sample_id,))
        # Rebuild this person's in-memory cache from DB to drop the embedding
        await self._reload_person_samples(pid)
        return True

    async def delete_voice_sample(self, sample_id: int) -> bool:
        row = await self._db.fetchone(
            "SELECT person_id FROM voice_samples WHERE id = ?", (sample_id,)
        )
        if row is None:
            return False
        pid = int(row["person_id"])
        await self._db.execute("DELETE FROM voice_samples WHERE id = ?", (sample_id,))
        await self._reload_person_samples(pid)
        return True

    async def _reload_person_samples(self, person_id: int) -> None:
        """Reload the in-memory caches for a single person after a sample delete."""
        self._face_samples[person_id] = []
        self._voice_samples[person_id] = []
        for r in await self._db.fetchall(
            "SELECT embedding FROM face_samples WHERE person_id = ?", (person_id,)
        ):
            try:
                emb = np.frombuffer(r["embedding"], dtype=np.float32).copy()
                if emb.size:
                    self._face_samples[person_id].append(emb)
            except Exception:
                continue
        for r in await self._db.fetchall(
            "SELECT embedding FROM voice_samples WHERE person_id = ?", (person_id,)
        ):
            try:
                emb = np.frombuffer(r["embedding"], dtype=np.float32).copy()
                if emb.size:
                    self._voice_samples[person_id].append(emb)
            except Exception:
                continue

    async def list_pending(self) -> list[dict]:
        """Pending review items (drift conflicts + unknown clusters).
        For cluster rows we also compute an unconstrained top-match
        suggestion so the dashboard dropdown can pre-select the most
        likely person — saves the user from picking blindly when the
        face is borderline."""
        rows = await self._db.fetchall(
            "SELECT id, kind, person_id, cluster_id, embedding, similarity, "
            "anchored_via, captured_at, face_bbox, "
            "image_jpeg IS NOT NULL AS has_image, "
            "audio_pcm16 IS NOT NULL AS has_audio "
            "FROM identity_pending WHERE resolved = 0 "
            "AND NOT (kind = 'pending_cluster_face' AND image_jpeg IS NULL) "
            "AND NOT (kind = 'pending_cluster_voice' AND audio_pcm16 IS NULL) "
            "ORDER BY captured_at DESC"
        )
        out: list[dict] = []
        for r in rows:
            pid = r["person_id"]
            suggested_id: Optional[int] = None
            suggested_name: Optional[str] = None
            suggested_sim: Optional[float] = None
            kind = r["kind"] or ""
            if pid is None and kind.startswith("pending_cluster_"):
                # Cluster row — no anchor person. Score the row's
                # embedding against the live centroid bank for the
                # right modality and surface the top hit.
                blob = r["embedding"]
                if blob:
                    try:
                        emb = np.frombuffer(blob, dtype=np.float32)
                        modality = "voice" if "voice" in kind else "face"
                        # Expected dim guard — voice (256 ECAPA) vs face
                        # (512 ArcFace). Skip suggestion if the blob's
                        # shape is wrong rather than silently misranking.
                        expected = (
                            256 if modality == "voice"
                            else ACTIVE_FACE_EMBEDDING_DIM
                        )
                        if emb.size == expected:
                            top = self._top_match_unconstrained(modality, emb)
                            if top is not None:
                                suggested_id, suggested_sim = top
                                suggested_name = self._persons.get(
                                    suggested_id
                                )
                    except Exception as e:
                        logger.debug(
                            f"[Identity] pending suggestion failed for "
                            f"row {r['id']}: {e}"
                        )
            face_bbox_raw = r["face_bbox"]
            face_bbox = None
            if face_bbox_raw:
                try:
                    face_bbox = json.loads(face_bbox_raw)
                except (ValueError, TypeError):
                    face_bbox = None
            out.append({
                "id": int(r["id"]),
                "kind": kind,
                "person_id": int(pid) if pid is not None else None,
                "person_name": self._persons.get(int(pid)) if pid is not None else None,
                "cluster_id": r["cluster_id"],
                "similarity": r["similarity"],
                "anchored_via": r["anchored_via"],
                "captured_at": r["captured_at"],
                "has_image": bool(r["has_image"]),
                "has_audio": bool(r["has_audio"]),
                "face_bbox": face_bbox,
                "suggested_person_id": suggested_id,
                "suggested_person_name": suggested_name,
                "suggested_similarity": suggested_sim,
            })
        return out

    async def get_pending_image(self, pending_id: int) -> Optional[bytes]:
        row = await self._db.fetchone(
            "SELECT image_jpeg FROM identity_pending WHERE id = ?", (pending_id,)
        )
        return row["image_jpeg"] if row and row["image_jpeg"] else None

    async def get_pending_audio(self, pending_id: int) -> Optional[bytes]:
        row = await self._db.fetchone(
            "SELECT audio_pcm16 FROM identity_pending WHERE id = ?", (pending_id,)
        )
        return row["audio_pcm16"] if row and row["audio_pcm16"] else None

    async def resolve_pending(
        self,
        pending_id: int,
        action: str,
        target_name: Optional[str] = None,
    ) -> bool:
        """Resolve a pending row.

        action='confirm'   → drift case: anchor was right, persist as drift_capture
        action='assign'    → cluster case: assign to a person (existing or new
                             via target_name)
        action='reject'    → mark resolved=2, do nothing
        """
        row = await self._db.fetchone(
            "SELECT * FROM identity_pending WHERE id = ?", (pending_id,)
        )
        if row is None:
            return False
        if int(row["resolved"]) != 0:
            return False

        kind = row["kind"]
        emb = np.frombuffer(row["embedding"], dtype=np.float32).copy()
        modality = "voice" if "voice" in kind else "face"

        # Auto-dismiss any "pending review" notifications pointing at
        # this id. Resolving the underlying review means the user has
        # already taken action — leaving a stale alert in the bell is
        # noise.
        async def _dismiss_notifs(pid: int) -> None:
            n = self._notifier
            if n is None:
                return
            try:
                # When the resolution involves a cluster, every row in
                # that cluster also gets resolved — sweep their alerts
                # too so 8 cluster siblings don't leave 8 stale rows.
                cid = row["cluster_id"] if "cluster_id" in row.keys() else None
                if cid is not None:
                    sibling_rows = await self._db.fetchall(
                        "SELECT id FROM identity_pending WHERE cluster_id = ?",
                        (cid,),
                    )
                    sibling_ids = [int(r["id"]) for r in sibling_rows]
                    if sibling_ids:
                        await n.dismiss_for_targets("pending", sibling_ids)
                        return
            except Exception:
                pass
            await n.dismiss_for_target("pending", pid)

        if action == "reject":
            await self._db.execute(
                "UPDATE identity_pending SET resolved = 2 WHERE id = ?", (pending_id,)
            )
            await _dismiss_notifs(pending_id)
            return True

        # Image bytes (if any) attached to this pending row — preserved on
        # the resulting face_sample so dashboard thumbnails work even for
        # samples that came from the pending queue.
        try:
            row_image = row["image_jpeg"]
        except (IndexError, KeyError):
            row_image = None

        # All non-reject branches converge on resolved=1 below; the
        # _dismiss_notifs helper is also called below to sweep alerts
        # tied to this pending id and any cluster siblings.

        if action == "confirm":
            # Drift confirm: row already has person_id
            pid_raw = row["person_id"]
            if pid_raw is None:
                return False
            pid = int(pid_raw)
            await self._save_sample(
                modality=modality,
                person_id=pid,
                emb=emb,
                source="drift_capture",
                image_jpeg=row_image,
            )
            await self._db.execute(
                "UPDATE identity_pending SET resolved = 1 WHERE id = ?", (pending_id,)
            )
            await _dismiss_notifs(pending_id)
            return True

        if action == "assign":
            if not target_name:
                return False
            pid = await self.ensure_person(target_name)
            # Quality gate: don't pollute the centroid bank with samples
            # too small to be reliable. The face bbox lets us compute a
            # cheap area check; full pose/blur gates would require
            # re-running InsightFace on the JPEG (skipped for cost).
            primary_bbox = None
            try:
                primary_bbox = json.loads(row["face_bbox"]) if row["face_bbox"] else None
            except Exception:
                primary_bbox = None
            if not _passes_pending_area_gate(primary_bbox):
                logger.info(
                    f"[Identity] pending {pending_id} sample skipped: "
                    f"face area below gate"
                )
            else:
                await self._save_sample(
                    modality=modality,
                    person_id=pid,
                    emb=emb,
                    source="live_question",
                    image_jpeg=row_image,
                )
            # Resolve all pending rows in the same cluster as well —
            # cluster rows that fail the quality gate still get the
            # `resolved=1` marker so they exit the queue (they were
            # already in this cluster; rejecting only the bad ones
            # in-place keeps the centroid bank clean while preventing
            # the user from seeing them again).
            cid = row["cluster_id"]
            if cid is not None:
                cluster_rows = await self._db.fetchall(
                    "SELECT id, embedding, image_jpeg, face_bbox FROM identity_pending "
                    "WHERE cluster_id = ? AND resolved = 0",
                    (cid,),
                )
                for cr in cluster_rows:
                    cemb = np.frombuffer(cr["embedding"], dtype=np.float32).copy()
                    if int(cr["id"]) == pending_id:
                        continue
                    try:
                        cr_image = cr["image_jpeg"]
                    except (IndexError, KeyError):
                        cr_image = None
                    cr_bbox = None
                    try:
                        cr_bbox = json.loads(cr["face_bbox"]) if cr["face_bbox"] else None
                    except Exception:
                        cr_bbox = None
                    if not _passes_pending_area_gate(cr_bbox):
                        continue  # mark resolved below, but skip save
                    await self._save_sample(
                        modality=modality,
                        person_id=pid,
                        emb=cemb,
                        source="live_question",
                        image_jpeg=cr_image,
                    )
                await self._db.execute(
                    "UPDATE identity_pending SET resolved = 1 WHERE cluster_id = ? AND resolved = 0",
                    (cid,),
                )
                # Drop the in-memory cluster (it's now a real person)
                self._pending_clusters.pop(int(cid), None)
            else:
                await self._db.execute(
                    "UPDATE identity_pending SET resolved = 1 WHERE id = ?",
                    (pending_id,),
                )
            await _dismiss_notifs(pending_id)
            return True

        return False

    async def resolve_pending_bulk(
        self,
        pending_ids: list[int],
        action: str,
        target_name: Optional[str] = None,
    ) -> dict:
        """Bulk variant of resolve_pending. Returns a summary
        {"ok": int, "skipped_quality": int, "failed": int, "ids": [...]}.
        Used by the dashboard's Pending Reviews tab to drain large
        backlogs efficiently. Same semantics as the singular call for
        each row — cluster-cascade still applies on assigns."""
        out = {"ok": 0, "skipped_quality": 0, "failed": 0, "ids": []}
        for pid in pending_ids:
            try:
                before_face_samples = sum(
                    len(v) for v in self._face_samples.values()
                )
                ok = await self.resolve_pending(pid, action, target_name)
                after_face_samples = sum(
                    len(v) for v in self._face_samples.values()
                )
                if not ok:
                    out["failed"] += 1
                    continue
                # If we assigned but no new sample landed, it was
                # quality-gated. Surface so the UI can show "150 done,
                # 12 skipped (too small)".
                if (action == "assign"
                        and after_face_samples == before_face_samples):
                    out["skipped_quality"] += 1
                else:
                    out["ok"] += 1
                out["ids"].append(int(pid))
            except Exception as e:
                logger.warning(
                    f"[Identity] bulk resolve {pid} failed: {e}"
                )
                out["failed"] += 1
        return out

    async def reject_all_unresolved_pending(self) -> int:
        """Mark every unresolved pending row as resolved=2 (rejected).
        Nuclear option to clear a runaway pending queue. Returns the
        count of rows affected."""
        # Snapshot the affected ids BEFORE the UPDATE so we can sweep
        # their notifications too (the notifications table doesn't
        # have a CASCADE — they'd otherwise stay in the bell).
        rows = await self._db.fetchall(
            "SELECT id FROM identity_pending WHERE resolved = 0"
        )
        n = len(rows)
        if n == 0:
            return 0
        affected_ids = [int(r["id"]) for r in rows]
        await self._db.execute(
            "UPDATE identity_pending SET resolved = 2 WHERE resolved = 0"
        )
        if self._notifier is not None:
            try:
                await self._notifier.dismiss_for_targets("pending", affected_ids)
            except Exception as e:
                logger.debug(f"[Identity] reject_all notif dismiss failed: {e}")
        logger.info(f"[Identity] auto-rejected {n} unresolved pending rows")
        return n

    async def collapse_pending_duplicates(
        self, modality: str = "face", min_sim: float = 0.35,
    ) -> dict:
        """Sweep unresolved pending rows of the given modality and
        collapse similar ones into single cluster representatives.
        Keeps the row with the highest similarity score against the
        live centroid bank as the cluster rep; marks the rest
        resolved=2. Returns {kept, rejected}. Cheap O(N^2) — fine for
        N in the low thousands."""
        kind = f"pending_cluster_{modality}"
        rows = await self._db.fetchall(
            "SELECT id, embedding, similarity, captured_at, image_jpeg IS NOT NULL AS has_image "
            "FROM identity_pending "
            "WHERE kind = ? AND resolved = 0 "
            "ORDER BY captured_at ASC",
            (kind,),
        )
        if not rows:
            return {"kept": 0, "rejected": 0, "scanned": 0}
        expected = (
            ACTIVE_FACE_EMBEDDING_DIM if modality == "face" else 256
        )
        items: list[tuple[int, np.ndarray, float, int]] = []
        for r in rows:
            try:
                emb = np.frombuffer(r["embedding"], dtype=np.float32)
                if emb.size != expected:
                    continue
                items.append((
                    int(r["id"]),
                    emb,
                    float(r["similarity"] or 0.0),
                    int(r["has_image"]),
                ))
            except Exception:
                continue
        # Greedy clustering: walk newest-best-first; each item either
        # joins an existing cluster (cosine >= min_sim to rep) or
        # spawns a new one with itself as rep.
        # Sort by has_image DESC then similarity DESC so the
        # highest-quality preview is kept as the rep.
        items.sort(key=lambda t: (-t[3], -t[2]))
        reps: list[tuple[int, np.ndarray]] = []
        kill: list[int] = []
        for pid, emb, _sim, _has_img in items:
            joined = False
            for rep_id, rep_emb in reps:
                if _cosine(emb, rep_emb) >= min_sim:
                    kill.append(pid)
                    joined = True
                    break
            if not joined:
                reps.append((pid, emb))
        if kill:
            placeholders = ",".join("?" for _ in kill)
            await self._db.execute(
                f"UPDATE identity_pending SET resolved = 2 "
                f"WHERE id IN ({placeholders})",
                tuple(kill),
            )
            # Auto-dismiss the corresponding notifications too —
            # collapsing duplicates fires the same UX intent as a
            # bulk-reject (the user no longer has these to triage).
            if self._notifier is not None:
                try:
                    await self._notifier.dismiss_for_targets("pending", kill)
                except Exception as e:
                    logger.debug(
                        f"[Identity] collapse notif dismiss failed: {e}"
                    )
        logger.info(
            f"[Identity] collapsed pending {modality}: "
            f"kept {len(reps)} reps, rejected {len(kill)} dupes "
            f"out of {len(items)} scanned"
        )
        return {
            "kept": len(reps),
            "rejected": len(kill),
            "scanned": len(items),
        }

    async def delete_person(self, person_id: int) -> bool:
        """Delete a person and (via FK ON DELETE CASCADE) all their samples."""
        await self._db.execute("DELETE FROM persons WHERE id = ?", (person_id,))
        self._persons.pop(person_id, None)
        self._voice_samples.pop(person_id, None)
        self._face_samples.pop(person_id, None)
        return True

    async def rename_person(self, person_id: int, new_name: str) -> bool:
        try:
            await self._db.execute(
                "UPDATE persons SET name = ? WHERE id = ?", (new_name, person_id)
            )
            self._persons[person_id] = new_name
            return True
        except Exception as e:
            logger.warning(f"[Identity] rename_person failed: {e}")
            return False


# ── helpers (sample preview encoding) ───────────────────────────────────────

def _jpeg(frame: np.ndarray) -> Optional[bytes]:
    """BGR frame → JPEG bytes for dashboard preview."""
    try:
        import cv2  # local import — cv2 may not be available at import time
    except Exception:
        return None
    try:
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return buf.tobytes() if ok else None
    except Exception:
        return None


def _pcm16(audio: np.ndarray) -> Optional[bytes]:
    """float32 [-1,1] mono → int16 PCM bytes for dashboard preview playback.
    Truncates to ~3s at 16kHz to keep DB rows small."""
    if audio is None:
        return None
    try:
        max_samples = 16000 * 3
        clip = audio[:max_samples]
        clip = np.clip(clip, -1.0, 1.0)
        return (clip * 32767.0).astype(np.int16).tobytes()
    except Exception:
        return None


def _count_faces(frame: np.ndarray) -> int:
    """Quick face count via deepface's extract_faces. Returns 0 on any error."""
    try:
        from deepface import DeepFace
        faces = DeepFace.extract_faces(
            frame,
            detector_backend="opencv",
            enforce_detection=False,
            align=False,
        )
        # extract_faces returns a phantom 'face' for the whole frame when none
        # detected; filter by facial_area.w.
        n = 0
        for f in faces or []:
            if not isinstance(f, dict):
                continue
            fa = f.get("facial_area") or {}
            if not isinstance(fa, dict):
                continue
            if int(fa.get("w", 0)) >= 30 and int(fa.get("h", 0)) >= 30:
                n += 1
        return n
    except Exception:
        return 0
