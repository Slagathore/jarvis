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
import time
from dataclasses import dataclass
from datetime import datetime, timezone
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

# Pending-cluster merge threshold — if a new unknown sample's cosine to an
# existing pending-cluster centroid is at least this, fold into that cluster
# rather than starting a new one.
PENDING_MERGE_THRESHOLD = {
    "voice": 0.65,
    "face":  0.55,
}


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
            await self._db.execute(
                "INSERT INTO face_samples (person_id, embedding, pose, captured_at, source, model_version) "
                "VALUES (?, ?, ?, ?, 'migration', ?)",
                (pid, emb_blob, "candid", _now_iso(), ACTIVE_FACE_MODEL_VERSION),
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
        for r in await self._db.fetchall(
            "SELECT person_id, embedding FROM face_samples WHERE model_version = ?",
            (ACTIVE_FACE_MODEL_VERSION,),
        ):
            pid = int(r["person_id"])
            try:
                emb = np.frombuffer(r["embedding"], dtype=np.float32).copy()
                if emb.size and pid in self._face_samples:
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
        emb = await self._face.embed_frame(frame) if self._face is not None else None
        if emb is None:
            return None
        match = self._match_against("face", emb)
        if match is None:
            await self._add_to_pending_cluster(
                "face", emb, image_jpeg=_jpeg(frame)
            )
        else:
            self._verify_pending[match.person_id] = "voice"
            match.pending_verification = "voice"
        return match

    async def identify_from_embedding_async(
        self, embedding: np.ndarray, modality: str = "face"
    ) -> Optional[PersonMatch]:
        """
        Match an already-extracted embedding against the centroid bank.

        Used by ObservationBuilder when the caller has already cropped the
        face out of the room frame and embedded it via FaceRecognizer.
        Skipping the embed step here means we don't do face detection
        twice (once for the person crop, once inside identify_face).

        Behavior mirrors `identify_face`: pending-cluster write on miss,
        voice-verification scheduling on hit. Differs in that no image is
        attached to the pending row (caller has the crop).
        """
        if embedding is None:
            return None
        match = self._match_against(modality, embedding)
        if match is None:
            await self._add_to_pending_cluster(modality, embedding)
        else:
            other = "voice" if modality == "face" else "face"
            self._verify_pending[match.person_id] = other
            match.pending_verification = other
        return match

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
        sample_id = await self._db.execute(
            "INSERT INTO face_samples (person_id, embedding, pose, captured_at, source, image_jpeg, model_version) "
            "VALUES (?, ?, ?, ?, 'enroll', ?, ?)",
            (
                pid, emb.astype(np.float32).tobytes(), pose, _now_iso(),
                _jpeg(frame), ACTIVE_FACE_MODEL_VERSION,
            ),
        )
        self._face_samples.setdefault(pid, []).append(emb.astype(np.float32))
        logger.info(f"[Identity] Enrolled face for '{name}' (pose={pose}, id={sample_id})")
        return sample_id

    async def enroll_voice(
        self, name: str, audio: np.ndarray, prompt_id: str = "wake"
    ) -> Optional[int]:
        emb = await self._spk.embed_audio(audio) if self._spk is not None else None
        if emb is None:
            logger.warning(f"[Identity] enroll_voice '{name}' — embedding failed")
            return None
        pid = await self.ensure_person(name)
        sample_id = await self._db.execute(
            "INSERT INTO voice_samples (person_id, embedding, prompt_id, captured_at, source) "
            "VALUES (?, ?, ?, ?, 'enroll')",
            (pid, emb.astype(np.float32).tobytes(), prompt_id, _now_iso()),
        )
        self._voice_samples.setdefault(pid, []).append(emb.astype(np.float32))
        logger.info(f"[Identity] Enrolled voice for '{name}' (prompt={prompt_id}, id={sample_id})")
        return sample_id

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
        emb_f32 = emb.astype(np.float32)
        if modality == "voice":
            await self._db.execute(
                "INSERT INTO voice_samples (person_id, embedding, prompt_id, captured_at, source) "
                "VALUES (?, ?, ?, ?, ?)",
                (person_id, emb_f32.tobytes(), prompt_id or "candid", _now_iso(), source),
            )
            self._voice_samples.setdefault(person_id, []).append(emb_f32)
        else:
            await self._db.execute(
                "INSERT INTO face_samples (person_id, embedding, pose, captured_at, source, image_jpeg, model_version) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    person_id, emb_f32.tobytes(), pose or "candid", _now_iso(),
                    source, image_jpeg, ACTIVE_FACE_MODEL_VERSION,
                ),
            )
            self._face_samples.setdefault(person_id, []).append(emb_f32)

    # ── pending clusters (for unknown samples) ──────────────────────────────

    async def _add_to_pending_cluster(
        self,
        modality: str,
        emb: np.ndarray,
        image_jpeg: Optional[bytes] = None,
        audio_pcm16: Optional[bytes] = None,
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
        )

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
    ) -> int:
        pending_id = await self._db.execute(
            "INSERT INTO identity_pending "
            "(kind, person_id, cluster_id, embedding, image_jpeg, audio_pcm16, "
            " similarity, anchored_via, captured_at, resolved) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
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
        """Pending review items (drift conflicts + unknown clusters)."""
        rows = await self._db.fetchall(
            "SELECT id, kind, person_id, cluster_id, similarity, anchored_via, "
            "captured_at, image_jpeg IS NOT NULL AS has_image, "
            "audio_pcm16 IS NOT NULL AS has_audio "
            "FROM identity_pending WHERE resolved = 0 ORDER BY captured_at DESC"
        )
        out: list[dict] = []
        for r in rows:
            pid = r["person_id"]
            out.append({
                "id": int(r["id"]),
                "kind": r["kind"],
                "person_id": int(pid) if pid is not None else None,
                "person_name": self._persons.get(int(pid)) if pid is not None else None,
                "cluster_id": r["cluster_id"],
                "similarity": r["similarity"],
                "anchored_via": r["anchored_via"],
                "captured_at": r["captured_at"],
                "has_image": bool(r["has_image"]),
                "has_audio": bool(r["has_audio"]),
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

        if action == "reject":
            await self._db.execute(
                "UPDATE identity_pending SET resolved = 2 WHERE id = ?", (pending_id,)
            )
            return True

        # Image bytes (if any) attached to this pending row — preserved on
        # the resulting face_sample so dashboard thumbnails work even for
        # samples that came from the pending queue.
        try:
            row_image = row["image_jpeg"]
        except (IndexError, KeyError):
            row_image = None

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
            return True

        if action == "assign":
            if not target_name:
                return False
            pid = await self.ensure_person(target_name)
            await self._save_sample(
                modality=modality,
                person_id=pid,
                emb=emb,
                source="live_question",
                image_jpeg=row_image,
            )
            # Resolve all pending rows in the same cluster as well
            cid = row["cluster_id"]
            if cid is not None:
                cluster_rows = await self._db.fetchall(
                    "SELECT id, embedding, image_jpeg FROM identity_pending "
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
            return True

        return False

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
