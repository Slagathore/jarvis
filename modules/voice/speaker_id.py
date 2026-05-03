"""
JARVIS — Ambient Home AI
========================
Mission: Identify who is speaking using voice embeddings (Resemblyzer's
         pretrained ECAPA-style encoder). After STT transcribes a wake-word
         capture, the same audio gets passed here to derive a 256-dim speaker
         embedding which we cosine-compare against enrolled speakers' centroid
         embeddings stored in the SQLite `speakers` table.

         Enrollment: pass an audio clip + name; first sample becomes the
         centroid, subsequent samples for the same name are averaged in.

         Inference: the closest enrolled speaker above similarity_threshold
         is returned. None if no enrolled speaker passes threshold.

Modules: modules/voice/speaker_id.py
Classes: SpeakerIdentifier
Functions:
    SpeakerIdentifier.__init__(db, threshold)         — Wire DB, set match threshold
    SpeakerIdentifier.load()                          — Load encoder + cache enrolled
    SpeakerIdentifier.enroll(name, audio)             — Add an enrollment sample
    SpeakerIdentifier.identify(audio)                 — Return (name|None, similarity)
    SpeakerIdentifier.list_enrolled()                 — Names + sample counts
    SpeakerIdentifier.delete(name)                    — Remove an enrollment
"""

import asyncio
import sys
import types
from typing import Any, Optional

import numpy as np
from loguru import logger

# Resemblyzer imports webrtcvad at module load even though we don't use its
# VAD-based preprocessing (record_until_silence already trims). webrtcvad
# doesn't have a Windows wheel, so we shim it before the import.
if "webrtcvad" not in sys.modules:
    sys.modules["webrtcvad"] = types.ModuleType("webrtcvad")

# Encoder runs at 16kHz mono — same as our STT/wake pipeline, so audio passed
# in from those paths needs no resampling.
RESEMBLYZER_SAMPLE_RATE = 16000

# Cosine similarity threshold above which a speaker is considered identified.
# 0.75 is conservative — Resemblyzer's authors suggest 0.65-0.75 for verification
# and 0.7+ for closed-set identification.
DEFAULT_SIMILARITY_THRESHOLD = 0.75


class SpeakerIdentifier:
    """
    Async wrapper around Resemblyzer's VoiceEncoder for speaker identification.

    Enrolled speakers are stored in the SQLite `speakers` table. Each row holds
    the running centroid of all enrollment samples for that name.
    """

    def __init__(self, db, threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> None:
        self._db = db
        self._encoder: Optional[Any] = None
        self._threshold: float = float(threshold)
        # name → 256-dim float32 centroid embedding
        self._known: dict[str, np.ndarray] = {}

    @property
    def is_loaded(self) -> bool:
        return self._encoder is not None

    @property
    def enrolled_count(self) -> int:
        return len(self._known)

    async def load(self) -> None:
        """Load Resemblyzer's pretrained voice encoder and cache enrolled speakers."""
        try:
            from resemblyzer import VoiceEncoder
        except ImportError as e:
            logger.warning(f"[SpeakerID] resemblyzer not installed: {e}")
            return

        try:
            # VoiceEncoder() loads weights — runs in a thread so init isn't blocked
            self._encoder = await asyncio.to_thread(VoiceEncoder)
            await self._refresh_cache()
            logger.info(
                f"[SpeakerID] Ready ({len(self._known)} enrolled: "
                + ", ".join(self._known.keys()) + ")"
            )
        except Exception as e:
            logger.error(f"[SpeakerID] Failed to load voice encoder: {e}")

    async def _refresh_cache(self) -> None:
        """Reload enrolled speaker centroids from the database into memory."""
        try:
            rows = await self._db.fetchall(
                "SELECT name, embedding FROM speakers"
            )
        except Exception as e:
            logger.warning(f"[SpeakerID] Could not load enrolled speakers: {e}")
            return
        self._known.clear()
        for row in rows:
            try:
                emb = np.frombuffer(row["embedding"], dtype=np.float32).copy()
                if emb.size > 0:
                    self._known[row["name"]] = emb
            except Exception as e:
                logger.warning(f"[SpeakerID] Bad embedding for '{row['name']}': {e}")

    async def enroll(self, name: str, audio: np.ndarray) -> bool:
        """
        Add an enrollment sample. If `name` already enrolled, the new sample
        is averaged into the existing centroid (sample_count tracked in DB).
        Audio must be float32 [-1, 1] mono at 16kHz (typical wake-recording output).
        """
        if not self.is_loaded:
            logger.warning("[SpeakerID] Encoder not loaded — enroll skipped")
            return False
        if audio is None or len(audio) < RESEMBLYZER_SAMPLE_RATE:
            logger.warning(
                f"[SpeakerID] Audio too short to enroll ({len(audio) if audio is not None else 0} samples)"
            )
            return False
        try:
            emb = await asyncio.to_thread(self._embed, audio)
        except Exception as e:
            logger.warning(f"[SpeakerID] Embedding failed: {e}")
            return False

        existing = self._known.get(name)
        prior_count = await self._get_sample_count(name) if existing is not None else 0
        new_count = prior_count + 1
        if existing is not None and prior_count > 0:
            centroid = (existing.astype(np.float64) * prior_count + emb.astype(np.float64)) / new_count
            centroid = centroid.astype(np.float32)
        else:
            centroid = emb.astype(np.float32)

        try:
            await self._db.execute(
                "INSERT OR REPLACE INTO speakers (name, embedding, sample_count) VALUES (?, ?, ?)",
                (name, centroid.tobytes(), new_count),
            )
        except Exception as e:
            logger.warning(f"[SpeakerID] DB upsert failed: {e}")
            return False
        self._known[name] = centroid
        logger.info(f"[SpeakerID] Enrolled '{name}' (sample {new_count})")
        return True

    async def identify(self, audio: np.ndarray) -> tuple[Optional[str], float]:
        """
        Return (best-matching name, similarity) or (None, best_sim) if nothing
        passes the threshold. Always returns the best similarity score so the
        caller can log it for tuning.
        """
        if not self.is_loaded or not self._known:
            return None, 0.0
        if audio is None or len(audio) < RESEMBLYZER_SAMPLE_RATE // 2:
            return None, 0.0
        try:
            emb = await asyncio.to_thread(self._embed, audio)
        except Exception as e:
            logger.debug(f"[SpeakerID] identify embedding failed: {e}")
            return None, 0.0

        best_name: Optional[str] = None
        best_sim = -1.0
        for name, ref in self._known.items():
            sim = float(_cosine(emb, ref))
            if sim > best_sim:
                best_name, best_sim = name, sim

        if best_sim >= self._threshold:
            return best_name, best_sim
        return None, best_sim

    async def list_enrolled(self) -> list[dict]:
        """Return a JSON-friendly list of enrolled speakers + sample counts."""
        try:
            rows = await self._db.fetchall(
                "SELECT name, sample_count FROM speakers ORDER BY name"
            )
        except Exception:
            return []
        return [{"name": r["name"], "sample_count": r["sample_count"]} for r in rows]

    async def delete(self, name: str) -> bool:
        """Remove a speaker enrollment by name."""
        try:
            await self._db.execute("DELETE FROM speakers WHERE name = ?", (name,))
        except Exception as e:
            logger.warning(f"[SpeakerID] delete failed: {e}")
            return False
        self._known.pop(name, None)
        logger.info(f"[SpeakerID] Deleted enrollment '{name}'")
        return True

    async def _get_sample_count(self, name: str) -> int:
        try:
            row = await self._db.fetchone(
                "SELECT sample_count FROM speakers WHERE name = ?",
                (name,),
            )
        except Exception:
            return 0
        return int(row["sample_count"]) if row else 0

    def _embed(self, audio: np.ndarray) -> np.ndarray:
        """Synchronous embedding helper — runs in a thread via asyncio.to_thread."""
        if self._encoder is None:
            raise RuntimeError("encoder not loaded")
        # Resemblyzer's embed_utterance expects float32 mono at 16kHz, [-1, 1]
        clean = np.asarray(audio, dtype=np.float32).flatten()
        return self._encoder.embed_utterance(clean)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D vectors. Returns 0 on degenerate input."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
