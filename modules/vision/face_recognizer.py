"""
JARVIS — Ambient Home AI
========================
Mission: Identify who is visible in a camera frame using face embeddings via
         the `deepface` library (Facenet model). Like SpeakerIdentifier this
         is centroid-based: each enrolled name has one running average of all
         enrollment-sample embeddings stored in the `faces` SQLite table.
         Live frames get the largest detected face cropped + embedded + cosine-
         compared to enrolled centroids.

         Recognition runs at the vision_loop cadence (default 60s), so this is
         "who is here right now" per minute, not per frame.

Modules: modules/vision/face_recognizer.py
Classes: FaceRecognizer
Functions:
    FaceRecognizer.__init__(db, threshold)         — Wire DB, set match threshold
    FaceRecognizer.load()                          — Warm the deepface model + cache
    FaceRecognizer.enroll(name, frame)             — Enroll a face from a BGR frame
    FaceRecognizer.identify(frame)                 — Return (name|None, similarity)
    FaceRecognizer.list_enrolled()                 — Names + sample counts
    FaceRecognizer.delete(name)                    — Remove an enrollment
"""

import asyncio
from typing import Any, Optional

import numpy as np
from loguru import logger

# Facenet's 128-dim embeddings work well for closed-set identification.
# Cosine threshold tuned slightly conservative — 0.7 corresponds to roughly
# the boundary deepface itself uses for Facenet/cosine in DeepFace.verify().
DEFAULT_SIMILARITY_THRESHOLD = 0.70

# Deepface model name. Facenet is fast, decent accuracy. Could swap to
# "ArcFace" for higher accuracy at the cost of slower model load.
_MODEL_NAME = "Facenet"
_DETECTOR_BACKEND = "opencv"  # fastest; degrades gracefully if no face found


class FaceRecognizer:
    """
    Async wrapper around deepface for face identification in BGR frames.

    Enrolled faces are stored in the SQLite `faces` table. Each row holds
    the running centroid of all enrollment-sample embeddings for that name.
    """

    def __init__(self, db, threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> None:
        self._db = db
        self._threshold: float = float(threshold)
        self._loaded: bool = False
        # name → 128-dim float32 centroid
        self._known: dict[str, np.ndarray] = {}

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def enrolled_count(self) -> int:
        return len(self._known)

    async def load(self) -> None:
        """
        Warm the deepface Facenet model with a tiny dummy call so the first
        real frame doesn't pay the ~5s model-build cost. Hydrate the centroid
        cache from DB.
        """
        try:
            # Trigger model build once
            await asyncio.to_thread(self._warm_model)
            self._loaded = True
        except Exception as e:
            logger.warning(f"[FaceRec] deepface warm-up failed: {e} — recognition disabled")
            return
        await self._refresh_cache()
        logger.info(
            f"[FaceRec] Ready ({len(self._known)} enrolled: "
            + ", ".join(self._known.keys()) + ")"
        )

    @staticmethod
    def _warm_model() -> None:
        """Force deepface to load Facenet weights + initialize TF graph."""
        from deepface import DeepFace
        dummy = (np.zeros((160, 160, 3), dtype=np.uint8))
        # enforce_detection=False so the dummy doesn't fail face detection
        DeepFace.represent(dummy, model_name=_MODEL_NAME, enforce_detection=False)

    async def _refresh_cache(self) -> None:
        """Reload enrolled face centroids from the DB into memory."""
        try:
            rows = await self._db.fetchall("SELECT name, embedding FROM faces")
        except Exception as e:
            logger.warning(f"[FaceRec] Could not load enrolled faces: {e}")
            return
        self._known.clear()
        for row in rows:
            try:
                emb = np.frombuffer(row["embedding"], dtype=np.float32).copy()
                if emb.size > 0:
                    self._known[row["name"]] = emb
            except Exception as e:
                logger.warning(f"[FaceRec] Bad embedding for '{row['name']}': {e}")

    async def enroll(self, name: str, frame: np.ndarray) -> bool:
        """
        Detect the largest face in `frame`, embed it, average into the existing
        centroid for `name` (or seed a new one). Returns False if no face detected.
        """
        if not self._loaded:
            return False
        emb = await asyncio.to_thread(self._embed_frame, frame)
        if emb is None:
            logger.warning(f"[FaceRec] No face detected in frame for enroll '{name}'")
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
                "INSERT OR REPLACE INTO faces (name, embedding, sample_count) VALUES (?, ?, ?)",
                (name, centroid.tobytes(), new_count),
            )
        except Exception as e:
            logger.warning(f"[FaceRec] DB upsert failed: {e}")
            return False
        self._known[name] = centroid
        logger.info(f"[FaceRec] Enrolled '{name}' (sample {new_count})")
        return True

    async def identify(self, frame: np.ndarray) -> tuple[Optional[str], float]:
        """
        Return (name, similarity) for the best match above threshold, or
        (None, best_sim) otherwise. Always returns the best similarity for
        tuning. (None, 0.0) if no face detected at all.
        """
        if not self._loaded or not self._known:
            return None, 0.0
        emb = await asyncio.to_thread(self._embed_frame, frame)
        if emb is None:
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
        try:
            rows = await self._db.fetchall(
                "SELECT name, sample_count FROM faces ORDER BY name"
            )
        except Exception:
            return []
        return [{"name": r["name"], "sample_count": r["sample_count"]} for r in rows]

    async def delete(self, name: str) -> bool:
        try:
            await self._db.execute("DELETE FROM faces WHERE name = ?", (name,))
        except Exception as e:
            logger.warning(f"[FaceRec] delete failed: {e}")
            return False
        self._known.pop(name, None)
        logger.info(f"[FaceRec] Deleted enrollment '{name}'")
        return True

    async def _get_sample_count(self, name: str) -> int:
        try:
            row = await self._db.fetchone(
                "SELECT sample_count FROM faces WHERE name = ?",
                (name,),
            )
        except Exception:
            return 0
        return int(row["sample_count"]) if row else 0

    def _embed_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Sync: detect face(s) in the BGR frame, take the largest, return its
        embedding as float32. None if no face detected.
        Runs in a thread via asyncio.to_thread.
        """
        from deepface import DeepFace
        try:
            # extract_faces returns dicts with 'face' (cropped+aligned), 'facial_area'
            faces = DeepFace.extract_faces(
                frame,
                detector_backend=_DETECTOR_BACKEND,
                enforce_detection=False,
                align=True,
            )
        except Exception as e:
            logger.debug(f"[FaceRec] extract_faces failed: {e}")
            return None
        if not faces:
            return None

        # Largest face wins (closest person to camera most likely)
        def _area(f: dict) -> int:
            fa = f.get("facial_area", {}) or {}
            return int(fa.get("w", 0)) * int(fa.get("h", 0))

        biggest = max(faces, key=_area)
        if _area(biggest) < 30 * 30:
            # Too small to be a real face; opencv detector false-positives on
            # noise crops sometimes. Skip rather than enroll garbage.
            return None

        face_crop = biggest.get("face")
        if face_crop is None:
            return None

        try:
            reps = DeepFace.represent(
                face_crop,
                model_name=_MODEL_NAME,
                enforce_detection=False,
                detector_backend="skip",   # already cropped
            )
        except Exception as e:
            logger.debug(f"[FaceRec] represent failed: {e}")
            return None
        if not reps:
            return None
        emb = np.asarray(reps[0]["embedding"], dtype=np.float32)
        return emb


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
