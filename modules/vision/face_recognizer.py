"""
JARVIS — Ambient Home AI
========================
Mission: Identify who is visible in a camera frame using ArcFace embeddings
         via InsightFace's `buffalo_l` model. World-Model spec §11 chooses
         this over the legacy DeepFace+Facenet stack for three reasons:
            - 512-dim embeddings vs 128-dim → better inter-class separation
            - InsightFace also ships pose (yaw/pitch/roll) + age/gender per
              face, which the auto-enrollment quality gates already want
            - 5-10 ms / face on a CUDA RTX, comparable to Facenet

         The public async surface stays compatible with IdentityManager
         (load/embed_frame/enroll/identify/list_enrolled/delete) so the
         IM contract isn't disturbed. Internals are a clean rewrite
         around InsightFace's FaceAnalysis app.

         Migration discipline:
            - Every embedding this class produces is L2-normalized 512-dim
              float32 — IdentityManager treats them as such.
            - The class advertises its `MODEL_VERSION` so IM can tag every
              face_samples row it writes; old `facenet_v1` rows survive in
              the DB for history but are filtered out of the active
              centroid bank.
            - IM must regenerate centroids from new ArcFace samples. The
              first day or two after this lands, recognition will be sparse
              until 5-10 ArcFace samples per resident accumulate (manual
              enrollment via dashboard, or passive drift capture).

Modules: modules/vision/face_recognizer.py
Classes: FaceRecognizer
Spec:    new 2.md §11 (The ArcFace Upgrade).

#todo: Detect when CUDAExecutionProvider isn't actually wired (e.g. cuDNN
       missing) and fall back to CPU with a logger.warning instead of
       silently using CPU. Today onnxruntime hides that downgrade.
#todo: ArcFace uses input size 112×112; if the largest detected face is
       smaller than that, log it — sub-resolution faces produce noisy
       embeddings that pollute the centroid bank.
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional

import numpy as np
from loguru import logger


# ── Model selection + tunables ──────────────────────────────────────────────
# `buffalo_l` is the standard ArcFace bundle: SCRFD detector + r100 ArcFace
# encoder + pose/age/gender heads. ~250 MB on disk, ~700 MB GPU memory at
# runtime. The first instantiation downloads to ~/.insightface/models/.
_MODEL_NAME = "buffalo_l"
_DET_SIZE = (640, 640)

# Tag stamped on every embedding this class produces. Used by IdentityManager
# to filter face_samples to the active model when computing centroids.
MODEL_VERSION = "arcface_buffalo_l_v1"

# Cosine similarity floor — below this, the match is too weak to count
# (returns "unknown"). Tighter than DeepFace/Facenet's 0.7 because the
# scale of cosines is different on ArcFace; 0.5 is the standard.
SIMILARITY_THRESHOLD = 0.5

# Margin gating: best - second_best must beat this for a confident match.
# Below it the system refuses to guess between two enrolled persons —
# IdentityManager flags ambiguous, the dashboard can surface for review.
MARGIN_THRESHOLD = 0.10

# Minimum face crop size we'll accept for an embedding. ArcFace's input
# is 112×112; anything appreciably smaller is upscaled by the encoder
# and produces noisy embeddings.
_MIN_FACE_AREA = 30 * 30


class FaceRecognizer:
    """
    ArcFace-based face recognition using InsightFace's buffalo_l model.
    Integrates with IdentityManager — provides detection + embedding +
    pose. Identity matching (centroid + margin gating) is IdentityManager's
    job; this class produces the embeddings IdentityManager compares against.
    """

    # Module-level constants exposed on the class for callers that hold a
    # reference but don't want to import the module.
    SIMILARITY_THRESHOLD = SIMILARITY_THRESHOLD
    MARGIN_THRESHOLD = MARGIN_THRESHOLD
    MODEL_VERSION = MODEL_VERSION

    def __init__(
        self,
        db: Any,
        threshold: float = SIMILARITY_THRESHOLD,
        use_gpu: bool = True,
        model_name: str = _MODEL_NAME,
    ) -> None:
        # Keep the same constructor signature as the legacy class so existing
        # call sites in IdentityManager / orchestrator don't change. `threshold`
        # is the legacy single-similarity threshold used by `identify()`;
        # IdentityManager applies its own match/stranger/margin policy on top
        # of `embed_frame()` and ignores this knob.
        self._db = db
        self._threshold: float = float(threshold)
        self._use_gpu = bool(use_gpu)
        self._model_name = model_name
        self._loaded: bool = False
        # Lazy-init: paying the ~700 MB GPU cost only after `load()` is
        # explicitly called keeps cold-import cheap.
        self._app: Optional[Any] = None
        # Centroid cache — populated by `_refresh_cache()` against the
        # legacy `faces` table. Identity v2 reads its samples from
        # `face_samples` directly via IdentityManager; we keep this for
        # the v1-style `identify()` API only.
        self._known: dict[str, np.ndarray] = {}

    # ── Public properties ───────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def enrolled_count(self) -> int:
        return len(self._known)

    @property
    def model_version(self) -> str:
        return MODEL_VERSION

    # ── Lifecycle ───────────────────────────────────────────────────────────

    async def load(self) -> None:
        """
        Build the InsightFace FaceAnalysis app, hydrate the legacy
        centroid cache, and mark loaded. Safe to call concurrently —
        InsightFace's prepare() is synchronous so we run it in a thread.
        """
        try:
            await asyncio.to_thread(self._build_app)
            self._loaded = True
        except Exception as e:
            logger.warning(
                f"[FaceRec] InsightFace load failed: {e} — recognition disabled"
            )
            return
        await self._refresh_cache()
        logger.info(
            f"[FaceRec] Ready ({self._model_name}, "
            f"{len(self._known)} legacy enrolled: "
            + (", ".join(self._known.keys()) or "none") + ")"
        )

    def _build_app(self) -> None:
        """Build the FaceAnalysis app on the calling thread. Heavy GPU work."""
        from insightface.app import FaceAnalysis  # type: ignore[import-untyped]
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self._use_gpu else ["CPUExecutionProvider"]
        )
        app = FaceAnalysis(name=self._model_name, providers=providers)
        # ctx_id=0 = first GPU, -1 = CPU. det_size is the SCRFD input.
        app.prepare(ctx_id=0 if self._use_gpu else -1, det_size=_DET_SIZE)
        self._app = app

    async def _refresh_cache(self) -> None:
        """Hydrate the legacy `faces` table centroids into memory. Used only
        by the v1-style `identify()`/`enroll()` path that some old call sites
        still hit. Identity v2 / IdentityManager doesn't go through this cache.
        """
        try:
            rows = await self._db.fetchall("SELECT name, embedding FROM faces")
        except Exception as e:
            logger.debug(f"[FaceRec] Could not load legacy faces: {e}")
            return
        self._known.clear()
        for row in rows:
            try:
                emb = np.frombuffer(row["embedding"], dtype=np.float32).copy()
                if emb.size > 0:
                    self._known[row["name"]] = emb
            except Exception as e:
                logger.warning(
                    f"[FaceRec] Bad legacy embedding for '{row['name']}': {e}"
                )

    # ── Detection + embedding (the core API IdentityManager hits) ───────────

    async def embed_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Public wrapper: detect the largest face in `frame`, return its
        L2-normalized 512-dim embedding. None if no face detected.

        Returns just the embedding for compatibility with IdentityManager;
        for richer per-detection state (pose, det_score) call
        `embed_largest_face_full()` instead.
        """
        if not self._loaded:
            return None
        result = await asyncio.to_thread(self._embed_largest_face_full, frame)
        return None if result is None else result["embedding"]

    async def embed_largest_face_full(
        self, frame: np.ndarray
    ) -> Optional[dict]:
        """Like `embed_frame` but returns the full detection dict — bbox,
        embedding, det_score, pose (yaw/pitch/roll), age, gender. Used by
        the auto-enrollment quality gate (e.g. reject samples with
        |yaw| > 25° to avoid polluting centroids with profile shots).
        """
        if not self._loaded:
            return None
        return await asyncio.to_thread(self._embed_largest_face_full, frame)

    async def detect_and_embed(self, frame: np.ndarray) -> list[dict]:
        """Return ALL detected faces (not just the largest). Same dict
        shape as `embed_largest_face_full`. Used when the orchestrator
        wants to associate multiple person bboxes with multiple identities
        in the same frame.
        """
        if not self._loaded:
            return []
        return await asyncio.to_thread(self._detect_all, frame)

    # ── Sync helpers (run inside asyncio.to_thread) ─────────────────────────

    def _detect_all(self, frame: np.ndarray) -> list[dict]:
        if frame is None or frame.size == 0 or self._app is None:
            return []
        try:
            faces = self._app.get(frame)
        except Exception as e:
            logger.debug(f"[FaceRec] InsightFace get() failed: {e}")
            return []
        out: list[dict] = []
        for f in faces:
            bbox = tuple(int(c) for c in f.bbox)
            area = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
            if area < _MIN_FACE_AREA:
                # Discard sub-30×30 detections — too small to embed reliably.
                continue
            out.append(_face_to_dict(f, bbox))
        return out

    def _embed_largest_face_full(self, frame: np.ndarray) -> Optional[dict]:
        faces = self._detect_all(frame)
        if not faces:
            return None
        # Largest face wins (closest to camera). Tie-broken arbitrarily.
        return max(
            faces,
            key=lambda f: (f["bbox"][2] - f["bbox"][0])
                          * (f["bbox"][3] - f["bbox"][1]),
        )

    # ── Legacy enroll/identify/list/delete (v1 `faces` table) ──────────────
    # These kept for backwards compatibility — the dashboard's old enroll
    # button and any pre-Identity-v2 call site. Identity v2 / IdentityManager
    # bypasses this and writes to face_samples directly.

    async def enroll(self, name: str, frame: np.ndarray) -> bool:
        """v1 enroll path. Updates the legacy `faces` table centroid + cache."""
        if not self._loaded:
            return False
        emb = await self.embed_frame(frame)
        if emb is None:
            logger.warning(
                f"[FaceRec] No face detected in frame for enroll '{name}'"
            )
            return False

        existing = self._known.get(name)
        prior_count = await self._get_sample_count(name) if existing is not None else 0
        new_count = prior_count + 1
        if existing is not None and prior_count > 0:
            centroid = (
                existing.astype(np.float64) * prior_count + emb.astype(np.float64)
            ) / new_count
            centroid = centroid.astype(np.float32)
        else:
            centroid = emb.astype(np.float32)

        try:
            await self._db.execute(
                "INSERT OR REPLACE INTO faces (name, embedding, sample_count) VALUES (?, ?, ?)",
                (name, centroid.tobytes(), new_count),
            )
        except Exception as e:
            logger.warning(f"[FaceRec] Legacy DB upsert failed: {e}")
            return False
        self._known[name] = centroid
        logger.info(f"[FaceRec] Enrolled '{name}' (sample {new_count})")
        return True

    async def identify(self, frame: np.ndarray) -> tuple[Optional[str], float]:
        """v1 identify path. Returns (name, similarity) for the best match
        above the legacy threshold, else (None, best_sim). (None, 0.0) if
        no face detected at all.
        """
        if not self._loaded or not self._known:
            return None, 0.0
        emb = await self.embed_frame(frame)
        if emb is None:
            return None, 0.0
        best_name: Optional[str] = None
        best_sim = -1.0
        for name, ref in self._known.items():
            sim = self.cosine_similarity(emb, ref)
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
        return [
            {"name": r["name"], "sample_count": r["sample_count"]}
            for r in rows
        ]

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

    # ── Cosine helper (centralized so callers can use the class's metric) ──

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """ArcFace embeddings are already L2-normalized → cosine = dot.
        Falls back to explicit normalization if either input isn't.
        """
        if a is None or b is None or a.size == 0 or b.size == 0:
            return 0.0
        # Detect non-normalized inputs (legacy Facenet centroids in the
        # cache) and normalize on the fly.
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if abs(na - 1.0) < 1e-3 and abs(nb - 1.0) < 1e-3:
            return float(np.dot(a, b))
        if na <= 0.0 or nb <= 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


def _face_to_dict(f: Any, bbox: tuple) -> dict:
    """Convert an InsightFace `Face` object into the plain dict the rest
    of the system passes around. Embedding is `normed_embedding` —
    already L2-normalized, 512-dim float32."""
    pose = getattr(f, "pose", None)  # array [pitch, yaw, roll] or None
    return {
        "bbox": bbox,
        "embedding": np.asarray(f.normed_embedding, dtype=np.float32),
        "det_score": float(f.det_score),
        "yaw":   float(pose[1]) if pose is not None else 0.0,
        "pitch": float(pose[0]) if pose is not None else 0.0,
        "roll":  float(pose[2]) if pose is not None else 0.0,
        "age":     int(f.age) if hasattr(f, "age") and f.age is not None else None,
        "gender":  int(f.gender) if hasattr(f, "gender") and f.gender is not None else None,
    }
