"""
JARVIS — Ambient Home AI
========================
Mission: Long-term semantic memory layer.

         The conversation_log table is a raw transcript. This is the
         *distilled* memory: facts, preferences, events, and instructions
         that should persist across conversations. Each memory carries an
         importance score (0..1) and a 384-dim embedding so retrieval is
         semantic, not just chronological.

         Three write paths:
           1. extract_from_turn() — LLM-extraction pass after each turn
              automatically pulls out anything worth remembering.
           2. add() — explicit save (e.g. via the 'remember' LLM tool when
              the model decides "I should remember this exactly").
           3. record_thought() — Jarvis's own self-realizations (curiosity
              engine when idle).

         Retrieval injects top-K memories into the prompt context, scored as:
           cosine_sim × importance × recency_decay
         So a high-importance fact beats a low-importance fact at the same
         similarity, and stale memories slowly fade to noise.

Modules: modules/memory/memory_v2.py
Classes: MemoryStore
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
from loguru import logger

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RECENCY_HALF_LIFE_DAYS = 30  # memory's recency weight halves every N days


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _recency_weight(captured_at: Optional[str]) -> float:
    """Exponential decay weight in (0, 1]. Anything < 1 day old is ~1.0;
    very old memories drop to ~0.1."""
    dt = _parse_iso(captured_at)
    if dt is None:
        return 0.5
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0)
    return float(0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass
class Memory:
    id: int
    kind: str
    subject: Optional[str]
    content: str
    importance: float
    created_at: str
    last_accessed_at: Optional[str]
    access_count: int


class MemoryStore:
    """Semantic + importance-scored memory layer.

    Lazy-loads the sentence-transformers encoder on first use (~80 MB
    first-time download, then cached locally). All DB writes are async.
    """

    def __init__(
        self,
        db: Any,
        llm: Optional[Any] = None,
        broadcast: Optional[Any] = None,
    ) -> None:
        self._db = db
        self._llm = llm
        # Async dashboard-broadcast callback. When set, every add/update/delete
        # fires the corresponding 'memory.*' event so the dashboard memory
        # card hot-reloads without waiting for a manual refresh. Without
        # this, auto-extracted memories (the conversation-curator pass after
        # every turn, plus the self-thought loop) silently land in the DB
        # and never appear until the next dashboard reload.
        self._broadcast = broadcast
        self._encoder: Optional[Any] = None
        # In-process cache of (id, embedding) so retrieve() doesn't pay
        # blob deserialization on every call. Rebuilt on add/delete.
        self._cache: list[tuple[int, np.ndarray]] = []
        self._cache_loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        return self._encoder is not None

    async def init(self) -> None:
        """Load encoder + warm the cache. Idempotent."""
        if self._encoder is None:
            try:
                await asyncio.to_thread(self._load_encoder)
            except Exception as e:
                logger.warning(f"[MemoryV2] encoder load failed: {e} — semantic recall disabled")
                return
        await self._reload_cache()
        logger.info(f"[MemoryV2] Ready ({len(self._cache)} memories)")

    def _load_encoder(self) -> None:
        from sentence_transformers import SentenceTransformer
        self._encoder = SentenceTransformer(EMBEDDING_MODEL)

    async def _reload_cache(self) -> None:
        try:
            rows = await self._db.fetchall(
                "SELECT id, embedding FROM memories WHERE archived = 0"
            )
        except Exception as e:
            logger.warning(f"[MemoryV2] cache reload failed: {e}")
            return
        cache: list[tuple[int, np.ndarray]] = []
        for r in rows:
            blob = r["embedding"]
            if not blob:
                continue
            try:
                arr = np.frombuffer(blob, dtype=np.float32).copy()
                if arr.size == EMBEDDING_DIM:
                    cache.append((int(r["id"]), arr))
            except Exception:
                continue
        self._cache = cache
        self._cache_loaded = True

    # ── Embedding ──────────────────────────────────────────────────────────

    async def embed(self, text: str) -> Optional[np.ndarray]:
        if self._encoder is None:
            return None
        try:
            arr = await asyncio.to_thread(
                self._encoder.encode, text, convert_to_numpy=True, show_progress_bar=False
            )
            return arr.astype(np.float32)
        except Exception as e:
            logger.debug(f"[MemoryV2] embed failed: {e}")
            return None

    # ── Add / mutate ────────────────────────────────────────────────────────

    async def add(
        self,
        kind: str,
        content: str,
        subject: Optional[str] = None,
        importance: float = 0.5,
        source_event_id: Optional[int] = None,
        source_kind: Optional[str] = None,
    ) -> Optional[int]:
        """Persist a memory. Embeds the content for semantic retrieval."""
        if not content or not content.strip():
            return None
        emb = await self.embed(content)
        emb_blob = emb.tobytes() if emb is not None else None
        try:
            mid = await self._db.execute(
                "INSERT INTO memories (kind, subject, content, importance, embedding, "
                "source_event_id, source_kind, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    kind, subject, content, float(importance), emb_blob,
                    source_event_id, source_kind, _now_iso(),
                ),
            )
        except Exception as e:
            logger.warning(f"[MemoryV2] add failed: {e}")
            return None
        if emb is not None:
            self._cache.append((int(mid), emb))
        logger.debug(f"[MemoryV2] +{kind} (imp={importance:.2f}): {content[:80]}")
        # Hot-load the dashboard's memory card. Fires for ALL writes:
        # auto-extraction after each turn, the LLM's explicit remember/
        # record_thought/record_question tools, and the self-thought loop.
        if self._broadcast is not None:
            try:
                await self._broadcast({
                    "type": "memory.added",
                    "id": int(mid),
                    "kind": kind,
                    "importance": float(importance),
                    "subject": subject,
                })
            except Exception as e:
                logger.debug(f"[MemoryV2] broadcast failed: {e}")
        return int(mid)

    async def update(
        self,
        memory_id: int,
        content: Optional[str] = None,
        importance: Optional[float] = None,
        kind: Optional[str] = None,
        subject: Optional[str] = None,
    ) -> bool:
        sets = []
        params: list = []
        if content is not None:
            sets.append("content = ?")
            params.append(content)
            emb = await self.embed(content)
            if emb is not None:
                sets.append("embedding = ?")
                params.append(emb.tobytes())
        if importance is not None:
            sets.append("importance = ?")
            params.append(float(importance))
        if kind is not None:
            sets.append("kind = ?")
            params.append(kind)
        if subject is not None:
            sets.append("subject = ?")
            params.append(subject)
        if not sets:
            return False
        params.append(memory_id)
        try:
            await self._db.execute(
                f"UPDATE memories SET {', '.join(sets)} WHERE id = ?", tuple(params)
            )
            await self._reload_cache()
        except Exception as e:
            logger.warning(f"[MemoryV2] update failed: {e}")
            return False
        if self._broadcast is not None:
            try:
                await self._broadcast({"type": "memory.updated", "id": int(memory_id)})
            except Exception:
                pass
        return True

    async def delete(self, memory_id: int) -> bool:
        try:
            await self._db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            await self._reload_cache()
        except Exception as e:
            logger.warning(f"[MemoryV2] delete failed: {e}")
            return False
        if self._broadcast is not None:
            try:
                await self._broadcast({"type": "memory.deleted", "id": int(memory_id)})
            except Exception:
                pass
        return True

    async def archive(self, memory_id: int) -> bool:
        try:
            await self._db.execute(
                "UPDATE memories SET archived = 1 WHERE id = ?", (memory_id,)
            )
            await self._reload_cache()
        except Exception:
            return False
        if self._broadcast is not None:
            try:
                await self._broadcast({"type": "memory.deleted", "id": int(memory_id)})
            except Exception:
                pass
        return True

    # ── Retrieval ───────────────────────────────────────────────────────────

    async def retrieve(self, query: str, k: int = 8) -> list[dict]:
        """Top-K semantic search.

        Score = cosine(q, mem) × importance × recency_decay
        Returns the most relevant non-archived memories with full metadata.
        Touches last_accessed_at + access_count on returned rows.
        """
        if not self._cache:
            await self._reload_cache()
        if not self._cache:
            return []
        q_emb = await self.embed(query)
        if q_emb is None:
            return []

        # Pull metadata in one go for the cached IDs
        ids = [i for i, _ in self._cache]
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = await self._db.fetchall(
            f"SELECT id, kind, subject, content, importance, created_at, "
            f"last_accessed_at, access_count "
            f"FROM memories WHERE id IN ({placeholders}) AND archived = 0",
            tuple(ids),
        )
        meta = {int(r["id"]): r for r in rows}

        scored: list[tuple[float, int]] = []
        for mid, emb in self._cache:
            r = meta.get(mid)
            if r is None:
                continue
            cos = _cosine(q_emb, emb)
            if cos <= 0.0:
                continue
            imp = float(r["importance"] or 0.5)
            rec = _recency_weight(r["created_at"])
            score = cos * (0.5 + 0.5 * imp) * (0.5 + 0.5 * rec)
            scored.append((score, mid))

        scored.sort(key=lambda t: t[0], reverse=True)
        top = scored[:k]
        if not top:
            return []
        # Touch access stats
        now = _now_iso()
        for _score, mid in top:
            try:
                await self._db.execute(
                    "UPDATE memories SET last_accessed_at = ?, access_count = access_count + 1 "
                    "WHERE id = ?",
                    (now, mid),
                )
            except Exception:
                pass

        out: list[dict] = []
        for score, mid in top:
            r = meta.get(mid)
            if r is None:
                continue
            out.append({
                "id": int(r["id"]),
                "kind": r["kind"],
                "subject": r["subject"],
                "content": r["content"],
                "importance": float(r["importance"] or 0.5),
                "created_at": r["created_at"],
                "score": float(score),
            })
        return out

    # ── LLM-driven extraction ──────────────────────────────────────────────

    async def extract_from_turn(
        self,
        user_text: str,
        assistant_text: str,
        room: Optional[str] = None,
        source_event_id: Optional[int] = None,
    ) -> list[int]:
        """After each conversation turn, ask the LLM what's worth remembering.

        Returns the list of memory IDs created. Errors are swallowed — this
        runs as a fire-and-forget background task; failure shouldn't block
        the user-facing response.
        """
        if self._llm is None:
            return []
        prompt = (
            "You are Jarvis's memory curator. Decide what (if anything) from "
            "this conversation turn should be remembered for future "
            "interactions with Cole.\n\n"
            "Reply with ONLY a JSON array. Each entry: {kind, subject, content, importance}.\n"
            "  kind:       'fact' (objective info), 'preference' (likes/dislikes), "
            "              'event' (something that happened), 'instruction' (how Cole wants "
            "              you to behave). Use lowercase.\n"
            "  subject:    person name, room, project, or topic the memory relates to. "
            "              Optional — null if generic.\n"
            "  content:    the memory itself, written as a single declarative sentence "
            "              (NOT 'Cole said X' — write the underlying fact).\n"
            "  importance: 0.0-1.0. 0.9+ for load-bearing personal info "
            "              (anniversary, allergy, hard preference). 0.6-0.8 for useful "
            "              context. 0.3-0.5 for incidentals. 0.1 for trivia.\n\n"
            "If nothing in this turn is worth remembering, reply with: []\n\n"
            f"User: {user_text!r}\n"
            f"Jarvis: {assistant_text!r}"
        )
        try:
            raw = await self._llm.chat([{"role": "user", "content": prompt}])
        except Exception as e:
            logger.debug(f"[MemoryV2] extraction LLM call failed: {e}")
            return []

        # Robust JSON extraction — model may wrap with prose or markdown
        match = re.search(r"\[.*\]", raw or "", re.DOTALL)
        if not match:
            return []
        try:
            items = json.loads(match.group(0))
        except Exception:
            return []
        if not isinstance(items, list):
            return []

        new_ids: list[int] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            kind = str(item.get("kind", "fact")).strip().lower() or "fact"
            subject = item.get("subject")
            if subject is not None:
                subject = str(subject).strip() or None
            try:
                importance = max(0.0, min(1.0, float(item.get("importance", 0.5))))
            except (TypeError, ValueError):
                importance = 0.5
            mid = await self.add(
                kind=kind, content=content, subject=subject,
                importance=importance,
                source_event_id=source_event_id,
                source_kind="conversation",
            )
            if mid is not None:
                new_ids.append(mid)
        if new_ids:
            logger.info(f"[MemoryV2] Extracted {len(new_ids)} memories from turn")
        return new_ids

    # ── Self-realization (Jarvis's own thoughts) ───────────────────────────

    async def record_thought(
        self,
        content: str,
        subject: Optional[str] = None,
        importance: float = 0.4,
    ) -> Optional[int]:
        """Save one of Jarvis's own self-generated reflections. Same store
        as facts, distinguished by kind='thought'."""
        return await self.add(
            kind="thought",
            content=content,
            subject=subject,
            importance=importance,
            source_kind="self_thought",
        )

    async def record_question(
        self,
        content: str,
        subject: Optional[str] = None,
        importance: float = 0.5,
    ) -> Optional[int]:
        """A question Jarvis wants answered (by Cole or by Claude)."""
        return await self.add(
            kind="question",
            content=content,
            subject=subject,
            importance=importance,
            source_kind="self_thought",
        )

    async def list_unsurfaced_thoughts_or_questions(self, limit: int = 20) -> list[dict]:
        try:
            rows = await self._db.fetchall(
                "SELECT id, kind, subject, content, importance, created_at "
                "FROM memories WHERE kind IN ('thought', 'question') "
                "AND surfaced_at IS NULL AND archived = 0 "
                "ORDER BY importance DESC, created_at DESC LIMIT ?",
                (int(limit),),
            )
        except Exception:
            return []
        return [
            {
                "id": int(r["id"]),
                "kind": r["kind"],
                "subject": r["subject"],
                "content": r["content"],
                "importance": float(r["importance"] or 0.5),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    async def mark_surfaced(self, memory_id: int) -> None:
        try:
            await self._db.execute(
                "UPDATE memories SET surfaced_at = ? WHERE id = ?",
                (_now_iso(), memory_id),
            )
        except Exception:
            pass

    # ── Listing for dashboard ──────────────────────────────────────────────

    async def list_recent(
        self,
        limit: int = 100,
        kind: Optional[str] = None,
        include_archived: bool = False,
    ) -> list[dict]:
        sql = "SELECT id, kind, subject, content, importance, created_at, last_accessed_at, access_count, archived FROM memories WHERE 1=1"
        params: list = []
        if not include_archived:
            sql += " AND archived = 0"
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))
        try:
            rows = await self._db.fetchall(sql, tuple(params))
        except Exception:
            return []
        return [
            {
                "id": int(r["id"]),
                "kind": r["kind"],
                "subject": r["subject"],
                "content": r["content"],
                "importance": float(r["importance"] or 0.5),
                "created_at": r["created_at"],
                "last_accessed_at": r["last_accessed_at"],
                "access_count": int(r["access_count"] or 0),
                "archived": bool(r["archived"]),
            }
            for r in rows
        ]
