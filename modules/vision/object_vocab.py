"""
JARVIS — Ambient Home AI
========================
Mission: ObjectVocabLearner — lets Jarvis grow its own object vocabulary
         by asking. The vision stack has two detectors: closed-vocab
         YOLO (fixed classes) and open-vocab OWLv2 (matches text queries).
         Anything that is neither a YOLO class nor a known OWLv2 query is
         an object Jarvis literally cannot name.

         This module is the learning loop for exactly that case:

           note_unknown()      — the vision loop saw an un-nameable
                                  object; record the sighting (keyed so
                                  recurrences of the SAME object count up)
           pending_question()  — an unknown has recurred enough to be
                                  worth interrupting Cole over
           record_answer()     — Cole said what it is; persist it
           learned_queries()   — the confirmed vocabulary, fed back into
                                  the OWLv2 query set so the object is
                                  recognised from now on

         The learned vocabulary persists to data/learned_objects.json so
         it survives restarts. This module is detector-agnostic — the
         caller decides what counts as "unknown" and supplies a stable
         key; the module owns recurrence counting, the ask/answer cycle,
         and persistence.

Modules: modules/vision/object_vocab.py
Classes: ObjectVocabLearner
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from loguru import logger


class ObjectVocabLearner:
    """Recurrence-tracked ask-to-learn store for unknown objects.

    Config keys (from config["vision"]["object_vocab"], all optional):
        enabled:              master switch (default True)
        ask_after_sightings:  recurrences before an unknown is asked (3)
        forget_unasked_after_s: drop a stale unknown never re-seen (1800)
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        *,
        store_path: str = "data/learned_objects.json",
    ) -> None:
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", True))
        self._ask_after = int(cfg.get("ask_after_sightings", 3))
        self._forget_after_s = float(cfg.get("forget_unasked_after_s", 1800))
        self._store_path = Path(store_path)
        # Confirmed vocabulary: [{name, query, room, learned_at, sightings}].
        self._learned: list[dict] = []
        # In-flight unknowns: key -> {count, room, first_ts, last_ts, asked}.
        self._pending: dict[str, dict] = {}
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._store_path.exists():
            return
        try:
            data = json.loads(self._store_path.read_text(encoding="utf-8"))
            self._learned = list(data.get("learned", []))
            logger.info(
                f"[ObjectVocab] loaded {len(self._learned)} learned object(s)"
            )
        except Exception as e:
            logger.warning(f"[ObjectVocab] store load failed: {e}")

    def _save(self) -> None:
        try:
            self._store_path.parent.mkdir(parents=True, exist_ok=True)
            self._store_path.write_text(
                json.dumps({"learned": self._learned}, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"[ObjectVocab] store save failed: {e}")

    # ── Sighting intake ──────────────────────────────────────────────────────

    def note_unknown(
        self, key: str, room: str, *, descriptor: Optional[dict] = None
    ) -> None:
        """Record one sighting of an un-nameable object. `key` must be
        stable across sightings of the SAME object (e.g. a tracked
        cluster id) so recurrences accumulate toward the ask threshold."""
        if not self.enabled or not key:
            return
        now = time.time()
        self._expire_stale(now)
        rec = self._pending.get(key)
        if rec is None:
            self._pending[key] = {
                "count": 1, "room": room, "first_ts": now, "last_ts": now,
                "asked": False, "descriptor": descriptor or {},
            }
        else:
            rec["count"] += 1
            rec["last_ts"] = now
            rec["room"] = room

    def _expire_stale(self, now: float) -> None:
        """Drop unknowns that stopped recurring before being asked —
        a one-off mis-detection should not pester Cole later."""
        for key in list(self._pending):
            rec = self._pending[key]
            if (not rec["asked"]
                    and now - rec["last_ts"] > self._forget_after_s):
                del self._pending[key]

    # ── Ask / answer cycle ───────────────────────────────────────────────────

    def pending_question(self) -> Optional[dict]:
        """The next unknown worth asking Cole about, or None. Returns
        {key, room, count} — the caller turns it into a spoken question."""
        if not self.enabled:
            return None
        for key, rec in self._pending.items():
            if not rec["asked"] and rec["count"] >= self._ask_after:
                return {"key": key, "room": rec["room"], "count": rec["count"]}
        return None

    def mark_asked(self, key: str) -> None:
        """Flag an unknown as asked so it is not asked again while we
        wait for Cole's answer."""
        rec = self._pending.get(key)
        if rec is not None:
            rec["asked"] = True

    def record_answer(
        self, key: str, name: str, *, query: Optional[str] = None
    ) -> Optional[dict]:
        """Cole named the object. Persist it to the learned vocabulary
        and return the new record. `query` is the OWLv2 prompt (defaults
        to the name). Returns None if the key is unknown/blank name."""
        name = (name or "").strip()
        if not name:
            return None
        rec = self._pending.pop(key, None)
        entry = {
            "name": name,
            "query": (query or name).strip().lower(),
            "room": rec.get("room", "") if rec else "",
            "learned_at": time.time(),
            "sightings": rec.get("count", 0) if rec else 0,
        }
        # Replace any prior entry with the same query rather than dup it.
        self._learned = [e for e in self._learned
                         if e.get("query") != entry["query"]]
        self._learned.append(entry)
        self._save()
        logger.info(f"[ObjectVocab] learned '{name}' (query='{entry['query']}')")
        return entry

    def dismiss(self, key: str) -> None:
        """Cole said it is not worth tracking — drop it, do not re-ask."""
        self._pending.pop(key, None)

    # ── Read-out for the detector ────────────────────────────────────────────

    def learned_queries(self) -> list[str]:
        """OWLv2 query strings for every learned object — merge these
        into the open-vocab detector's query set so learned objects are
        recognised going forward."""
        return [e["query"] for e in self._learned if e.get("query")]

    def learned_names(self) -> list[str]:
        return [e["name"] for e in self._learned if e.get("name")]

    def snapshot(self) -> dict:
        """Dashboard view — learned vocabulary + in-flight unknowns."""
        return {
            "learned": list(self._learned),
            "pending": [
                {"key": k, "count": r["count"], "room": r["room"],
                 "asked": r["asked"]}
                for k, r in self._pending.items()
            ],
        }
