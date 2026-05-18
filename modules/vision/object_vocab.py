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
        ignore_classes:       YOLO classes never worth an ask (furniture)
        min_confidence:       detection confidence floor for noting (0.45)
    """

    # Static furniture / fixtures YOLO sees constantly — never worth an
    # ask-to-learn prompt. Overridable via config.vision.object_vocab.
    _DEFAULT_IGNORE_CLASSES: frozenset[str] = frozenset({
        "chair", "couch", "bed", "dining table", "tv", "potted plant",
        "clock", "sink", "toilet", "refrigerator", "microwave", "oven",
        "keyboard", "mouse", "vase", "bench", "toaster",
    })

    # How many recent detection bboxes to keep per unknown — the spatial
    # trail behind the persistence / "it keeps showing up here" check.
    _MAX_BBOXES: int = 24

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
        # Caller-side "is this worth noting" policy (see should_note): an
        # ignore list of always-present classes + a confidence floor.
        ignore = cfg.get("ignore_classes")
        self._ignore_classes: set[str] = (
            {str(c) for c in ignore} if ignore is not None
            else set(self._DEFAULT_IGNORE_CLASSES)
        )
        self._min_confidence = float(cfg.get("min_confidence", 0.45))
        self._store_path = Path(store_path)
        # Confirmed vocabulary: [{name, query, room, learned_at, sightings}].
        self._learned: list[dict] = []
        # In-flight unknowns: key -> {count, room, first_ts, last_ts, asked}.
        self._pending: dict[str, dict] = {}
        # Keys Cole has explicitly dismissed — note_unknown ignores them so
        # a "don't track that" answer sticks across future sightings.
        self._dismissed: set[str] = set()
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._store_path.exists():
            return
        try:
            data = json.loads(self._store_path.read_text(encoding="utf-8"))
            self._learned = list(data.get("learned", []))
            self._dismissed = {str(k) for k in data.get("dismissed", [])}
            logger.info(
                f"[ObjectVocab] loaded {len(self._learned)} learned object(s)"
                + (f", {len(self._dismissed)} dismissed"
                   if self._dismissed else "")
            )
        except Exception as e:
            logger.warning(f"[ObjectVocab] store load failed: {e}")

    def _save(self) -> None:
        try:
            self._store_path.parent.mkdir(parents=True, exist_ok=True)
            self._store_path.write_text(
                json.dumps(
                    {"learned": self._learned,
                     "dismissed": sorted(self._dismissed)},
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"[ObjectVocab] store save failed: {e}")

    # ── Caller-side noting policy ────────────────────────────────────────────

    def should_note(self, yolo_class: str, confidence: float) -> bool:
        """Gate for note_unknown the vision loop calls per detection.
        True when a detection is worth counting toward an ask — False for
        a disabled learner, an ignored (always-present furniture) class,
        or a detection below the confidence floor."""
        if not self.enabled:
            return False
        if not yolo_class or yolo_class in self._ignore_classes:
            return False
        return float(confidence) >= self._min_confidence

    # ── Sighting intake ──────────────────────────────────────────────────────

    def note_unknown(
        self, key: str, room: str, *,
        descriptor: Optional[dict] = None,
        crop_path: Optional[str] = None,
    ) -> None:
        """Record one sighting of an un-nameable object. `key` must be
        stable across sightings of the SAME object (e.g. room:class) so
        recurrences accumulate toward the ask threshold.

        Each sighting also appends the detection's bbox to a rolling
        spatial trail — the persistence Cole asked for: an unknown that
        keeps re-appearing in the SAME spot is a real object worth a
        photo and a question, not a one-off mis-detection. `crop_path`,
        when given, is kept as the freshest picture evidence.
        """
        if not self.enabled or not key:
            return
        if key in self._dismissed:
            return  # Cole already said this one is not worth tracking
        now = time.time()
        self._expire_stale(now)
        desc = descriptor or {}
        bbox = desc.get("bbox")
        rec = self._pending.get(key)
        if rec is None:
            self._pending[key] = {
                "count": 1, "room": room, "first_ts": now, "last_ts": now,
                "asked": False, "descriptor": desc,
                "bboxes": [list(bbox)] if bbox else [],
                "crop_path": crop_path,
            }
        else:
            rec["count"] += 1
            rec["last_ts"] = now
            rec["room"] = room
            if bbox:
                rec.setdefault("bboxes", []).append(list(bbox))
                rec["bboxes"] = rec["bboxes"][-self._MAX_BBOXES:]
            if crop_path:
                rec["crop_path"] = crop_path
            if desc:
                # Refresh confidence/bbox; the YOLO class stays put.
                rec["descriptor"] = {**rec.get("descriptor", {}), **desc}

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
        {key, room, count, descriptor, crop_path, location} — the caller
        turns it into a spoken question; crop_path/location also give the
        dashboard Review tab its picture + 'where it keeps showing up'."""
        if not self.enabled:
            return None
        for key, rec in self._pending.items():
            if not rec["asked"] and rec["count"] >= self._ask_after:
                return {
                    "key": key, "room": rec["room"], "count": rec["count"],
                    "descriptor": dict(rec.get("descriptor") or {}),
                    "crop_path": rec.get("crop_path"),
                    "location": self._location_summary(
                        rec.get("bboxes") or []
                    ),
                }
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
        """Cole said it is not worth tracking — drop it AND remember the
        dismissal (persisted) so note_unknown won't re-add it on the next
        sighting and pester him again."""
        self._pending.pop(key, None)
        if key:
            self._dismissed.add(key)
            self._save()

    # ── Read-out for the detector ────────────────────────────────────────────

    def learned_queries(self) -> list[str]:
        """OWLv2 query strings for every learned object — merge these
        into the open-vocab detector's query set so learned objects are
        recognised going forward."""
        return [e["query"] for e in self._learned if e.get("query")]

    def learned_names(self) -> list[str]:
        return [e["name"] for e in self._learned if e.get("name")]

    def learned_query_names(self) -> dict[str, str]:
        """{query: friendly name} for every learned object — lets the
        open-vocab loop tag an OWLv2 hit with the name Cole gave it."""
        return {e["query"]: e.get("name", e["query"])
                for e in self._learned if e.get("query")}

    def snapshot(self) -> dict:
        """Dashboard view — learned vocabulary + in-flight unknowns
        (with picture + spatial evidence, via review_items)."""
        return {
            "learned": list(self._learned),
            "pending": self.review_items(),
        }

    def review_items(self) -> list[dict]:
        """Every in-flight unknown, richest form — for the dashboard
        Review tab. Each carries its YOLO class, freshest crop, sighting
        count, ask state, and a spatial summary (parked in one spot, or
        scattered?). Sorted most-sighted first."""
        out: list[dict] = []
        for key, rec in self._pending.items():
            desc = rec.get("descriptor") or {}
            out.append({
                "key": key,
                "room": rec["room"],
                "count": rec["count"],
                "asked": rec["asked"],
                "yolo_class": desc.get("yolo_class"),
                "confidence": desc.get("confidence"),
                "crop_path": rec.get("crop_path"),
                "location": self._location_summary(rec.get("bboxes") or []),
                "first_ts": rec["first_ts"],
                "last_ts": rec["last_ts"],
            })
        out.sort(key=lambda x: x["count"], reverse=True)
        return out

    @staticmethod
    def _location_summary(bboxes: list) -> dict:
        """Collapse a spatial trail of bboxes into {center, stability, n}.

        stability is 0..1 — how tightly the sightings cluster in space.
        Near 1.0: the object keeps appearing in the SAME place (a real
        thing sitting somewhere — we can be confident it's there even
        between detections). Near 0: scattered sightings, more likely
        detector noise than one persistent object."""
        pts: list[tuple[float, float]] = []
        diag = 0.0
        for b in bboxes:
            try:
                x1, y1, x2, y2 = (float(v) for v in b)
            except Exception:
                continue
            pts.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
            diag = max(diag, ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        if not pts:
            return {"center": None, "stability": 0.0, "n": 0}
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        center = [round(cx, 1), round(cy, 1)]
        if len(pts) < 2 or diag <= 0:
            return {"center": center,
                    "stability": 1.0 if len(pts) == 1 else 0.5,
                    "n": len(pts)}
        spread = sum(
            ((p[0] - cx) ** 2 + (p[1] - cy) ** 2) ** 0.5 for p in pts
        ) / len(pts)
        # Spread within one object-diagonal == "same spot". Clamp 0..1.
        stability = max(0.0, min(1.0, 1.0 - spread / diag))
        return {"center": center, "stability": round(stability, 3),
                "n": len(pts)}
