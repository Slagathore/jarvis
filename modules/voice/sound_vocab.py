"""
JARVIS — Ambient Home AI
========================
Mission: SoundVocabLearner — the sound-side analogue of ObjectVocabLearner.

         The voice cascade VAD-segments audio and routes each segment:
         wake word, watched safety event (alarm/glass/cry), or speech.
         A segment that is none of those — loud enough for the VAD to
         open, but not a wake word, not a watched event, and carrying no
         transcribable speech — is a sound Jarvis HEARD but cannot name.

         This store keeps the recent ones, each with a saved audio clip,
         so the dashboard Review tab can play them back: Cole names the
         ones worth knowing ("that's the dishwasher beep") or dismisses
         the one-offs. Named sounds persist to data/learned_sounds.json.

         Unlike objects, sounds are events, not things-in-a-place — so
         this is a bounded recency log, not a recurrence counter.

Modules: modules/voice/sound_vocab.py
Classes: SoundVocabLearner
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from loguru import logger


class SoundVocabLearner:
    """Bounded recency log of unidentified sounds, for the Review tab.

    Config keys (config["voice"]["sound_vocab"], all optional):
        enabled:      master switch (default True)
        max_unknown:  how many recent unknown sounds to retain (40)
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        *,
        store_path: str = "data/learned_sounds.json",
    ) -> None:
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", True))
        self._max_unknown = int(cfg.get("max_unknown", 40))
        self._store_path = Path(store_path)
        # Confirmed vocabulary: [{name, room, guess, learned_at}].
        self._learned: list[dict] = []
        # Recent unidentified sounds (newest appended last). In-memory —
        # ephemeral, like ObjectVocabLearner's pending set.
        self._unknown: list[dict] = []
        self._next_id = 1
        self._load()

    # ── Persistence (learned vocabulary only) ────────────────────────────────

    def _load(self) -> None:
        if not self._store_path.exists():
            return
        try:
            data = json.loads(self._store_path.read_text(encoding="utf-8"))
            self._learned = list(data.get("learned", []))
            logger.info(
                f"[SoundVocab] loaded {len(self._learned)} learned sound(s)"
            )
        except Exception as e:
            logger.warning(f"[SoundVocab] store load failed: {e}")

    def _save(self) -> None:
        try:
            self._store_path.parent.mkdir(parents=True, exist_ok=True)
            self._store_path.write_text(
                json.dumps({"learned": self._learned}, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"[SoundVocab] store save failed: {e}")

    # ── Intake ───────────────────────────────────────────────────────────────

    def note_unknown(
        self,
        room: str,
        *,
        clip_path: Optional[str] = None,
        guess: Optional[str] = None,
        score: float = 0.0,
        duration_s: float = 0.0,
    ) -> Optional[dict]:
        """Record one unidentified sound. `clip_path` is a saved WAV the
        Review tab plays back; `guess` is the classifier's best (low-
        confidence) label, if any. Returns the stored item."""
        if not self.enabled:
            return None
        item = {
            "id": self._next_id,
            "room": room,
            "ts": time.time(),
            "clip_path": clip_path,
            "guess": guess,
            "score": round(float(score), 3),
            "duration_s": round(float(duration_s), 2),
        }
        self._next_id += 1
        self._unknown.append(item)
        if len(self._unknown) > self._max_unknown:
            # Drop the oldest — bounded recency log.
            self._unknown = self._unknown[-self._max_unknown:]
        return item

    # ── Review / answer ──────────────────────────────────────────────────────

    def review_items(self) -> list[dict]:
        """Recent unknown sounds, newest first — the Review-tab feed."""
        return list(reversed(self._unknown))

    def _take(self, item_id: int) -> Optional[dict]:
        for i, it in enumerate(self._unknown):
            if it["id"] == int(item_id):
                return self._unknown.pop(i)
        return None

    def clip_path_for(self, item_id: int) -> Optional[str]:
        for it in self._unknown:
            if it["id"] == int(item_id):
                return it.get("clip_path")
        return None

    def record_answer(self, item_id: int, name: str) -> Optional[dict]:
        """Cole named the sound. Persist it and drop it from the review
        list. Returns the learned record, or None on a blank name."""
        name = (name or "").strip()
        if not name:
            return None
        item = self._take(item_id)
        entry = {
            "name": name,
            "room": item.get("room", "") if item else "",
            "guess": item.get("guess") if item else None,
            "learned_at": time.time(),
        }
        self._learned.append(entry)
        self._save()
        logger.info(f"[SoundVocab] learned sound '{name}'")
        return entry

    def dismiss(self, item_id: int) -> None:
        """Drop an unknown sound — a one-off not worth remembering."""
        self._take(item_id)

    def snapshot(self) -> dict:
        """Dashboard view — learned vocabulary + recent unknowns."""
        return {
            "learned": list(self._learned),
            "pending": self.review_items(),
        }
