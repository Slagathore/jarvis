"""
JARVIS — Ambient Home AI
========================
Mission: Score how messy a room looks via the LLM. Independent of the room's
         long-term baseline (that's AnomalyDetector's job) — this answers the
         absolute "is it tidy or a disaster right now" question, not "did
         anything change". When a room crosses the alert threshold, fires
         room_messy event so the dashboard / curiosity engine can react.

         Runs at most once every cooldown_seconds per room to keep LLM cost
         bounded. Score is 0 (immaculate) to 10 (disaster).

Modules: modules/vision/mess_detector.py
Classes: MessDetector
Functions:
    MessDetector.__init__(config, llm)        — Wire LLM and thresholds
    MessDetector.should_check(room)           — Cooldown gate
    MessDetector.score(room, scene_desc)      — Async LLM-based score (0-10) + reason
"""

import re
import time
from typing import Optional

from loguru import logger


class MessDetector:
    """
    LLM-based absolute tidiness scoring of a room. Uses the same scene-
    description text that SceneAnalyzer already generates — no second vision
    call needed.
    """

    def __init__(self, config: dict, llm) -> None:
        cfg = config.get("mess", {}) if isinstance(config.get("mess"), dict) else {}
        self._llm = llm
        self._threshold: float = float(cfg.get("alert_threshold", 6.5))
        self._cooldown_seconds: float = float(cfg.get("cooldown_seconds", 1800))
        self._last_check: dict[str, float] = {}
        self._last_score: dict[str, float] = {}

    @property
    def threshold(self) -> float:
        return self._threshold

    def should_check(self, room: str) -> bool:
        last = self._last_check.get(room)
        if last is None:
            return True
        return (time.monotonic() - last) >= self._cooldown_seconds

    def last_score(self, room: str) -> Optional[float]:
        return self._last_score.get(room)

    async def score(self, room: str, scene_description: str) -> Optional[tuple[float, str]]:
        """
        Score the absolute tidiness of the room from 0 (immaculate) to 10
        (disaster). Returns (score, reason) on success, None on LLM failure.
        Records the check time only on success so should_check() respects
        the cooldown.
        """
        if self._llm is None or not scene_description.strip():
            return None

        system_prompt = (
            "You are a tidiness-scoring assistant. Given a description of the "
            "current state of a room, score its tidiness on a 0-10 scale where "
            "0 = immaculate, organized, looks like a magazine; 5 = lived-in, "
            "average; 10 = disaster, can't see the floor.\n\n"
            "Score only what the description actually says. Do not invent "
            "objects or assume mess that isn't described. Reply EXACTLY in "
            "this format:\n"
            "SCORE: <number>\n"
            "REASON: <one short sentence>\n"
        )
        user_prompt = (
            f"Room: {room}\n\n"
            f"Scene description:\n{scene_description.strip()}"
        )
        try:
            response = await self._llm.chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])
        except Exception as e:
            logger.warning(f"[Mess] LLM call failed for '{room}': {e}")
            return None

        parsed = _parse_score_reason(response)
        if parsed is None:
            logger.debug(f"[Mess] Could not parse LLM output: {response[:120]!r}")
            return None
        score, reason = parsed
        self._last_check[room] = time.monotonic()
        self._last_score[room] = score
        return score, reason


_SCORE_RE = re.compile(r"SCORE:\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE)
_REASON_RE = re.compile(r"REASON:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL)


def _parse_score_reason(text: str) -> Optional[tuple[float, str]]:
    if not text:
        return None
    m_score = _SCORE_RE.search(text)
    if not m_score:
        return None
    try:
        score = float(m_score.group(1))
    except ValueError:
        return None
    score = max(0.0, min(10.0, score))
    m_reason = _REASON_RE.search(text)
    reason = m_reason.group(1).strip() if m_reason else ""
    return score, reason
