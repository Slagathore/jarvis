"""
JARVIS — Ambient Home AI
========================
Mission: Detect when a room looks unusual compared to its long-term baseline.
         The vision pipeline already generates per-frame natural-language
         descriptions and stores per-room baselines via RoomBaselines. This
         module asks the LLM to score the divergence from baseline (0=identical,
         10=radically different) so Jarvis can proactively flag "kitchen looks
         unusually messy" without us hardcoding what messy means.

         Scoring runs at most once every cooldown_seconds per room so the LLM
         doesn't get hammered. Results above the alert threshold get broadcast
         as `room_anomaly` events.

Modules: modules/vision/anomaly_detector.py
Classes: AnomalyDetector
Functions:
    AnomalyDetector.__init__(config, llm)         — Wire LLM and thresholds
    AnomalyDetector.score(room, baseline, current) — Async LLM-based score (0-10) + reason
    AnomalyDetector.should_check(room)            — Cooldown gate
"""

import re
import time
from typing import Optional

from loguru import logger


class AnomalyDetector:
    """
    LLM-based anomaly scoring of a current room description vs its baseline.

    Args (from config['anomaly']):
        alert_threshold:   Score >= this triggers a `room_anomaly` event. Default 6.
        cooldown_seconds:  Min seconds between scores for the same room. Default 600.
    """

    def __init__(self, config: dict, llm) -> None:
        cfg = config.get("anomaly", {}) if isinstance(config.get("anomaly"), dict) else {}
        self._llm = llm
        self._threshold: float = float(cfg.get("alert_threshold", 6.0))
        self._cooldown_seconds: float = float(cfg.get("cooldown_seconds", 600))
        self._last_check: dict[str, float] = {}  # room → monotonic timestamp

    @property
    def threshold(self) -> float:
        return self._threshold

    def should_check(self, room: str) -> bool:
        """Cooldown gate — return True if we're allowed to score this room now."""
        last = self._last_check.get(room)
        if last is None:
            return True
        return (time.monotonic() - last) >= self._cooldown_seconds

    async def score(
        self,
        room: str,
        baseline: str,
        current: str,
    ) -> Optional[tuple[float, str]]:
        """
        Ask the LLM to compare the current description to the baseline.
        Returns (score 0-10, brief reason) on success, None on LLM failure.

        Records the check time on success so should_check() respects cooldown.
        """
        if self._llm is None:
            return None
        if not baseline.strip() or not current.strip():
            return None

        # Strict, anti-hallucination prompt — score numerically, give one short reason
        system_prompt = (
            "You are an anomaly-scoring assistant. You will be given two "
            "descriptions of the same room: a long-term baseline and the "
            "current state. Score how different the current state is from "
            "baseline on a 0-10 scale, where 0 = identical, 5 = noticeably "
            "different, 10 = radically different. Be calibrated and specific.\n\n"
            "Reply EXACTLY in this format:\n"
            "SCORE: <number>\n"
            "REASON: <one short sentence describing the most significant change>\n"
            "\n"
            "Do not invent objects or activities not mentioned in the inputs."
        )
        user_prompt = (
            f"Room: {room}\n\n"
            f"BASELINE:\n{baseline.strip()}\n\n"
            f"CURRENT:\n{current.strip()}"
        )
        try:
            response = await self._llm.chat([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])
        except Exception as e:
            logger.warning(f"[Anomaly] LLM call failed for '{room}': {e}")
            return None

        parsed = _parse_score_reason(response)
        if parsed is None:
            logger.debug(f"[Anomaly] Could not parse LLM output: {response[:120]!r}")
            return None
        score, reason = parsed
        self._last_check[room] = time.monotonic()
        return score, reason


_SCORE_RE = re.compile(r"SCORE:\s*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE)
_REASON_RE = re.compile(r"REASON:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL)


def _parse_score_reason(text: str) -> Optional[tuple[float, str]]:
    """Extract SCORE and REASON from the LLM's response. Tolerates extra text."""
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
