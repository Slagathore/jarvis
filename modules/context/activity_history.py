"""
JARVIS — Ambient Home AI
========================
Mission: Track activity transitions over time so we can answer two questions:

         (1) Predicted remaining duration — "Cole usually games for ~2h, he's
             ~30 min in, so ~1h 30m left."
         (2) User routine — "Cole usually starts coding around 9am on Tuesdays."

         Both use the same `activity_log` SQLite table. Each row is one
         continuous run of a single activity: opened when activity changes,
         closed when activity changes again.

Modules: modules/context/activity_history.py
Classes: ActivityHistory
Functions:
    ActivityHistory.__init__(db)                 — Wire DB
    ActivityHistory.record_change(activity, room) — Close prev row + open new
    ActivityHistory.current_run()                — (activity, started_at) of open row
    ActivityHistory.mean_duration_seconds(activity)
                                                 — Mean over all closed rows
    ActivityHistory.predict_remaining_seconds(activity, started_at)
                                                 — mean - elapsed (clamped >= 0)
    ActivityHistory.typical_activity(day_of_week, hour, top_n)
                                                 — Most-common activity at
                                                   that time slot, sorted by frequency
    ActivityHistory.summary_for_prompt(state)    — Single-line context blurb
                                                   for the LLM system prompt
"""

from datetime import datetime
from typing import Any, Optional

from loguru import logger


class ActivityHistory:
    """Async wrapper over the `activity_log` table for routine + duration analytics."""

    # Don't bother predicting until we have at least this many closed rows for
    # an activity. With <3 samples the mean is noise.
    _MIN_SAMPLES_FOR_PREDICTION = 3

    def __init__(self, db) -> None:
        self._db = db

    async def record_change(self, new_activity: str, room: Optional[str] = None) -> None:
        """
        Close any open row whose activity != new_activity, then open a new
        row for new_activity if no open one matches. Idempotent — repeated
        calls with the same activity are a no-op.
        """
        if not new_activity or new_activity == "unknown":
            return
        try:
            current = await self._db.fetchone(
                "SELECT id, activity, started_at FROM activity_log "
                "WHERE ended_at IS NULL ORDER BY started_at DESC LIMIT 1"
            )
        except Exception as e:
            logger.debug(f"[ActivityHistory] fetch open row failed: {e}")
            return

        now_iso = datetime.now().isoformat()
        if current and current["activity"] == new_activity:
            return  # Already in this activity

        # Close the previous open run
        if current:
            try:
                started = datetime.fromisoformat(current["started_at"])
                duration_s = max(0, int((datetime.now() - started).total_seconds()))
            except (TypeError, ValueError):
                duration_s = None
            try:
                await self._db.execute(
                    "UPDATE activity_log SET ended_at = ?, duration_seconds = ? WHERE id = ?",
                    (now_iso, duration_s, current["id"]),
                )
            except Exception as e:
                logger.debug(f"[ActivityHistory] close prev row failed: {e}")

        # Open the new run
        try:
            await self._db.execute(
                "INSERT INTO activity_log (activity, started_at, room) VALUES (?, ?, ?)",
                (new_activity, now_iso, room),
            )
        except Exception as e:
            logger.debug(f"[ActivityHistory] open new row failed: {e}")

    async def current_run(self) -> Optional[dict[str, Any]]:
        """Return the open row {activity, started_at} or None."""
        try:
            row = await self._db.fetchone(
                "SELECT activity, started_at FROM activity_log "
                "WHERE ended_at IS NULL ORDER BY started_at DESC LIMIT 1"
            )
        except Exception:
            return None
        if not row:
            return None
        return {"activity": row["activity"], "started_at": row["started_at"]}

    async def mean_duration_seconds(self, activity: str) -> Optional[float]:
        """Mean duration over all closed runs of `activity`. None if not enough samples."""
        try:
            row = await self._db.fetchone(
                "SELECT COUNT(*) AS n, AVG(duration_seconds) AS avg_s "
                "FROM activity_log WHERE activity = ? AND duration_seconds IS NOT NULL",
                (activity,),
            )
        except Exception:
            return None
        if not row or row["n"] is None or int(row["n"]) < self._MIN_SAMPLES_FOR_PREDICTION:
            return None
        return float(row["avg_s"])

    async def predict_remaining_seconds(
        self,
        activity: str,
        started_at: datetime,
    ) -> Optional[float]:
        """
        Estimate how many seconds remain in the current run based on the mean
        duration of past runs. None if not enough data. Negative results
        (already exceeded mean) get clamped to 0.
        """
        mean_s = await self.mean_duration_seconds(activity)
        if mean_s is None:
            return None
        elapsed = (datetime.now() - started_at).total_seconds()
        return max(0.0, mean_s - elapsed)

    async def typical_activity(
        self,
        day_of_week: int,
        hour: int,
        top_n: int = 3,
    ) -> list[tuple[str, int]]:
        """
        Most common activities at this day-of-week / hour. Returns list of
        (activity, count) tuples sorted by count descending.
        SQLite's strftime: %w = day of week (0=Sun), %H = hour.
        """
        try:
            rows = await self._db.fetchall(
                "SELECT activity, COUNT(*) AS n FROM activity_log "
                "WHERE CAST(strftime('%w', started_at) AS INTEGER) = ? "
                "  AND CAST(strftime('%H', started_at) AS INTEGER) = ? "
                "GROUP BY activity "
                "ORDER BY n DESC "
                "LIMIT ?",
                (day_of_week, hour, top_n),
            )
        except Exception:
            return []
        return [(r["activity"], int(r["n"])) for r in rows]

    async def summary_for_prompt(self, current_activity: Optional[str]) -> Optional[str]:
        """
        Build a one-or-two-line context blurb for the LLM system prompt that
        captures predicted-remaining-duration + typical-activity-now.
        Returns None if there's nothing useful to say (no history yet, or
        current activity is unknown/idle).
        """
        lines: list[str] = []
        if current_activity and current_activity not in ("unknown", "idle", "away"):
            run = await self.current_run()
            if run and run["activity"] == current_activity:
                try:
                    started = datetime.fromisoformat(run["started_at"])
                except (TypeError, ValueError):
                    started = None
                if started is not None:
                    elapsed = (datetime.now() - started).total_seconds()
                    remaining = await self.predict_remaining_seconds(
                        current_activity, started
                    )
                    elapsed_str = _humanize_seconds(elapsed)
                    if remaining is not None:
                        if remaining > 0:
                            lines.append(
                                f"Cole has been {current_activity.replace('_', ' ')} "
                                f"for {elapsed_str}. Based on past sessions, "
                                f"~{_humanize_seconds(remaining)} likely remaining."
                            )
                        else:
                            lines.append(
                                f"Cole has been {current_activity.replace('_', ' ')} "
                                f"for {elapsed_str} — already past his typical session length."
                            )
                    else:
                        lines.append(
                            f"Cole has been {current_activity.replace('_', ' ')} "
                            f"for {elapsed_str}."
                        )

        # Typical activity for this time-slot
        now = datetime.now()
        typical = await self.typical_activity(
            day_of_week=int(now.strftime("%w")),
            hour=now.hour,
            top_n=2,
        )
        if typical and typical[0][1] >= 3:
            top_act, top_count = typical[0]
            if top_act != current_activity:
                lines.append(
                    f"At this time on {now.strftime('%A')}s Cole typically "
                    f"{top_act.replace('_', ' ')} ({top_count} prior occurrences)."
                )

        if not lines:
            return None
        return " ".join(lines)


def _humanize_seconds(s: float) -> str:
    """'2h 15m' / '45m' / '12s' style."""
    s = int(max(0, s))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    h = s // 3600
    m = (s % 3600) // 60
    return f"{h}h {m}m" if m else f"{h}h"
