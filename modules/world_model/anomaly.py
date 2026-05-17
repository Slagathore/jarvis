"""
JARVIS — World Model / Anomaly
==============================
Mission: AnomalyScorer — the live half of audit roadmap §25. It
         subscribes to `world.entity_event`, scores each event against
         the resident's behavioral profile (built nightly by
         PatternMiner), and publishes `world.anomaly` when an event is
         unusual enough to merit attention.

         The score is a weighted blend of components, each ~0-10:
           time_of_day   — is anyone normally active at this hour?
           room_at_time  — is this room normal for this weekday-hour?
           arrival/departure lateness — vs the usual distribution.

         Guards against nagging:
           - min_history_days: do not score against a thin profile.
           - per-entity cooldown: one alert per entity per N minutes.
           - auto_tune: a nightly pass raises the threshold if the
             user keeps marking anomalies "not actually unusual".

         Anomalies persist to `world_anomalies` so the dashboard can
         show a review queue and auto_tune can measure the FP rate.

Modules: modules/world_model/anomaly.py
Classes: AnomalyScorer
"""

from __future__ import annotations

import json
import math
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from loguru import logger


def _parse_ts(value: Any) -> datetime:
    try:
        dt = datetime.fromisoformat(str(value))
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


class AnomalyScorer:
    """Scores live world events against resident behavioral profiles."""

    def __init__(self, bus: Any, world: Any, db: Any,
                 config: Optional[dict] = None) -> None:
        self.bus = bus
        self.world = world
        self._db = db
        cfg = config or {}
        self.threshold = float(cfg.get("anomaly_threshold", 6.0))
        self.cooldown_seconds = float(cfg.get("anomaly_cooldown_seconds", 600))
        self.min_history_days = int(cfg.get("anomaly_min_history_days", 14))
        self._fp_high = float(cfg.get("anomaly_auto_tune_fp_high", 0.30))
        self._fp_low = float(cfg.get("anomaly_auto_tune_fp_low", 0.05))
        self._last_alert_ts: dict[str, datetime] = {}
        self._sub = None

    async def start(self) -> None:
        self._sub = self.bus.subscribe("world.entity_event", self._on_event)
        logger.info(
            f"[AnomalyScorer] started (threshold={self.threshold:.1f}, "
            f"min_history={self.min_history_days}d)"
        )

    async def stop(self) -> None:
        if self._sub is not None:
            try:
                self._sub.unsubscribe()
            except Exception:
                pass
            self._sub = None

    # ── Live scoring ─────────────────────────────────────────────────────────

    async def _on_event(self, event: dict) -> None:
        try:
            ent = self.world.entities.get(event.get("entity_id"))
            # People only; residents only — cats/visitors are excluded.
            if (ent is None or not getattr(ent, "is_resident", False)
                    or getattr(ent, "entity_type", None) != "person"):
                return
            profile = (ent.metadata or {}).get("pattern_profile")
            if not profile or profile.get("n_events", 0) == 0:
                return
            # Do not trust a profile built from too little history.
            wstart = profile.get("window_start")
            if wstart:
                age = datetime.now(timezone.utc) - _parse_ts(wstart)
                if age < timedelta(days=self.min_history_days):
                    return

            score, components = self._score(event, profile)
            if score < self.threshold:
                return
            # Per-entity cooldown — do not spam during an unusual day.
            now = datetime.now(timezone.utc)
            last = self._last_alert_ts.get(ent.id)
            if last and (now - last).total_seconds() < self.cooldown_seconds:
                return
            self._last_alert_ts[ent.id] = now

            anomaly_id = str(uuid.uuid4())
            payload = {
                "id": anomaly_id,
                "entity_id": ent.id,
                "entity_name": ent.display_name,
                "event": event,
                "score": round(score, 3),
                "components": components,
                "ts": event.get("ts"),
            }
            await self._persist(payload)
            await self.bus.publish("world.anomaly", payload)
            logger.info(
                f"[AnomalyScorer] anomaly for {ent.display_name}: "
                f"score={score:.2f} {components}"
            )
        except Exception:
            logger.exception("[AnomalyScorer] scoring failed")

    def _score(self, event: dict, profile: dict) -> tuple[float, dict]:
        """Weighted blend of anomaly components, each roughly 0-10."""
        components: dict[str, float] = {}
        ts = _parse_ts(event.get("ts"))
        weekday, hour = ts.weekday(), ts.hour
        room = event.get("room")
        et = event.get("event_type")

        # Time-of-day: how unusual is any activity at this hour?
        active = (profile.get("weekly_active_hours", {}) or {}).get(weekday, []) \
            or (profile.get("weekly_active_hours", {}) or {}).get(str(weekday), [])
        if active and hour not in active:
            dists = [min(abs(hour - h), 24 - abs(hour - h)) for h in active]
            components["time_of_day"] = float(min(min(dists) * 2.0, 10.0))
        else:
            components["time_of_day"] = 0.0

        # Room-given-time: how unusual is this room at this weekday-hour?
        rbwh = profile.get("room_by_weekday_hour", {}) or {}
        room_dist = (rbwh.get(weekday) or rbwh.get(str(weekday)) or {})
        room_dist = room_dist.get(hour) or room_dist.get(str(hour)) or {}
        if not room:
            components["room_at_time"] = 0.0
        elif not room_dist:
            # The weekday-hour bucket has NO history at all — the profile
            # has no expectation for this time, so being in any room now
            # is unmodeled. (The spec left this case at 0; that let a
            # 3 AM appearance under-score.)
            components["room_at_time"] = 8.0
        else:
            p_room = room_dist.get(room, 0.0)
            components["room_at_time"] = (
                float(min(-math.log(p_room + 0.01), 8.0)) if p_room > 0 else 8.0
            )

        # Arrival / departure lateness vs the usual distribution.
        if et == "departed":
            dep = profile.get("departure_by_weekday", {}) or {}
            components["departure_time"] = self._outlier_score(
                hour, dep.get(weekday) or dep.get(str(weekday)) or {})
        elif et == "reappeared":
            arr = profile.get("arrival_by_weekday", {}) or {}
            components["arrival_time"] = self._outlier_score(
                hour, arr.get(weekday) or arr.get(str(weekday)) or {})

        weights = {
            "time_of_day": 0.30, "room_at_time": 0.45,
            "arrival_time": 0.15, "departure_time": 0.15,
        }
        score = sum(weights.get(k, 0.0) * v for k, v in components.items())
        return score, components

    @staticmethod
    def _outlier_score(value: int, dist: dict) -> float:
        """Score an hour against a {hour: count} histogram."""
        if not dist:
            return 0.0
        # Keys may be int or str depending on the JSON round-trip.
        norm = {int(k): v for k, v in dist.items()}
        total = sum(norm.values()) or 1
        p = norm.get(value, 0) / total
        if p == 0:
            observed = sorted(norm.keys())
            if not observed:
                return 0.0
            dists = [min(abs(value - h), 24 - abs(value - h)) for h in observed]
            return float(min(min(dists) * 1.5, 8.0))
        return float(min(-math.log(p + 0.01), 6.0))

    # ── Persistence + feedback loop ──────────────────────────────────────────

    async def _persist(self, payload: dict) -> None:
        if self._db is None:
            return
        try:
            await self._db.execute(
                "INSERT INTO world_anomalies "
                "(id, ts, entity_id, entity_name, score, components, event) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (payload["id"], payload.get("ts") or _iso_now(),
                 payload.get("entity_id"), payload.get("entity_name"),
                 float(payload.get("score", 0.0)),
                 json.dumps(payload.get("components", {}), default=str),
                 json.dumps(payload.get("event", {}), default=str)),
            )
        except Exception as e:
            logger.debug(f"[AnomalyScorer] anomaly persist failed: {e}")

    async def invalidate(self, anomaly_id: str, reason: str = "") -> bool:
        """Mark an anomaly as a false positive (dashboard 'not unusual'
        button). Feeds auto_tune. Returns True if a row was updated."""
        if self._db is None:
            return False
        try:
            await self._db.execute(
                "UPDATE world_anomalies SET invalidated = 1, "
                "invalidated_reason = ?, invalidated_ts = ? WHERE id = ?",
                (reason, _iso_now(), anomaly_id),
            )
            await self.bus.publish("world.anomaly_invalidated", {
                "anomaly_id": anomaly_id, "reason": reason,
            })
            logger.info(f"[AnomalyScorer] anomaly {anomaly_id[:8]} invalidated")
            return True
        except Exception as e:
            logger.debug(f"[AnomalyScorer] invalidate failed: {e}")
            return False

    async def recent_anomalies(self, limit: int = 50) -> list[dict]:
        """Highest-score-first review queue for the dashboard."""
        if self._db is None:
            return []
        try:
            rows = await self._db.fetchall(
                "SELECT * FROM world_anomalies ORDER BY ts DESC LIMIT ?",
                (int(limit),),
            )
            return [dict(r) for r in rows]
        except Exception as e:
            logger.debug(f"[AnomalyScorer] recent query failed: {e}")
            return []

    async def auto_tune(self, days_back: int = 7) -> None:
        """Nightly: if the user keeps marking anomalies 'not unusual',
        the threshold is too low — raise it. If almost nothing is ever
        invalidated and we still barely fire, lower it (floor 3.0)."""
        if self._db is None:
            return
        cutoff = (datetime.now(timezone.utc)
                  - timedelta(days=days_back)).isoformat()
        try:
            total = await self._db.fetchone(
                "SELECT COUNT(*) AS n FROM world_anomalies WHERE ts >= ?",
                (cutoff,))
            fp = await self._db.fetchone(
                "SELECT COUNT(*) AS n FROM world_anomalies "
                "WHERE ts >= ? AND invalidated = 1", (cutoff,))
        except Exception as e:
            logger.debug(f"[AnomalyScorer] auto-tune query failed: {e}")
            return
        n = int(total["n"]) if total else 0
        n_fp = int(fp["n"]) if fp else 0
        if n < 10:
            return  # too little signal to tune on
        fp_rate = n_fp / max(n, 1)
        if fp_rate > self._fp_high:
            self.threshold += 0.5
            logger.info(f"[AnomalyScorer] auto-tune: FP rate {fp_rate:.2f} "
                        f"high → threshold {self.threshold:.2f}")
        elif fp_rate < self._fp_low:
            self.threshold = max(self.threshold - 0.25, 3.0)
            logger.info(f"[AnomalyScorer] auto-tune: FP rate {fp_rate:.2f} "
                        f"low → threshold {self.threshold:.2f}")


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()
