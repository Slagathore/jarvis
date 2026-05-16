"""
JARVIS — World Model / Belief
=============================
Mission: BeliefResolver — the belief-state tracker (audit roadmap D4).

         It subscribes to `vision.observation`, turns every observation
         into an EvidenceFrame, and maintains competing hypotheses about
         each entity with the four-axis confidence model from
         belief.types. Movement between rooms requires the new evidence
         to clear the old location confidence by a margin, so a weak
         stray detection never yanks a confident pin.

SHADOW MODE (D4a — current):
         The resolver runs read-only. It ingests evidence, updates
         hypotheses, persists them to entity_beliefs / belief_evidence,
         and logs state transitions — but it publishes nothing on the
         bus and does not touch WorldModel.world_entities. The live
         system is completely unaffected. This lets the belief model be
         observed against the running WorldModel before D4b flips the
         projection over to it.

         SCOPE: people (keyed by resolved person_id) AND pets (cat/dog,
         keyed coarsely by obj_class:room). Pet *identity* — which cat —
         is deliberately NOT resolved here; that per-animal matching is
         the D4b job. Shadow-mode pet ingest exists so belief_evidence
         accumulates real pet sightings (with colour/size/bbox features)
         for D4b to be built and tuned against. Objects are still D4b.

Modules: modules/world_model/belief/resolver.py
Classes: BeliefResolver
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from loguru import logger

from modules.world_model.belief.types import (
    BeliefHypothesis,
    BeliefState,
    EvidenceFrame,
)

# Tuning. Conservative defaults; overridable via the world_model.
# belief_resolver config block.
_DEFAULTS = {
    "shadow": True,                  # D4a: never leave shadow mode without D4b
    "sighting_location_gain": 0.45,  # how hard a sighting pulls location conf
    "sighting_visibility_gain": 0.6,
    "absence_visibility_decay": 0.55,  # multiply visibility on a missed frame
    "absence_location_decay": 0.97,    # location barely moves on a missed frame
    "cross_room_move_margin": 0.30,    # new room must beat old location by this
    "present_unseen_threshold": 0.30,  # visibility below this → PRESENT_UNSEEN
    # Hysteresis: the discrete CONFIRMED<->UNSEEN state flips only on a RUN
    # of consecutive misses / hits, not a single noisy frame. A full flap
    # now costs unseen_misses + confirmed_hits detector cycles, which keeps
    # an intermittently-detected entity's belief stable instead of
    # oscillating every cycle.
    "present_unseen_misses": 6,        # consec. misses: CONFIRMED → UNSEEN
    "present_confirmed_hits": 3,       # consec. hits:   UNSEEN → CONFIRMED
    "decay_interval_s": 30.0,
    "stale_location_decay": 0.92,      # per decay tick with no evidence
    "departed_threshold": 0.12,        # location below this → DEPARTED
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class BeliefResolver:
    """Belief-state tracker. Constructed after WorldModel; started/stopped
    by the orchestrator. In shadow mode it only reads + persists beliefs."""

    def __init__(self, bus: Any, db: Any, config: Optional[dict] = None) -> None:
        self._bus = bus
        self._db = db
        cfg = dict(_DEFAULTS)
        if config:
            cfg.update({k: v for k, v in config.items() if k in _DEFAULTS})
        self._cfg = cfg
        self.shadow: bool = bool(cfg["shadow"])
        # entity_key → primary hypothesis. (D4a tracks one per entity;
        # competing secondaries arrive in D4b.)
        self._hyp: dict[str, BeliefHypothesis] = {}
        self._sub = None
        self._decay_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        await self._load_existing()
        self._sub = self._bus.subscribe("vision.observation", self._on_observation)
        self._decay_task = asyncio.create_task(
            self._decay_loop(), name="belief_resolver:decay"
        )
        logger.info(
            f"[BeliefResolver] started (shadow={self.shadow}, "
            f"{len(self._hyp)} hypothesis(es) restored)"
        )

    async def stop(self) -> None:
        if self._sub is not None:
            try:
                self._sub.unsubscribe()
            except Exception:
                pass
            self._sub = None
        if self._decay_task is not None and not self._decay_task.done():
            self._decay_task.cancel()
            try:
                await self._decay_task
            except (asyncio.CancelledError, Exception):
                pass
            self._decay_task = None
        logger.debug("[BeliefResolver] stopped")

    # ── Event ingest ─────────────────────────────────────────────────────────

    async def _on_observation(self, payload: dict) -> None:
        """Handle one `vision.observation` batch (one camera, one room)."""
        try:
            room = payload.get("room")
            camera = payload.get("camera")
            ts = payload.get("ts") or _utcnow()
            observations = payload.get("observations") or []

            # Build evidence from each observation. People are keyed by their
            # resolved person_id; pets (cat/dog) are keyed coarsely by
            # obj_class:room. The coarse pet key means multi-animal identity
            # is NOT resolved here — proper per-animal matching is the D4b
            # job. What matters in shadow mode is that every pet sighting
            # still lands a belief_evidence row carrying the colour / size /
            # bbox features D4b will need, so one walk-around gathers data
            # for people and pets alike.
            seen_keys: set[str] = set()
            async with self._lock:
                for obs in observations:
                    pid = getattr(obs, "person_id", None)
                    obj_class = getattr(obs, "obj_class", None)
                    if pid is not None:
                        key = f"person:{pid}"
                        etype = "person"
                        score = float(
                            getattr(obs, "confidence", 0.0)
                            * max(0.05,
                                  getattr(obs, "person_match_confidence", 0.0))
                        )
                        meta: dict = {
                            "person_name": getattr(obs, "person_name", None),
                        }
                    elif obj_class in ("cat", "dog"):
                        key = f"{obj_class}:{room}"
                        etype = obj_class
                        score = float(getattr(obs, "confidence", 0.0))
                        omd = getattr(obs, "metadata", {}) or {}
                        meta = {
                            "color_class": omd.get("color_class"),
                            "size_normalized": omd.get("size_normalized"),
                        }
                    else:
                        continue
                    seen_keys.add(key)
                    ev = EvidenceFrame(
                        ts=ts, entity_key=key, entity_type=etype,
                        source="vision.observation", evidence_type="sighting",
                        room=room, camera=camera, score=score,
                        bbox=getattr(obs, "bbox", None),
                        payload=meta,
                    )
                    await self._ingest_sighting(ev)

                # Absence: anyone we believe is PRESENT in this room but who
                # was not in this batch gets weak negative evidence.
                for key, hyp in list(self._hyp.items()):
                    if (key not in seen_keys and hyp.room == room
                            and hyp.state in (BeliefState.PRESENT_CONFIRMED,
                                              BeliefState.PRESENT_UNSEEN)):
                        ev = EvidenceFrame(
                            ts=ts, entity_key=key, entity_type=hyp.entity_type,
                            source="vision.observation", evidence_type="absence",
                            room=room, camera=camera,
                            score=self._cfg["absence_visibility_decay"],
                        )
                        await self._ingest_absence(ev, hyp)
        except Exception:
            logger.exception("[BeliefResolver] observation ingest failed")

    async def _ingest_sighting(self, ev: EvidenceFrame) -> None:
        hyp = self._hyp.get(ev.entity_key)
        if hyp is None:
            hyp = BeliefHypothesis(
                hypothesis_id=str(uuid.uuid4()),
                entity_key=ev.entity_key, entity_type=ev.entity_type,
            )
            self._hyp[ev.entity_key] = hyp

        prev_state, prev_room = hyp.state, hyp.room
        gain_loc = self._cfg["sighting_location_gain"]
        gain_vis = self._cfg["sighting_visibility_gain"]

        if hyp.room is None or hyp.room == ev.room:
            # Same room (or first sighting) — reinforce. Confidences track
            # every frame; the discrete STATE is hysteretic (run-counters).
            hyp.room, hyp.camera = ev.room, ev.camera
            hyp.confidence_location = _lift(hyp.confidence_location,
                                            ev.score * gain_loc)
            hyp.confidence_visibility = _lift(hyp.confidence_visibility,
                                              ev.score * gain_vis)
            hyp.consecutive_hits += 1
            hyp.consecutive_misses = 0
            if hyp.state == BeliefState.PRESENT_UNSEEN:
                # Re-confirm only after a run of hits — one stray frame in
                # a quiet gap should not flip the label back to confirmed.
                if hyp.consecutive_hits >= self._cfg["present_confirmed_hits"]:
                    hyp.state = BeliefState.PRESENT_CONFIRMED
            elif hyp.state != BeliefState.PRESENT_CONFIRMED:
                # UNKNOWN / DEPARTED / SUSPECTED_ELSEWHERE / TRANSITIONING —
                # a same-room sighting is an unambiguous (re)acquisition.
                hyp.state = BeliefState.PRESENT_CONFIRMED
            # else already PRESENT_CONFIRMED — hold.
        else:
            # Different room — only move if this sighting beats the old
            # location confidence by the move margin. Otherwise the old
            # pin holds and we note a competing hypothesis elsewhere.
            margin = self._cfg["cross_room_move_margin"]
            if ev.score > hyp.confidence_location + margin:
                logger.info(
                    f"[BeliefResolver] {ev.entity_key} moved "
                    f"{hyp.room} → {ev.room} "
                    f"(evidence {ev.score:.2f} > {hyp.confidence_location:.2f}+{margin})"
                )
                hyp.room, hyp.camera = ev.room, ev.camera
                hyp.confidence_location = ev.score
                hyp.confidence_visibility = ev.score * gain_vis
                hyp.state = BeliefState.PRESENT_CONFIRMED
            else:
                hyp.state = BeliefState.SUSPECTED_ELSEWHERE
                logger.debug(
                    f"[BeliefResolver] {ev.entity_key} weak sighting in "
                    f"{ev.room} ({ev.score:.2f}) — pin holds at {hyp.room}"
                )

        hyp.confidence_identity = _lift(hyp.confidence_identity, ev.score)
        hyp.last_confirmed_ts = ev.ts
        hyp.last_evidence_ts = ev.ts
        hyp.recompute_state_confidence()
        hyp.evidence_breakdown = {"last": "sighting", "score": round(ev.score, 3)}
        await self._persist_evidence(ev)
        await self._persist_belief(hyp)
        await self._on_transition(hyp, prev_state, prev_room)

    async def _ingest_absence(self, ev: EvidenceFrame, hyp: BeliefHypothesis) -> None:
        prev_state, prev_room = hyp.state, hyp.room
        hyp.consecutive_misses += 1
        hyp.consecutive_hits = 0
        hyp.confidence_visibility = round(
            hyp.confidence_visibility * self._cfg["absence_visibility_decay"], 4
        )
        hyp.confidence_location = round(
            hyp.confidence_location * self._cfg["absence_location_decay"], 4
        )
        if hyp.confidence_location < self._cfg["departed_threshold"]:
            # Location confidence has decayed past the floor — absent long
            # enough that we no longer believe it is even in the room.
            # This check must live here, not only in _decay_loop: the decay
            # loop skips entities with recent evidence, and a continuous
            # stream of absence frames IS recent evidence — so without this
            # an entity that left frame would decay toward zero confidence
            # but stay PRESENT_UNSEEN forever. Going DEPARTED also stops the
            # absence spam: the _on_observation absence loop only feeds
            # hypotheses still in a PRESENT_* state.
            hyp.state = BeliefState.DEPARTED
        elif (hyp.state == BeliefState.PRESENT_CONFIRMED
                and hyp.consecutive_misses >= self._cfg["present_unseen_misses"]
                and hyp.confidence_visibility
                < self._cfg["present_unseen_threshold"]):
            # Not detected for a sustained RUN of frames — believed still
            # present (location confidence holds), just not currently
            # observable. The miss-run gate is what stops the flapping: a
            # 1-2 frame YOLO gap no longer flips the label. (Door-
            # disappearance / white-dog-on-white-blanket rule.)
            hyp.state = BeliefState.PRESENT_UNSEEN
        hyp.last_evidence_ts = ev.ts
        hyp.recompute_state_confidence()
        hyp.evidence_breakdown = {"last": "absence"}
        await self._persist_evidence(ev)
        await self._persist_belief(hyp)
        await self._on_transition(hyp, prev_state, prev_room)

    # ── Decay ────────────────────────────────────────────────────────────────

    async def _decay_loop(self) -> None:
        """Slowly decay location confidence for hypotheses that have had no
        evidence at all for a while (camera off, entity outside coverage)."""
        interval = float(self._cfg["decay_interval_s"])
        while True:
            try:
                await asyncio.sleep(interval)
                now = _utcnow()
                async with self._lock:
                    for hyp in list(self._hyp.values()):
                        if hyp.last_evidence_ts is None:
                            continue
                        age = (now - _aware(hyp.last_evidence_ts)).total_seconds()
                        if age < interval:
                            continue
                        prev_state, prev_room = hyp.state, hyp.room
                        hyp.confidence_location = round(
                            hyp.confidence_location
                            * self._cfg["stale_location_decay"], 4
                        )
                        hyp.confidence_visibility = round(
                            hyp.confidence_visibility * 0.8, 4
                        )
                        if hyp.confidence_location < self._cfg["departed_threshold"]:
                            hyp.state = BeliefState.DEPARTED
                        elif hyp.state == BeliefState.PRESENT_CONFIRMED:
                            hyp.state = BeliefState.PRESENT_UNSEEN
                        hyp.recompute_state_confidence()
                        await self._persist_belief(hyp)
                        await self._on_transition(hyp, prev_state, prev_room)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("[BeliefResolver] decay loop iteration failed")

    # ── Persistence ──────────────────────────────────────────────────────────

    async def _persist_evidence(self, ev: EvidenceFrame) -> None:
        try:
            await self._db.execute(
                "INSERT INTO belief_evidence "
                "(ts, entity_key, source, room, camera, evidence_type, "
                " score, payload) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (_iso(ev.ts), ev.entity_key, ev.source, ev.room, ev.camera,
                 ev.evidence_type, float(ev.score), ev.payload_json()),
            )
        except Exception as e:
            logger.debug(f"[BeliefResolver] evidence persist failed: {e}")

    async def prune_evidence(self, *, retain_days: int = 30) -> int:
        """Delete `belief_evidence` rows older than `retain_days`.

        `belief_evidence` is an append-only evidence log — the durable
        belief state lives in `entity_beliefs`. It is the D4-era sibling
        of `world_entity_events` and grows just as fast (~90k rows within
        days of going live). The nightly maintenance pass calls this so
        the table cannot grow without bound. Returns rows deleted.
        """
        if self._db is None:
            return 0
        cutoff = (_utcnow() - timedelta(days=int(retain_days))).isoformat()
        try:
            row = await self._db.fetchone(
                "SELECT COUNT(*) AS n FROM belief_evidence WHERE ts < ?",
                (cutoff,),
            )
            n = int(row["n"]) if row else 0
            if n:
                await self._db.execute(
                    "DELETE FROM belief_evidence WHERE ts < ?", (cutoff,)
                )
            return n
        except Exception as e:
            logger.debug(f"[BeliefResolver] evidence prune failed: {e}")
            return 0

    async def _persist_belief(self, hyp: BeliefHypothesis) -> None:
        try:
            await self._db.execute(
                "INSERT OR REPLACE INTO entity_beliefs "
                "(hypothesis_id, entity_key, entity_type, state, room, camera, "
                " confidence_identity, confidence_location, confidence_visibility, "
                " confidence_state, is_primary, last_confirmed_ts, "
                " last_evidence_ts, evidence_breakdown, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (hyp.hypothesis_id, hyp.entity_key, hyp.entity_type, hyp.state,
                 hyp.room, hyp.camera, hyp.confidence_identity,
                 hyp.confidence_location, hyp.confidence_visibility,
                 hyp.confidence_state, 1 if hyp.is_primary else 0,
                 _iso(hyp.last_confirmed_ts), _iso(hyp.last_evidence_ts),
                 hyp.evidence_breakdown_json(), _iso(_utcnow())),
            )
        except Exception as e:
            logger.debug(f"[BeliefResolver] belief persist failed: {e}")

    async def _load_existing(self) -> None:
        try:
            rows = await self._db.fetchall(
                "SELECT * FROM entity_beliefs WHERE is_primary = 1"
            )
        except Exception as e:
            logger.debug(f"[BeliefResolver] load skipped: {e}")
            return
        for r in rows:
            # Restore last_*_ts too: the decay loop skips any hypothesis
            # with last_evidence_ts is None, so a belief restored without
            # its timestamps would never decay after a restart until fresh
            # evidence happened to arrive.
            self._hyp[r["entity_key"]] = BeliefHypothesis(
                hypothesis_id=r["hypothesis_id"], entity_key=r["entity_key"],
                entity_type=r["entity_type"], state=r["state"],
                room=r["room"], camera=r["camera"],
                confidence_identity=r["confidence_identity"],
                confidence_location=r["confidence_location"],
                confidence_visibility=r["confidence_visibility"],
                confidence_state=r["confidence_state"],
                last_confirmed_ts=_parse_dt(r["last_confirmed_ts"]),
                last_evidence_ts=_parse_dt(r["last_evidence_ts"]),
            )

    # ── Introspection (dashboard / D4b hand-off) ─────────────────────────────

    def snapshot(self) -> list[dict]:
        """Current primary hypotheses — for a dashboard panel or a
        shadow-vs-live comparison."""
        return [
            {
                "entity_key": h.entity_key, "state": h.state, "room": h.room,
                "confidence_location": h.confidence_location,
                "confidence_visibility": h.confidence_visibility,
                "confidence_state": h.confidence_state,
            }
            for h in self._hyp.values()
        ]

    async def _on_transition(
        self, hyp: BeliefHypothesis, prev_state: str, prev_room: Optional[str],
    ) -> None:
        """Handle a belief change: always log it; when LIVE (not shadow),
        also publish a world.belief_changed event so consumers (dashboard,
        and eventually presence) can react. In shadow mode nothing is
        published — the resolver stays observe-only."""
        if hyp.state == prev_state and hyp.room == prev_room:
            return
        logger.info(
            f"[BeliefResolver]{' (shadow)' if self.shadow else ' (live)'} "
            f"{hyp.entity_key}: {prev_state}@{prev_room} → "
            f"{hyp.state}@{hyp.room} "
            f"(loc={hyp.confidence_location:.2f} "
            f"vis={hyp.confidence_visibility:.2f})"
        )
        if self.shadow:
            return
        try:
            await self._bus.publish("world.belief_changed", {
                "entity_key": hyp.entity_key,
                "entity_type": hyp.entity_type,
                "state": hyp.state,
                "room": hyp.room,
                "prev_state": prev_state,
                "prev_room": prev_room,
                "confidence_state": hyp.confidence_state,
                "confidence_location": hyp.confidence_location,
                "ts": _iso(hyp.last_evidence_ts),
            })
        except Exception as e:
            logger.debug(f"[BeliefResolver] belief_changed publish failed: {e}")


def _lift(current: float, gain: float) -> float:
    """Raise a confidence toward 1.0 by `gain` of the remaining headroom."""
    return round(min(1.0, current + (1.0 - current) * max(0.0, min(1.0, gain))), 4)


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


def _iso(ts: Optional[datetime]) -> Optional[str]:
    return _aware(ts).isoformat() if ts is not None else None


def _parse_dt(value: Any) -> Optional[datetime]:
    """Parse an ISO timestamp stored in SQLite back to an aware datetime.
    Returns None for empty/garbage so callers stay None-safe."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None
