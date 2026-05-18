"""
JARVIS — World Model / Belief
=============================
Mission: BeliefResolver — the belief-state tracker (audit roadmap D4).

         It subscribes to `vision.observation`, turns every observation
         into an EvidenceFrame, and maintains COMPETING HYPOTHESES about
         each entity with the four-axis confidence model from
         belief.types.

MULTI-HYPOTHESIS MODEL (D4b):
         Each entity carries a *list* of hypotheses — exactly one
         `is_primary` (the projection: "where Jarvis believes X is") plus
         zero or more competitors. A weak sighting of X in a new room
         does NOT yank the confident primary pin; it creates a SECONDARY
         hypothesis there in state SUSPECTED_ELSEWHERE. The secondary is
         only promoted to primary once its location confidence clears the
         primary's by `cross_room_move_margin` — i.e. the entity is now
         more likely there than where it was pinned. So the resolver can
         hold, correctly:

             Summer  primary   : PRESENT_UNSEEN      @ bedroom
             Summer  secondary : SUSPECTED_ELSEWHERE @ kitchen

         Dead secondaries are pruned; if a primary decays to DEPARTED and
         a live secondary exists, the secondary takes over.

LIVE vs SHADOW:
         `shadow: false` → on every primary transition the resolver
         publishes `world.belief_changed`. `shadow: true` → observe-only.

         SCOPE: people (keyed by person_id) and pets (cat/dog, keyed
         coarsely obj_class:room). Per-animal identity is not resolved
         here. Objects are out of scope.

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

_PRESENT_STATES = (BeliefState.PRESENT_CONFIRMED, BeliefState.PRESENT_UNSEEN)

# Tuning. Conservative defaults; overridable via the world_model.
# belief_resolver config block.
_DEFAULTS = {
    "shadow": True,                  # never leave shadow without D4b done
    "sighting_location_gain": 0.45,  # how hard a sighting pulls location conf
    "sighting_visibility_gain": 0.6,
    "absence_visibility_decay": 0.55,  # multiply visibility on a missed frame
    "absence_location_decay": 0.97,    # location barely moves on a missed frame
    "cross_room_move_margin": 0.30,    # a secondary must beat the primary's
                                       # location confidence by this to be promoted
    "present_unseen_threshold": 0.30,  # visibility below this → PRESENT_UNSEEN
    # Hysteresis: the discrete CONFIRMED<->UNSEEN state flips only on a RUN
    # of consecutive misses / hits, not a single noisy frame.
    "present_unseen_misses": 6,        # consec. misses: CONFIRMED → UNSEEN
    "present_confirmed_hits": 3,       # consec. hits:   UNSEEN → CONFIRMED
    "decay_interval_s": 30.0,
    "stale_location_decay": 0.92,      # per decay tick with no evidence
    "departed_threshold": 0.12,        # location below this → DEPARTED
    "max_hypotheses": 4,               # cap competitors per entity
    # belief_evidence write throttle (D4b tuning data, not a per-frame log).
    "evidence_min_interval_s": 15.0,
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class BeliefResolver:
    """Multi-hypothesis belief-state tracker. Constructed after
    WorldModel; started/stopped by the orchestrator."""

    def __init__(self, bus: Any, db: Any, config: Optional[dict] = None) -> None:
        self._bus = bus
        self._db = db
        cfg = dict(_DEFAULTS)
        if config:
            cfg.update({k: v for k, v in config.items() if k in _DEFAULTS})
        self._cfg = cfg
        self.shadow: bool = bool(cfg["shadow"])
        self._evidence_min_interval_s = float(cfg["evidence_min_interval_s"])
        self._max_hyps = int(cfg["max_hypotheses"])
        # entity_key → list of hypotheses; exactly one has is_primary=True.
        self._entities: dict[str, list[BeliefHypothesis]] = {}
        # entity_key → ts of its last persisted belief_evidence row.
        self._last_evidence_persist: dict[str, datetime] = {}
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
        n_hyp = sum(len(v) for v in self._entities.values())
        logger.info(
            f"[BeliefResolver] started (shadow={self.shadow}, "
            f"{len(self._entities)} entit(ies) / {n_hyp} hypothesis(es) restored)"
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

    # ── Hypothesis-set helpers ───────────────────────────────────────────────

    def _primary(self, key: str) -> Optional[BeliefHypothesis]:
        for h in self._entities.get(key, []):
            if h.is_primary:
                return h
        return None

    def _hyp_in_room(self, key: str, room: Optional[str]) -> Optional[BeliefHypothesis]:
        for h in self._entities.get(key, []):
            if h.room == room:
                return h
        return None

    def _new_hyp(
        self, key: str, etype: str, room: Optional[str],
        camera: Optional[str], *, primary: bool,
    ) -> BeliefHypothesis:
        h = BeliefHypothesis(
            hypothesis_id=str(uuid.uuid4()), entity_key=key,
            entity_type=etype, room=room, camera=camera, is_primary=primary,
        )
        self._entities.setdefault(key, []).append(h)
        return h

    def _promote(self, key: str, hyp: BeliefHypothesis) -> None:
        """Make `hyp` the sole primary for its entity."""
        for h in self._entities.get(key, []):
            h.is_primary = (h is hyp)

    def _primary_view(self, key: str) -> tuple[Optional[str], Optional[str]]:
        p = self._primary(key)
        return (p.state, p.room) if p is not None else (None, None)

    # ── Event ingest ─────────────────────────────────────────────────────────

    async def _on_observation(self, payload: dict) -> None:
        """Handle one `vision.observation` batch (one camera, one room)."""
        try:
            room = payload.get("room")
            camera = payload.get("camera")
            ts = payload.get("ts") or _utcnow()
            observations = payload.get("observations") or []

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
                        etype = obj_class
                        score = float(getattr(obs, "confidence", 0.0))
                        omd = getattr(obs, "metadata", {}) or {}
                        # Individuate the pet. The default key is the coarse
                        # `species:room` — which collapses two cats in one
                        # room into a single "a cat is here" hypothesis. If
                        # the confirmed-sample bank can name this animal, key
                        # by its identity so each pet gets its own belief.
                        # Falls back to the coarse key on no match / no
                        # samples / any error.
                        pet_id, pet_name = await self._resolve_pet(
                            obj_class, room, obs, omd
                        )
                        key = (
                            f"pet:{pet_id}" if pet_id is not None
                            else f"{obj_class}:{room}"
                        )
                        meta = {
                            "color_class": omd.get("color_class"),
                            "size_normalized": omd.get("size_normalized"),
                            "pet_id": pet_id,
                            "pet_name": pet_name,
                        }
                    else:
                        continue
                    seen_keys.add(key)
                    ev = EvidenceFrame(
                        ts=ts, entity_key=key, entity_type=etype,
                        source="vision.observation", evidence_type="sighting",
                        room=room, camera=camera, score=score,
                        bbox=getattr(obs, "bbox", None), payload=meta,
                    )
                    await self._ingest_sighting(ev)

                # Absence: a hypothesis pinned to THIS room whose entity was
                # not in this batch gets weak negative evidence. Only the
                # `room` hypothesis is affected — competitors elsewhere are
                # not observable by this camera.
                for key in list(self._entities.keys()):
                    if key in seen_keys:
                        continue
                    for hyp in list(self._entities.get(key, [])):
                        if hyp.room == room and hyp.state in _PRESENT_STATES:
                            ev = EvidenceFrame(
                                ts=ts, entity_key=key,
                                entity_type=hyp.entity_type,
                                source="vision.observation",
                                evidence_type="absence", room=room,
                                camera=camera,
                                score=self._cfg["absence_visibility_decay"],
                            )
                            await self._ingest_absence(ev, hyp)
        except Exception:
            logger.exception("[BeliefResolver] observation ingest failed")

    async def _resolve_pet(
        self, species: str, room: Optional[str], obs: Any, omd: dict,
    ) -> tuple[Optional[Any], Optional[str]]:
        """Best-effort per-animal identity for a cat/dog observation.
        Returns (pet_entity_id, pet_name) when the confirmed-sample bank
        names this pet with enough confidence, else (None, None). Reuses the
        visual descriptor already computed into the observation metadata —
        no frame, no re-cropping."""
        try:
            from modules.world_model.pet_identity import (
                match_pet_from_descriptor,
            )
            bbox = getattr(obs, "bbox", None)
            query: dict[str, Any] = {
                "species": species,
                "room": room,
                "bbox": list(bbox) if bbox else None,
                "frame_width": omd.get("frame_width"),
                "frame_height": omd.get("frame_height"),
                "size_normalized": omd.get("size_normalized"),
                "color_class": omd.get("color_class", "unknown"),
                "color_histogram": omd.get("color_histogram"),
            }
            if species == "dog":
                query["breed_class"] = omd.get("breed_class")
            else:
                query["coat_texture"] = omd.get("coat_texture")
            match = await match_pet_from_descriptor(
                db=self._db, species=species, room=room or "", query=query,
            )
            if (match and match.get("accepted")
                    and match.get("entity_id") is not None):
                return match["entity_id"], match.get("pet_name")
        except Exception as e:
            logger.debug(f"[BeliefResolver] pet identity resolve failed: {e}")
        return None, None

    async def _ingest_sighting(self, ev: EvidenceFrame) -> None:
        key = ev.entity_key
        prev_state, prev_room = self._primary_view(key)
        gain_loc = self._cfg["sighting_location_gain"]
        gain_vis = self._cfg["sighting_visibility_gain"]

        # First-ever sighting of this entity → the founding primary.
        if not self._entities.get(key):
            h = self._new_hyp(key, ev.entity_type, ev.room, ev.camera,
                               primary=True)
            h.state = BeliefState.PRESENT_CONFIRMED
            self._reinforce(h, ev, gain_loc, gain_vis)
            await self._commit(key, ev, prev_state, prev_room)
            return

        primary = self._primary(key)
        target = self._hyp_in_room(key, ev.room)
        if target is None:
            # A new room for this entity — a competing SECONDARY hypothesis.
            target = self._new_hyp(key, ev.entity_type, ev.room, ev.camera,
                                   primary=False)
            target.state = BeliefState.SUSPECTED_ELSEWHERE

        self._reinforce(target, ev, gain_loc, gain_vis)

        if primary is None or target is primary:
            # Reinforcing the primary in its own room — hysteretic re-confirm.
            if target.state == BeliefState.PRESENT_UNSEEN:
                if target.consecutive_hits >= self._cfg["present_confirmed_hits"]:
                    target.state = BeliefState.PRESENT_CONFIRMED
            elif target.state != BeliefState.PRESENT_CONFIRMED:
                target.state = BeliefState.PRESENT_CONFIRMED
        else:
            # `target` is a secondary — does the entity now belong there?
            margin = self._cfg["cross_room_move_margin"]
            if target.confidence_location > primary.confidence_location + margin:
                # The new room beats the old pin → the entity moved.
                target.state = BeliefState.PRESENT_CONFIRMED
                primary.state = BeliefState.PRESENT_UNSEEN
                self._promote(key, target)
                logger.info(
                    f"[BeliefResolver] {key} moved {primary.room} → {ev.room} "
                    f"(loc {target.confidence_location:.2f} > "
                    f"{primary.confidence_location:.2f}+{margin})"
                )
            else:
                # Weak — the secondary stands as a competitor, pin holds.
                target.state = BeliefState.SUSPECTED_ELSEWHERE
                logger.debug(
                    f"[BeliefResolver] {key} weak sighting in {ev.room} "
                    f"({ev.score:.2f}) — primary holds at {primary.room}"
                )

        await self._commit(key, ev, prev_state, prev_room)

    def _reinforce(
        self, hyp: BeliefHypothesis, ev: EvidenceFrame,
        gain_loc: float, gain_vis: float,
    ) -> None:
        """Apply a positive sighting to one hypothesis."""
        hyp.camera = ev.camera
        hyp.confidence_location = _lift(hyp.confidence_location,
                                        ev.score * gain_loc)
        hyp.confidence_visibility = _lift(hyp.confidence_visibility,
                                          ev.score * gain_vis)
        hyp.confidence_identity = _lift(hyp.confidence_identity, ev.score)
        hyp.consecutive_hits += 1
        hyp.consecutive_misses = 0
        hyp.last_confirmed_ts = ev.ts
        hyp.last_evidence_ts = ev.ts
        hyp.evidence_breakdown = {"last": "sighting", "score": round(ev.score, 3)}
        # Refresh the human-readable name from this sighting: person_name for
        # people, the individuated pet_name for pets. Falls back to the
        # entity_type ("cat"/"dog"/"person") only when nothing better is known.
        name = (ev.payload or {}).get("person_name") or (ev.payload or {}).get("pet_name")
        if name:
            hyp.display_name = name
        elif not hyp.display_name:
            hyp.display_name = hyp.entity_type

    async def _ingest_absence(
        self, ev: EvidenceFrame, hyp: BeliefHypothesis
    ) -> None:
        key = ev.entity_key
        prev_state, prev_room = self._primary_view(key)
        hyp.consecutive_misses += 1
        hyp.consecutive_hits = 0
        hyp.confidence_visibility = round(
            hyp.confidence_visibility * self._cfg["absence_visibility_decay"], 4
        )
        hyp.confidence_location = round(
            hyp.confidence_location * self._cfg["absence_location_decay"], 4
        )
        if hyp.confidence_location < self._cfg["departed_threshold"]:
            hyp.state = BeliefState.DEPARTED
        elif (hyp.state == BeliefState.PRESENT_CONFIRMED
                and hyp.consecutive_misses >= self._cfg["present_unseen_misses"]
                and hyp.confidence_visibility
                < self._cfg["present_unseen_threshold"]):
            hyp.state = BeliefState.PRESENT_UNSEEN
        hyp.last_evidence_ts = ev.ts
        hyp.evidence_breakdown = {"last": "absence"}
        await self._commit(key, ev, prev_state, prev_room)

    # ── Reconcile + commit ───────────────────────────────────────────────────

    def _reconcile(self, key: str) -> list[str]:
        """Keep an entity's hypothesis set sane. Returns hypothesis_ids
        removed (so the caller deletes their rows):
          - prune DEPARTED secondaries;
          - if the primary is DEPARTED and a live secondary exists,
            promote the strongest live secondary;
          - cap the set at max_hypotheses (drop the weakest secondaries).
        """
        hyps = self._entities.get(key, [])
        if not hyps:
            return []
        removed: list[str] = []

        kept = [h for h in hyps
                if h.is_primary or h.state != BeliefState.DEPARTED]
        removed += [h.hypothesis_id for h in hyps if h not in kept]

        # Re-promote if the primary has departed but a competitor is alive.
        primary = next((h for h in kept if h.is_primary), None)
        if primary is not None and primary.state == BeliefState.DEPARTED:
            live = [h for h in kept
                    if h is not primary and h.state in _PRESENT_STATES]
            if live:
                best = max(live, key=lambda h: h.confidence_location)
                for h in kept:
                    h.is_primary = (h is best)

        # Cap the competitor set — drop the weakest non-primary extras.
        if len(kept) > self._max_hyps:
            secondaries = sorted(
                (h for h in kept if not h.is_primary),
                key=lambda h: h.confidence_location,
            )
            drop = secondaries[: len(kept) - self._max_hyps]
            removed += [h.hypothesis_id for h in drop]
            kept = [h for h in kept if h not in drop]

        self._entities[key] = kept
        return removed

    async def _commit(
        self, key: str, ev: EvidenceFrame,
        prev_state: Optional[str], prev_room: Optional[str],
    ) -> None:
        """Reconcile the entity, persist every live hypothesis, delete
        pruned rows, persist evidence, and fire a transition if the
        PRIMARY (the projection) changed."""
        removed = self._reconcile(key)
        for h in self._entities.get(key, []):
            h.recompute_state_confidence()

        new_state, new_room = self._primary_view(key)
        transitioned = (new_state != prev_state or new_room != prev_room)

        if self._should_persist_evidence(ev, transitioned):
            await self._persist_evidence(ev)
        for hid in removed:
            await self._delete_belief(hid)
        for h in self._entities.get(key, []):
            await self._persist_belief(h)

        primary = self._primary(key)
        if primary is not None and transitioned:
            await self._on_transition(primary, prev_state, prev_room)

    # ── Decay ────────────────────────────────────────────────────────────────

    async def _decay_loop(self) -> None:
        """Slowly decay confidence for hypotheses with no recent evidence
        (camera off, entity outside coverage)."""
        interval = float(self._cfg["decay_interval_s"])
        while True:
            try:
                await asyncio.sleep(interval)
                now = _utcnow()
                async with self._lock:
                    for key in list(self._entities.keys()):
                        prev_state, prev_room = self._primary_view(key)
                        changed = False
                        for hyp in list(self._entities.get(key, [])):
                            if hyp.last_evidence_ts is None:
                                continue
                            age = (now - _aware(hyp.last_evidence_ts)
                                   ).total_seconds()
                            if age < interval:
                                continue
                            changed = True
                            hyp.confidence_location = round(
                                hyp.confidence_location
                                * self._cfg["stale_location_decay"], 4)
                            hyp.confidence_visibility = round(
                                hyp.confidence_visibility * 0.8, 4)
                            if (hyp.confidence_location
                                    < self._cfg["departed_threshold"]):
                                hyp.state = BeliefState.DEPARTED
                            elif hyp.state == BeliefState.PRESENT_CONFIRMED:
                                hyp.state = BeliefState.PRESENT_UNSEEN
                        if not changed:
                            continue
                        removed = self._reconcile(key)
                        for h in self._entities.get(key, []):
                            h.recompute_state_confidence()
                        for hid in removed:
                            await self._delete_belief(hid)
                        for h in self._entities.get(key, []):
                            await self._persist_belief(h)
                        primary = self._primary(key)
                        new_state, new_room = self._primary_view(key)
                        if (primary is not None
                                and (new_state != prev_state
                                     or new_room != prev_room)):
                            await self._on_transition(
                                primary, prev_state, prev_room)
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("[BeliefResolver] decay loop iteration failed")

    # ── Persistence ──────────────────────────────────────────────────────────

    def _should_persist_evidence(
        self, ev: EvidenceFrame, transitioned: bool
    ) -> bool:
        """Gate belief_evidence writes. Transitions always persist; a
        sighting persists at most once per evidence_min_interval_s per
        entity; non-transition absences never persist."""
        if transitioned:
            self._last_evidence_persist[ev.entity_key] = ev.ts
            return True
        if ev.evidence_type != "sighting":
            return False
        last = self._last_evidence_persist.get(ev.entity_key)
        if last is not None:
            elapsed = (_aware(ev.ts) - _aware(last)).total_seconds()
            if elapsed < self._evidence_min_interval_s:
                return False
        self._last_evidence_persist[ev.entity_key] = ev.ts
        return True

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
        """Delete `belief_evidence` rows older than `retain_days`. The
        durable belief state lives in `entity_beliefs`; this table is an
        append-only evidence log the nightly pass keeps bounded."""
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

    async def _delete_belief(self, hypothesis_id: str) -> None:
        try:
            await self._db.execute(
                "DELETE FROM entity_beliefs WHERE hypothesis_id = ?",
                (hypothesis_id,),
            )
        except Exception as e:
            logger.debug(f"[BeliefResolver] belief delete failed: {e}")

    async def _load_existing(self) -> None:
        """Restore every hypothesis (primary + competitors), grouped per
        entity. Each entity is guaranteed exactly one primary afterwards."""
        try:
            rows = await self._db.fetchall("SELECT * FROM entity_beliefs")
        except Exception as e:
            logger.debug(f"[BeliefResolver] load skipped: {e}")
            return
        for r in rows:
            hyp = BeliefHypothesis(
                hypothesis_id=r["hypothesis_id"], entity_key=r["entity_key"],
                entity_type=r["entity_type"], state=r["state"],
                room=r["room"], camera=r["camera"],
                confidence_identity=r["confidence_identity"],
                confidence_location=r["confidence_location"],
                confidence_visibility=r["confidence_visibility"],
                confidence_state=r["confidence_state"],
                is_primary=bool(r["is_primary"]),
                last_confirmed_ts=_parse_dt(r["last_confirmed_ts"]),
                last_evidence_ts=_parse_dt(r["last_evidence_ts"]),
            )
            self._entities.setdefault(r["entity_key"], []).append(hyp)
        # Repair the invariant: exactly one primary per entity.
        for key, hyps in self._entities.items():
            primaries = [h for h in hyps if h.is_primary]
            if len(primaries) == 1:
                continue
            best = max(hyps, key=lambda h: h.confidence_location)
            for h in hyps:
                h.is_primary = (h is best)

    # ── Introspection (dashboard) ────────────────────────────────────────────

    def snapshot(self) -> list[dict]:
        """Per-entity belief view — the primary plus its competitors."""
        out: list[dict] = []
        for key, hyps in self._entities.items():
            primary = self._primary(key)
            out.append({
                "entity_key": key,
                "display_name": primary.display_name if primary else None,
                "primary": self._hyp_dict(primary) if primary else None,
                "competitors": [
                    self._hyp_dict(h) for h in hyps if not h.is_primary
                ],
            })
        return out

    @staticmethod
    def _hyp_dict(h: BeliefHypothesis) -> dict:
        return {
            "state": h.state, "room": h.room,
            "display_name": h.display_name,
            "confidence_location": h.confidence_location,
            "confidence_visibility": h.confidence_visibility,
            "confidence_state": h.confidence_state,
        }

    async def _on_transition(
        self, hyp: BeliefHypothesis, prev_state: Optional[str],
        prev_room: Optional[str],
    ) -> None:
        """The projection (primary hypothesis) changed. Always log it;
        when LIVE, also publish `world.belief_changed`."""
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
                "entity_name": hyp.display_name,
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
    """Parse an ISO timestamp from SQLite back to an aware datetime."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None
