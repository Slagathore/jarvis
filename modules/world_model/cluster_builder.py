"""
JARVIS — World Model
====================
Mission: Phase 4 (§22.5) — cold-start cluster protocol for animal
         disambiguation. On day 1 the system has zero behavioral data
         and can't tell Spooky from Velcro (both black) or Smudge from
         Onyx (also both black, in the example household). Rather than
         guessing, we COLLECT — every animal detection of a tracked
         species accumulates as an unattributed observation cluster
         until we have ~200 events. Then we cluster the embeddings,
         surface the clusters in the dashboard, the user labels them,
         and we re-attribute the events.

         AnimalClusterBuilder runs on-demand (called by the dashboard
         "build clusters" button). apply_cluster_labels() is invoked
         by the dashboard's submit handler with a {cluster_id: name}
         dict. Both are async.

Modules: modules/world_model/cluster_builder.py
Classes: AnimalClusterBuilder
Funcs:   apply_cluster_labels
Spec:    new 2.md §22.5.

Note: For the K-means input vector, we read color_class + room from
      the event row's TOP-LEVEL columns (always present), and read
      size_normalized from the event's `metadata` JSON column. The
      visual_embedding is currently None until §23 lands the CLIP
      encoder; the clustering still works because color_class +
      room one-hots dominate for the household's discriminating
      pairs (Spooky/Velcro by room, Sparta/Serval by size).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np
from loguru import logger

from modules.world_model.store import WorldStore


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# Default rooms used for the room one-hot. Override via config in
# `cluster_builder.rooms` if your topology differs. Order matters
# (it's the index into the one-hot), so don't reorder casually.
DEFAULT_ROOMS = ["office", "living_room", "bedroom", "kitchen", "laundry_room"]
DEFAULT_COLORS = ["striped", "tabby-and-white", "black", "tuxedo",
                  "silver-tabby", "cream", "brindle", "tan", "brown",
                  "white", "tricolor", "merle", "unknown"]


class AnimalClusterBuilder:
    """
    Cold-start clusterer for cats or dogs. Generic over species — pass
    `species="cat"` or `species="dog"` (or extend `tracked_species`).
    """

    def __init__(self, store: WorldStore, config: dict) -> None:
        self.store = store
        self.cfg = config or {}
        self.rooms: list[str] = list(
            self.cfg.get("rooms") or DEFAULT_ROOMS
        )
        self.colors: list[str] = list(
            self.cfg.get("colors") or DEFAULT_COLORS
        )

    async def cluster_unattributed(
        self,
        species: str = "cat",
        n_clusters: Optional[int] = None,
        days_back: int = 7,
    ) -> dict[int, list[str]]:
        """
        Pull every unattributed event for `species` from the last
        `days_back` days. K-means cluster their feature vectors.
        Returns {cluster_id: [event_id, ...]}. Empty dict when the
        threshold isn't met.

        `n_clusters` defaults to the number of declared resident
        entities of this species (so K=4 for 4 declared cats); the
        dashboard can pass an override if it wants a different K.
        """
        cutoff = _utcnow() - timedelta(days=days_back)
        rows = await self.store.db.fetchall(
            "SELECT * FROM world_entity_events "
            "WHERE entity_type = ? "
            "AND ts >= ? "
            "AND event_type IN ('first_seen', 'moved_within_room', "
            "'reappeared', 'moved_to') "
            "AND (entity_name IS NULL OR entity_name LIKE 'unknown_%') "
            "ORDER BY ts DESC LIMIT ?",
            (species, cutoff.isoformat(), 20000),
        )
        unattrib = [dict(r) for r in rows]
        threshold = int(self.cfg.get("cluster_min_observations", 200))
        if len(unattrib) < threshold:
            logger.info(
                f"[ClusterBuilder] {species}: {len(unattrib)}/{threshold} "
                "unattributed events — below threshold"
            )
            return {}

        # K = declared residents of this species, unless caller overrode.
        if n_clusters is None:
            n_clusters = await self._declared_resident_count(species)
            if n_clusters < 2:
                logger.info(
                    f"[ClusterBuilder] {species}: need ≥2 resident pets "
                    "for K-means — skipping"
                )
                return {}

        features: list[np.ndarray] = []
        for e in unattrib:
            features.append(self._featurize(e))
        X = np.stack(features)

        # Lazy import — sklearn pulls scipy bits we'd rather not require
        # at module load time for callers that never cluster.
        from sklearn.cluster import KMeans
        km = KMeans(
            n_clusters=int(n_clusters), n_init="auto", random_state=42,
        ).fit(X)

        clusters: dict[int, list[str]] = {i: [] for i in range(int(n_clusters))}
        labels = km.labels_
        if labels is None:
            logger.warning(f"[ClusterBuilder] {species}: KMeans produced no labels")
            return {}
        for label, e in zip(labels, unattrib):
            clusters[int(label)].append(e["id"])
        logger.info(
            f"[ClusterBuilder] {species}: built {len(clusters)} clusters "
            f"from {len(unattrib)} events"
        )
        return clusters

    # ── helpers ────────────────────────────────────────────────────────────

    def _featurize(self, event: dict) -> np.ndarray:
        meta = _decode_metadata(event.get("metadata"))
        color = meta.get("color_class", "unknown")
        room = event.get("room")
        size = meta.get("size_normalized")
        visual = meta.get("visual_embedding")  # None until §23

        color_oh = self._color_one_hot(color)
        room_oh = self._room_one_hot(room)
        if visual is not None:
            try:
                visual_vec = np.asarray(visual, dtype=np.float32)
                visual_vec = visual_vec / (np.linalg.norm(visual_vec) + 1e-9)
            except Exception:
                visual_vec = np.zeros(512, dtype=np.float32)
        else:
            # Fall back to color-histogram if we have it (cheap), else zeros.
            hist = meta.get("color_histogram")
            if hist is not None:
                try:
                    visual_vec = np.asarray(hist, dtype=np.float32)
                except Exception:
                    visual_vec = np.zeros(64, dtype=np.float32)
            else:
                visual_vec = np.zeros(64, dtype=np.float32)

        size_vec = np.array(
            [float(size) if size is not None else 0.04], dtype=np.float32
        )
        # Weights match §22.5: visual base, color×5 (hard signal), room×2.
        return np.concatenate([
            visual_vec,
            color_oh * 5.0,
            room_oh * 2.0,
            size_vec * 3.0,
        ]).astype(np.float32)

    def _color_one_hot(self, color: Optional[str]) -> np.ndarray:
        v = np.zeros(len(self.colors), dtype=np.float32)
        token = color or "unknown"
        if token in self.colors:
            v[self.colors.index(token)] = 1.0
        else:
            v[self.colors.index("unknown")] = 1.0
        return v

    def _room_one_hot(self, room: Optional[str]) -> np.ndarray:
        v = np.zeros(len(self.rooms), dtype=np.float32)
        if room and room in self.rooms:
            v[self.rooms.index(room)] = 1.0
        return v

    async def _declared_resident_count(self, species: str) -> int:
        """How many resident pets of this species exist in the entity
        table? K-means K defaults to this so we don't over-cluster."""
        rows = await self.store.db.fetchall(
            "SELECT COUNT(*) AS n FROM world_entities "
            "WHERE entity_type = ? AND is_resident = 1 "
            "AND archived_at IS NULL",
            (species,),
        )
        if not rows:
            return 0
        return int(rows[0]["n"] or 0)


# ── apply_cluster_labels ────────────────────────────────────────────────────


async def apply_cluster_labels(
    store: WorldStore,
    cluster_to_pet_name: dict[int, str],
    clusters: dict[int, list[str]],
    species: str = "cat",
) -> int:
    """
    Re-attribute every event in a cluster to the named resident pet.
    Returns the number of events updated. The caller (dashboard) owns
    triggering a profile rebuild after this.
    """
    pets_by_name = {
        e.display_name: e for e in await store.load_entities()
        if e.entity_type == species and e.display_name
    }

    total_updated = 0
    for cluster_id, pet_name in cluster_to_pet_name.items():
        target = pets_by_name.get(pet_name)
        if target is None:
            logger.warning(
                f"[ClusterBuilder] no resident {species} named '{pet_name}'; "
                "skipping cluster"
            )
            continue
        event_ids = clusters.get(cluster_id) or []
        if not event_ids:
            continue
        placeholders = ",".join("?" for _ in event_ids)
        await store.db.execute(
            f"UPDATE world_entity_events "
            f"SET entity_id = ?, entity_name = ? "
            f"WHERE id IN ({placeholders})",
            (target.id, pet_name, *event_ids),
        )
        total_updated += len(event_ids)
        logger.info(
            f"[ClusterBuilder] re-attributed {len(event_ids)} events "
            f"to '{pet_name}' ({species})"
        )
    return total_updated


# ── helpers ─────────────────────────────────────────────────────────────────


def _decode_metadata(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            return json.loads(raw)
        except Exception:
            return {}
    return {}
