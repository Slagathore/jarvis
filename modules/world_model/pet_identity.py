"""
JARVIS — World Model
====================
Mission: Lightweight resident-pet visual prototype bank.

Confirmed dashboard tags write one crop descriptor per pet. Later live
cat/dog boxes compare against those confirmed examples before the UI
shows a suggested name. This is intentionally conservative: it is not a
deep animal re-identification model, but it gives the system a real
persistent visual memory instead of relying only on recent event overlap.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from modules.world_model.geometry import bbox_iou


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clamp_crop(
    frame: np.ndarray,
    bbox: list[float] | tuple[float, float, float, float],
) -> tuple[Optional[np.ndarray], tuple[int, int, int, int]]:
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
    x1 = max(0, min(fw - 1, x1))
    y1 = max(0, min(fh - 1, y1))
    x2 = max(0, min(fw, x2))
    y2 = max(0, min(fh, y2))
    if x2 <= x1 or y2 <= y1:
        return None, (x1, y1, x2, y2)
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def descriptor_from_crop(
    species: str,
    crop: Optional[np.ndarray],
    *,
    room: Optional[str],
    bbox: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
) -> dict:
    """Build the same cheap visual descriptor family used by the animal
    observation path. Values are JSON-serializable for storage."""
    from modules.vision.observation_builder import (
        _classify_cat_color,
        _classify_dog_color,
        _coat_texture_descriptor,
        _coarse_breed_class,
        _color_histogram,
    )

    x1, y1, x2, y2 = bbox
    frame_area = max(frame_width * frame_height, 1)
    size_norm = max(0, x2 - x1) * max(0, y2 - y1) / frame_area
    hist = _color_histogram(crop)
    color = (
        _classify_dog_color(crop)
        if species == "dog"
        else _classify_cat_color(crop)
    )
    out: dict[str, Any] = {
        "species": species,
        "room": room,
        "bbox": [x1, y1, x2, y2],
        "frame_width": int(frame_width),
        "frame_height": int(frame_height),
        "size_normalized": float(size_norm),
        "color_class": color,
        "color_histogram": hist.tolist() if hist is not None else None,
    }
    if species == "dog":
        out["breed_class"] = _coarse_breed_class(
            crop, max(0, x2 - x1), max(0, y2 - y1)
        )
    else:
        out["coat_texture"] = _coat_texture_descriptor(crop)
    return out


async def save_confirmed_pet_sample(
    *,
    db: Any,
    pet_entity: Any,
    frame: np.ndarray,
    room: str,
    bbox: list[float] | tuple[float, float, float, float],
    sample_dir: Path,
    source: str = "manual_tag",
) -> Optional[int]:
    crop, clamped = clamp_crop(frame, bbox)
    if crop is None or crop.size == 0:
        return None
    sample_dir.mkdir(parents=True, exist_ok=True)
    name_stem = "".join(
        c if c.isalnum() or c in ("-", "_") else "_"
        for c in str(pet_entity.display_name or pet_entity.id)
    ).strip("_") or str(pet_entity.id)
    path = sample_dir / (
        f"{pet_entity.entity_type}_{name_stem}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_"
        f"{uuid.uuid4().hex[:6]}.jpg"
    )
    try:
        import cv2
        if not cv2.imwrite(str(path), crop):
            path = None  # type: ignore[assignment]
    except Exception:
        path = None  # type: ignore[assignment]

    fh, fw = frame.shape[:2]
    desc = descriptor_from_crop(
        pet_entity.entity_type,
        crop,
        room=room,
        bbox=clamped,
        frame_width=fw,
        frame_height=fh,
    )
    return await db.execute(
        "INSERT INTO pet_visual_samples "
        "(pet_entity_id, pet_name, species, created_at, room, bbox, "
        "crop_path, descriptor_json, source) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            pet_entity.id,
            pet_entity.display_name,
            pet_entity.entity_type,
            _now_iso(),
            room,
            json.dumps(list(clamped)),
            str(path) if path is not None else None,
            json.dumps(desc),
            source,
        ),
    )


async def match_pet_from_crop(
    *,
    db: Any,
    species: str,
    frame: np.ndarray,
    room: str,
    bbox: list[float] | tuple[float, float, float, float],
    max_samples_per_pet: int = 20,
) -> Optional[dict]:
    crop, clamped = clamp_crop(frame, bbox)
    if crop is None or crop.size == 0:
        return None
    fh, fw = frame.shape[:2]
    query = descriptor_from_crop(
        species,
        crop,
        room=room,
        bbox=clamped,
        frame_width=fw,
        frame_height=fh,
    )
    rows = await db.fetchall(
        "SELECT pet_entity_id, pet_name, species, room, bbox, descriptor_json "
        "FROM pet_visual_samples "
        "WHERE species = ? "
        "ORDER BY created_at DESC LIMIT ?",
        (species, max(50, max_samples_per_pet * 12)),
    )
    per_pet_seen: dict[str, int] = {}
    costs: dict[str, list[float]] = {}
    entity_for_name: dict[str, str] = {}
    for r in rows:
        name = r["pet_name"]
        if not name:
            continue
        n_seen = per_pet_seen.get(name, 0)
        if n_seen >= max_samples_per_pet:
            continue
        per_pet_seen[name] = n_seen + 1
        try:
            desc = json.loads(r["descriptor_json"] or "{}")
        except Exception:
            continue
        cost = _descriptor_cost(query, desc, current_room=room)
        costs.setdefault(name, []).append(cost)
        entity_for_name[name] = r["pet_entity_id"]
    if not costs:
        return None

    ranked: list[tuple[str, float]] = []
    for name, vals in costs.items():
        vals_sorted = sorted(vals)
        # Average up to the three best samples for this pet. This keeps
        # one ugly crop from dominating, but still rewards repeated agreement.
        ranked.append((name, float(np.mean(vals_sorted[:3]))))
    ranked.sort(key=lambda x: x[1])
    best_name, best_cost = ranked[0]
    second_cost = ranked[1][1] if len(ranked) > 1 else 1.0
    margin = second_cost - best_cost
    score = max(0.0, min(1.0, 1.0 - best_cost))

    # Conservative thresholds: weak descriptors can hint, but should not
    # confidently rename a pet unless the nearest prototype is both good
    # and separated from the runner-up.
    accepted = best_cost <= 0.42 and margin >= 0.08
    return {
        "pet_name": best_name,
        "entity_id": entity_for_name.get(best_name),
        "score": round(score, 3),
        "margin": round(float(margin), 3),
        "accepted": accepted,
        "sample_count": sum(len(v) for v in costs.values()),
    }


def _descriptor_cost(query: dict, sample: dict, *, current_room: str) -> float:
    q_hist = query.get("color_histogram")
    s_hist = sample.get("color_histogram")
    hist_cost = _hist_bhattacharyya(q_hist, s_hist)

    q_color = query.get("color_class", "unknown")
    s_color = sample.get("color_class", "unknown")
    if q_color != "unknown" and s_color != "unknown" and q_color != s_color:
        color_cost = 0.55
    else:
        color_cost = 0.0 if q_color == s_color and q_color != "unknown" else 0.18

    q_size = query.get("size_normalized")
    s_size = sample.get("size_normalized")
    if q_size is not None and s_size is not None:
        size_cost = min(
            abs(np.log(max(float(q_size), 1e-4) / max(float(s_size), 1e-4))) / 2.0,
            1.0,
        )
    else:
        size_cost = 0.5

    room_cost = 0.0 if sample.get("room") == current_room else 0.18

    texture_cost = _texture_cost(query, sample)

    q_bbox = query.get("bbox")
    s_bbox = sample.get("bbox")
    if sample.get("room") == current_room and q_bbox and s_bbox:
        spatial_cost = 1.0 - bbox_iou(tuple(q_bbox), tuple(s_bbox))
    else:
        spatial_cost = 0.5

    return float(
        0.36 * hist_cost
        + 0.20 * color_cost
        + 0.16 * size_cost
        + 0.12 * texture_cost
        + 0.10 * room_cost
        + 0.06 * spatial_cost
    )


def _hist_bhattacharyya(h1: Any, h2: Any) -> float:
    if h1 is None or h2 is None:
        return 0.5
    a = np.asarray(h1, dtype=np.float32)
    b = np.asarray(h2, dtype=np.float32)
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return 0.5
    bc = float(np.sum(np.sqrt(a * b)))
    return float(min(1.0, np.sqrt(max(0.0, 1.0 - bc))))


def _texture_cost(query: dict, sample: dict) -> float:
    for key in ("coat_texture", "breed_class"):
        q = query.get(key)
        s = sample.get(key)
        if q is None or s is None:
            continue
        if isinstance(q, str) or isinstance(s, str):
            return 0.0 if q == s else 0.35
        qa = np.asarray(q, dtype=np.float32)
        sa = np.asarray(s, dtype=np.float32)
        if qa.size and qa.shape == sa.shape:
            denom = float(np.linalg.norm(qa) * np.linalg.norm(sa) + 1e-9)
            sim = float(np.dot(qa, sa) / denom)
            return max(0.0, min(1.0, 1.0 - sim))
    return 0.5
