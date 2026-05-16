"""
JARVIS — Ambient Home AI
========================
Mission: Per-camera "ignore zone" support. A framed painting, a TV, a poster
         of a person — anything static that YOLO keeps detecting as a person
         or pet — can be masked out by drawing a polygon over it in the
         dashboard polygon editor. Detections whose box-center falls inside
         an ignore zone for that room are dropped before they ever reach the
         entity layer.

         Zones are stored per room in data/polygon_overrides.json under the
         `ignore_zones` key (written by the polygon editor, same file as
         exits/landmarks). This module loads them with an mtime check so an
         edit in the dashboard takes effect on the next frame — no restart.

Modules: modules/vision/ignore_zones.py
Functions:
    load_ignore_zones()           — {room: [polygon, ...]}, mtime-cached
    point_in_polygon(pt, polygon) — ray-cast point-in-polygon test
    filter_detections(dets, room) — drop detections inside a room's zones
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from loguru import logger

# Same file the polygon editor writes (exits / landmarks / ignore_zones).
_OVERRIDES_PATH = Path("data/polygon_overrides.json")

# mtime-keyed cache so a dashboard edit applies on the next frame without
# re-parsing the JSON on every detection call.
_cache: dict[str, list[list[list[float]]]] = {}
_cache_mtime: float = -1.0


def _refresh() -> None:
    """Reload ignore zones from disk if the overrides file changed."""
    global _cache, _cache_mtime
    try:
        mtime = _OVERRIDES_PATH.stat().st_mtime
    except OSError:
        # File doesn't exist — no zones configured anywhere.
        if _cache:
            _cache = {}
        _cache_mtime = -1.0
        return
    if mtime == _cache_mtime:
        return
    try:
        raw = json.loads(_OVERRIDES_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug(f"[IgnoreZones] overrides parse failed: {e}")
        _cache_mtime = mtime
        return
    zones: dict[str, list[list[list[float]]]] = {}
    for room, cfg in (raw or {}).items():
        if not isinstance(cfg, dict):
            continue
        polys = cfg.get("ignore_zones") or []
        clean: list[list[list[float]]] = []
        for poly in polys:
            # The polygon editor stores each zone as {"polygon": [[x,y],...]}
            # (same shape as exits/landmarks). Accept a raw list too.
            pts = poly.get("polygon") if isinstance(poly, dict) else poly
            if isinstance(pts, list) and len(pts) >= 3:
                try:
                    clean.append(
                        [[float(p[0]), float(p[1])] for p in pts]
                    )
                except (TypeError, ValueError, IndexError):
                    continue
        if clean:
            zones[room] = clean
    _cache = zones
    _cache_mtime = mtime
    if zones:
        logger.info(
            "[IgnoreZones] loaded: "
            + ", ".join(f"{r}×{len(p)}" for r, p in zones.items())
        )


def load_ignore_zones() -> dict[str, list[list[list[float]]]]:
    """Return {room: [polygon, ...]} of currently-configured ignore zones.
    Each polygon is a list of [x, y] in camera-frame pixels."""
    _refresh()
    return _cache


def point_in_polygon(x: float, y: float, polygon: list[list[float]]) -> bool:
    """Ray-casting point-in-polygon test. polygon is [[x, y], ...]."""
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i][0], polygon[i][1]
        xj, yj = polygon[j][0], polygon[j][1]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
        ):
            inside = not inside
        j = i
    return inside


def _box_center(box) -> Optional[tuple[float, float]]:
    try:
        x1, y1, x2, y2 = box
        return (float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0
    except (TypeError, ValueError):
        return None


def filter_detections(detections: list[dict], room: str) -> list[dict]:
    """Drop detections whose box-center lies inside an ignore zone for
    `room`. Returns the kept detections (a new list). Detection dicts use
    the `box` = [x1, y1, x2, y2] shape produced by ObjectDetector."""
    zones = load_ignore_zones().get(room)
    if not zones or not detections:
        return detections
    kept: list[dict] = []
    dropped = 0
    for det in detections:
        center = _box_center(det.get("box"))
        if center is not None and any(
            point_in_polygon(center[0], center[1], poly) for poly in zones
        ):
            dropped += 1
            continue
        kept.append(det)
    if dropped:
        logger.debug(
            f"[IgnoreZones] '{room}': dropped {dropped} detection(s) inside "
            f"an ignore zone"
        )
    return kept
