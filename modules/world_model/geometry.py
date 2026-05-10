"""
JARVIS — World Model
====================
Mission: Pure geometry helpers. No I/O, no imports from other world_model
         modules. Used by WorldModel for exit/landmark detection.

Modules: modules/world_model/geometry.py
Spec:    new 2.md §15 (Full Code: geometry.py).
"""

from typing import Sequence


def point_in_polygon(x: float, y: float, polygon: Sequence[tuple[float, float]]) -> bool:
    """
    Standard ray-casting algorithm. Returns True if (x, y) is inside the polygon.
    Polygon is a list of (x, y) tuples; the polygon closes implicitly.
    """
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def bbox_center(bbox: tuple) -> tuple[float, float]:
    """Return (cx, cy) for a bbox (x1, y1, x2, y2)."""
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def bbox_iou(a: tuple, b: tuple) -> float:
    """Intersection-over-union for two bboxes (x1, y1, x2, y2)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def bbox_area_normalized(bbox: tuple, frame_w: int, frame_h: int) -> float:
    """Bbox area as fraction of frame area. Used for cat size estimation."""
    return ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / (frame_w * frame_h)
