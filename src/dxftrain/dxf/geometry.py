"""Name-free geometry helpers (shoelace area, rotated-rectangle shape measures, canonical vertex lists)."""

from __future__ import annotations

import hashlib
import math

from shapely.geometry import Polygon


def shoelace(pts: list[tuple[float, float]]) -> float:
    s = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def shape_measures(poly: Polygon) -> dict:
    """aspect = long/short side of the minimum rotated rectangle; rectangularity = area / that rectangle."""
    mrr = poly.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)
    if len(coords) < 4:
        return {"aspect": None, "rectangularity": None}
    e1 = math.dist(coords[0], coords[1])
    e2 = math.dist(coords[1], coords[2])
    long_, short = max(e1, e2), min(e1, e2)
    return {
        "aspect": round(long_ / short, 3) if short > 1e-9 else None,
        "rectangularity": round(poly.area / mrr.area, 3) if mrr.area > 1e-12 else None,
    }


def canonical_vertices(pts: list[tuple[float, float]], tol: float = 0.001) -> str:
    """Hash of the vertex list up to start index and direction, rounded to `tol` (H1: identical polygons)."""
    q = [(round(x / tol), round(y / tol)) for x, y in pts]
    if len(q) > 1 and q[0] == q[-1]:
        q = q[:-1]
    best = None
    for seq in (q, q[::-1]):
        for i in range(len(seq)):
            r = tuple(seq[i:] + seq[:i])
            if best is None or r < best:
                best = r
    return hashlib.sha256(repr(best).encode()).hexdigest()[:16]


def drawing_side(u: float, v: float) -> str:
    """Nearest side of the plot's bounding box in DRAWING orientation (not compass, not road side)."""
    d = {"left": u, "right": 1 - u, "bottom": v, "top": 1 - v}
    return min(d, key=d.get)
