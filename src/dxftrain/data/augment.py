"""Train-only augmentation: colour dropout, whole-drawing rotations/flips, corrupted text labels.

Rotations and flips are applied to the whole drawing BEFORE position features are read. Every position
feature is relative to the plot's axis-aligned bounding box, and a 90-degree rotation or a flip maps that box
onto the transformed drawing's box, so transforming (u, v) in the unit square is exact.
"""

from __future__ import annotations

import copy
import random

from dxftrain.dxf.geometry import drawing_side

# The 8 symmetries of the square, as maps of (u, v) in [0,1]^2. Index 0 is the identity.
D4 = [
    ("id", lambda u, v: (u, v), False),
    ("rot90", lambda u, v: (1 - v, u), True),
    ("rot180", lambda u, v: (1 - u, 1 - v), False),
    ("rot270", lambda u, v: (v, 1 - u), True),
    ("flip_x", lambda u, v: (1 - u, v), False),
    ("flip_y", lambda u, v: (u, 1 - v), False),
    ("transpose", lambda u, v: (v, u), True),
    ("anti_transpose", lambda u, v: (1 - v, 1 - u), True),
]


def transform(features: dict, k: int) -> dict:
    name, fn, swaps_axes = D4[k]
    f = copy.deepcopy(features)
    if f.get("rel_pos") is not None:
        u, v = fn(*f["rel_pos"])
        f["rel_pos"] = [round(u, 3), round(v, 3)]
    if swaps_axes:
        if f.get("orientation") in ("horizontal", "vertical"):
            f["orientation"] = "vertical" if f["orientation"] == "horizontal" else "horizontal"
        if "value_to_plot_width" in f:
            f["value_to_plot_width"], f["value_to_plot_height"] = f.get("value_to_plot_height"), f.get("value_to_plot_width")
    return f


def with_side(features: dict) -> dict:
    if features.get("rel_pos") is not None:
        features["nearest_side"] = drawing_side(*features["rel_pos"])
    return features


def corrupt_text(text: str, rng: random.Random) -> str:
    op = rng.choice(["lower", "upper", "title", "truncate"])
    if op == "lower":
        return text.lower()
    if op == "upper":
        return text.upper()
    if op == "title":
        return text.title()
    return text[: rng.randint(3, max(3, len(text) - 1))] if len(text) > 3 else text


def augment(features: dict, cfg: dict, rng: random.Random) -> tuple[dict, dict]:
    """One augmented copy. Returns (features, aug_record)."""
    rec: dict = {}
    k = rng.randint(1, 7) if cfg.get("rotate_flip", True) else 0
    f = transform(features, k)
    rec["d4"] = D4[k][0]
    if cfg.get("colour_dropout_p", 0) and rng.random() < cfg["colour_dropout_p"]:
        f["colour"] = None
        rec["colour_dropout"] = True
    for key in ("text", "dim_text_override"):
        if f.get(key) and rng.random() < cfg.get("text_corrupt_p", 0):
            f[key] = corrupt_text(f[key], rng)
            rec[f"{key}_corrupt"] = True
    return with_side(f), rec
