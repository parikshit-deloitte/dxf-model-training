"""B1: deterministic code baseline on per-object, name-free features.

llm_layer_mapper.py was not available, so this is NOT a port: it is written from the Manual alone.
  - A polygon whose area matches the declared plot area (the code's own verification) -> PLOT_BOUNDARY.
  - A text holding PLOT_AREA_M2=... -> PLAN_INFO.
  - With colour: (object type, Manual colour) pairs that only ONE Manual category uses -> that category.
Everything else is "unknown". Confidence is 1.0 for a rule hit; there is nothing to calibrate.
"""

from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache

from dxftrain.dxf import manual
from dxftrain.eval.metrics import UNKNOWN, Pred


@lru_cache(maxsize=1)
def unique_colour_rules() -> dict[tuple[str, int], str]:
    owners: dict[tuple[str, int], set] = defaultdict(set)
    for row in manual.manual()["layers"]:
        if row["Rule_No"] in manual.CONTEXT_RULES or not str(row["Color_Code"]).isdigit():
            continue
        cat = manual.classify_layer(row["Layer_Name"].replace("_n", "_1")).category
        if cat is None:
            continue
        for t in manual._types_from_element(row["Element_Type"]):
            owners[(t, int(row["Color_Code"]))].add(cat)
    # colour 7 (default white/black) is what an unstyled drawing uses everywhere: never evidence.
    return {k: next(iter(v)) for k, v in owners.items() if len(v) == 1 and k[1] != 7}


def predict_one(f: dict) -> str:
    if f["type"] == "POLYGON" and f.get("is_plot_candidate"):
        return "PLOT_BOUNDARY"
    if f["type"] == "TEXT" and re.search(r"PLOT_AREA_M2\s*=", f.get("text") or "", re.I):
        return "PLAN_INFO"
    col = f.get("colour")
    if isinstance(col, int):
        return unique_colour_rules().get((f["type"], col), UNKNOWN)
    return UNKNOWN


def predict(rows: list[dict]) -> list[Pred]:
    out = []
    for r in rows:
        lab = predict_one(r["features"])
        out.append(Pred(r["object_id"], r["project"], r["label"], lab, 1.0 if lab != UNKNOWN else 0.0))
    return out
