"""Project keys (LESSEE_NAME + PLOT_NO) and pseudonymous project ids, and file triage."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

_REV = re.compile(r"(\b(rev|r|revised|final|v)\s*\d*\b|\(\d+\)|\d{1,2}[\s._-]*[a-z]{3}[\s._-]*\d{2,4}|\d{1,2}[._-]\d{1,2}[._-]\d{2,4}|_r\d+)",
                  re.IGNORECASE)


def norm(s: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def project_key(meta: dict) -> tuple[str, str]:
    """(key, how). Uses PLAN_INFO LESSEE_NAME + PLOT_NO, else the filename stem without revision/date tokens."""
    p = meta.get("_plan_info_private") or {}
    if p.get("LESSEE_NAME") and p.get("PLOT_NO"):
        return f"{norm(p['LESSEE_NAME'])}|{norm(p['PLOT_NO'])}", "plan_info"
    stem = Path(meta["source_name"]).stem
    stem = _REV.sub(" ", stem)
    return "file|" + norm(stem), "filename"


def pseudonym(key: str) -> str:
    return "P-" + hashlib.sha256(key.encode()).hexdigest()[:10]


SKIP_LAYERS = {"0", "DEFPOINTS"}


def triage(meta: dict, conforming_min: float, unlabeled_max: float) -> dict:
    used = {l: v for l, v in meta["layers"].items() if l.upper() not in SKIP_LAYERS}
    match = [l for l, v in used.items() if v["class"] in ("category", "context")]
    cats = [l for l, v in used.items() if v["class"] == "category"]
    pct = len(match) / len(used) if used else 0.0
    upper = {l.upper() for l in used}
    has_plot, has_info = "PLOT_BOUNDARY" in upper, "PLAN_INFO" in upper
    if pct >= conforming_min and has_plot and has_info:
        tier = "CONFORMING"
    elif pct < unlabeled_max or not cats:
        tier = "UNLABELED"
    else:
        tier = "NEAR_CONFORMING"
    return {"tier": tier, "layer_match_pct": round(pct, 3), "used_layers": len(used),
            "has_plot_boundary": has_plot, "has_plan_info": has_info,
            "near_miss_layers": sorted(l for l, v in used.items() if v["class"] == "near_miss")}
