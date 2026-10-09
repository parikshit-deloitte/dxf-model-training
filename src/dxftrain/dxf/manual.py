"""Manual layer names -> object categories, and near-miss detection.

The only source of layer names is contracts/manual_layers.v1.json (Drawing Manual v1.0). No alias for a
misspelling is ever added here: a near-miss layer is reported and its objects are dropped from training.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MANUAL_PATH = ROOT / "contracts" / "manual_layers.v1.json"

OBJECT_TYPES = ("POLYGON", "DIMENSION", "TEXT")

# Manual rules whose layers are drawing frames or free text, not parameters. Objects on them are "unknown"
# (decision Q-A3b in the Stage A report).
CONTEXT_RULES = {46, 47, 48, 49, 50}

# The Manual prints Rule 14 as BKL_n_SETBACK_DIM (Data_Quality_Notes #1) while its naming convention is BLK_n.
# Both spellings are the Manual's own; they map to one category. Reported, not silently assumed.
MANUAL_SPELLING_VARIANTS = {"BKL_n_SETBACK_DIM": "BLK_n_SETBACK_DIM"}

# PLAN_INFO is a Manual layer (Plan_Info_Keys section), not in the Layers table.
EXTRA_CATEGORIES = {"PLAN_INFO": {"TEXT"}}

# Explicit near-miss names seen in real drawings (docs/manual_proposals.md of the scrutiny repo).
EXPLICIT_NEAR_MISS = {"COMPUND_WALL", "COMPOND_WALL_EXISTING"}

UNKNOWN = "unknown"


def _types_from_element(element: str) -> set[str]:
    e = element.lower()
    out = set()
    if "polygon" in e:
        out.add("POLYGON")
    if "dimension" in e:
        out.add("DIMENSION")
    if "mtext" in e:
        out.add("TEXT")
    return out


def _template_regex(template: str) -> re.Pattern:
    parts = re.split(r"(?<=_)n(?=_|$)|^n(?=_)", template)
    return re.compile("".join(re.escape(p) + ("[A-Z0-9]+" if i < len(parts) - 1 else "")
                              for i, p in enumerate(parts)), re.IGNORECASE)


@dataclass(frozen=True)
class Category:
    name: str
    rule_nos: tuple[int, ...]
    object_types: frozenset[str]
    templates: tuple[str, ...]


@lru_cache(maxsize=1)
def manual() -> dict:
    return json.loads(MANUAL_PATH.read_text())


@lru_cache(maxsize=1)
def categories() -> dict[str, Category]:
    acc: dict[str, dict] = {}
    for row in manual()["layers"]:
        if row["Rule_No"] in CONTEXT_RULES:
            continue
        template = row["Layer_Name"]
        name = MANUAL_SPELLING_VARIANTS.get(template, template)
        c = acc.setdefault(name, {"rules": set(), "types": set(), "templates": {name}})
        c["rules"].add(row["Rule_No"])
        c["types"] |= _types_from_element(row["Element_Type"])
        c["templates"].add(template)
    for name, types in EXTRA_CATEGORIES.items():
        acc[name] = {"rules": set(), "types": set(types), "templates": {name}}
    return {n: Category(n, tuple(sorted(v["rules"])), frozenset(v["types"]), tuple(sorted(v["templates"])))
            for n, v in sorted(acc.items()) if v["types"]}


@lru_cache(maxsize=1)
def context_templates() -> tuple[str, ...]:
    return tuple(sorted({r["Layer_Name"] for r in manual()["layers"] if r["Rule_No"] in CONTEXT_RULES}))


@lru_cache(maxsize=1)
def _compiled() -> list[tuple[re.Pattern, str, str]]:
    """(regex, category or 'context', template); longest template first so a fullmatch is unambiguous."""
    out = []
    for name, c in categories().items():
        for t in c.templates:
            out.append((_template_regex(t), name, t))
    for t in context_templates():
        out.append((_template_regex(t), "context", t))
    out.sort(key=lambda x: -len(x[2]))
    return out


def all_templates() -> list[str]:
    return [t for _, _, t in _compiled()]


def edit_distance(a: str, b: str) -> int:
    a, b = a.upper(), b.upper()
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _shape(layer: str) -> str:
    """Replace the variable parts of a layer name with 'n' so it can be compared to a template."""
    return re.sub(r"(?<=_)[0-9]+(?=_|$)|(?<=^BLK_)[A-Z](?=_)", "n", layer.upper())


@dataclass(frozen=True)
class LayerClass:
    kind: str            # "category" | "context" | "near_miss" | "other"
    category: str | None
    template: str | None
    note: str = ""


@lru_cache(maxsize=100_000)
def classify_layer(layer: str) -> LayerClass:
    name = layer.strip()
    for rx, cat, t in _compiled():
        if rx.fullmatch(name):
            return LayerClass("context" if cat == "context" else "category", None if cat == "context" else cat, t)
    if name.upper() in EXPLICIT_NEAR_MISS:
        return LayerClass("near_miss", None, None, "explicit near-miss list")
    shaped = _shape(name)
    for t in all_templates():
        if len(t) < 6:  # short names (RWH, MTEXT): a fuzzy match would catch unrelated layers
            if shaped.startswith(t.upper() + "_"):
                return LayerClass("near_miss", None, t, f"starts with Manual name {t}")
            continue
        d_full = edit_distance(shaped, t)
        if d_full <= 2:
            return LayerClass("near_miss", None, t, f"edit distance {d_full} to {t}")
        for k in (len(t) - 2, len(t) - 1, len(t), len(t) + 1, len(t) + 2):
            if 0 < k < len(shaped) and shaped[k] in "_- " and edit_distance(shaped[:k], t) <= 2:
                return LayerClass("near_miss", None, t, f"prefix '{shaped[:k]}' within 2 edits of {t}")
    return LayerClass("other", None, None)


def label_for(layer: str, object_type: str) -> tuple[str | None, str]:
    """Gold label for an object on a CONFORMING/NEAR_CONFORMING file's layer.

    Returns (label, reason). label None means DROP the object (near-miss layer, or an object type the Manual
    does not define for that layer).
    """
    lc = classify_layer(layer)
    if lc.kind == "near_miss":
        return None, "near_miss"
    if lc.kind == "category":
        if object_type in categories()[lc.category].object_types:
            return lc.category, "manual"
        return None, "type_mismatch"
    return UNKNOWN, lc.kind  # context or other
