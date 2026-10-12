"""Features -> chat messages. The system prompt is prompts/classify_system.v1.txt with the category list filled in;
the same rendered text is used for training, validation and serving, and its SHA-256 travels with every artefact.
"""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

from dxftrain.dxf import manual

ROOT = Path(__file__).resolve().parents[3]
PROMPT_TEMPLATE = ROOT / "prompts" / "classify_system.v1.txt"
CATEGORIES_PATH = ROOT / "contracts" / "categories.v1.json"

# Field order and names as the model sees them. Nothing else from a record is rendered.
FIELDS = [
    ("type", "type"), ("colour", "colour"),
    ("area_m2", "area_m2"), ("area_ratio_plot", "area_ratio_to_plot"), ("perimeter_m", "perimeter_m"),
    ("n_vertices", "vertices"), ("aspect", "aspect"), ("rectangularity", "rectangularity"),
    ("is_plot_candidate", "area_matches_declared_plot"), ("contained_by_n", "inside_n_polygons"),
    ("contains_n", "contains_n_polygons"),
    ("dim_value_m", "dimension_value_m"), ("dim_kind", "dimension_kind"), ("orientation", "orientation"),
    ("dim_rank", "dimension_rank_desc"), ("dims_in_file", "dimensions_in_drawing"),
    ("value_to_plot_width", "value_to_plot_width"), ("value_to_plot_height", "value_to_plot_height"),
    ("dim_text_override", "dimension_text"),
    ("text", "text"), ("text_height", "text_height"),
    ("inside_plot", "inside_plot"), ("touches_plot_edge", "touches_plot_edge"),
    ("rel_pos", "position_in_plot_box"), ("nearest_side", "nearest_plot_box_side"),
]


@lru_cache(maxsize=1)
def category_names() -> tuple[str, ...]:
    return tuple(json.loads(CATEGORIES_PATH.read_text())["categories"])


def write_categories_contract() -> None:
    cats = sorted(manual.categories())
    CATEGORIES_PATH.write_text(json.dumps({"version": "1", "source": "contracts/manual_layers.v1.json",
                                           "categories": cats + [manual.UNKNOWN]}, indent=1))


@lru_cache(maxsize=1)
def system_prompt() -> str:
    cats = [c for c in category_names() if c != manual.UNKNOWN]
    return PROMPT_TEMPLATE.read_text().replace("{CATEGORIES}", "\n".join(cats)).strip()


@lru_cache(maxsize=1)
def prompt_sha256() -> str:
    return hashlib.sha256(system_prompt().encode()).hexdigest()


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, list):
        return ", ".join(_fmt(x) for x in v)
    return str(v)


def user_message(features: dict, declared_plot_area: float) -> str:
    lines = [f"declared_plot_area_m2: {declared_plot_area:g}"]
    for key, label in FIELDS:
        v = features.get(key)
        if v is None:
            if key == "colour":
                lines.append("colour: not available")
            continue
        lines.append(f"{label}: {_fmt(v)}")
    return "\n".join(lines)


def redact_layer(text: str | None, layer: str) -> tuple[str | None, bool]:
    """Remove the object's own layer name from free text so the answer never leaks into the prompt."""
    if not text or len(layer) < 3:
        return text, False
    pat = re.compile(re.escape(layer).replace(r"\_", r"[_ ]"), re.IGNORECASE)
    new = pat.sub("[…]", text)
    return new, new != text


def answer(category: str) -> str:
    return json.dumps({"category": category})


def messages(features: dict, declared: float, category: str | None) -> list[dict]:
    m = [{"role": "system", "content": system_prompt()},
         {"role": "user", "content": user_message(features, declared)}]
    if category is not None:
        m.append({"role": "assistant", "content": answer(category)})
    return m
