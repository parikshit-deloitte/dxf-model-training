"""Per-file object extraction (replaces build_object_dataset.py, which is not available).

Run as a subprocess per file so a timeout or a crash on one large drawing never stops the batch:

    python -m dxftrain.data.extract_objects <file.dxf> <out_dir>

Writes <out_dir>/<sha256>.meta.json and <sha256>.objects.jsonl. Objects carry their layer ONLY in the
`_layer` field, which is used for the gold label and is never rendered into a prompt.
Top-level model-space entities only (INSERT contents are not exploded in this version; counted in meta).
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import ezdxf
from ezdxf import recover
from shapely import STRtree
from shapely.geometry import Point, Polygon

from dxftrain.dxf import manual
from dxftrain.dxf.geometry import canonical_vertices, shape_measures, shoelace
from dxftrain.dxf.mtext import parse_plan_info, plain, to_float

EXTRACTOR_VERSION = "1.1.0"            # 1.1.0: duplicate-plot rule H1 (decision Q-A1, approved 2026-10-09)
PLOT_MATCH_TOL = 0.02          # declared plot area vs polygon area, relative (rule unchanged until approved)
EDGE_TOL_M = 0.05              # "touches plot edge"
TEXT_MAX = 80
ROOT = Path(__file__).resolve().parents[3]


def duplicate_policy() -> str:
    """configs/data.yaml duplicate_policy: 'none' or 'H1' (identical copies of one outline count as one plot)."""
    import yaml
    return yaml.safe_load((ROOT / "configs" / "data.yaml").read_text()).get("duplicate_policy", "none")


def resolve_plot(cands: list[dict], policy: str) -> tuple[list[dict], str | None]:
    """The plot polygon(s) among the candidates and how they were resolved.

    unique: exactly one candidate. H1: two or more candidates that are all the same outline (canonical vertex list,
    any start index or direction, 1 mm); every copy is the plot. Anything else (different shapes): unresolved.
    """
    if len(cands) == 1:
        return cands, "unique"
    if len(cands) > 1 and policy == "H1" and len({c["vertex_hash"] for c in cands}) == 1:
        return cands, "H1"
    return [], None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def colour_of(e, layer_colours: dict[str, int]) -> dict:
    if e.dxf.hasattr("true_color"):
        r, g, b = ezdxf.colors.int2rgb(e.dxf.true_color)
        return {"colour": f"rgb:{r},{g},{b}", "colour_source": "entity"}
    c = e.dxf.get("color", 256)
    if c == 256:
        return {"colour": layer_colours.get(e.dxf.layer.upper(), 7), "colour_source": "bylayer"}
    if c == 0:
        return {"colour": 7, "colour_source": "byblock"}
    return {"colour": c, "colour_source": "entity"}


def polygon_points(e) -> list[tuple[float, float]] | None:
    t = e.dxftype()
    try:
        if t == "LWPOLYLINE":
            closed = e.closed
            pts_b = list(e.get_points("xyb"))
            if any(abs(b) > 1e-9 for _, _, b in pts_b):
                pts = [(v.x, v.y) for v in ezdxf.path.make_path(e).flattening(0.01)]
            else:
                pts = [(x, y) for x, y, _ in pts_b]
        elif t == "POLYLINE" and e.is_2d_polyline:
            closed = e.is_closed
            pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
        else:
            return None
    except Exception:
        return None
    if len(pts) >= 2 and math.dist(pts[0], pts[-1]) < 1e-6:
        closed = True
        pts = pts[:-1]
    if not closed or len(pts) < 3:
        return None
    return pts


def dimension_info(e) -> dict | None:
    try:
        value = float(e.get_measurement())
    except Exception:
        return None
    dimtype = e.dimtype & 7
    kind = {0: "linear", 1: "aligned", 2: "angular", 3: "diameter", 4: "radius", 5: "angular3p", 6: "ordinate"}.get(dimtype, "other")
    p2, p3 = e.dxf.get("defpoint2"), e.dxf.get("defpoint3")
    if p2 is not None and p3 is not None:
        cx, cy = (p2.x + p3.x) / 2, (p2.y + p3.y) / 2
        ang = math.degrees(math.atan2(p3.y - p2.y, p3.x - p2.x)) if kind == "aligned" else e.dxf.get("angle", 0.0)
        pts = [(p2.x, p2.y), (p3.x, p3.y)]
    else:
        tp = e.dxf.get("text_midpoint") or e.dxf.get("defpoint")
        cx, cy, ang, pts = tp.x, tp.y, 0.0, [(tp.x, tp.y)]
    a = abs(ang) % 180
    orient = "horizontal" if min(a, 180 - a) < 5 else "vertical" if abs(a - 90) < 5 else "inclined"
    override = (e.dxf.get("text", "") or "").strip()
    return {"value": round(value, 4) if kind in ("linear", "aligned") else round(value, 4), "dim_kind": kind,
            "orientation": orient if kind in ("linear", "aligned") else None, "cx": cx, "cy": cy, "pts": pts,
            "override": override if override not in ("", "<>") else None}


def collect_raw(msp, layer_colours: dict[str, int]):
    """Top-level model-space polygons, dimensions and texts, with layer kept only in `_layer`."""
    raw: list[dict] = []
    type_counts: Counter = Counter()
    layer_use: Counter = Counter()
    plan_info_candidates = []
    for e in msp:
        t = e.dxftype()
        type_counts[t] += 1
        layer = e.dxf.get("layer", "0")
        layer_use[layer] += 1
        base = {"handle": e.dxf.handle, "_layer": layer, **colour_of(e, layer_colours)}
        if t in ("LWPOLYLINE", "POLYLINE"):
            pts = polygon_points(e)
            if pts is None:
                continue
            area = shoelace(pts)
            if area <= 1e-9:
                continue
            raw.append({**base, "type": "POLYGON", "pts": pts, "area": area, "n_vertices": len(pts)})
        elif t == "DIMENSION":
            d = dimension_info(e)
            if d:
                raw.append({**base, "type": "DIMENSION", **d})
        elif t in ("TEXT", "MTEXT"):
            try:
                txt = e.dxf.text if t == "TEXT" else e.plain_text()
                ins = e.dxf.insert
            except Exception:
                continue
            txt = " ".join((txt or "").split())
            if not txt:
                continue
            if "PLOT_AREA_M2" in txt.upper():
                plan_info_candidates.append((layer, e.text if t == "MTEXT" else txt, e.dxf.handle))
            raw.append({**base, "type": "TEXT", "text_full": txt, "cx": ins.x, "cy": ins.y,
                        "text_height": round(e.dxf.get("height" if t == "TEXT" else "char_height", 0) or 0, 3)})
    return raw, type_counts, layer_use, plan_info_candidates


def extract(path: Path, policy: str | None = None) -> tuple[dict, list[dict]]:
    t0 = time.time()
    sha = sha256_file(path)
    meta: dict = {"file_sha256": sha, "size_bytes": path.stat().st_size, "extractor_version": EXTRACTOR_VERSION,
                  "source_name": path.name}
    doc, auditor = recover.readfile(str(path))
    msp = doc.modelspace()
    meta["dxf_version"] = doc.dxfversion
    meta["insunits"] = doc.header.get("$INSUNITS", 0)
    meta["audit_errors"] = len(auditor.errors)
    layer_colours = {l.dxf.name.upper(): abs(l.dxf.get("color", 7)) for l in doc.layers}

    raw, type_counts, layer_use, plan_info_candidates = collect_raw(msp, layer_colours)
    meta["entity_counts"] = dict(type_counts)
    meta["inserts_not_exploded"] = type_counts.get("INSERT", 0)
    meta["layers"] = {l: {"entities": n, "class": manual.classify_layer(l).kind,
                          "category": manual.classify_layer(l).category,
                          "near_miss_of": manual.classify_layer(l).template if manual.classify_layer(l).kind == "near_miss" else None,
                          "note": manual.classify_layer(l).note}
                      for l, n in layer_use.items()}

    # PLAN_INFO: the Manual layer first; content (KEY=VALUE with PLOT_AREA_M2) only for no-layer drawings.
    by_layer = [c for c in plan_info_candidates if c[0].upper() == "PLAN_INFO"]
    chosen = by_layer or plan_info_candidates
    meta["plan_info_found_by"] = "layer" if by_layer else ("content" if plan_info_candidates else None)
    meta["plan_info_candidates"] = len(chosen)
    info = parse_plan_info(chosen[0][1]) if len(chosen) == 1 else {}
    if len(chosen) > 1:
        parsed = [parse_plan_info(c[1]) for c in chosen]
        areas = {p.get("PLOT_AREA_M2") for p in parsed}
        info = parsed[0] if len(areas) == 1 else {}
        meta["plan_info_conflict"] = len(areas) > 1
    declared = to_float(info.get("PLOT_AREA_M2"))
    meta["declared_plot_area_m2"] = declared
    meta["_plan_info_private"] = {"LESSEE_NAME": info.get("LESSEE_NAME"), "PLOT_NO": info.get("PLOT_NO"),
                                  "PROJECT_FALLING_UNDER_ZONE": info.get("PROJECT_FALLING_UNDER_ZONE")}
    meta["plan_info_keys"] = sorted(info)

    polys = [o for o in raw if o["type"] == "POLYGON"]
    cands = []
    if declared:
        for o in polys:
            if abs(o["area"] - declared) / declared <= PLOT_MATCH_TOL:
                xs, ys = zip(*o["pts"])
                cands.append({"handle": o["handle"], "layer": o["_layer"], "area": round(o["area"], 3),
                              "n_vertices": o["n_vertices"], "bbox": [round(min(xs), 3), round(min(ys), 3),
                                                                       round(max(xs), 3), round(max(ys), 3)],
                              "vertex_hash": canonical_vertices(o["pts"])})
    meta["plot_candidates"] = cands
    meta["plot_match_count"] = len(cands)

    plot_cands, how = resolve_plot(cands, duplicate_policy() if policy is None else policy)
    meta["plot_resolved_by"] = how
    objects: list[dict] = []
    if plot_cands:
        handles = {c["handle"] for c in plot_cands}
        plot_o = next(o for o in polys if o["handle"] == min(handles))
        objects = build_features(raw, plot_o, declared, handles)
    meta["objects"] = len(objects)
    meta["seconds"] = round(time.time() - t0, 2)
    return meta, objects


def build_features(raw: list[dict], plot_o: dict, declared: float, plot_handles: set[str] | None = None) -> list[dict]:
    plot_handles = plot_handles or {plot_o["handle"]}
    plot = Polygon(plot_o["pts"]).buffer(0)
    minx, miny, maxx, maxy = plot.bounds
    w, h = max(maxx - minx, 1e-9), max(maxy - miny, 1e-9)
    shapes = []
    for o in raw:
        if o["type"] == "POLYGON":
            p = Polygon(o["pts"]).buffer(0)
            o["_poly"] = p
            c = p.representative_point() if not p.is_empty else Point(o["pts"][0])
            o["cx"], o["cy"] = p.centroid.x if not p.is_empty else c.x, p.centroid.y if not p.is_empty else c.y
            shapes.append(o)
    tree = STRtree([o["_poly"] for o in shapes]) if shapes else None
    dims = sorted([o for o in raw if o["type"] == "DIMENSION" and o["dim_kind"] in ("linear", "aligned")],
                  key=lambda o: -o["value"])
    for i, o in enumerate(dims):
        o["dim_rank"] = i + 1
        o["dim_rank_pct"] = round(i / (len(dims) - 1), 3) if len(dims) > 1 else 0.0
    n_dims = len(dims)

    out = []
    for o in raw:
        f: dict = {"type": o["type"], "colour": o["colour"], "colour_source": o["colour_source"]}
        pt = Point(o["cx"], o["cy"])
        f["rel_pos"] = [round((o["cx"] - minx) / w, 3), round((o["cy"] - miny) / h, 3)]
        f["inside_plot"] = bool(plot.covers(pt))
        geom = None
        if o["type"] == "POLYGON":
            p = o["_poly"]
            f["area_m2"] = round(o["area"], 3)
            f["area_ratio_plot"] = round(o["area"] / declared, 4)
            f["perimeter_m"] = round(p.length, 3)
            f["n_vertices"] = o["n_vertices"]
            f.update(shape_measures(p) if not p.is_empty else {"aspect": None, "rectangularity": None})
            f["touches_plot_edge"] = bool(p.exterior.distance(plot.exterior) < EDGE_TOL_M) if not p.is_empty and p.geom_type == "Polygon" else bool(p.distance(plot.exterior) < EDGE_TOL_M)
            f["is_plot_candidate"] = o["handle"] in plot_handles
            containing = contained = 0
            if tree is not None and not p.is_empty:
                rp = p.representative_point()
                for j in tree.query(rp):
                    q = shapes[j]
                    if q is o or abs(q["area"] - o["area"]) <= 1e-3 * max(o["area"], 1e-9):
                        continue
                    if q["area"] > o["area"] and q["_poly"].covers(rp):
                        containing += 1
                for j in tree.query(p, predicate="covers"):
                    q = shapes[j]
                    if q is not o and q["area"] < o["area"] * (1 - 1e-3):
                        contained += 1
            f["contained_by_n"] = containing
            f["contains_n"] = contained
            geom = [[round(x, 3), round(y, 3)] for x, y in o["pts"]]
        elif o["type"] == "DIMENSION":
            f["dim_value_m"] = o["value"]
            f["dim_kind"] = o["dim_kind"]
            f["orientation"] = o["orientation"]
            f["dim_rank"] = o.get("dim_rank")
            f["dim_rank_pct"] = o.get("dim_rank_pct")
            f["dims_in_file"] = n_dims
            f["dim_text_override"] = o["override"][:TEXT_MAX] if o["override"] else None
            f["touches_plot_edge"] = any(plot.exterior.distance(Point(x, y)) < EDGE_TOL_M for x, y in o["pts"])
            f["value_to_plot_width"] = round(o["value"] / w, 4)
            f["value_to_plot_height"] = round(o["value"] / h, 4)
            geom = [[round(x, 3), round(y, 3)] for x, y in o["pts"]]
        else:
            f["text"] = o["text_full"][:TEXT_MAX]
            f["text_len"] = len(o["text_full"])
            f["text_height"] = o["text_height"]
            f["touches_plot_edge"] = bool(plot.exterior.distance(pt) < EDGE_TOL_M)
            geom = [[round(o["cx"], 3), round(o["cy"], 3)]]
        out.append({"handle": o["handle"], "_layer": o["_layer"], "features": f, "_geom": geom})
    return out


def main(argv: list[str]) -> int:
    path, out_dir = Path(argv[0]), Path(argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    meta, objects = extract(path)
    sha = meta["file_sha256"]
    tmp = out_dir / f"{sha}.objects.jsonl.tmp"
    with open(tmp, "w") as fh:
        for o in objects:
            fh.write(json.dumps(o, separators=(",", ":")) + "\n")
    tmp.rename(out_dir / f"{sha}.objects.jsonl")
    (out_dir / f"{sha}.meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps({"sha": sha, "objects": len(objects), "plot_match_count": meta["plot_match_count"],
                      "seconds": meta["seconds"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
