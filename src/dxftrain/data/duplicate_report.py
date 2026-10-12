"""Stage A step 1b: every file skipped by the plot rule, with its candidates and the duplicate hypotheses.

Nothing here changes the matching rule. It reports what each proposed policy WOULD do.
  H1 identical : all candidates share one canonical vertex list (same shape drawn twice, any start/direction)
  H2 nested    : candidates are offset copies (bboxes nested, area difference < 2%)
  H3 by layer  : exactly one candidate is on the Manual PLOT_BOUNDARY layer
For 0-match files it shows the closest polygons, so a wrong declaration or wrong units can be seen.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dxftrain.data.extract_objects import EXTRACTOR_VERSION, polygon_points
from dxftrain.dxf.geometry import shoelace

ROOT = Path(__file__).resolve().parents[3]
OBJ_DIR = ROOT / "data" / "objects" / f"extractor-{EXTRACTOR_VERSION}"


def nested(a: list, b: list) -> bool:
    return (a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]) or \
           (b[0] <= a[0] and b[1] <= a[1] and b[2] >= a[2] and b[3] >= a[3])


def hypotheses(cands: list[dict]) -> dict:
    h1 = len({c["vertex_hash"] for c in cands}) == 1
    h2 = all(nested(cands[0]["bbox"], c["bbox"]) for c in cands[1:]) and \
        (max(c["area"] for c in cands) - min(c["area"] for c in cands)) / max(c["area"] for c in cands) < 0.02
    on_plot = [c for c in cands if c["layer"].upper() == "PLOT_BOUNDARY"]
    return {"H1_identical": h1, "H2_nested": h2, "H3_one_on_PLOT_BOUNDARY": len(on_plot) == 1}


def closest_polygons(path: str, declared: float, k: int = 3) -> list[dict]:
    from ezdxf import recover
    doc, _ = recover.readfile(path)
    out = []
    for e in doc.modelspace().query("LWPOLYLINE POLYLINE"):
        pts = polygon_points(e)
        if pts:
            a = shoelace(pts)
            out.append({"layer": e.dxf.layer, "area": round(a, 3), "rel_diff": round((a - declared) / declared, 4)})
    out.sort(key=lambda o: abs(o["rel_diff"]))
    on_plot = [o for o in out if o["layer"].upper() == "PLOT_BOUNDARY"][:2]
    return out[:k] + [o for o in on_plot if o not in out[:k]]


def main(argv: list[str]) -> int:
    files = json.loads((ROOT / "data" / "files.json").read_text())
    report = []
    for mp in sorted(OBJ_DIR.glob("*.meta.json")):
        m = json.loads(mp.read_text())
        n = m["plot_match_count"]
        if n == 1 or not m["declared_plot_area_m2"]:  # H1-resolved files are reported too, with their hypotheses
            continue
        rec = {"file_sha256": m["file_sha256"], "name": m["source_name"], "declared": m["declared_plot_area_m2"],
               "matches": n, "candidates": m["plot_candidates"]}
        if n >= 2:
            rec["hypotheses"] = hypotheses(m["plot_candidates"])
        else:
            rec["closest"] = closest_polygons(files[m["file_sha256"]]["paths"][0], m["declared_plot_area_m2"])
        report.append(rec)
    out = ROOT / "data" / "plot_match_report.json"
    out.write_text(json.dumps(report, indent=1))
    for r in report:
        print(f"\n{r['name'][:45]:45s} declared={r['declared']} matches={r['matches']}")
        for c in r["candidates"]:
            print(f"   cand layer={c['layer'][:28]:28s} area={c['area']:<12} verts={c['n_vertices']:<4} hash={c['vertex_hash']} bbox={c['bbox']}")
        if "hypotheses" in r:
            print("   ", r["hypotheses"])
        for c in r.get("closest", []):
            print(f"   closest layer={c['layer'][:28]:28s} area={c['area']:<12} diff={c['rel_diff']:+.2%}")
    print("\n->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
