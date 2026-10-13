"""Stage A step 5: labelling sheet (CSV + thumbnails) from real NO-LAYER drawings, for an officer to label.

    python -m dxftrain.data.labelling_sheet [--per-file 60]

Writes data/sheets/sheet-v<n>/ with sheet.csv, thumbs/, status.json (status=unlabelled). Nothing reads a sheet
for training or testing until status.json says "approved" (feedback/ingest_sheet.py enforces it).
The sheet never shows a layer name. Its rows are a seeded sample: every dimension (capped), the largest
polygons, a random sample of the remaining polygons, and a random sample of texts.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from ezdxf import recover  # noqa: E402

from dxftrain.data.extract_objects import EXTRACTOR_VERSION, collect_raw  # noqa: E402
from dxftrain.data.projects import pseudonym, project_key, triage  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
OBJ_DIR = DATA / "objects" / f"extractor-{EXTRACTOR_VERSION}"
COLUMNS = ["object_id", "file_sha256", "project", "handle", "type", "colour", "area_m2", "n_vertices",
           "dim_value_m", "text", "thumbnail", "label", "labeller", "confidence_of_labeller", "notes"]


def extent(raw: list[dict]) -> tuple[float, float, float, float]:
    xs, ys = [], []
    for o in raw:
        for x, y in (o.get("pts") or [(o["cx"], o["cy"])]):
            xs.append(x)
            ys.append(y)
    xs, ys = np.array(xs), np.array(ys)
    lo_x, hi_x = np.percentile(xs, [2, 98])
    lo_y, hi_y = np.percentile(ys, [2, 98])
    pad = 0.05 * max(hi_x - lo_x, hi_y - lo_y, 1.0)
    return lo_x - pad, lo_y - pad, hi_x + pad, hi_y + pad


def thumbnail(raw: list[dict], target: dict, box, path: Path) -> None:
    fig = plt.figure(figsize=(6, 6), dpi=110)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("white")
    for o in raw:
        if o["type"] == "POLYGON" and o is not target:
            p = np.array(o["pts"] + [o["pts"][0]])
            ax.plot(p[:, 0], p[:, 1], lw=0.3, color="#b0b0b0")
        elif o["type"] == "DIMENSION" and o is not target and len(o["pts"]) == 2:
            p = np.array(o["pts"])
            ax.plot(p[:, 0], p[:, 1], lw=0.3, color="#c8c8c8")
    if target["type"] == "POLYGON":
        p = np.array(target["pts"] + [target["pts"][0]])
        ax.fill(p[:, 0], p[:, 1], color="#e4572e", alpha=0.25)
        ax.plot(p[:, 0], p[:, 1], lw=1.6, color="#e4572e")
    elif target["type"] == "DIMENSION":
        p = np.array(target["pts"])
        ax.plot(p[:, 0], p[:, 1], lw=2.0, color="#e4572e", marker="o", ms=4)
    else:
        ax.plot([target["cx"]], [target["cy"]], marker="s", ms=10, mfc="none", mec="#e4572e", mew=2)
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0)
    ax.set_xlim(x0, x0 + side)
    ax.set_ylim(y0, y0 + side)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(path)
    plt.close(fig)


def sample(raw: list[dict], k: int, rng: random.Random) -> list[dict]:
    dims = [o for o in raw if o["type"] == "DIMENSION"]
    polys = sorted([o for o in raw if o["type"] == "POLYGON"], key=lambda o: -o["area"])
    texts = [o for o in raw if o["type"] == "TEXT"]
    pick = rng.sample(dims, min(len(dims), k // 3))
    pick += polys[: k // 6]
    rest = polys[k // 6:]
    pick += rng.sample(rest, min(len(rest), k // 6))
    pick += rng.sample(texts, min(len(texts), k - len(pick)))
    return pick


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-file", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20261007)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    files = json.loads((DATA / "files.json").read_text())
    sheets = DATA / "sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    n = 1 + max([int(p.name.split("-v")[1]) for p in sheets.glob("sheet-v*")] or [0])
    out = sheets / f"sheet-v{n}"
    (out / "thumbs").mkdir(parents=True)
    tri = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())["triage"]  # same thresholds as the dataset
    rows = []
    for mp in sorted(OBJ_DIR.glob("*.meta.json")):
        meta = json.loads(mp.read_text())
        if triage(meta, tri["conforming_min_match"], tri["unlabeled_max_match"])["tier"] != "UNLABELED":
            continue
        doc, _ = recover.readfile(files[meta["file_sha256"]]["paths"][0])
        layer_colours = {l.dxf.name.upper(): abs(l.dxf.get("color", 7)) for l in doc.layers}
        raw, *_ = collect_raw(doc.modelspace(), layer_colours)
        if not raw:
            continue
        box = extent(raw)
        project = pseudonym(project_key(meta)[0])
        for o in sample(raw, a.per_file, rng):
            oid = f"{meta['file_sha256']}:{o['handle']}"
            thumb = out / "thumbs" / f"{meta['file_sha256'][:12]}_{o['handle']}.png"
            thumbnail(raw, o, box, thumb)
            rows.append({"object_id": oid, "file_sha256": meta["file_sha256"], "project": project,
                         "handle": o["handle"], "type": o["type"], "colour": o["colour"],
                         "area_m2": round(o["area"], 3) if o["type"] == "POLYGON" else "",
                         "n_vertices": o.get("n_vertices", ""),
                         "dim_value_m": o.get("value", "") if o["type"] == "DIMENSION" else "",
                         "text": (o.get("text_full") or "")[:120], "thumbnail": f"thumbs/{thumb.name}",
                         "label": "", "labeller": "", "confidence_of_labeller": "", "notes": ""})
    with open(out / "sheet.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    cats = json.loads((ROOT / "contracts" / "categories.v1.json").read_text())["categories"]
    (out / "status.json").write_text(json.dumps({
        "status": "unlabelled", "created": dt.datetime.now().isoformat(), "rows": len(rows),
        "allowed_labels": cats + ["not_sure"],
        "instructions": "Fill 'label' with one allowed label per row (the red object in the thumbnail). Use "
                        "'not_sure' rather than guessing. Set status to 'labelled' when done; a second officer "
                        "reviews and sets 'approved'. Until then no code uses this sheet."}, indent=1))
    print(out, len(rows), "rows")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
