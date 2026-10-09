"""Stage G: feedback loop.

    python -m dxftrain.feedback.feedback ingest-corrections <export.jsonl>
    python -m dxftrain.feedback.feedback ingest-sheet data/sheets/sheet-v1
    python -m dxftrain.feedback.feedback retrain-check

Officer corrections arrive as an anonymised export file (no DB access, no credentials). Each line:
  {"file_sha256", "handle", "project": "<pseudonymous id>", "category", "corrected_at", "features", "declared_plot_area_m2"}
No names, plot numbers or e-mails may appear; the importer rejects a line that has any key it does not know.
Both importers write new, versioned, append-only object sets tagged source=officer / source=officer_sheet.
The sheet importer accepts only status "approved", and its rows go to the NO-LAYER TEST set (Q-G2: test-only).
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

from dxftrain.data import render_prompt as rp

ROOT = Path(__file__).resolve().parents[3]
FB = ROOT / "data" / "feedback"
ALLOWED = {"file_sha256", "handle", "project", "category", "corrected_at", "features", "declared_plot_area_m2"}
PSEUDO = re.compile(r"^P-[0-9a-f]{10}$")


def _next_dir(prefix: str) -> Path:
    FB.mkdir(parents=True, exist_ok=True)
    n = 1 + max([int(p.name.rsplit("-v", 1)[1]) for p in FB.glob(f"{prefix}-v*")] or [0])
    d = FB / f"{prefix}-v{n}"
    d.mkdir()
    return d


def ingest_corrections(path: Path) -> Path:
    cats = set(rp.category_names())
    out, bad = [], Counter()
    for i, line in enumerate(open(path), 1):
        r = json.loads(line)
        if set(r) - ALLOWED:
            bad["unknown_keys(not anonymised?)"] += 1
            continue
        if not PSEUDO.match(r.get("project", "")):
            bad["project_not_pseudonymous"] += 1
            continue
        if r["category"] not in cats:
            bad["category_not_in_contract"] += 1
            continue
        if any(k.startswith("_") or k == "layer" for k in r["features"]):
            bad["layer_in_features"] += 1
            continue
        out.append({**r, "object_id": f"{r['file_sha256']}:{r['handle']}", "label": r["category"], "source": "officer"})
    d = _next_dir("officer")
    (d / "objects.jsonl").write_text("".join(json.dumps(o) + "\n" for o in out))
    (d / "import.json").write_text(json.dumps({"from": path.name, "imported": len(out), "rejected": dict(bad),
                                              "at": dt.datetime.now().isoformat()}, indent=1))
    print(d, len(out), "imported", dict(bad))
    return d


def ingest_sheet(sheet: Path) -> Path:
    status = json.loads((sheet / "status.json").read_text())
    if status["status"] != "approved":
        raise SystemExit(f"sheet status is '{status['status']}': only an approved sheet can be used")
    allowed = set(status["allowed_labels"])
    rows = list(csv.DictReader(open(sheet / "sheet.csv")))
    labelled = [r for r in rows if r["label"] and r["label"] in allowed and r["label"] != "not_sure"]
    d = _next_dir("nolayer_test")
    (d / "objects.jsonl").write_text("".join(json.dumps({**r, "source": "officer_sheet", "use": "test_only"}) + "\n"
                                             for r in labelled))
    (d / "import.json").write_text(json.dumps({"sheet": sheet.name, "rows": len(rows), "labelled": len(labelled),
                                              "not_sure": sum(r["label"] == "not_sure" for r in rows)}, indent=1))
    print(d, len(labelled), "labelled rows (test-only)")
    return d


def retrain_check() -> str:
    cfg = yaml.safe_load((ROOT / "configs" / "feedback.yaml").read_text())
    from dxftrain.registry.registry import active
    act = active()
    base_counts: Counter = Counter()
    if act:
        ds = next((p for p in (ROOT / "data" / "datasets").glob("ds-*")
                   if json.loads((p / "manifest.json").read_text())["dataset_sha256"] == act["dataset_sha256"]), None)
        if ds:
            for c, v in json.loads((ds / "learnability.json").read_text()).items():
                base_counts[c] = v["real"]
    new: Counter = Counter()
    projects = set()
    for d in sorted(FB.glob("officer-v*")) if FB.exists() else []:
        for l in open(d / "objects.jsonl"):
            o = json.loads(l)
            new[o["label"]] += 1
            projects.add(o["project"])
    total_base = sum(base_counts.values())
    reasons = []
    for c, n in new.items():
        if base_counts[c] < cfg["not_learnable_min_real"] <= base_counts[c] + n:
            reasons.append(f"{c}: {base_counts[c]} -> {base_counts[c] + n} real (crosses {cfg['not_learnable_min_real']})")
    if total_base and sum(new.values()) / total_base >= cfg["min_relative_growth"]:
        reasons.append(f"real objects +{sum(new.values())} on {total_base} (>= {cfg['min_relative_growth']:.0%})")
    verdict = "RETRAIN" if reasons else "WAIT"
    print(json.dumps({"verdict": verdict, "reasons": reasons, "new_real_per_class": dict(new),
                      "new_projects": len(projects), "active_model": act["id"] if act else None}, indent=1))
    return verdict


def main(argv: list[str]) -> int:
    cmd = argv[0]
    if cmd == "ingest-corrections":
        ingest_corrections(Path(argv[1]))
    elif cmd == "ingest-sheet":
        ingest_sheet(Path(argv[1]))
    elif cmd == "retrain-check":
        retrain_check()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
