"""Stage A step 1: run the per-object extractor over every real DXF, one subprocess per file, with a timeout.

Idempotent: a file whose <sha>.meta.json already exists for this extractor version is not re-run.
Every file gets one record in the skip log, including the ones that succeeded (status=ok).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from dxftrain.data.extract_objects import EXTRACTOR_VERSION, sha256_file

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
OBJ_DIR = DATA / "objects" / f"extractor-{EXTRACTOR_VERSION}"


def collect(sources_path: Path) -> dict[str, dict]:
    cfg = yaml.safe_load(sources_path.read_text())
    files: dict[str, dict] = {}
    for f in cfg["folders"]:
        root = Path(os.path.expanduser(f["path"]))
        if not root.exists():
            continue
        it = root.rglob("*") if f.get("recursive") else root.iterdir()
        for p in it:
            if p.is_file() and p.suffix.lower() == ".dxf":
                sha = sha256_file(p)
                rec = files.setdefault(sha, {"sha256": sha, "paths": [], "sources": set(), "size": p.stat().st_size})
                rec["paths"].append(str(p))
                rec["sources"].add(f["source"])
    for r in files.values():
        r["sources"] = sorted(r["sources"])
    return files


def run_one(rec: dict, timeout: int) -> dict:
    sha = rec["sha256"]
    meta_path = OBJ_DIR / f"{sha}.meta.json"
    base = {"sha256": sha, "size": rec["size"], "name": Path(rec["paths"][0]).name}
    if "synthetic" in rec["sources"]:
        return {**base, "status": "excluded_synthetic"}
    if not meta_path.exists():
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
        try:
            p = subprocess.run([sys.executable, "-m", "dxftrain.data.extract_objects", rec["paths"][0], str(OBJ_DIR)],
                               capture_output=True, text=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            return {**base, "status": "timeout", "detail": f">{timeout}s"}
        if p.returncode != 0:
            tail = (p.stderr or "").strip().splitlines()[-1:] or ["no stderr"]
            return {**base, "status": "unreadable", "detail": tail[0][:300]}
    meta = json.loads(meta_path.read_text())
    if meta["insunits"] not in (0, 6):
        status = "units"
    elif meta["declared_plot_area_m2"] is None:
        status = "no_plan_info" if not meta["plan_info_found_by"] else "no_declared_area"
    else:
        n = meta["plot_match_count"]
        status = "ok" if meta.get("plot_resolved_by") else ("plot_0" if n == 0 else "plot_2" if n == 2 else "plot_3+")
    return {**base, "status": status, "seconds": meta["seconds"], "plot_match_count": meta["plot_match_count"], "plot_resolved_by": meta.get("plot_resolved_by"),
            "declared_plot_area_m2": meta["declared_plot_area_m2"], "objects": meta["objects"],
            "insunits": meta["insunits"]}


def main(argv: list[str]) -> int:
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    files = collect(DATA / "sources.yaml")
    OBJ_DIR.mkdir(parents=True, exist_ok=True)
    (DATA / "files.json").write_text(json.dumps(files, indent=1))
    from collections import Counter

    from dxftrain.common.progress import Progress
    todo = sorted(files.values(), key=lambda r: r["size"])
    print(f"[extract] {len(todo)} unique DXF files, {cfg['workers']} workers, timeout {cfg['file_timeout_s']}s/file",
          flush=True)
    prog, counts, results = Progress("extract", len(todo), every_s=0), Counter(), []
    with ThreadPoolExecutor(max_workers=cfg["workers"]) as ex:
        for r in ex.map(lambda r: run_one(r, cfg["file_timeout_s"]), todo):
            results.append(r)
            counts[r["status"]] += 1
            prog.update(len(results), extra=f"{r['status']:<10} {r['name']}  ({dict(counts)})")
    log_dir = DATA / "skip_logs"
    log_dir.mkdir(exist_ok=True)
    out = log_dir / f"{dt.datetime.now():%Y%m%d-%H%M%S}.jsonl"
    with open(out, "w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps(Counter(r["status"] for r in results)), "->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
