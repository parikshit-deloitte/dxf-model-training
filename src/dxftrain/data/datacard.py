"""DATACARD.md: per-class x split x source counts, per-project counts, real kept apart from augmented,
classes with fewer than N real examples flagged NOT LEARNABLE YET (reported, never hidden)."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from dxftrain.dxf import manual
from dxftrain.data import render_prompt as rp


def learnability(rows: dict, min_real: int) -> dict[str, dict]:
    """Real (non-augmented) example counts per class across ALL splits, from the no_colour variant."""
    c: Counter = Counter()
    for (split, variant), rs in rows.items():
        if variant != "no_colour":
            continue
        for r in rs:
            if r["source"] == "real":
                c[r["label"]] += 1
    out = {}
    for cat in rp.category_names():
        n = c.get(cat, 0)
        out[cat] = {"real": n, "learnable": n >= min_real or cat == manual.UNKNOWN}
    return out


def write_card(out: Path, rows: dict, files: list, usable: list, split_of: dict, drops, manifest: dict, cfg: dict) -> None:
    L = []
    L.append(f"# Data card — {out.name}\n")
    L.append(f"- dataset_sha256: `{manifest['dataset_sha256']}`")
    L.append(f"- prompt_sha256: `{manifest['prompt_sha256']}` (prompts/classify_system.v1.txt + contracts/categories.v1.json)")
    L.append(f"- git: `{manifest['git_commit']}` · extractor {manifest['extractor_version']} · created {manifest['created']}")
    L.append("- Real data only from drawings; augmented rows are train-only and counted separately. No synthetic files.\n")

    tiers = Counter(m["triage"]["tier"] for m in files)
    L.append("## Files\n")
    L.append(f"Real DXF files extracted: {len(files)} · tiers: {dict(tiers)} · usable for labels (plot rule passed, conforming/near): {len(usable)}\n")

    learn = learnability(rows, cfg["not_learnable_min_real"])
    (out / "learnability.json").write_text(json.dumps(learn, indent=1, sort_keys=True))
    L.append("## Per class (real examples, no_colour variant; augmented in brackets)\n")
    L.append("| class | train | val | test | total real | learnable (≥%d real) |" % cfg["not_learnable_min_real"])
    L.append("|---|---:|---:|---:|---:|---|")
    cnt = defaultdict(Counter)
    augc = Counter()
    for (split, variant), rs in rows.items():
        if variant != "no_colour":
            continue
        for r in rs:
            if r["source"] == "real":
                cnt[r["label"]][split] += 1
            else:
                augc[r["label"]] += 1
    for cat in sorted(rp.category_names(), key=lambda c: -learn[c]["real"]):
        c = cnt[cat]
        if learn[cat]["real"] == 0:
            continue
        L.append(f"| {cat} | {c['train']} [{augc[cat]}] | {c['val']} | {c['test']} | {learn[cat]['real']} | "
                 f"{'yes' if learn[cat]['learnable'] else '**NOT LEARNABLE YET**'} |")
    zero = [c for c in rp.category_names() if learn[c]["real"] == 0]
    L.append(f"\nClasses with ZERO real examples ({len(zero)}): " + ", ".join(zero) + "\n")

    L.append("## Per project (real objects, no_colour)\n")
    L.append("| project | split | files | objects | labelled (non-unknown) |")
    L.append("|---|---|---:|---:|---:|")
    pf = Counter(m["project"] for m in usable)
    po, pl = Counter(), Counter()
    for (split, variant), rs in rows.items():
        if variant == "no_colour":
            for r in rs:
                if r["source"] == "real":
                    po[r["project"]] += 1
                    pl[r["project"]] += r["label"] != manual.UNKNOWN
    for p in sorted(pf, key=lambda p: (split_of[p], p)):
        L.append(f"| {p} | {split_of[p]} | {pf[p]} | {po[p]} | {pl[p]} |")

    L.append("\n## Dropped objects (not labelled unknown — dropped)\n")
    for k, v in sorted(drops.items()):
        L.append(f"- {k}: {v}")
    L.append("\n## Rows per file in this dataset\n")
    for (split, variant), rs in sorted(rows.items()):
        src = Counter(r["source"] for r in rs)
        L.append(f"- {split}.{variant}.jsonl: {len(rs)} rows {dict(src)}")
    (out / "DATACARD.md").write_text("\n".join(L) + "\n")
