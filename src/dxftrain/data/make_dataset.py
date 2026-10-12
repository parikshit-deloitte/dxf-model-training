"""Stage A steps 2-4: labels, project split, augmentation, immutable dataset + data card.

    python -m dxftrain.data.make_dataset

Reads data/objects/extractor-<v>/ (from build.py). Writes data/datasets/ds-<YYYYMMDD>-<sha8>/ and refuses to
write into an existing directory. Only files whose plot was resolved (exactly one polygon within
2% of PLOT_AREA_M2, or identical copies of one outline under H1) and are CONFORMING or NEAR_CONFORMING supply
labelled objects.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import random
import stat
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from dxftrain.common.versions import environment
from dxftrain.data import augment as aug
from dxftrain.data import render_prompt as rp
from dxftrain.data.extract_objects import EXTRACTOR_VERSION
from dxftrain.data.projects import project_key, pseudonym, triage
from dxftrain.dxf import manual

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
OBJ_DIR = DATA / "objects" / f"extractor-{EXTRACTOR_VERSION}"
VARIANTS = ("with_colour", "no_colour")


def load_files(cfg: dict) -> list[dict]:
    files = []
    for mp in sorted(OBJ_DIR.glob("*.meta.json")):
        meta = json.loads(mp.read_text())
        key, how = project_key(meta)
        meta["project_key"], meta["project_key_from"] = key, how
        meta["project"] = pseudonym(key)
        meta["triage"] = triage(meta, cfg["triage"]["conforming_min_match"], cfg["triage"]["unlabeled_max_match"])
        files.append(meta)
    return files


def label_objects(meta: dict, drops: Counter) -> list[dict]:
    objs = [json.loads(l) for l in open(OBJ_DIR / f"{meta['file_sha256']}.objects.jsonl")]
    tier = meta["triage"]["tier"]
    labelled, unknown = [], []
    for o in objs:
        f = o["features"]
        label, why = manual.label_for(o["_layer"], f["type"])
        if label is None:
            drops[why] += 1
            continue
        if label == manual.UNKNOWN and why == "other" and tier == "NEAR_CONFORMING":
            # A non-Manual layer in a file that only partly follows the Manual may hold real parameters under
            # another name: not trustworthy as 'unknown'.
            drops["other_layer_in_near_conforming"] += 1
            continue
        o["label"], o["label_reason"] = label, why
        (unknown if label == manual.UNKNOWN else labelled).append(o)
    # An 'unknown' polygon that duplicates a labelled polygon (same area within 0.5%, centroid within 0.10 m)
    # would teach two answers for one shape: drop it.
    sig = [(o["features"]["area_m2"], o["features"]["rel_pos"]) for o in labelled if o["features"]["type"] == "POLYGON"]
    kept_unknown = []
    for o in unknown:
        f = o["features"]
        if f["type"] == "POLYGON" and any(abs(a - f["area_m2"]) <= 0.005 * a and
                                           abs(p[0] - f["rel_pos"][0]) * 1 + abs(p[1] - f["rel_pos"][1]) * 1 < 1e-3
                                           for a, p in sig):
            drops["duplicate_of_labelled"] += 1
            continue
        kept_unknown.append(o)
    out = labelled + kept_unknown
    for o in out:
        o["features"] = aug.with_side(o["features"])
        for key in ("text", "dim_text_override"):
            if o["features"].get(key):
                o["features"][key], red = rp.redact_layer(o["features"][key], o["_layer"])
                if red:
                    drops["_text_redacted_not_dropped"] += 1
    return out


def split_projects(projects: list[str], cfg: dict) -> dict[str, str]:
    frozen_path = ROOT / cfg["frozen_test_projects"]
    rng = random.Random(cfg["seed"])
    order = sorted(projects)
    rng.shuffle(order)
    frozen = set(json.loads(frozen_path.read_text())["test"]) if frozen_path.exists() else None
    n = len(order)
    n_test = max(1, round(n * cfg["ratios"]["test"]))
    n_val = max(1, round(n * cfg["ratios"]["val"]))
    if frozen is None:
        frozen = set(order[:n_test])
        frozen_path.write_text(json.dumps({"created": dt.date.today().isoformat(),
                                           "note": "Pseudonymous project ids frozen as TEST. Never edit; new projects are assigned by the seeded shuffle.",
                                           "test": sorted(frozen)}, indent=1))
    rest = [p for p in order if p not in frozen]
    out = {p: "test" for p in projects if p in frozen}
    for i, p in enumerate(rest):
        out[p] = "val" if i < n_val else "train"
    return out


def example(o: dict, meta: dict, split: str, variant: str, source: str, aug_rec: dict | None) -> dict:
    f = dict(o["features"])
    if variant == "no_colour":
        f["colour"] = None
    msgs = rp.messages(f, meta["declared_plot_area_m2"], o["label"])
    free_text = " ".join(str(f.get(k) or "") for k in ("text", "dim_text_override")).upper()
    leak = len(o["_layer"]) >= 3 and rp.redact_layer(free_text, o["_layer"])[1]
    if leak:
        raise AssertionError(f"layer name leaked into prompt: {meta['file_sha256']}:{o['handle']}")
    return {"object_id": f"{meta['file_sha256']}:{o['handle']}", "file_sha256": meta["file_sha256"],
            "project": meta["project"], "split": split, "variant": variant, "source": source, "aug": aug_rec,
            "label": o["label"], "type": f["type"], "features": f,
            "declared_plot_area_m2": meta["declared_plot_area_m2"],
            "prompt_sha256": rp.prompt_sha256(), "messages": msgs}


def main(argv: list[str]) -> int:
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text())
    scfg = yaml.safe_load((ROOT / "configs" / "split.yaml").read_text())
    acfg = yaml.safe_load((ROOT / "configs" / "augment.yaml").read_text())
    rng = random.Random(cfg["seed"])
    files = load_files(cfg)

    usable = [m for m in files if m.get("plot_resolved_by") and m["declared_plot_area_m2"]
              and m["insunits"] in (0, 6) and m["triage"]["tier"] in ("CONFORMING", "NEAR_CONFORMING")]
    drops: Counter = Counter()
    per_file = {m["file_sha256"]: label_objects(m, drops) for m in usable}
    projects = sorted({m["project"] for m in usable})
    split_of = split_projects(projects, scfg)

    rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for m in usable:
        split = split_of[m["project"]]
        objs = per_file[m["file_sha256"]]
        if split == "train":
            lab = [o for o in objs if o["label"] != manual.UNKNOWN]
            unk = [o for o in objs if o["label"] == manual.UNKNOWN]
            cap = cfg["unknown_per_labelled_train"] * max(1, len(lab))
            if len(unk) > cap:
                drops["train_unknown_capped"] += len(unk) - cap
                unk = rng.sample(unk, cap)
            objs = lab + unk
        for variant in VARIANTS:
            for o in objs:
                rows[(split, variant)].append(example(o, m, split, variant, "real", None))
                if split == "train" and acfg["enabled"]:
                    for _ in range(acfg["copies_per_object"]):
                        f2, rec = aug.augment(o["features"], {**acfg, "colour_dropout_p": acfg["colour_dropout_p"] if variant == "with_colour" else 0}, rng)
                        rows[(split, variant)].append(example({**o, "features": f2}, m, split, variant, "augmented", rec))

    # project-split invariant
    seen: dict[str, set] = defaultdict(set)
    for (split, _), rs in rows.items():
        for r in rs:
            seen[r["project"]].add(split)
    assert all(len(s) == 1 for s in seen.values()), "a project is on two sides of the split"

    payload = {f"{s}.{v}.jsonl": "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows[(s, v)])
               for s in ("train", "val", "test") for v in VARIANTS}
    ds_sha = hashlib.sha256("".join(k + hashlib.sha256(v.encode()).hexdigest() for k, v in sorted(payload.items())).encode()).hexdigest()
    out = DATA / "datasets" / f"ds-{dt.date.today():%Y%m%d}-{ds_sha[:8]}"
    if out.exists():
        print(f"dataset {out.name} already exists (identical content); not overwritten")
        return 0
    out.mkdir(parents=True)
    for name, text in payload.items():
        (out / name).write_text(text)

    from dxftrain.data.datacard import write_card
    manifest = {"dataset_sha256": ds_sha, "prompt_sha256": rp.prompt_sha256(), "created": dt.datetime.now().isoformat(),
                "extractor_version": EXTRACTOR_VERSION, "configs": {"data": cfg, "split": scfg, "augment": acfg},
                "categories_sha256": hashlib.sha256(rp.CATEGORIES_PATH.read_bytes()).hexdigest(),
                "files": {m["file_sha256"]: {"project": m["project"], "split": split_of[m["project"]],
                                             "tier": m["triage"]["tier"]} for m in usable},
                "split_of_project": split_of, "drops": dict(drops), **environment(ROOT)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    write_card(out, rows, files, usable, split_of, drops, manifest, cfg)
    (DATA / "project_keys.local.json").write_text(json.dumps(
        {m["project"]: {"key": m["project_key"], "from": m["project_key_from"], "files": [m["source_name"]]} for m in files}, indent=1))
    for p in out.iterdir():
        os.chmod(p, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    os.chmod(out, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
