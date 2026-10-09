"""Stage B: baselines on train+val projects, leave-one-project-out when there are fewer than 40 projects.

    python -m dxftrain.baselines.stage_b --dataset data/datasets/ds-... [--only b1,b2,b3]

Test projects are never touched here (they are the Stage E gate). Scored rows are real only; the GBM trains on
real + augmented rows of the training projects of each fold, as the fine-tuned model would.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from dxftrain.baselines import gbm, rules_mapper
from dxftrain.calibrate import temperature
from dxftrain.eval import metrics
from dxftrain.eval.metrics import Pred

ROOT = Path(__file__).resolve().parents[3]
VARIANTS = ("no_colour", "with_colour")
BACKEND = "auto"  # set by --backend


def load(ds: Path, variant: str) -> list[dict]:
    rows = []
    for split in ("train", "val"):
        rows += [json.loads(l) for l in open(ds / f"{split}.{variant}.jsonl")]
    return rows


def not_learnable(ds: Path) -> set[str]:
    return {c for c, v in json.loads((ds / "learnability.json").read_text()).items() if not v["learnable"]}


def lopo(rows: list[dict], fn) -> list[Pred]:
    projects = sorted({r["project"] for r in rows})
    preds = []
    for p in projects:
        train = [r for r in rows if r["project"] != p]
        test = [r for r in rows if r["project"] == p and r["source"] == "real"]
        preds += fn(train, test)
    return preds


def llm_zero_shot(rows: list[dict], ds: Path, variant: str, out_dir: Path) -> list[Pred]:
    """Zero-shot Qwen2.5-7B-Instruct: label log-likelihood scores; T fitted leave-one-project-out too."""
    from dxftrain.infer import backend as bk
    from dxftrain.infer.scorer import score_rows
    cfg = yaml.safe_load((ROOT / "configs" / "baselines.yaml").read_text())
    real = [r for r in rows if r["source"] == "real"]
    backend = bk.resolve(BACKEND)
    scorer = bk.make_scorer(cfg["zero_shot_model"] if backend == "mlx" else bk.base_model(backend), backend=backend)
    zrun = bk.zero_shot_dir(ds, backend)
    zrun.mkdir(parents=True, exist_ok=True)
    S = score_rows(scorer, real, zrun / f"scores.{variant}.jsonl")  # shared with Stage D/E
    labels = scorer.labels
    idx = {c: i for i, c in enumerate(labels)}
    gold = np.array([idx[r["label"]] for r in real])
    projects = np.array([r["project"] for r in real])
    preds = []
    for p in sorted(set(projects)):
        tr, te = projects != p, projects == p
        T = temperature.fit(S[tr], gold[tr])
        P = temperature.softmax(S[te], T)
        for r, prob in zip([r for r, m in zip(real, te) if m], P):
            j = int(prob.argmax())
            preds.append(Pred(r["object_id"], r["project"], r["label"], labels[j], float(prob[j])))
    return preds


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--only", default="b1,b2,b3")
    ap.add_argument("--backend", default="auto", help="B3 only: mlx | cuda | cpu | auto")
    a = ap.parse_args(argv)
    global BACKEND
    BACKEND = a.backend
    ds = Path(a.dataset)
    nl = not_learnable(ds)
    out_dir = ROOT / "reports" / "stage_b" / ds.name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}  # --only b3 adds to b1/b2
    for variant in VARIANTS:
        rows = load(ds, variant)
        n_proj = len({r["project"] for r in rows})
        assert n_proj < 40, "use GroupKFold(5) above 40 projects"
        runs = {}
        if "b1" in a.only:
            runs["B1_rules"] = rules_mapper.predict([r for r in rows if r["source"] == "real"])
        if "b2" in a.only:
            runs["B2_gbm"] = lopo(rows, lambda tr, te: gbm.fit_predict(tr, te))
        if "b3" in a.only:
            runs["B3_zero_shot_qwen7b"] = llm_zero_shot(rows, ds, variant, out_dir)
        for name, preds in runs.items():
            rep = metrics.report(preds, nl)
            rep["projects"] = n_proj
            (out_dir / f"{name}.{variant}.json").write_text(json.dumps(rep, indent=1))
            (out_dir / f"{name}.{variant}.preds.jsonl").write_text(
                "".join(json.dumps(p.__dict__) + "\n" for p in preds))
            w0 = rep["wrong0"]
            summary[f"{name}.{variant}"] = {
                "n": rep["n"], "at_tau0": {k: rep["at_tau_0"][k] for k in ("correct", "wrong", "unknown", "wrong_rate", "coverage")},
                "wrong0": {k: w0[k] for k in ("tau", "coverage", "labelled_recall", "n_labelled", "correct")},
                "fold_spread_own_wrong0_coverage": rep["per_fold"]["spread"]["own_wrong0_coverage"]}
            print(name, variant, json.dumps(summary[f"{name}.{variant}"]))
    summary_path.write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
