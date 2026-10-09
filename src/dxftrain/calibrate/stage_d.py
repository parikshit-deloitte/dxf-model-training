"""Stage D: confidence for a trained model (or the zero-shot base).

    python -m dxftrain.calibrate.stage_d --dataset data/datasets/ds-... --run runs/<run_id> [--adapter-dir adapter_best]
    python -m dxftrain.calibrate.stage_d --dataset ... --zero-shot

1. Score every VAL object (both variants) by label log-likelihood (infer/scorer.py) — never the model's words.
2. Fit one temperature T on val (NLL).
3. tau = the largest calibrated confidence of any WRONG val prediction; every prediction above it was right.
   Print the number of val objects behind tau per class. A class with fewer than 200 val objects makes tau
   PROVISIONAL.
Writes <run>/calibration.json and <run>/val_preds.<variant>.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from dxftrain.calibrate import temperature
from dxftrain.data import render_prompt as rp
from dxftrain.eval import metrics
from dxftrain.eval.metrics import Pred

ROOT = Path(__file__).resolve().parents[3]
PROVISIONAL_BELOW = 200


def preds_from_scores(rows: list[dict], S: np.ndarray, labels: list[str], T: float) -> list[Pred]:
    P = temperature.softmax(S, T)
    out = []
    for r, p in zip(rows, P):
        j = int(p.argmax())
        out.append(Pred(r["object_id"], r["project"], r["label"], labels[j], float(p[j])))
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run")
    ap.add_argument("--adapter-dir", default="adapter_best")
    ap.add_argument("--zero-shot", action="store_true")
    ap.add_argument("--model", default=None, help="override base model path (e.g. a fused model)")
    ap.add_argument("--backend", default="auto", help="zero-shot only: mlx | cuda | cpu | auto (a run uses its own)")
    a = ap.parse_args(argv)
    from dxftrain.infer import backend as bk
    from dxftrain.infer.scorer import score_rows

    ds = Path(a.dataset)
    if a.zero_shot:
        backend = bk.resolve(a.backend)
        run = bk.zero_shot_dir(ds, backend)
        run.mkdir(parents=True, exist_ok=True)
        base = a.model or (yaml.safe_load((ROOT / "configs" / "baselines.yaml").read_text())["zero_shot_model"]
                           if backend == "mlx" else bk.base_model(backend))
        adapter = None
    else:
        run = Path(a.run)
        meta = json.loads((run / "run.json").read_text())
        backend = meta.get("backend", "mlx")  # a model is scored by the backend it was trained with
        base = a.model or meta.get("model") or meta.get("lora_cfg", {}).get("model") or bk.base_model(backend)
        adapter = str(run / a.adapter_dir)
    nl = {c for c, v in json.loads((ds / "learnability.json").read_text()).items() if not v["learnable"]}
    scorer = bk.make_scorer(base, adapter_path=adapter, backend=backend)
    labels = scorer.labels
    idx = {c: i for i, c in enumerate(labels)}

    rows, S = [], []
    for variant in ("no_colour", "with_colour"):
        rs = [json.loads(l) for l in open(ds / f"val.{variant}.jsonl")]
        S.append(score_rows(scorer, rs, run / f"scores.{variant}.jsonl"))
        rows += rs
    S = np.concatenate(S)
    gold = np.array([idx[r["label"]] for r in rows])
    T = temperature.fit(S, gold)
    preds = preds_from_scores(rows, S, labels, T)
    tau = metrics.tau_wrong0(preds, nl)
    rep = metrics.report(preds, nl, tau)
    val_n = Counter(r["label"] for r in rows)
    learnable = [c for c in labels if c not in nl and c != metrics.UNKNOWN]
    small = {c: val_n.get(c, 0) for c in learnable if val_n.get(c, 0) < PROVISIONAL_BELOW}
    cal = {"T": T, "tau": tau, "provisional": bool(small), "val_objects": len(rows),
           "val_n_per_learnable_class": {c: val_n.get(c, 0) for c in learnable},
           "classes_below_200_val": small, "objects_behind_tau_per_class": rep["wrong0"]["objects_behind_tau_per_class"],
           "val_at_tau": rep["at_tau"], "nll_T1": temperature.nll(S, gold, 1.0), "nll_T": temperature.nll(S, gold, T),
           "not_learnable": sorted(nl), "labels": labels, "prompt_sha256": rp.prompt_sha256(),
           "base_model": base, "adapter": adapter, "backend": backend, "dataset": ds.name}
    (run / "calibration.json").write_text(json.dumps(cal, indent=1))
    (run / "val_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: cal[k] for k in ("T", "tau", "provisional", "val_objects", "val_at_tau",
                                          "objects_behind_tau_per_class", "nll_T1", "nll_T")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
