"""Stage E: evaluation gate on held-out TEST projects. Writes gate_report.json and MODEL_CARD.md.

    python -m dxftrain.gate.gate --dataset data/datasets/ds-... --run runs/<run_id> \
        --zero-shot-run runs/zeroshot-<ds> [--served models/<id>/served]

Every system uses a threshold chosen on VALIDATION and applied unchanged to TEST:
  model / B3 : T and tau from their Stage D calibration.json
  B2 (GBM)   : tau = tau_wrong0 of its Stage B leave-one-project-out predictions; refit on train+val for test
  B1 (rules) : tau = tau_wrong0 of its Stage B predictions
  active     : the registry's active model, with its own calibration (G4)
Checks: G1 WRONG <= best baseline · G2 coverage > best baseline · G3 served == evaluated within tolerance ·
G4 G1+G2 against the active model · G5 enough test objects, else INCONCLUSIVE.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from dxftrain.baselines import gbm, rules_mapper
from dxftrain.calibrate.stage_d import preds_from_scores
from dxftrain.eval import metrics
from dxftrain.eval.metrics import Pred

ROOT = Path(__file__).resolve().parents[3]


def load(ds: Path, split: str, variant: str) -> list[dict]:
    return [json.loads(l) for l in open(ds / f"{split}.{variant}.jsonl")]


def stage_b_preds(ds: Path, name: str, variant: str) -> list[Pred]:
    p = ROOT / "reports" / "stage_b" / ds.name / f"{name}.{variant}.preds.jsonl"
    return [Pred(**json.loads(l)) for l in open(p)]


def scored(run: Path, model: str, adapter: str | None, rows: list[dict], variant: str, tag: str) -> list[Pred]:
    from dxftrain.infer.backend import make_scorer
    from dxftrain.infer.scorer import score_rows
    cal = json.loads((run / "calibration.json").read_text())
    scorer = make_scorer(model, adapter_path=adapter, backend=cal.get("backend", "mlx"))
    assert scorer.labels == cal["labels"]
    name = f"scores.{variant}.jsonl" if tag in ("model", "zeroshot", "active") else f"scores.{tag}.{variant}.jsonl"
    S = score_rows(scorer, rows, run / name)
    return preds_from_scores(rows, S, cal["labels"], cal["T"])


def card(path: Path, rep: dict) -> None:
    m = rep["model"]
    L = [f"# Model card — {m['id']}\n",
         f"- **Gate: {rep['verdict']}**" + (f" — {'; '.join(rep['reasons'])}" if rep["reasons"] else ""),
         f"- base: `{m['base']}` · adapter: `{m['adapter']}` · backend: {m['backend']}",
         f"- dataset: `{m['dataset']}` sha256 `{m['dataset_sha256']}`",
         f"- prompt sha256: `{m['prompt_sha256']}`",
         f"- git commit: `{m['git_commit']}` · seed: {m['seed']}",
         f"- calibration (val): T={m['T']}, tau={m['tau']:.4f}, "
         f"{'**PROVISIONAL** (classes < 200 val objects: ' + str(len(m['classes_below_200_val'])) + ')' if m['provisional'] else 'final'}",
         f"- not learnable (always 'unknown'): {', '.join(m['not_learnable'])}\n", "## Test metrics (threshold from validation)\n",
         "| variant | system | n | correct | wrong | unknown | WRONG rate | coverage | labelled recall (n) |",
         "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for v, sysm in rep["test"].items():
        for name, r in sysm.items():
            t = r["at_tau"]
            L.append(f"| {v} | {name} | {t['n']} | {t['correct']} | {t['wrong']} | {t['unknown']} | {t['wrong_rate']} | "
                     f"{t['coverage']} | {t['labelled_recall']} ({t['n_labelled']}) |")
    L += ["\n## Checks\n"] + [f"- {k}: {'PASS' if v['pass'] else 'FAIL' if v['pass'] is False else 'INCONCLUSIVE'} — {v['detail']}"
                              for k, v in rep["checks"].items()]
    L += ["\n## Known limits\n",
          "- Setback SIDE is never predicted; it comes only from DXF colour downstream. Without colour it is an officer item.",
          "- Trained and tested on CONFORMING/NEAR_CONFORMING drawings with layer names hidden. That approximates, but is "
          "not, the no-layer use case; the approved labelling sheet is the real test set (not available yet).",
          f"- Test set: {rep['test_projects']} projects. Per-class test counts are small; see gate_report.json.",
          "- Trained with MLX on Apple Silicon (affine 4-bit, AdamW), not the specified nf4/paged-8-bit CUDA recipe."
          if m["backend"] == "mlx" else "- CUDA QLoRA recipe as specified."]
    path.write_text("\n".join(L) + "\n")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--zero-shot-run", required=True)
    ap.add_argument("--served", default=None, help="served artefact dir (fused model) for G3, or 'same' when the "
                                                     "server loads exactly the evaluated base + adapter")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    ds, run, zrun = Path(a.dataset), Path(a.run), Path(a.zero_shot_run)
    gcfg = yaml.safe_load((ROOT / "configs" / "gate.yaml").read_text())
    cal = json.loads((run / "calibration.json").read_text())
    zcal = json.loads((zrun / "calibration.json").read_text())
    runmeta = json.loads((run / "run.json").read_text())
    nl = set(cal["not_learnable"])
    from dxftrain.registry.registry import active
    act = active()

    test_rep, checks, reasons = {}, {}, []
    g1 = g2 = g3 = g4 = True
    g5_detail = []
    for v in gcfg["variants_required"]:
        test = load(ds, "test", v)
        trainval = load(ds, "train", v) + load(ds, "val", v)
        systems: dict[str, tuple[list[Pred], float]] = {}
        systems["MODEL"] = (scored(run, cal["base_model"], cal["adapter"], test, v, "model"), cal["tau"])
        systems["B3_zero_shot"] = (scored(zrun, zcal["base_model"], None, test, v, "zeroshot"), zcal["tau"])
        b2_tau = metrics.tau_wrong0(stage_b_preds(ds, "B2_gbm", v), nl)
        systems["B2_gbm"] = (gbm.fit_predict(trainval, test), b2_tau)
        systems["B1_rules"] = (rules_mapper.predict(test), metrics.tau_wrong0(stage_b_preds(ds, "B1_rules", v), nl))
        if act and act["id"] != runmeta.get("run_id"):
            arun = ROOT / act["run_dir"]
            acal = json.loads((arun / "calibration.json").read_text())
            systems["ACTIVE"] = (scored(arun, acal["base_model"], acal["adapter"], test, v, "active"), acal["tau"])
        reps = {k: metrics.report(p, nl, tau) for k, (p, tau) in systems.items()}
        test_rep[v] = reps
        m = reps["MODEL"]["at_tau"]
        base = {k: r["at_tau"] for k, r in reps.items() if k.startswith("B")}
        best_wrong = min(b["wrong_rate"] for b in base.values())
        best_cov = max(b["coverage"] for b in base.values())
        ok1, ok2 = m["wrong_rate"] <= best_wrong, m["coverage"] > best_cov
        g1 &= ok1
        g2 &= ok2
        checks[f"G1_wrong[{v}]"] = {"pass": ok1, "detail": f"model {m['wrong_rate']} (n={m['n']}) vs best baseline {best_wrong}"}
        checks[f"G2_coverage[{v}]"] = {"pass": ok2, "detail": f"model {m['coverage']} vs best baseline {best_cov}"}
        if "ACTIVE" in reps:
            ac = reps["ACTIVE"]["at_tau"]
            ok4 = m["wrong_rate"] <= ac["wrong_rate"] and m["coverage"] > ac["coverage"]
            g4 &= ok4
            checks[f"G4_vs_active[{v}]"] = {"pass": ok4, "detail": f"active {act['id']}: wrong {ac['wrong_rate']}, coverage {ac['coverage']}"}
        else:
            checks[f"G4_vs_active[{v}]"] = {"pass": True, "detail": "no active model in the registry"}
        # G3: served artefact vs the evaluated model
        if a.served == "same":
            checks[f"G3_quantisation[{v}]"] = {"pass": True, "detail": "the served artefact IS the evaluated one "
                                                                       "(same base + adapter, same loader): nothing to compare"}
        elif a.served:
            served = scored(run, a.served, None, test, v, "served")
            ev = systems["MODEL"][0]
            agree = sum(x.label == y.label for x, y in zip(ev, served)) / len(ev)
            dconf = max(abs(x.confidence - y.confidence) for x, y in zip(ev, served))
            same = all(metrics.effective(x, cal["tau"], nl) == metrics.effective(y, cal["tau"], nl) for x, y in zip(ev, served))
            q = gcfg["quantisation"]
            ok3 = agree >= q["min_top1_agreement"] and dconf <= q["max_abs_conf_delta"] and (same or not q["require_identical_outcomes_at_tau"])
            g3 &= ok3
            checks[f"G3_quantisation[{v}]"] = {"pass": ok3, "detail": f"top-1 agreement {agree:.4f}, max |Δconf| {dconf:.4f}, identical outcomes at tau: {same} (n={len(ev)})"}
        else:
            g3 = None
            checks[f"G3_quantisation[{v}]"] = {"pass": None, "detail": "no served artefact given"}
        auto = {c for c, s in reps["MODEL"]["per_class_at_tau"].items() if s["n_pred"] > 0}
        n_test_class = {}
        for r in test:
            n_test_class[r["label"]] = n_test_class.get(r["label"], 0) + 1
        thin = {c: n_test_class.get(c, 0) for c in (auto or {}) if n_test_class.get(c, 0) < gcfg["min_n_per_auto_class"]}
        if len(test) < gcfg["min_n_test_objects"]:
            g5_detail.append(f"{v}: only {len(test)} test objects")
        if thin:
            g5_detail.append(f"{v}: auto-labelled classes with < {gcfg['min_n_per_auto_class']} test objects: {thin}")
    checks["G5_sample_size"] = {"pass": None if g5_detail else True, "detail": "; ".join(g5_detail) or "enough test objects"}

    if not (g1 and g2 and g4) or g3 is False:
        verdict = "FAIL"
        reasons = [k for k, v in checks.items() if v["pass"] is False]
    elif g5_detail or g3 is None:
        verdict = "INCONCLUSIVE"
        reasons = [k for k, v in checks.items() if v["pass"] is None]
    else:
        verdict = "PASS"
    model = {"id": runmeta["run_id"], "base": cal["base_model"], "adapter": cal["adapter"], "backend": runmeta.get("backend"),
             "dataset": ds.name, "dataset_sha256": runmeta["dataset_sha256"], "prompt_sha256": cal["prompt_sha256"],
             "git_commit": runmeta.get("git_commit"), "seed": runmeta["seed"], "T": cal["T"], "tau": cal["tau"],
             "provisional": cal["provisional"], "classes_below_200_val": cal["classes_below_200_val"],
             "not_learnable": cal["not_learnable"]}
    rep = {"verdict": verdict, "reasons": reasons, "checks": checks, "model": model,
           "test_projects": len({r["project"] for r in load(ds, "test", "no_colour")}),
           "test": {v: {k: {"at_tau": r["at_tau"], "per_class_at_tau": r["per_class_at_tau"],
                            "confusion_at_tau": r["confusion_at_tau"], "per_fold": r["per_fold"]["spread"]}
                        for k, r in reps.items()} for v, reps in test_rep.items()}}
    out = Path(a.out) if a.out else ROOT / "models" / runmeta["run_id"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate_report.json").write_text(json.dumps(rep, indent=1))
    card(out / "MODEL_CARD.md", rep)
    print(json.dumps({"verdict": verdict, "reasons": reasons, "checks": checks}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
