"""The only place outcomes are defined. Every number leaves this module with its sample size.

Outcome of one object at threshold tau:
  effective = predicted label, or "unknown" if the model said unknown, confidence <= tau, or the label is
              marked not learnable.
  UNKNOWN  if effective == "unknown"            (safe, whatever the gold label)
  CORRECT  if effective == gold
  WRONG    otherwise                             (a confident label on a gold-unknown object is WRONG too)
coverage          = (CORRECT + WRONG) / n
labelled_recall   = CORRECT on gold != unknown / n(gold != unknown)   -> what the downstream code actually gets
tau_wrong0        = the largest confidence of any WRONG prediction (so every prediction strictly above it is
                    right); coverage@WRONG=0 is the coverage at that tau.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

UNKNOWN = "unknown"


@dataclass
class Pred:
    object_id: str
    project: str
    gold: str
    label: str          # argmax label (may be "unknown")
    confidence: float   # calibrated probability of `label`


def effective(p: Pred, tau: float, not_learnable: set[str]) -> str:
    if p.label == UNKNOWN or p.label in not_learnable or p.confidence <= tau:
        return UNKNOWN
    return p.label


def outcomes(preds: list[Pred], tau: float, not_learnable: set[str]) -> dict:
    c = Counter()
    lab_n = lab_ok = 0
    for p in preds:
        e = effective(p, tau, not_learnable)
        o = "unknown" if e == UNKNOWN else ("correct" if e == p.gold else "wrong")
        c[o] += 1
        if p.gold != UNKNOWN:
            lab_n += 1
            lab_ok += o == "correct"
    n = len(preds)
    return {"n": n, "correct": c["correct"], "wrong": c["wrong"], "unknown": c["unknown"],
            "wrong_rate": round(c["wrong"] / n, 4) if n else None,
            "coverage": round((c["correct"] + c["wrong"]) / n, 4) if n else None,
            "n_labelled": lab_n, "labelled_recall": round(lab_ok / lab_n, 4) if lab_n else None, "tau": tau}


def tau_wrong0(preds: list[Pred], not_learnable: set[str]) -> float:
    wrong = [p.confidence for p in preds
             if p.label != UNKNOWN and p.label not in not_learnable and p.label != p.gold]
    return max(wrong) if wrong else 0.0


def at_wrong0(preds: list[Pred], not_learnable: set[str]) -> dict:
    t = tau_wrong0(preds, not_learnable)
    r = outcomes(preds, t, not_learnable)
    behind = Counter(effective(p, t, not_learnable) for p in preds)
    behind.pop(UNKNOWN, None)
    r["objects_behind_tau_per_class"] = dict(behind)
    return r


def per_class(preds: list[Pred], tau: float, not_learnable: set[str]) -> dict:
    tp, fp, gold_n = Counter(), Counter(), Counter()
    for p in preds:
        e = effective(p, tau, not_learnable)
        gold_n[p.gold] += 1
        if e != UNKNOWN:
            (tp if e == p.gold else fp)[e] += 1
    out = {}
    for c in sorted(set(gold_n) | set(tp) | set(fp)):
        if c == UNKNOWN:
            continue
        pred_n = tp[c] + fp[c]
        out[c] = {"n_gold": gold_n[c], "n_pred": pred_n,
                  "precision": round(tp[c] / pred_n, 3) if pred_n else None,
                  "recall": round(tp[c] / gold_n[c], 3) if gold_n[c] else None}
    return out


def confusion(preds: list[Pred], tau: float, not_learnable: set[str]) -> dict:
    m: dict = defaultdict(Counter)
    for p in preds:
        m[p.gold][effective(p, tau, not_learnable)] += 1
    return {g: dict(r) for g, r in m.items()}


def per_fold(preds: list[Pred], tau: float, not_learnable: set[str]) -> dict:
    by = defaultdict(list)
    for p in preds:
        by[p.project].append(p)
    folds = {k: {**outcomes(v, tau, not_learnable), "own_wrong0_coverage": at_wrong0(v, not_learnable)["coverage"]}
             for k, v in sorted(by.items())}
    def spread(key):
        xs = [f[key] for f in folds.values() if f[key] is not None]
        return {"mean": round(sum(xs) / len(xs), 4), "min": min(xs), "max": max(xs), "folds": len(xs)} if xs else None
    return {"folds": folds, "spread": {k: spread(k) for k in ("wrong_rate", "coverage", "labelled_recall",
                                                             "own_wrong0_coverage")}}


def nll(probs_of_gold: list[float]) -> float:
    return -sum(math.log(max(p, 1e-12)) for p in probs_of_gold) / max(len(probs_of_gold), 1)


def report(preds: list[Pred], not_learnable: set[str], tau: float | None = None) -> dict:
    """Full report. tau=None: use tau_wrong0 on these preds (validation). Pass a tau to apply a fixed one (test)."""
    t = tau_wrong0(preds, not_learnable) if tau is None else tau
    return {"n": len(preds), "at_tau_0": outcomes(preds, 0.0, not_learnable), "tau": t,
            "at_tau": {**outcomes(preds, t, not_learnable)},
            "wrong0": at_wrong0(preds, not_learnable),
            "per_class_at_tau": per_class(preds, t, not_learnable),
            "confusion_at_tau": confusion(preds, t, not_learnable),
            "per_fold": per_fold(preds, t, not_learnable)}
