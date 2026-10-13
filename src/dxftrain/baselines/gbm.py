"""B2: gradient boosting on the same name-free features, with calibrated probabilities.

sklearn HistGradientBoostingClassifier (LightGBM needs libomp, not installed here). Calibration: temperature
scaling fitted on a project-held-out quarter of each training fold, never on the fold being scored.
"""

from __future__ import annotations

import random
import re
from collections import Counter

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from dxftrain.calibrate import temperature
from dxftrain.eval.metrics import Pred

NUMERIC = ["area_m2", "area_ratio_plot", "perimeter_m", "n_vertices", "aspect", "rectangularity",
           "is_plot_candidate", "contained_by_n", "contains_n", "dim_value_m", "dim_rank", "dim_rank_pct",
           "dims_in_file", "value_to_plot_width", "value_to_plot_height", "text_len", "text_height",
           "inside_plot", "touches_plot_edge"]
MAX_CATEGORIES = 254  # codes 0..253: under HistGradientBoostingClassifier's max_bins (255), one bin left for missing
CATEG = ["type", "dim_kind", "orientation", "nearest_side", "colour"]
KEYWORDS = {"kw_plot_area": r"PLOT_AREA", "kw_eq": r"=", "kw_north": r"^\s*N\s*$|NORTH", "kw_m2": r"M2|SQ\.?\s*M",
            "kw_block": r"\bBL(OC)?K", "kw_floor": r"FLOOR|FLR", "kw_gate": r"GATE", "kw_park": r"PARK",
            "kw_setback": r"SET\s*BACK", "kw_height": r"\bHT\b|HEIGHT", "kw_guard": r"GUARD|SECURITY|CABIN",
            "kw_stair": r"STAIR", "kw_wc": r"\bW\.?C\b|TOILET|BATH", "kw_road": r"ROAD", "kw_digit": r"\d"}


class Vectoriser:
    def __init__(self) -> None:
        self.vocab: dict[str, dict] = {c: {} for c in CATEG}

    def _cat(self, c: str, v) -> float:
        """Vocabulary code, or NaN (sklearn's 'missing') for None and for values not in the fitted vocabulary."""
        code = None if v is None else self.vocab[c].get(str(v))
        return np.nan if code is None else float(code)

    def transform(self, rows: list[dict], fit: bool = False) -> np.ndarray:
        if fit:
            # HistGradientBoosting takes at most MAX_CATEGORIES codes per categorical feature (true colours alone can
            # exceed that across drawings): keep the most frequent, the rest count as missing.
            for c in CATEG:
                seen = Counter(str(r["features"][c]) for r in rows if r["features"].get(c) is not None)
                self.vocab[c] = {k: i for i, (k, _) in enumerate(seen.most_common(MAX_CATEGORIES))}
        X = []
        for r in rows:
            f = r["features"]
            v = [float(f[k]) if isinstance(f.get(k), (int, float)) and not isinstance(f.get(k), str) else np.nan for k in NUMERIC]
            rp = f.get("rel_pos") or [np.nan, np.nan]
            v += [rp[0], rp[1]]
            v += [self._cat(c, f.get(c)) for c in CATEG]
            text = (f.get("text") or "") + " " + (f.get("dim_text_override") or "")
            v += [1.0 if re.search(p, text, re.I) else 0.0 for p in KEYWORDS.values()]
            X.append(v)
        return np.array(X, dtype=float)

    @property
    def categorical_mask(self) -> list[bool]:
        return [False] * (len(NUMERIC) + 2) + [True] * len(CATEG) + [False] * len(KEYWORDS)


def fit_predict(train: list[dict], test: list[dict], seed: int = 0) -> list[Pred]:
    """Fit on `train` (with a project-held-out calibration quarter), predict `test`."""
    projects = sorted({r["project"] for r in train})
    rng = random.Random(seed)
    rng.shuffle(projects)
    cal_p = set(projects[: max(1, len(projects) // 4)]) if len(projects) >= 4 else set()
    fit_rows = [r for r in train if r["project"] not in cal_p]
    cal_rows = [r for r in train if r["project"] in cal_p and r["source"] == "real"]

    vec = Vectoriser()
    Xf = vec.transform(fit_rows, fit=True)
    labels = sorted({r["label"] for r in fit_rows})
    idx = {c: i for i, c in enumerate(labels)}
    yf = np.array([idx[r["label"]] for r in fit_rows])
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=5,
                                         l2_regularization=1.0, categorical_features=vec.categorical_mask,
                                         random_state=seed)
    clf.fit(Xf, yf)

    def scores(rows):
        if not rows:
            return np.zeros((0, len(labels)))
        p = np.full((len(rows), len(labels)), 1e-9)
        p[:, clf.classes_] = np.clip(clf.predict_proba(vec.transform(rows)), 1e-9, None)
        return np.log(p)

    T = 1.0
    known = [r for r in cal_rows if r["label"] in idx]
    if known:
        T = temperature.fit(scores(known), np.array([idx[r["label"]] for r in known]))
    probs = temperature.softmax(scores(test), T) if test else np.zeros((0, len(labels)))
    out = []
    for r, p in zip(test, probs):
        j = int(p.argmax())
        out.append(Pred(r["object_id"], r["project"], r["label"], labels[j], float(p[j])))
    return out
