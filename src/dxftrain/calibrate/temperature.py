"""Temperature scaling: one scalar T, fitted by minimising NLL of the gold label on held-out data.

Works on any per-label score vector (LLM label log-likelihoods, or log-probabilities from the GBM).
"""

from __future__ import annotations

import math

import numpy as np


def softmax(scores: np.ndarray, T: float) -> np.ndarray:
    z = scores / T
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def nll(scores: np.ndarray, gold: np.ndarray, T: float) -> float:
    p = softmax(scores, T)[np.arange(len(gold)), gold]
    return float(-np.log(np.clip(p, 1e-12, None)).mean())


def fit(scores: np.ndarray, gold: np.ndarray, lo: float = 0.05, hi: float = 20.0) -> float:
    """Golden-section search over log T. Returns T (1.0 if there is nothing to fit)."""
    if len(gold) == 0:
        return 1.0
    a, b = math.log(lo), math.log(hi)
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    for _ in range(80):
        if nll(scores, gold, math.exp(c)) < nll(scores, gold, math.exp(d)):
            b = d
        else:
            a = c
        c, d = b - g * (b - a), a + g * (b - a)
    return round(math.exp((a + b) / 2), 4)
