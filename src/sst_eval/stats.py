"""Significance testing helpers: exact McNemar and bootstrap CIs.

Pure-Python (stdlib ``math``/``random``) so they can be reused by lightweight
scripts and unit tests without numpy/scipy.
"""

from __future__ import annotations

import random
from math import comb


def exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact (binomial) McNemar p-value for discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) * (0.5 ** n) for i in range(k + 1)))


def mcnemar_from_pairs(reference, method):
    """Compare two equal-length 0/1 outcome sequences on matched items.

    Returns a dict with discordant counts, the method/reference accuracies and
    the exact p-value. ``method_wins`` = method right where reference wrong.
    """
    if len(reference) != len(method):
        raise ValueError("paired sequences must be the same length")
    b = sum(1 for r, m in zip(reference, method) if r == 0 and m == 1)
    c = sum(1 for r, m in zip(reference, method) if r == 1 and m == 0)
    p = exact_mcnemar_p(b, c)
    n = len(method)
    return {
        "method_acc": sum(method) / n if n else 0.0,
        "reference_acc": sum(reference) / n if n else 0.0,
        "method_wins": b,
        "reference_wins": c,
        "discordant": b + c,
        "p_value": p,
        "sig_at_0.05": p < 0.05,
        "interpretation": "significantly different" if p < 0.05 else "not significantly different",
    }


def bootstrap_ci(values, n_boot: int = 5000, seed: int = 427, alpha: float = 0.05):
    """Percentile bootstrap CI for the mean of a 0/1 (or numeric) sequence."""
    vals = list(values)
    n = len(vals)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = sum(vals[rng.randrange(n)] for _ in range(n)) / n
        boots.append(s)
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (lo, hi)
