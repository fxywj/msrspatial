"""Multiple-comparison correction."""

from __future__ import annotations

import numpy as np

__all__ = ["bh_fdr"]


def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR, returning q-values in the input order.

    Monotone step-up implementation, identical to the project's original
    ``bh_fdr`` (NO statsmodels dependency).
    """
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1:
        raise ValueError("pvalues must be 1-D")
    if np.any((p < 0) | (p > 1) | ~np.isfinite(p)):
        raise ValueError("pvalues must be finite and within [0, 1]")
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    thresholds = ranked * n / np.arange(1, n + 1)
    monotone = np.minimum.accumulate(thresholds[::-1])[::-1]
    q = np.empty(n, dtype=float)
    q[order] = monotone
    return np.minimum(q, 1.0)
