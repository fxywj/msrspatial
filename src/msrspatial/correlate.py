"""Vectorised pairwise correlations between column sets.

``colwise_spearman`` ranks each column then does a normalised matrix product,
which is ~100x faster than looping over ``scipy.stats.spearmanr`` and is
required for the permutation null (thousands of surrogates x dozens of maps).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = ["colwise_pearson", "colwise_spearman"]


def _normalise_columns(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(x, axis=0, keepdims=True)
    return x / np.where(norm == 0, 1.0, norm)


def colwise_pearson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson correlation between every column of ``x`` and every column of ``y``.

    Returns ``(x.shape[1], y.shape[1])``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"row count mismatch: {x.shape[0]} vs {y.shape[0]}")
    return _normalise_columns(x).T @ _normalise_columns(y)


def colwise_spearman(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spearman correlation, implemented as Pearson on ranks (vectorised)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"row count mismatch: {x.shape[0]} vs {y.shape[0]}")
    xr = stats.rankdata(x, axis=0)
    yr = stats.rankdata(y, axis=0)
    return _normalise_columns(xr).T @ _normalise_columns(yr)
