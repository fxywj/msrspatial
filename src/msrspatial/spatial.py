"""Spatial weights and Moran spectral decomposition.

Ported verbatim from the project's ``moran.py`` (a faithful re-implementation of
brainspace 0.1.10 / Wagner & Dray 2015). No behaviour change is intended: any
numerical difference against the original pipeline is a bug.
"""

from __future__ import annotations

import numpy as np

__all__ = ["weight_matrix_invdist", "compute_mem", "morans_i"]


def weight_matrix_invdist(distance: np.ndarray, knn: int = 15) -> np.ndarray:
    """Inverse-distance k-NN spatial weights, symmetrised.

    Parameters
    ----------
    distance
        ``(n, n)`` Euclidean distance matrix between parcel centroids.
    knn
        Number of nearest neighbours. Clipped to ``n - 1``.

    Returns
    -------
    ``(n, n)`` symmetric weight matrix with a zero diagonal.
    """
    d = np.asarray(distance, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError(f"distance must be square, got shape {d.shape}")
    if not np.isfinite(d).all():
        raise ValueError("distance contains NaN or inf")
    n = d.shape[0]
    idx = np.argsort(d, axis=1)
    w = np.zeros_like(d)
    k = min(int(knn), n - 1)
    for i in range(n):
        nn = idx[i, 1 : k + 1]
        w[i, nn] = 1.0 / np.maximum(d[i, nn], 1e-6)
    return (w + w.T) / 2


def compute_mem(w: np.ndarray, tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Moran eigenvector maps (MEM) from a symmetric spatial weight matrix.

    Double-centres ``w`` (= M W M), eigendecomposes, drops zero eigenvalues.

    Returns
    -------
    mem
        ``(n, n_components)`` eigenvectors, descending eigenvalue order.
    ev
        ``(n_components,)`` non-zero eigenvalues.
    """
    w = np.asarray(w, dtype=float)
    if not np.allclose(w, w.T):
        w = (w + w.T) / 2
    m = w.mean(axis=0, keepdims=True)
    wc = w.mean() - m - m.T + w
    ev, mem = np.linalg.eigh(wc)
    ev, mem = ev[::-1], mem[:, ::-1]
    keep = np.abs(ev) >= tol
    return mem[:, keep], ev[keep]


def morans_i(x: np.ndarray, w: np.ndarray) -> float:
    """Moran's I of a single map ``x`` under weights ``w``."""
    z = np.asarray(x, dtype=float)
    z = z - z.mean()
    s0 = w.sum()
    denom = z @ z
    if denom == 0:
        return float("nan")
    return float((z.shape[0] / s0) * (z @ (w @ z)) / denom)
