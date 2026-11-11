"""Spatial null models (Moran Spectral Randomization).

The surrogate maps preserve the spatial autocorrelation of the observed map
while randomising its spatial phase, so that a correlation between two brain
maps is tested against "equally smooth" chance maps rather than against
white-noise permutations.

Reference: Wagner & Dray (2015) Methods in Ecology and Evolution 6:1169-1179.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

__all__ = ["moran_surrogates", "spin_pvalue", "null_diagnostics"]


def moran_surrogates(
    x: np.ndarray,
    mem: np.ndarray,
    n_rep: int = 1000,
    random_state: int | None = None,
) -> np.ndarray:
    """Generate MSR surrogates of ``x``.

    Parameters
    ----------
    x
        ``(n,)`` or ``(n, n_features)`` observed map(s).
    mem
        ``(n, n_components)`` Moran eigenvector maps from :func:`compute_mem`.
    n_rep
        Number of surrogates.
    random_state
        Seed. Integer for reproducibility.

    Returns
    -------
    ``(n_rep, n, n_features)`` float array. Shape is always 3-D even for a
    single input map, so downstream code never has to special-case ``squeeze``.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"x must be 1-D or 2-D, got {x.ndim}-D")
    n_rows = x.shape[0]
    if mem.shape[0] != n_rows:
        raise ValueError(f"mem has {mem.shape[0]} rows, x has {n_rows}")
    if n_rep < 1:
        raise ValueError("n_rep must be >= 1")

    rng = np.random.default_rng(random_state)
    n_comp = mem.shape[1]
    n_cols = x.shape[1]

    # correlation of each eigenvector with each input map
    c = 1.0 - cdist(x.T, mem.T, "correlation").T  # (n_comp, n_cols)
    signs = rng.choice([-1.0, 1.0], size=(n_rep, n_comp, n_cols))
    coeff = c[None, :, :] * signs  # (n_rep, n_comp, n_cols)

    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0, ddof=1)
    # (n_rep, n, n_comp) @ (n_rep, n_comp, n_cols) -> (n_rep, n, n_cols)
    reconstructed = np.matmul(mem, coeff)
    sim = x_mean + reconstructed * (np.sqrt(n_rows - 1) * x_std)
    return sim


def spin_pvalue(observed: np.ndarray, null: np.ndarray, two_sided: bool = True) -> np.ndarray:
    """Permutation p-value with the +1 conservative correction.

    ``p = (1 + #{|null| >= |obs|}) / (K + 1)`` when ``two_sided``.
    """
    obs = np.asarray(observed, dtype=float)
    nul = np.asarray(null, dtype=float)
    if two_sided:
        obs, nul = np.abs(obs), np.abs(nul)
    exceed = (nul >= obs[None, ...]).sum(axis=0)
    return (1.0 + exceed) / (nul.shape[0] + 1.0)


def null_diagnostics(
    observed: np.ndarray,
    surrogates: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Quality control for a spatial null.

    Two things must hold for MSR to be valid:

    1. **Marginal distribution preserved** -- ``mean_ratio`` and ``sd_ratio``
       should both be close to 1.
    2. **Spatial autocorrelation preserved** -- ``morans_i_ratio`` (surrogate /
       observed) should be close to 1. This is the property that makes MSR a
       *spatial* null rather than a white-noise permutation.

    ``mean_corr`` is the signed average correlation between the observed map and
    its surrogates; it should be near zero. Note that ``sd_corr`` can be large
    (a surrogate may be close to ``+x`` or ``-x``): MSR randomises the spatial
    phase, it does not guarantee decorrelation for any single draw.
    """
    from .spatial import morans_i

    obs = np.asarray(observed, dtype=float).ravel()
    sur = np.asarray(surrogates, dtype=float)
    if sur.ndim == 2:
        sur = sur[:, :, None]
    flat = sur.reshape(sur.shape[0], -1)

    corrs = np.array([np.corrcoef(obs, s)[0, 1] for s in flat[: min(200, flat.shape[0])]])
    out = {
        "mean_ratio": float(np.nanmean(flat) / np.nanmean(obs)) if np.nanmean(obs) != 0 else float("nan"),
        "sd_ratio": float(np.nanmean(flat.std(axis=1)) / np.nanstd(obs, ddof=1)),
        "mean_corr": float(np.nanmean(corrs)),
        "sd_corr": float(np.nanstd(corrs)),
        "mean_abs_corr": float(np.nanmean(np.abs(corrs))),
    }
    if weights is not None:
        i_obs = morans_i(obs, weights)
        i_sur = np.array([morans_i(s, weights) for s in flat[: min(200, flat.shape[0])]])
        out["morans_i_observed"] = float(i_obs)
        out["morans_i_ratio"] = float(np.nanmean(i_sur) / i_obs) if i_obs != 0 else float("nan")
    return out
