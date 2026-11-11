"""Unit tests for the numerical core."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from msrspatial import (
    bh_fdr,
    colwise_pearson,
    colwise_spearman,
    compute_mem,
    morans_i,
    moran_surrogates,
    null_diagnostics,
    spin_pvalue,
    weight_matrix_invdist,
)


@pytest.fixture()
def grid() -> tuple[np.ndarray, np.ndarray]:
    """6x6 grid of parcels -> (distance, coords)."""
    xs, ys = np.meshgrid(np.arange(6), np.arange(6))
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    return d, coords


def test_weight_matrix_is_symmetric_with_zero_diagonal(grid):
    d, _ = grid
    w = weight_matrix_invdist(d, knn=4)
    assert np.allclose(w, w.T)
    assert np.allclose(np.diag(w), 0.0)
    assert (w > 0).sum() > 0


def test_weight_matrix_rejects_non_square():
    with pytest.raises(ValueError):
        weight_matrix_invdist(np.ones((3, 4)))


def test_mem_components_orthogonal(grid):
    d, _ = grid
    w = weight_matrix_invdist(d, knn=4)
    mem, ev = compute_mem(w)
    assert mem.shape[0] == d.shape[0]
    assert mem.shape[1] == len(ev)
    # eigenvectors of a symmetric matrix are orthonormal
    gram = mem.T @ mem
    assert np.allclose(gram, np.eye(mem.shape[1]), atol=1e-8)


def test_morans_i_high_for_smooth_map(grid):
    d, coords = grid
    w = weight_matrix_invdist(d, knn=4)
    smooth = coords[:, 0]  # varies slowly across the grid
    noisy = np.random.default_rng(0).standard_normal(len(coords))
    assert morans_i(smooth, w) > morans_i(noisy, w)


def test_colwise_correlations_match_scipy():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((60, 3))
    y = rng.standard_normal((60, 4))
    r = colwise_pearson(x, y)
    rho = colwise_spearman(x, y)
    for i in range(3):
        for j in range(4):
            assert r[i, j] == pytest.approx(stats.pearsonr(x[:, i], y[:, j])[0], abs=1e-10)
            assert rho[i, j] == pytest.approx(stats.spearmanr(x[:, i], y[:, j])[0], abs=1e-10)


def test_colwise_shape_mismatch_raises():
    with pytest.raises(ValueError):
        colwise_pearson(np.ones((10, 2)), np.ones((9, 3)))


def test_surrogates_preserve_marginals_and_autocorrelation(grid):
    d, coords = grid
    w = weight_matrix_invdist(d, knn=4)
    mem, _ = compute_mem(w)
    x = coords[:, 0] + 0.3 * np.random.default_rng(2).standard_normal(len(coords))
    sur = moran_surrogates(x, mem, n_rep=200, random_state=3)
    assert sur.shape == (200, len(x), 1)
    diag = null_diagnostics(x, sur, weights=w)
    assert 0.9 < diag["mean_ratio"] < 1.1
    assert 0.8 < diag["sd_ratio"] < 1.2
    # signed mean correlation must be near zero even though individual draws vary
    assert abs(diag["mean_corr"]) < 0.25
    # the defining property of MSR: spatial autocorrelation is preserved
    assert 0.8 < diag["morans_i_ratio"] < 1.2


def test_surrogates_are_not_identical_to_input(grid):
    d, coords = grid
    w = weight_matrix_invdist(d, knn=4)
    mem, _ = compute_mem(w)
    x = coords[:, 0]
    sur = moran_surrogates(x, mem, n_rep=5, random_state=4)[:, :, 0]
    assert not np.allclose(sur[0], x)


def test_spin_pvalue_bounds():
    obs = np.array([0.5, -0.3])
    null = np.array([[0.0, 0.0], [0.1, -0.1], [0.6, 0.4]])
    p = spin_pvalue(obs, null)
    assert np.all(p > 0) and np.all(p <= 1)
    # |0.5| is exceeded once out of three -> (1+1)/(3+1)
    assert p[0] == pytest.approx(0.5)


def test_bh_fdr_matches_known_example():
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    q = bh_fdr(p)
    expected = p * 5 / np.arange(1, 6)
    expected = np.minimum.accumulate(expected[::-1])[::-1]
    assert np.allclose(q, np.minimum(expected, 1.0))


def test_bh_fdr_rejects_out_of_range():
    with pytest.raises(ValueError):
        bh_fdr(np.array([0.1, 1.2]))
