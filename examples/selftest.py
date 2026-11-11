"""Dependency-light self-check (no pytest required).

Run after ``pip install -e .`` (or with ``PYTHONPATH=src``)::

    python examples/selftest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from msrspatial import (  # noqa: E402
    bh_fdr,
    colwise_pearson,
    colwise_spearman,
    compute_mem,
    load_matrix,
    load_table,
    morans_i,
    moran_surrogates,
    null_diagnostics,
    spatial_association,
    spin_pvalue,
    weight_matrix_invdist,
)


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not condition:
        raise SystemExit(f"self-test failed: {name}")


def main() -> None:
    # --- spatial weights & MEM -------------------------------------------------
    xs, ys = np.meshgrid(np.arange(6), np.arange(6))
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    w = weight_matrix_invdist(d, knn=4)
    check("weights symmetric", np.allclose(w, w.T))
    check("weights zero diagonal", np.allclose(np.diag(w), 0.0))

    mem, ev = compute_mem(w)
    gram = mem.T @ mem
    check("MEM orthonormal", np.allclose(gram, np.eye(mem.shape[1]), atol=1e-8),
          f"(n_comp={mem.shape[1]})")

    smooth = coords[:, 0]
    noisy = np.random.default_rng(0).standard_normal(len(coords))
    check("Moran's I smooth > noisy", morans_i(smooth, w) > morans_i(noisy, w),
          f"({morans_i(smooth, w):.3f} vs {morans_i(noisy, w):.3f})")

    # --- correlations ----------------------------------------------------------
    rng = np.random.default_rng(1)
    x = rng.standard_normal((60, 3))
    y = rng.standard_normal((60, 4))
    r = colwise_pearson(x, y)
    rho = colwise_spearman(x, y)
    err_r = max(abs(r[i, j] - stats.pearsonr(x[:, i], y[:, j])[0])
                for i in range(3) for j in range(4))
    err_rho = max(abs(rho[i, j] - stats.spearmanr(x[:, i], y[:, j])[0])
                  for i in range(3) for j in range(4))
    check("Pearson vs scipy", err_r < 1e-10, f"(max err={err_r:.2e})")
    check("Spearman vs scipy", err_rho < 1e-10, f"(max err={err_rho:.2e})")

    # --- null model ------------------------------------------------------------
    obs = coords[:, 0] + 0.3 * rng.standard_normal(len(coords))
    sur = moran_surrogates(obs, mem, n_rep=200, random_state=3)
    diag = null_diagnostics(obs, sur, weights=w)
    check("surrogate shape", sur.shape == (200, len(obs), 1), f"{sur.shape}")
    check("surrogate mean preserved", 0.9 < diag["mean_ratio"] < 1.1,
          f"(ratio={diag['mean_ratio']:.3f})")
    check("surrogate sd preserved", 0.8 < diag["sd_ratio"] < 1.2,
          f"(ratio={diag['sd_ratio']:.3f})")
    check("surrogate autocorrelation preserved", 0.8 < diag["morans_i_ratio"] < 1.2,
          f"(Moran ratio={diag['morans_i_ratio']:.3f})")
    check("surrogate signed mean corr near zero", abs(diag["mean_corr"]) < 0.25,
          f"(mean r={diag['mean_corr']:.3f}, sd={diag['sd_corr']:.3f})")

    p = spin_pvalue(np.array([0.5, -0.3]), np.array([[0.0, 0.0], [0.1, -0.1], [0.6, 0.4]]))
    check("spin p-value", np.allclose(p, [0.5, 0.5]), f"{p}")

    # --- FDR -------------------------------------------------------------------
    q = bh_fdr(np.array([0.01, 0.02, 0.03, 0.04, 0.05]))
    check("BH monotone", np.all(np.diff(q) >= -1e-12), f"{q}")

    # --- end-to-end ------------------------------------------------------------
    data = ROOT / "src" / "msrspatial" / "data"
    maps = load_matrix(data / "demo_maps.csv", id_column="region")
    _, features = load_table(data / "demo_features.csv", id_column="region")
    distance = load_matrix(data / "demo_distance.csv")
    names = ["receptor_1", "receptor_2", "receptor_3"]
    res = spatial_association(maps, features, distance,
                              map_names=["coupling_A", "coupling_B"],
                              feature_names=names, n_rep=500, seed=2026)
    check("pipeline rows", len(res.table) == 6, f"({len(res.table)} pairs)")
    r_planted = res.table.query("map == 'coupling_A' and feature == 'receptor_1'")["r"].iloc[0]
    r_null = res.table.query("map == 'coupling_A' and feature == 'receptor_3'")["r"].iloc[0]
    check("planted signal recovered", abs(r_planted) > abs(r_null) and abs(r_planted) > 0.5,
          f"(r={r_planted:.3f} vs noise {r_null:.3f})")
    print("\nall self-checks passed")


if __name__ == "__main__":
    main()
