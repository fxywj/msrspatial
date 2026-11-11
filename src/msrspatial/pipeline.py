"""End-to-end spatial association with a Moran spectral null.

This is the reusable core of the A-line analysis: for every coupling map,
test its spatial correlation with every molecular map under a null that
preserves spatial autocorrelation, then correct within the declared family.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .correct import bh_fdr
from .correlate import colwise_pearson, colwise_spearman
from .io import validate_alignment
from .nulls import moran_surrogates, null_diagnostics, spin_pvalue
from .spatial import compute_mem, weight_matrix_invdist

__all__ = ["SpatialResult", "spatial_association"]


@dataclass
class SpatialResult:
    """Container holding the results table and run diagnostics."""

    table: pd.DataFrame
    diagnostics: dict = field(default_factory=dict)

    def significant(self, alpha: float = 0.05) -> pd.DataFrame:
        return self.table[self.table["q_fdr"] < alpha].reset_index(drop=True)


def spatial_association(
    maps: np.ndarray,
    features: np.ndarray,
    distance: np.ndarray,
    *,
    map_names: list[str] | None = None,
    feature_names: list[str] | None = None,
    knn: int = 15,
    n_rep: int = 1000,
    seed: int = 2026,
    method: str = "spearman",
    fdr_scope: str = "per_map",
    fdr_alpha: float = 0.05,
) -> SpatialResult:
    """Test spatial association between coupling maps and molecular maps.

    Parameters
    ----------
    maps
        ``(n_parcels, n_maps)`` coupling maps (one column per metric/state).
    features
        ``(n_parcels, n_features)`` molecular maps (receptors, gene scores...).
    distance
        ``(n_parcels, n_parcels)`` centroid distance matrix.
    knn
        Neighbours for the inverse-distance spatial weights.
    n_rep
        Number of MSR surrogates (1000 is the published default).
    method
        ``"spearman"`` (default) or ``"pearson"``.
    fdr_scope
        ``"per_map"`` corrects across features within each map (matches the
        original A-line analysis); ``"global"`` corrects across all pairs.
    """
    maps = np.asarray(maps, dtype=float)
    features = np.asarray(features, dtype=float)
    distance = np.asarray(distance, dtype=float)
    if maps.ndim == 1:
        maps = maps[:, None]
    if features.ndim == 1:
        features = features[:, None]
    validate_alignment(maps, features, distance)

    if method not in ("spearman", "pearson"):
        raise ValueError("method must be 'spearman' or 'pearson'")
    correlate = colwise_spearman if method == "spearman" else colwise_pearson

    n_maps = maps.shape[1]
    n_feat = features.shape[1]
    map_names = list(map_names) if map_names is not None else [f"map_{i}" for i in range(n_maps)]
    feature_names = (
        list(feature_names) if feature_names is not None else [f"feature_{i}" for i in range(n_feat)]
    )
    if len(map_names) != n_maps or len(feature_names) != n_feat:
        raise ValueError("name lists must match the number of columns")

    weights = weight_matrix_invdist(distance, knn=knn)
    mem, ev = compute_mem(weights)

    rows: list[dict] = []
    diag: dict = {
        "n_parcels": int(maps.shape[0]),
        "n_maps": int(n_maps),
        "n_features": int(n_feat),
        "n_surrogates": int(n_rep),
        "knn": int(knn),
        "n_mem_components": int(mem.shape[1]),
        "method": method,
        "fdr_scope": fdr_scope,
        "per_map_diagnostics": {},
    }

    for j in range(n_maps):
        observed_map = maps[:, j]
        observed = correlate(observed_map[:, None], features)[0]

        surrogates = moran_surrogates(observed_map, mem, n_rep=n_rep, random_state=seed + j)
        # (n_rep, n) -> (n, n_rep) so colwise_* yields (n_rep, n_features)
        null = correlate(surrogates[:, :, 0].T, features)
        p_spin = spin_pvalue(observed, null, two_sided=True)

        diag["per_map_diagnostics"][map_names[j]] = null_diagnostics(
            observed_map, surrogates, weights=weights
        )

        for k in range(n_feat):
            rows.append(
                {
                    "map": map_names[j],
                    "feature": feature_names[k],
                    "r": float(observed[k]),
                    "p_spin": float(p_spin[k]),
                }
            )

    table = pd.DataFrame(rows)
    if fdr_scope == "per_map":
        table["q_fdr"] = np.nan
        for name, group in table.groupby("map", sort=False):
            table.loc[group.index, "q_fdr"] = bh_fdr(group["p_spin"].to_numpy())
    elif fdr_scope == "global":
        table["q_fdr"] = bh_fdr(table["p_spin"].to_numpy())
    else:
        raise ValueError("fdr_scope must be 'per_map' or 'global'")

    table["significant"] = table["q_fdr"] < fdr_alpha
    table = table[["map", "feature", "r", "p_spin", "q_fdr", "significant"]]
    diag["n_significant"] = int(table["significant"].sum())
    diag["fdr_alpha"] = float(fdr_alpha)

    return SpatialResult(table=table, diagnostics=diag)
