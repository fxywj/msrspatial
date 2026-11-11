"""msrspatial · spatial association of brain maps under a Moran spectral null.

Minimal public API::

    from msrspatial import spatial_association
    result = spatial_association(maps, features, distance, n_rep=1000, seed=2026)
    result.table          # map x feature results with p_spin and q_fdr
"""

from __future__ import annotations

from .correct import bh_fdr
from .correlate import colwise_pearson, colwise_spearman
from .io import load_matrix, load_table, validate_alignment, write_results
from .nulls import moran_surrogates, null_diagnostics, spin_pvalue
from .pipeline import SpatialResult, spatial_association
from .spatial import compute_mem, morans_i, weight_matrix_invdist

__version__ = "0.1.0"

__all__ = [
    "SpatialResult",
    "spatial_association",
    "weight_matrix_invdist",
    "compute_mem",
    "morans_i",
    "moran_surrogates",
    "spin_pvalue",
    "null_diagnostics",
    "colwise_pearson",
    "colwise_spearman",
    "bh_fdr",
    "load_matrix",
    "load_table",
    "validate_alignment",
    "write_results",
    "__version__",
]
