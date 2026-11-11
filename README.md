# msrspatial

[中文](README.zh-CN.md) | **English**

**Spatial association of brain maps under a Moran spectral null.**

`msrspatial` tests whether two sets of parcel-level brain maps are spatially
associated, using a null model that preserves spatial autocorrelation (Moran
Spectral Randomization) instead of permuting values at random. It is the
reusable core of an imaging-transcriptomics pipeline: couple a set of coupling
maps to a set of molecular maps (receptor densities, gene-expression scores) and
get back effect sizes, spatial p-values and FDR-corrected q-values.

## Why a spatial null

Two smooth cortical maps can correlate at |r| > 0.5 by chance alone. Naive
permutation destroys that smoothness and therefore produces anti-conservative
p-values. MSR generates surrogate maps with the same spatial autocorrelation
structure as the observed map, so the test asks the right question: *is the
association stronger than expected for maps that are equally smooth?*

## Install

```bash
pip install msrspatial
# development
pip install -e ".[dev,plot,enrich]"
```

Requires Python >= 3.9, numpy, pandas, scipy.

## Quick start (Python)

```python
from msrspatial import spatial_association, load_matrix

maps = load_matrix("coupling_maps.csv", id_column="region")   # (n_parcels, n_maps)
features = load_matrix("receptors.csv", id_column="region")   # (n_parcels, n_features)
distance = load_matrix("distance.csv")                        # (n_parcels, n_parcels)

result = spatial_association(
    maps, features, distance,
    map_names=["MFPT", "Reliance"],
    feature_names=["D1", "DAT", "MOR"],
    knn=15, n_rep=1000, seed=2026,
)

result.table            # map, feature, r, p_spin, q_fdr, significant
result.significant()    # q_fdr < 0.05
result.diagnostics      # null-model QC per map
```

## Quick start (CLI)

```bash
msrspatial run \
  --maps coupling_maps.csv --features receptors.csv --distance distance.csv \
  --id-column region --knn 15 --n-perm 1000 --seed 2026 \
  --out results/

msrspatial demo --out demo_results      # bundled toy data, no download
```

Outputs `spatial_association.csv` and `diagnostics.json`.

## Input formats

| Input | Shape | Accepted |
|---|---|---|
| `maps` | parcels × maps | `.npy`, CSV/TSV (headered or not) |
| `features` | parcels × features | CSV/TSV with an id column |
| `distance` | parcels × parcels | `.npy`, CSV/TSV (headered or not) |

Row order **must** match across the three inputs. `validate_alignment` raises on
any mismatch, non-finite value, asymmetric or non-zero-diagonal distance matrix —
silent row swaps are the most common cause of irreproducible spatial results.

## Null-model quality control

`result.diagnostics["per_map_diagnostics"]` reports, per map:

| Metric | Expected | Meaning |
|---|---|---|
| `mean_ratio` | ≈ 1 | surrogate marginal mean preserved |
| `sd_ratio` | ≈ 1 | surrogate marginal SD preserved |
| `morans_i_ratio` | ≈ 1 | **spatial autocorrelation preserved** (defining property) |
| `mean_corr` | ≈ 0 | signed mean correlation with the observed map |
| `sd_corr` | can be large | a single surrogate may be close to +x or −x; MSR randomises phase, not amplitude |

If `morans_i_ratio` departs from 1, the null is not spatial and p-values are not
interpretable.

## Method

1. Spatial weights: inverse-distance k-nearest-neighbour graph, symmetrised.
2. Moran eigenvector maps (MEM) from the double-centred weight matrix.
3. For each observed map, generate `n_rep` surrogates by projecting onto the MEM
   basis, randomly flipping the sign of each coefficient, and reconstructing.
4. Correlate surrogates with the features (vectorised Spearman/Pearson).
5. Two-sided p-value with the conservative `(1 + exceed) / (K + 1)` correction.
6. Benjamini–Hochberg FDR, either within each map (`per_map`, default) or across
   all pairs (`global`).

References: Wagner & Dray (2015) *Methods in Ecology and Evolution* 6:1169–1179;
Vos de Wael et al. (2020) *NeuroImage* 221:117112 (brainspace).

## Scope and limitations

- Designed for **parcel-level** maps (e.g. HCP-MMP 360). Vertex-level data must
  be parcellated first.
- Assumes a **symmetric** distance matrix with a zero diagonal.
- `n_rep=1000` gives a p-value resolution of 0.001; use more for formal
  publication thresholds.
- FDR family must be declared **before** looking at the results.

## Testing

```bash
pytest -q            # if pytest is installed
python examples/selftest.py   # dependency-light alternative
```

## Citation

If you use this software, please cite the paper describing it (see
`CITATION.cff`) and the method paper above.

## License

BSD-3-Clause.
