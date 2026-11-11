# Changelog

All notable changes to this project are documented here.
This project follows [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-11

### Added

- `spatial`: inverse-distance k-NN weights, Moran eigenvector maps (MEM),
  Moran's I.
- `nulls`: Moran Spectral Randomization surrogates, two-sided permutation
  p-values with the conservative `(1 + exceed) / (K + 1)` correction,
  and null-model diagnostics (marginal and autocorrelation preservation).
- `correlate`: vectorised Pearson and Spearman between column sets.
- `correct`: Benjamini–Hochberg FDR with a declared family scope
  (`per_map` or `global`).
- `io`: matrix/table loaders that accept `.npy`, CSV and TSV, plus
  `validate_alignment` which rejects row-count mismatch, non-finite values,
  asymmetric distance matrices and non-zero diagonals.
- `pipeline`: `spatial_association()` end-to-end entry point returning a
  results table and run diagnostics.
- `cli`: `msrspatial run` and `msrspatial demo`.
- Bundled synthetic demo dataset (20 parcels, 2 maps, 3 features) so the
  package can be exercised without any human data.
- 12 unit/end-to-end tests and a dependency-light `examples/selftest.py`.
- Bilingual documentation: `README.md` (English) and `README.zh-CN.md`.

### Notes

- No human neuroimaging data is distributed with this package. The demo data
  are synthetic and intended only for smoke tests.
- `n_rep=1000` gives a p-value resolution of 0.001.

[0.1.0]: https://github.com/fxywj/msrspatial/releases/tag/v0.1.0
