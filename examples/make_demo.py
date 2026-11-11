"""Generate the bundled toy dataset (20 parcels, 2 maps, 3 molecular features).

Run from the package root::

    python examples/make_demo.py

The data are synthetic and exist only so that ``msrspatial demo`` works without
shipping any human data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def smooth_field(coords: np.ndarray, rng: np.random.Generator, length: float = 0.6) -> np.ndarray:
    """Gaussian-kernel-smoothed random field -> spatially autocorrelated map."""
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    k = np.exp(-(d**2) / (2 * length**2))
    k /= k.sum(axis=1, keepdims=True)
    return k @ rng.standard_normal(coords.shape[0])


def multi_scale_field(coords: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sum of smooth fields at three scales.

    A single-scale field is dominated by one Moran eigenvector, which makes MSR
    surrogates collapse to approximately +x or -x. Real cortical maps are
    multi-scale, so the demo data should be too.
    """
    field = np.zeros(coords.shape[0])
    for length, weight in ((0.45, 1.0), (0.9, 0.7), (1.8, 0.45)):
        field += weight * smooth_field(coords, rng, length=length)
    return field


def main() -> None:
    rng = np.random.default_rng(7)
    out = Path(__file__).resolve().parents[1] / "src" / "msrspatial" / "data"
    out.mkdir(parents=True, exist_ok=True)

    n = 20
    coords = rng.uniform(0, 3, size=(n, 2))
    distance = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

    map_a = multi_scale_field(coords, rng)
    map_b = multi_scale_field(coords, rng)

    # feature 1 tracks map_a; feature 2 is weakly related; feature 3 is noise
    f1 = 0.85 * map_a + 0.53 * multi_scale_field(coords, rng)
    f2 = 0.45 * map_a + 0.89 * multi_scale_field(coords, rng)
    f3 = multi_scale_field(coords, rng)

    pd.DataFrame({"region": [f"R{i + 1:02d}" for i in range(n)],
                  "coupling_A": map_a, "coupling_B": map_b}).to_csv(
        out / "demo_maps.csv", index=False, encoding="utf-8"
    )
    pd.DataFrame({"region": [f"R{i + 1:02d}" for i in range(n)],
                  "receptor_1": f1, "receptor_2": f2, "receptor_3": f3}).to_csv(
        out / "demo_features.csv", index=False, encoding="utf-8"
    )
    pd.DataFrame(distance).to_csv(out / "demo_distance.csv", index=False, header=False)
    print(f"wrote demo data to {out}")


if __name__ == "__main__":
    main()
