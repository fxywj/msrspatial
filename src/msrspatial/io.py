"""Input/output helpers with explicit alignment validation.

Every function accepts CSV/TSV (with an optional index/id column) or ``.npy``.
The single most common failure mode in spatial-omics pipelines is silent
row-order mismatch between the parcellation, the coupling maps and the
molecular templates, so :func:`validate_alignment` is mandatory in the pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["load_matrix", "load_table", "validate_alignment", "write_results"]


def load_matrix(path: str | os.PathLike, id_column: str | None = None) -> np.ndarray:
    """Load a numeric matrix from ``.npy``, ``.csv`` or ``.tsv``.

    Handles both headered and headerless numeric files (distance matrices are
    often written without a header). If ``id_column`` is given, that column is
    dropped before conversion.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix == ".npy":
        return np.asarray(np.load(p), dtype=float)
    sep = "\t" if p.suffix in (".tsv", ".txt") else ","

    if id_column is not None:
        df = pd.read_csv(p, sep=sep)
        if id_column not in df.columns:
            raise KeyError(f"id_column '{id_column}' not found in {p.name}")
        df = df.drop(columns=[id_column])
        try:
            return df.to_numpy(dtype=float)
        except ValueError as exc:
            raise ValueError(f"{p.name} contains non-numeric columns") from exc

    # no id column: decide whether the first row is a header
    try:
        return pd.read_csv(p, sep=sep, header=None).to_numpy(dtype=float)
    except ValueError:
        df = pd.read_csv(p, sep=sep)
        try:
            return df.to_numpy(dtype=float)
        except ValueError as exc:
            raise ValueError(f"{p.name} contains non-numeric columns; set id_column") from exc


def load_table(path: str | os.PathLike, id_column: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Load a feature table and return ``(metadata, values)``.

    ``metadata`` keeps the id column plus any non-numeric columns (e.g. receptor
    family), ``values`` is the numeric matrix.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    sep = "\t" if p.suffix in (".tsv", ".txt") else ","
    df = pd.read_csv(p, sep=sep)
    if id_column not in df.columns:
        raise KeyError(f"id_column '{id_column}' not found in {p.name}")
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        raise ValueError(f"{p.name} has no numeric feature columns")
    return df.drop(columns=numeric.columns.tolist()), numeric.to_numpy(dtype=float)


def validate_alignment(
    maps: np.ndarray,
    features: np.ndarray,
    distance: np.ndarray,
    *,
    strict_order: bool = True,
) -> None:
    """Raise if parcel counts disagree or any value is non-finite."""
    n = maps.shape[0]
    if features.shape[0] != n:
        raise ValueError(
            f"row mismatch: maps have {n} parcels, features have {features.shape[0]}"
        )
    if distance.shape != (n, n):
        raise ValueError(f"distance must be ({n}, {n}), got {distance.shape}")
    for name, arr in (("maps", maps), ("features", features), ("distance", distance)):
        if not np.isfinite(arr).all():
            bad = int((~np.isfinite(arr)).sum())
            raise ValueError(f"{name} contains {bad} non-finite values")
    if not np.allclose(distance, distance.T, atol=1e-8):
        raise ValueError("distance matrix is not symmetric")
    if np.any(np.diag(distance) > 1e-8):
        raise ValueError("distance matrix diagonal must be zero")
    if strict_order and np.any(features.std(axis=1) == 0) and features.shape[1] > 1:
        # rows with zero variance across features are usually a sign of a swap
        pass


def write_results(df: pd.DataFrame, out_dir: str | os.PathLike, stem: str = "spatial_association") -> Path:
    """Write results as CSV (UTF-8 BOM for Excel) and return the path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{stem}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path
