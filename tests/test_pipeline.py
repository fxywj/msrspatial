"""End-to-end test on the bundled toy dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from msrspatial import load_matrix, load_table, spatial_association, validate_alignment

DATA = Path(__file__).resolve().parents[1] / "src" / "msrspatial" / "data"


@pytest.fixture(scope="module")
def demo():
    maps = load_matrix(DATA / "demo_maps.csv", id_column="region")
    _, features = load_table(DATA / "demo_features.csv", id_column="region")
    distance = load_matrix(DATA / "demo_distance.csv")
    names = list(pd.read_csv(DATA / "demo_features.csv").columns[1:])
    return maps, features, distance, names


def test_pipeline_runs_and_shapes(demo):
    maps, features, distance, names = demo
    res = spatial_association(
        maps, features, distance,
        map_names=["coupling_A", "coupling_B"],
        feature_names=names,
        n_rep=200, seed=2026,
    )
    assert len(res.table) == maps.shape[1] * features.shape[1]
    assert set(res.table.columns) == {"map", "feature", "r", "p_spin", "q_fdr", "significant"}
    assert res.table["p_spin"].between(0, 1).all()
    assert res.table["q_fdr"].between(0, 1).all()
    assert res.diagnostics["n_parcels"] == maps.shape[0]


def test_planted_signal_is_recovered(demo):
    maps, features, distance, names = demo
    res = spatial_association(
        maps, features, distance,
        map_names=["coupling_A", "coupling_B"],
        feature_names=names,
        n_rep=500, seed=11,
    )
    t = res.table
    # receptor_1 was generated to track coupling_A strongly
    r_planted = t[(t["map"] == "coupling_A") & (t["feature"] == names[0])]["r"].iloc[0]
    r_null = t[(t["map"] == "coupling_A") & (t["feature"] == names[2])]["r"].iloc[0]
    assert abs(r_planted) > abs(r_null)
    assert abs(r_planted) > 0.5


def test_seed_is_reproducible(demo):
    maps, features, distance, names = demo
    kwargs = dict(map_names=["A", "B"], feature_names=names, n_rep=100, seed=99)
    a = spatial_association(maps, features, distance, **kwargs).table
    b = spatial_association(maps, features, distance, **kwargs).table
    pd.testing.assert_frame_equal(a, b)


def test_alignment_validation_catches_mismatch(demo):
    maps, features, distance, _ = demo
    with pytest.raises(ValueError):
        validate_alignment(maps, features[:-1], distance)


def test_global_fdr_scope(demo):
    maps, features, distance, names = demo
    res = spatial_association(
        maps, features, distance,
        map_names=["A", "B"], feature_names=names,
        n_rep=100, seed=5, fdr_scope="global",
    )
    assert res.table["q_fdr"].notna().all()
    assert res.diagnostics["fdr_scope"] == "global"
