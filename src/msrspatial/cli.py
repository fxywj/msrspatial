"""Command-line interface: ``msrspatial run`` and ``msrspatial demo``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .io import load_matrix, load_table, write_results
from .pipeline import spatial_association


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="msrspatial",
        description="Spatial association between brain maps under a Moran spectral null.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run an association analysis")
    run.add_argument("--maps", required=True, help="coupling maps: .npy or CSV matrix")
    run.add_argument("--features", required=True, help="molecular maps: CSV with id column")
    run.add_argument("--distance", required=True, help="parcel distance matrix (.npy or CSV)")
    run.add_argument("--id-column", default="region", help="id column in --features (default: region)")
    run.add_argument("--map-names", default=None, help="comma-separated names for map columns")
    run.add_argument("--knn", type=int, default=15)
    run.add_argument("--n-perm", type=int, default=1000)
    run.add_argument("--seed", type=int, default=2026)
    run.add_argument("--method", choices=["spearman", "pearson"], default="spearman")
    run.add_argument("--fdr-scope", choices=["per_map", "global"], default="per_map")
    run.add_argument("--alpha", type=float, default=0.05)
    run.add_argument("--out", default="results", help="output directory")

    demo = sub.add_parser("demo", help="run on bundled toy data")
    demo.add_argument("--out", default="demo_results")
    demo.add_argument("--n-perm", type=int, default=200)

    return p


def _run(args: argparse.Namespace) -> int:
    maps = load_matrix(args.maps)
    _, features = load_table(args.features, id_column=args.id_column)
    distance = load_matrix(args.distance)

    sep = "\t" if Path(args.features).suffix in (".tsv", ".txt") else ","
    header = pd.read_csv(args.features, sep=sep)
    feature_names = list(header.select_dtypes(include=[np.number]).columns)

    map_names = args.map_names.split(",") if args.map_names else None

    result = spatial_association(
        maps,
        features,
        distance,
        map_names=map_names,
        feature_names=feature_names,
        knn=args.knn,
        n_rep=args.n_perm,
        seed=args.seed,
        method=args.method,
        fdr_scope=args.fdr_scope,
        fdr_alpha=args.alpha,
    )

    out = Path(args.out)
    path = write_results(result.table, out)
    (out / "diagnostics.json").write_text(
        json.dumps(result.diagnostics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {path}")
    print(f"significant pairs (q < {args.alpha}): {result.diagnostics['n_significant']}")
    return 0


def _demo(args: argparse.Namespace) -> int:
    here = Path(__file__).resolve().parent
    data = here / "data"
    maps = load_matrix(data / "demo_maps.csv", id_column="region")
    meta, features = load_table(data / "demo_features.csv", id_column="region")
    distance = load_matrix(data / "demo_distance.csv")
    result = spatial_association(
        maps,
        features,
        distance,
        map_names=["coupling_A", "coupling_B"],
        feature_names=[c for c in pd.read_csv(data / "demo_features.csv").columns if c != "region"],
        n_rep=args.n_perm,
        seed=2026,
    )
    out = Path(args.out)
    path = write_results(result.table, out, stem="demo_results")
    print(f"wrote {path}")
    print(result.table.to_string(index=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args)
    if args.command == "demo":
        return _demo(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
