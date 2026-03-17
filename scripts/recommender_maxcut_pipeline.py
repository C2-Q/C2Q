#!/usr/bin/env python3
"""
Run the MaxCut recommender sweep and the downstream CSV-based selection step.

This combines the logic that previously lived separately in:
- src/tests/tests_recommender_maxcut.py
- src/tests/ex2_test_1.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.tests.ex2_test_1 import DEFAULT_CAPS, DEFAULT_DELTA, DEFAULT_LAMBDA, DEFAULT_TAU, TRAPPED_ION, load_wide, run_recommender
from src.tests.tests_recommender_maxcut import run_recommender_maxcut_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MaxCut multi-device recommender variation experiment."
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/recommender_maxcut",
        help="Root output directory for plots, raw CSVs, and Algorithm 1 outputs.",
    )
    parser.add_argument("--min-qubits", type=int, default=4, help="Smallest graph size.")
    parser.add_argument("--max-qubits", type=int, default=58, help="Largest graph size.")
    parser.add_argument("--step", type=int, default=2, help="Qubit-size step.")
    parser.add_argument("--degree", type=int, default=3, help="Regular-graph degree.")
    parser.add_argument("--graph-seed", type=int, default=100, help="Seed for graph generation.")
    parser.add_argument("--qaoa-layers", type=int, default=1, help="Number of QAOA layers.")
    parser.add_argument(
        "--save-device-figures",
        action="store_true",
        help="Save per-run recommender bar charts for every qubit size.",
    )
    parser.add_argument(
        "--lambda-weights",
        default="0.6,0.3,0.1",
        help="Algorithm 1 weights as lambda1,lambda2,lambda3.",
    )
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU, help="Algorithm 1 fidelity threshold.")
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA, help="Algorithm 1 trapped-ion bias.")
    parser.add_argument(
        "--exclude",
        default="",
        help="Comma-separated device names to exclude during Algorithm 1 post-processing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw_csv"
    algorithm1_dir = output_dir / "algorithm1"
    per_run_figures_dir = output_dir / "device_figures"

    lambda_weights = tuple(float(x) for x in args.lambda_weights.split(","))
    excludes = {s.strip() for s in args.exclude.split(",") if s.strip()}

    print("[step] Generating recommender sweep outputs")
    run_recommender_maxcut_experiment(
        outdir=str(raw_dir),
        min_qubits=args.min_qubits,
        max_qubits=args.max_qubits,
        step=args.step,
        degree=args.degree,
        graph_seed=args.graph_seed,
        qaoa_layers=args.qaoa_layers,
        save_device_figures=args.save_device_figures,
        figures_dir=str(per_run_figures_dir),
    )

    print("[step] Running CSV-based Algorithm 1 post-processing")
    err_wide, time_wide, price_wide = load_wide(
        raw_dir / "errors_wide.csv",
        raw_dir / "times_wide.csv",
        raw_dir / "prices_wide.csv",
    )
    winners_df, details_df = run_recommender(
        err_wide,
        time_wide,
        price_wide,
        lam=lambda_weights,
        tau=args.tau,
        delta=args.delta,
        caps=DEFAULT_CAPS,
        trapped_ion=TRAPPED_ION,
        exclude=excludes,
    )

    algorithm1_dir.mkdir(parents=True, exist_ok=True)
    winners_path = algorithm1_dir / "winners.csv"
    details_path = algorithm1_dir / "details.csv"
    winners_df.to_csv(winners_path, index=False)
    details_df.to_csv(details_path, index=False)

    print("[done] Outputs written")
    print(f" - raw_csv: {raw_dir}")
    print(f" - winners: {winners_path}")
    print(f" - details: {details_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
