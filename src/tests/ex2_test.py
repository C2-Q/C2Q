#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recommender from CSVs (errors, times, prices)
---------------------------------------------

Reads either WIDE or LONG CSV triplets produced by save_recommender_csvs() and
computes the best device for each qubit size using Algorithm 1:

- Feasible if capacity allows and (1 - error) >= tau
- Normalize per-qubit on feasible set: hat(E,T,P)
- Score = lambda1*Ehat + lambda2*That + lambda3*Phat
- Soft bias: subtract delta for trapped-ion if (n>10) or (lambda1>=0.8)
- Tie-break: lexicographic by (Ehat, Phat, That)

Outputs:
- winners.csv
- details.csv

Usage examples:

# WIDE format (files with rows=qubits, columns=devices; a 'qubits' column present):
python recommender_from_csv.py \
  --format wide \
  --errors ex2_recommender.csv/errors_wide.csv \
  --times  ex2_recommender.csv/times_wide.csv \
  --prices ex2_recommender.csv/prices_wide.csv \
  --out winner

# LONG format (qubits,device,value):
python recommender_from_csv.py \
  --format long \
  --errors ex2_recommender.csv/errors_long.csv \
  --times  ex2_recommender.csv/times_long.csv \
  --prices ex2_recommender.csv/prices_long.csv \
  --out out_long \
  --exclude "Rigetti Ankaa-2"
"""

import argparse
import pandas as pd
from pathlib import Path

# ----------------------------
# Defaults (edit if desired)
# ----------------------------
DEFAULT_LAMBDA = (0.6, 0.3, 0.1)  # (lambda1, lambda2, lambda3)
DEFAULT_TAU = 0.5  # reject if 1 - error < tau
DEFAULT_DELTA = 0.05  # trapped-ion soft bias (applies when n>10 or lambda1>=0.8)

# Capacity caps
DEFAULT_CAPS = {
    "Quantinuum H1": 20,
    "Quantinuum H2": 56,
    "IonQ Aria (Amazon)": 25,
    "IonQ Aria (Azure)": 25,
    "IQM Garnet": 20,
    # Common IBM names used in your data / plots:
    "ibm_brisbane": 127,
    "ibm_sherbrooke": 127,
    "ibm_kyiv": 127,
    # If present:
    "Rigetti Ankaa-9Q-3": 9,
    "Rigetti Ankaa-2": 9,
}

# Trapped-ion flag (used for δ bias)
TRAPPED_ION = {
    "Quantinuum H1": True,
    "Quantinuum H2": True,
    "IonQ Aria (Amazon)": True,
    "IonQ Aria (Azure)": True,
    # others default False
}


# ----------------------------
# Data loading
# ----------------------------
def load_wide(errors_csv: Path, times_csv: Path, prices_csv: Path):
    """Returns (err_df, time_df, price_df) indexed by qubits, columns = devices."""
    err = pd.read_csv(errors_csv)
    tim = pd.read_csv(times_csv)
    pri = pd.read_csv(prices_csv)

    # Expect a 'qubits' column
    for df, nm in [(err, "errors"), (tim, "times"), (pri, "prices")]:
        if "qubits" not in df.columns:
            raise ValueError(f"{nm} CSV must include a 'qubits' column in WIDE format.")
        df.set_index("qubits", inplace=True)

    # Align on index, union of devices handled naturally by columns
    # (Missing entries stay as NaN; they will be ignored if infeasible.)
    return err, tim, pri


def load_long(errors_csv: Path, times_csv: Path, prices_csv: Path):
    """Returns (err_df, time_df, price_df) in WIDE shape (index=qubits, columns=devices)."""

    def pivot_long(path, label):
        df = pd.read_csv(path)
        # Expect columns: qubits, device, value
        exp = {"qubits", "device", "value"}
        if exp - set(df.columns):
            raise ValueError(f"{label} LONG CSV must have columns: qubits, device, value")
        return df.pivot_table(index="qubits", columns="device", values="value", aggfunc="first")

    return pivot_long(errors_csv, "errors"), pivot_long(times_csv, "times"), pivot_long(prices_csv, "prices")


# ----------------------------
# Algorithm 1 core
# ----------------------------
def feasible(n: int, device: str, error_pct: float, caps: dict, tau: float) -> bool:
    """Capacity and fidelity (1 - error) >= tau."""
    if device not in caps:
        # Unknown device: treat as not feasible unless you want to assume huge cap
        return False
    if n > int(caps[device]):
        return False
    e = float(error_pct) / 100.0
    return (1.0 - e) >= tau


def compute_for_qubit(n: int,
                      err_row: pd.Series,
                      time_row: pd.Series,
                      price_row: pd.Series,
                      lam, tau, delta, caps, trapped_ion, ion_bias_condition: bool):
    """
    Compute scores for a single qubit size.
    Returns a DataFrame rows per feasible device with Ehat, That, Phat, score, rank.
    """
    rows = []
    # Devices observed at this n = union of columns present in any of the three series
    devices = sorted(set(err_row.dropna().index) |
                     set(time_row.dropna().index) |
                     set(price_row.dropna().index))

    # Build raw rows first (and filter infeasible later)
    for d in devices:
        e = err_row.get(d, float("nan"))
        t = time_row.get(d, float("nan"))
        p = price_row.get(d, float("nan"))
        if pd.isna(e) or pd.isna(t) or pd.isna(p):
            continue
        if not feasible(n, d, e, caps, tau):
            continue
        rows.append({"qubits": n, "device": d, "E": float(e) / 100.0, "T": float(t), "P": float(p)})

    if not rows:
        return pd.DataFrame(columns=["qubits", "device", "E", "T", "P", "Ehat", "That", "Phat", "score", "rank"])

    df = pd.DataFrame(rows)

    # Normalization maxima (on feasible set)
    Emax = df["E"].max()
    Tmax = df["T"].max()
    Pmax = df["P"].max()

    # Normalized features
    df["Ehat"] = df["E"] / Emax if Emax > 0 else 0.0
    df["That"] = df["T"] / Tmax if Tmax > 0 else 0.0
    df["Phat"] = df["P"] / Pmax if Pmax > 0 else 0.0

    # Score
    df["score"] = lam[0] * df["Ehat"] + lam[1] * df["That"] + lam[2] * df["Phat"]

    # Soft bias for trapped-ion if condition holds
    if ion_bias_condition:
        df["score"] = df.apply(
            lambda r: r["score"] - delta if trapped_ion.get(r["device"], False) else r["score"],
            axis=1
        )

    # Tie-break: by (score, Ehat, Phat, That)
    df = df.sort_values(by=["score", "Ehat", "Phat", "That"], ascending=True).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def run_recommender(err_wide: pd.DataFrame,
                    time_wide: pd.DataFrame,
                    price_wide: pd.DataFrame,
                    lam=DEFAULT_LAMBDA,
                    tau=DEFAULT_TAU,
                    delta=DEFAULT_DELTA,
                    caps=None,
                    trapped_ion=None,
                    exclude=None):
    """
    Iterate over all qubit sizes (rows) and compute winners + details.
    """
    if trapped_ion is None:
        trapped_ion = TRAPPED_ION
    if caps is None:
        caps = DEFAULT_CAPS
    if exclude is None:
        exclude = set()
    # Ensure aligned indices; reindex the others to the union, fill with NaN
    all_q = sorted(set(err_wide.index) | set(time_wide.index) | set(price_wide.index))
    err_wide = err_wide.reindex(all_q)
    time_wide = time_wide.reindex(all_q)
    price_wide = price_wide.reindex(all_q)

    winners = []
    detail_frames = []

    for n in all_q:
        err_row = err_wide.loc[n] if n in err_wide.index else pd.Series(dtype=float)
        time_row = time_wide.loc[n] if n in time_wide.index else pd.Series(dtype=float)
        price_row = price_wide.loc[n] if n in price_wide.index else pd.Series(dtype=float)

        # Drop excluded devices (by masking columns)
        if exclude:
            err_row = err_row.drop(index=[c for c in err_row.index if c in exclude], errors="ignore")
            time_row = time_row.drop(index=[c for c in time_row.index if c in exclude], errors="ignore")
            price_row = price_row.drop(index=[c for c in price_row.index if c in exclude], errors="ignore")

        ion_bias_condition = (lam[0] >= 0.8) or (int(n) > 10)

        df_n = compute_for_qubit(n, err_row, time_row, price_row,
                                 lam, tau, delta, caps, trapped_ion, ion_bias_condition)

        if df_n.empty:
            winners.append({"qubits": int(n), "winner": None})
        else:
            winners.append({"qubits": int(n), "winner": df_n.iloc[0]["device"]})
            detail_frames.append(df_n)

    winners_df = pd.DataFrame(winners).sort_values("qubits")
    details_df = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    return winners_df, details_df


# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Run Algorithm 1 recommender from CSVs (wide or long).")
    ap.add_argument("--format", choices=["wide", "long"], required=True, help="CSV format type")
    ap.add_argument("--errors", required=True, help="Path to errors CSV")
    ap.add_argument("--times", required=True, help="Path to times CSV")
    ap.add_argument("--prices", required=True, help="Path to prices CSV")
    ap.add_argument("--out", default="out", help="Output directory")
    ap.add_argument("--lambda", dest="lam", default="0.6,0.3,0.1",
                    help="Weights as 'lambda1,lambda2,lambda3'")
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU, help="Fidelity threshold τ")
    ap.add_argument("--delta", type=float, default=DEFAULT_DELTA, help="Trapped-ion bias δ")
    ap.add_argument("--exclude", default="", help="Comma-separated device names to exclude")
    args = ap.parse_args()

    lam = tuple(float(x) for x in args.lam.split(","))
    tau = float(args.tau)
    delta = float(args.delta)
    excludes = {s.strip() for s in args.exclude.split(",")} if args.exclude.strip() else set()

    errors_csv = Path(args.errors)
    times_csv = Path(args.times)
    prices_csv = Path(args.prices)

    if args.format == "wide":
        err_wide, time_wide, price_wide = load_wide(errors_csv, times_csv, prices_csv)
    else:
        err_wide, time_wide, price_wide = load_long(errors_csv, times_csv, prices_csv)

    winners_df, details_df = run_recommender(
        err_wide, time_wide, price_wide,
        lam=lam, tau=tau, delta=delta,
        caps=DEFAULT_CAPS, trapped_ion=TRAPPED_ION,
        exclude=excludes
    )

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    winners_df.to_csv(outdir / "winners.csv", index=False)
    details_df.to_csv(outdir / "details.csv", index=False)

    print(f"Wrote {outdir / 'winners.csv'} and {outdir / 'details.csv'}")
    print(f"λ={lam}, τ={tau}, δ={delta}, excludes={list(excludes)}")


if __name__ == "__main__":
    main()
