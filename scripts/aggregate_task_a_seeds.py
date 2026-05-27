"""Aggregate multi-seed Task A results into mean/std tables.

This script is intentionally lightweight and offline-only:
- input: Task A CSV with one row per seed/config
- output: aggregated CSV with mean/std/count over seeds

It is designed for the building benchmark main tables after we rerun
`scripts/run_task_a.py` with multiple seeds.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


GROUP_COLS = ["task", "feature_set", "model", "split_type"]
EXCLUDE_METRIC_COLS = {"overall_n_train", "overall_n_test"}


def pick_metric_columns(df: pd.DataFrame) -> list[str]:
    metric_cols = []
    for col in df.columns:
        if col in GROUP_COLS or col == "seed":
            continue
        if col in EXCLUDE_METRIC_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            metric_cols.append(col)
    return metric_cols


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = pick_metric_columns(df)
    pieces = []

    grouped = df.groupby(GROUP_COLS, dropna=False)
    for keys, sub in grouped:
        row = dict(zip(GROUP_COLS, keys))
        row["n_seeds"] = int(sub["seed"].nunique()) if "seed" in sub.columns else len(sub)
        if "seed" in sub.columns:
            row["seeds"] = ",".join(str(int(x)) for x in sorted(sub["seed"].dropna().unique()))
        for col in metric_cols:
            vals = pd.to_numeric(sub[col], errors="coerce")
            row[f"{col}_n"] = int(vals.notna().sum())
            row[f"{col}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
            row[f"{col}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() >= 2 else np.nan
        pieces.append(row)

    out = pd.DataFrame(pieces)
    sort_cols = [c for c in GROUP_COLS if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_path",
        type=str,
        default="results/clean_building/task_a_results.csv",
        help="Input Task A results CSV with one row per seed/config "
             "(default points to clean_building/; the legacy root path "
             "results/task_a_results.csv is a historical merge-polluted file).",
    )
    ap.add_argument(
        "--out_path",
        type=str,
        default="results/clean_building/task_a_results_seed_summary.csv",
        help="Output aggregated CSV with mean/std over seeds.",
    )
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    df = pd.read_csv(in_path)
    if df.empty:
        raise ValueError(f"Input file is empty: {in_path}")

    agg = aggregate(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_path, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Aggregated rows: {len(agg)}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
