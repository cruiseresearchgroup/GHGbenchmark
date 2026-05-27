"""Aggregate multi-seed Task C results into mean/std tables.

This mirrors `aggregate_task_a_seeds.py` but supports the extra dimensions
used by Task C:
- C1: target_city
- C2: target_city + source_cities
- C3: target_city + k

We keep `dropna=False` so mixed-subtask schemas aggregate correctly.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


GROUP_COLS = ["task", "feature_set", "model", "target_city", "source_cities", "k"]
EXCLUDE_METRIC_COLS = {"n_train", "n_test"}


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
    rows = []

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
        rows.append(row)

    out = pd.DataFrame(rows)
    sort_cols = [c for c in GROUP_COLS if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", type=str, default="results/task_c_results.csv")
    ap.add_argument("--out_path", type=str, default="results/task_c_results_seed_summary.csv")
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
