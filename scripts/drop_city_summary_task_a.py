"""Summarize Task A rows after dropping one city from city-level metrics.

Important:
- This is an offline summary over already-saved per-city metrics such as
  `r2_<city>` / `mae_<city>`.
- It does NOT reconstruct sample-level predictions.
- Therefore it can provide a "drop-city macro/median" view, but not a true
  recomputed sample-level overall R² after removing that city.

This is still useful for quickly checking whether a headline is dominated by
one city in the city-level summary sense.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE_COLS = ["task", "feature_set", "model", "seed", "split_type"]


def extract_city_metric_cols(df: pd.DataFrame, prefix: str) -> dict[str, str]:
    out = {}
    for col in df.columns:
        if col.startswith(prefix):
            out[col[len(prefix):]] = col
    return out


def summarize_row(row: pd.Series, drop_city: str) -> dict:
    r2_cols = extract_city_metric_cols(pd.DataFrame([row]), "r2_")
    cities = sorted(c for c in r2_cols if c != drop_city)
    vals = []
    for city in cities:
        v = row.get(f"r2_{city}", np.nan)
        if pd.notna(v):
            vals.append(float(v))
    out = {k: row[k] for k in BASE_COLS if k in row.index}
    out["drop_city"] = drop_city
    out["drop_city_was_present"] = bool(
        f"r2_{drop_city}" in row.index and pd.notna(row.get(f"r2_{drop_city}"))
    )
    out["n_cities_retained"] = len(vals)
    out["macro_r2_dropcity"] = float(np.mean(vals)) if vals else np.nan
    out["median_r2_dropcity"] = float(np.median(vals)) if vals else np.nan
    out["q1_r2_dropcity"] = float(np.quantile(vals, 0.25)) if vals else np.nan
    out["q3_r2_dropcity"] = float(np.quantile(vals, 0.75)) if vals else np.nan
    out["min_r2_dropcity"] = float(np.min(vals)) if vals else np.nan
    out["max_r2_dropcity"] = float(np.max(vals)) if vals else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_path",
        type=str,
        default="results/clean_building/task_a_results.csv",
        help="Default points to clean_building/; the legacy root path "
             "results/task_a_results.csv is a historical merge-polluted file.",
    )
    ap.add_argument("--drop_city", type=str, default="nyc")
    ap.add_argument(
        "--out_path",
        type=str,
        default="results/clean_building/task_a_drop_city_summary.csv",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.in_path)
    if df.empty:
        raise ValueError(f"Input file is empty: {args.in_path}")

    keep = f"r2_{args.drop_city}"
    if keep in df.columns:
        df = df[df[keep].notna()].copy()

    rows = [summarize_row(row, args.drop_city) for _, row in df.iterrows()]
    out = pd.DataFrame(rows)
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_path, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Saved drop-city summary to: {args.out_path}")
    print("Note: this is based on saved per-city metrics only; it does not recompute sample-level overall R².")


if __name__ == "__main__":
    main()
