"""Organize clean building benchmark result tables into a dedicated directory.

Outputs a clean, audit-friendly result bundle under `results/clean_building/`.

Design goals:
- rebuild Task A from clean sources only
- keep other tasks separated rather than merged into one polluted CSV
- distinguish stable tables from provisional / single-seed multimodal tables
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

TASK_A_FEATURE_SETS = [
    "core_all_cities",
    "core_all_cities_climate_plus",
    "us_core",
    "us_metadata",
    "us_leaky_eui",
    "us_leaky_full",
    "au_core",
    "au_eui",
    "au_full",
]


def aggregate_with_seeds(df: pd.DataFrame, group_cols: list[str], exclude_metric_cols: set[str]) -> pd.DataFrame:
    metric_cols = []
    for col in df.columns:
        if col in group_cols or col == "seed":
            continue
        if col in exclude_metric_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            metric_cols.append(col)

    rows = []
    grouped = df.groupby(group_cols, dropna=False)
    for keys, sub in grouped:
        row = dict(zip(group_cols, keys))
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
    if len(out):
        out = out.sort_values(group_cols).reset_index(drop=True)
    return out


def build_task_a_clean(out_dir: Path) -> list[str]:
    notes = []
    raw_frames = []
    missing = []
    for fs in TASK_A_FEATURE_SETS:
        path = RESULTS / f"task_a_results_{fs}.csv"
        if path.exists():
            raw_frames.append(pd.read_csv(path))
        else:
            missing.append(path.name)
    mlp_path = RESULTS / "task_a_results_mlp_3seeds.csv"
    if mlp_path.exists():
        raw_frames.append(pd.read_csv(mlp_path))
    else:
        missing.append(mlp_path.name)

    if missing:
        notes.append("Missing Task A source files: " + ", ".join(missing))

    raw = pd.concat(raw_frames, ignore_index=True)
    raw_out = out_dir / "task_a_clean_3seeds_raw.csv"
    raw.to_csv(raw_out, index=False)

    summary = aggregate_with_seeds(
        raw,
        group_cols=["task", "feature_set", "model", "split_type"],
        exclude_metric_cols={"overall_n_train", "overall_n_test"},
    )
    summary_out = out_dir / "task_a_clean_3seeds_summary.csv"
    summary.to_csv(summary_out, index=False)

    headline = summary[
        summary["feature_set"].isin(["core_all_cities", "core_all_cities_climate_plus"])
    ].copy()
    headline_cols = [
        "task", "feature_set", "model", "split_type", "n_seeds", "seeds",
        "overall_r2_mean", "overall_r2_std",
        "macro_r2_mean", "macro_r2_std",
        "overall_mae_mean", "overall_mae_std",
        "macro_mae_mean", "macro_mae_std",
    ]
    headline = headline[[c for c in headline_cols if c in headline.columns]]
    headline.to_csv(out_dir / "task_a_clean_headline_table.csv", index=False)

    notes.append(f"Task A clean raw rows: {len(raw)}")
    notes.append(f"Task A clean aggregated rows: {len(summary)}")
    return notes


def copy_if_exists(src_name: str, dst: Path, notes: list[str]):
    src = RESULTS / src_name
    if src.exists():
        shutil.copy2(src, dst / src_name)
        notes.append(f"Copied {src_name}")
    else:
        notes.append(f"Missing {src_name}")


def write_index(out_dir: Path, notes: list[str]):
    content = f"""# Clean Building Results Bundle

This directory contains the cleaned, separated building benchmark result tables.

## Stable / main-task files

- `task_a_clean_3seeds_raw.csv`
- `task_a_clean_3seeds_summary.csv`
- `task_a_clean_headline_table.csv`
- `task_b_results_core.csv`
- `task_c_results_core.csv`
- `task_d_results_core.csv`
- `task_e_results_core.csv`

## Multimodal files

- `task_a_results_s2_all.csv`
- `task_a_s2_summary_grouped.csv`
- `task_a_s2_summary_random.csv`
- `task_c_results_s2_all.csv`
- `task_c_s2_summary_by_model.csv`
- `task_c_s2_headline_candidates.csv`

Important:
- The copied S2 tables are the current organized summaries, but they are not all multi-seed yet.
- `task_a_clean_3seeds_summary.csv` is the clean Task A seed-aggregated table and should be preferred over the polluted top-level `results/task_a_results.csv`.

## Notes

{chr(10).join('- ' + n for n in notes)}
"""
    (out_dir / "README.md").write_text(content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="results/clean_building")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    notes.extend(build_task_a_clean(out_dir))

    for name in [
        "task_b_results_core.csv",
        "task_c_results_core.csv",
        "task_d_results_core.csv",
        "task_e_results_core.csv",
        "task_a_results_s2_all.csv",
        "task_a_s2_summary_grouped.csv",
        "task_a_s2_summary_random.csv",
        "task_c_results_s2_all.csv",
        "task_c_s2_summary_by_model.csv",
        "task_c_s2_headline_candidates.csv",
        "building_final_summary_zh.md",
    ]:
        copy_if_exists(name, out_dir, notes)

    write_index(out_dir, notes)
    print(f"Organized clean building bundle at: {out_dir}")


if __name__ == "__main__":
    main()
