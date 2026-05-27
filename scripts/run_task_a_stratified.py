"""
Task A stratified R² — break headline tuned test performance by:
  * city
  * property_type (where available; AU rows lack it → labeled 'unknown')
  * HDD zone (4 buckets: cold / cool / mixed / warm, by test-set HDD quartile)

For each (feature_set, model) tuned best_cfg: refit on train+val, predict test,
then compute R² and sample sizes inside each stratum.

Why this matters:
  Reviewers at NeurIPS D&B will ask "where does the benchmark fail?" —
  point-R²=0.42 on 26 cities is useful, but a per-stratum view reveals
  whether the model is uniformly mediocre (model-limited) or driven by
  a few outlier cities/types (dataset-heterogeneity-limited).

Output:
  results/clean_building/task_a_stratified_r2.csv
    columns: feature_set, model, stratum_type, stratum, n, r2, mae, log_mae
"""
from __future__ import annotations
import sys, json, argparse, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.data.preprocessing import load_and_prepare, prepare_features
from src.data.feature_sets import TARGET, get_feature_set
from src.data.splitters import grouped_split
from src.evaluation.metrics import r2, mae, log_mae
from src.models.tree import RandomForestBaseline, XGBoostBaseline, LightGBMBaseline

warnings.filterwarnings("ignore")

CLIMATE_FEATURE_COLS = {
    "hdd", "cdd",
    "annual_mean_temp_c", "annual_rh_mean",
    "annual_ssrd_mj_m2_day", "annual_wind_ms",
}

OUT_DIR = Path("results/clean_building")
SUMMARY_CSV = OUT_DIR / "task_a_hpsearch_summary.csv"
DEFAULT_OUT_CSV = OUT_DIR / "task_a_stratified_r2.csv"

BUILDERS = {
    "LightGBM":     lambda **kw: LightGBMBaseline(n_jobs=8, early_stopping_rounds=30, **kw),
    "XGBoost":      lambda **kw: XGBoostBaseline(n_jobs=8, early_stopping_rounds=30, **kw),
    "RandomForest": lambda **kw: RandomForestBaseline(n_jobs=8, **kw),
}

MIN_STRATUM_N = 50  # skip strata with < 50 test rows


def split_with_test_df(df, fs_name, seed=42):
    """Same as hpsearch's prepare_split, but also returns the raw test_df so we
    can stratify on its non-feature columns (city, property_type, hdd)."""
    splits = grouped_split(df, test_size=0.2, val_size=0.1, seed=seed)
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
    X_tr, y_tr, *_ , encoders, medians, _ = prepare_features(
        train_df, fs_name, encoders=None
    )
    X_va, y_va, *_ = prepare_features(val_df,  fs_name, encoders=encoders, medians=medians)
    X_te, y_te, *_ = prepare_features(test_df, fs_name, encoders=encoders, medians=medians)
    # prepare_features drops NaN-target rows; re-apply the same filter to test_df
    # so row-order matches X_te / y_te.
    test_df_valid = test_df[test_df[TARGET].notna()].reset_index(drop=True)
    assert len(test_df_valid) == len(y_te), f"{len(test_df_valid)} vs {len(y_te)}"
    return X_tr, y_tr, X_va, y_va, X_te, y_te, test_df_valid


def refit_and_predict(model_name, cfg, X_tr, y_tr, X_va, y_va, X_te, clip_max):
    m = BUILDERS[model_name](**cfg)
    X_tv = np.concatenate([X_tr, X_va], axis=0)
    y_tv = np.concatenate([y_tr, y_va], axis=0)
    if model_name in ("LightGBM", "XGBoost"):
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(y_tv))
        cut = int(0.92 * len(idx))
        m.fit(X_tv[idx[:cut]], y_tv[idx[:cut]],
              X_val=X_tv[idx[cut:]], y_val=y_tv[idx[cut:]])
    else:
        m.fit(X_tv, y_tv)
    return np.clip(m.predict(X_te), 0, clip_max)


def metrics_for(y_true, y_pred):
    return {
        "n": int(len(y_true)),
        "r2": r2(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "log_mae": log_mae(y_true, y_pred),
    }


def stratify_rows(test_df, y_true, y_pred):
    """Emit rows: one per (stratum_type, stratum)."""
    out = []
    # Per-city
    for city in sorted(test_df["city"].dropna().unique()):
        mask = (test_df["city"] == city).values
        if mask.sum() < MIN_STRATUM_N:
            continue
        out.append({"stratum_type": "city", "stratum": city,
                    **metrics_for(y_true[mask], y_pred[mask])})

    # Per-property-type (if column exists and has non-null)
    if "property_type" in test_df.columns:
        pt = test_df["property_type"].fillna("unknown").astype(str).str.strip()
        for p in sorted(pt.unique()):
            mask = (pt == p).values
            if mask.sum() < MIN_STRATUM_N:
                continue
            out.append({"stratum_type": "property_type", "stratum": p,
                        **metrics_for(y_true[mask], y_pred[mask])})

    # Per-HDD zone (quartile buckets on test set)
    if "hdd" in test_df.columns and test_df["hdd"].notna().sum() >= 4 * MIN_STRATUM_N:
        hdd_vals = test_df["hdd"].values
        finite = np.isfinite(hdd_vals)
        qs = np.quantile(hdd_vals[finite], [0.25, 0.5, 0.75])
        labels = ["hdd_q1_warm", "hdd_q2_mixed", "hdd_q3_cool", "hdd_q4_cold"]
        bin_idx = np.digitize(hdd_vals, qs)  # 0..3
        for i, lab in enumerate(labels):
            mask = (bin_idx == i) & finite
            if mask.sum() < MIN_STRATUM_N:
                continue
            hdd_lo = hdd_vals[mask].min()
            hdd_hi = hdd_vals[mask].max()
            out.append({"stratum_type": "hdd_zone",
                        "stratum": f"{lab} [{hdd_lo:.0f}-{hdd_hi:.0f}]",
                        **metrics_for(y_true[mask], y_pred[mask])})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_path", type=str, default=str(DEFAULT_OUT_CSV))
    p.add_argument("--feature_sets", type=str,
                   default="core_all_cities,us_core,us_metadata,us_leaky_full")
    args = p.parse_args()

    summary = pd.read_csv(SUMMARY_CSV)
    rows = []
    cached = {}

    for i, r in summary.iterrows():
        fs_name, model_name = r["feature_set"], r["model"]
        if fs_name not in args.feature_sets.split(","):
            continue
        best_cfg = json.loads(r["best_cfg"])

        if fs_name not in cached:
            needs_ext = bool(set(get_feature_set(fs_name)) & CLIMATE_FEATURE_COLS)
            df = load_and_prepare(fs_name, join_external=needs_ext)
            df = df[df[TARGET].notna()].copy()
            X_tr, y_tr, X_va, y_va, X_te, y_te, test_df = split_with_test_df(
                df, fs_name, seed=args.seed
            )
            clip_max = float(np.max(y_tr) * 2)
            cached[fs_name] = (X_tr, y_tr, X_va, y_va, X_te, y_te, test_df, clip_max)
            print(f"[{fs_name}] loaded | test={len(X_te):,}")
        X_tr, y_tr, X_va, y_va, X_te, y_te, test_df, clip_max = cached[fs_name]

        print(f"  [{model_name}] refit + stratify ...", flush=True)
        y_pred = refit_and_predict(model_name, best_cfg, X_tr, y_tr, X_va, y_va, X_te, clip_max)
        y_te_arr = np.asarray(y_te, dtype=float)
        y_pr_arr = np.asarray(y_pred, dtype=float)

        for s in stratify_rows(test_df, y_te_arr, y_pr_arr):
            rows.append({"feature_set": fs_name, "model": model_name, **s})

        pd.DataFrame(rows).to_csv(args.out_path, index=False)

    print(f"\nDone → {args.out_path}")


if __name__ == "__main__":
    main()
