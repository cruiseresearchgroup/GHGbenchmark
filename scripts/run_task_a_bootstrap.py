"""
Task A bootstrap CI — for each tuned (feature_set, model) headline row,
refit the tuned best_cfg on train+val, predict on test once, then
bootstrap-resample the test rows 1000× with replacement to get
95% CIs for R², MAE, log_mae.

Bootstrap is over *test rows* (not seeds): this quantifies sampling
noise in the test set, which is what a reader wants when asking
"is Δ=0.02 R² between two models meaningful?". Seed-variance is a
separate question answered by the 3-seed clean run.

Inputs:
  results/clean_building/task_a_hpsearch_summary.csv  — tuned best_cfg

Outputs:
  results/clean_building/task_a_bootstrap_ci.csv      — one row per (fs, model)

Usage:
  python scripts/run_task_a_bootstrap.py --n_boot 1000
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
OUT_CSV = OUT_DIR / "task_a_bootstrap_ci.csv"

BUILDERS = {
    "LightGBM":     lambda **kw: LightGBMBaseline(n_jobs=8, early_stopping_rounds=30, **kw),
    "XGBoost":      lambda **kw: XGBoostBaseline(n_jobs=8, early_stopping_rounds=30, **kw),
    "RandomForest": lambda **kw: RandomForestBaseline(n_jobs=8, **kw),
}


def prepare_split(df, fs_name, seed=42):
    splits = grouped_split(df, test_size=0.2, val_size=0.1, seed=seed)
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
    X_tr, y_tr, feat_names, encoders, medians, _ = prepare_features(
        train_df, fs_name, encoders=None
    )
    X_va, y_va, *_ = prepare_features(val_df,  fs_name, encoders=encoders, medians=medians)
    X_te, y_te, *_ = prepare_features(test_df, fs_name, encoders=encoders, medians=medians)
    return X_tr, y_tr, X_va, y_va, X_te, y_te


def refit_and_predict(model_name, cfg, X_tr, y_tr, X_va, y_va, X_te, y_te, clip_max):
    m = BUILDERS[model_name](**cfg)
    X_tv = np.concatenate([X_tr, X_va], axis=0)
    y_tv = np.concatenate([y_tr, y_va], axis=0)
    if model_name in ("LightGBM", "XGBoost"):
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(y_tv))
        cut = int(0.92 * len(idx))
        tr_idx, es_idx = idx[:cut], idx[cut:]
        m.fit(X_tv[tr_idx], y_tv[tr_idx], X_val=X_tv[es_idx], y_val=y_tv[es_idx])
    else:
        m.fit(X_tv, y_tv)
    return np.clip(m.predict(X_te), 0, clip_max)


def bootstrap_ci(y_true, y_pred, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    r2s, maes, logmaes = np.empty(n_boot), np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        r2s[b]     = r2(yt, yp)
        maes[b]    = mae(yt, yp)
        logmaes[b] = log_mae(yt, yp)
    def q(a):
        return float(np.mean(a)), float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))
    return {"r2": q(r2s), "mae": q(maes), "log_mae": q(logmaes)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    summary = pd.read_csv(SUMMARY_CSV)
    print(f"[bootstrap] {len(summary)} (fs, model) headline rows | n_boot={args.n_boot}")

    rows = []
    cached = {}  # fs_name -> (X_tr, y_tr, X_va, y_va, X_te, y_te, clip_max, df)

    for i, r in summary.iterrows():
        fs_name, model_name = r["feature_set"], r["model"]
        best_cfg = json.loads(r["best_cfg"])

        if fs_name not in cached:
            needs_ext = bool(set(get_feature_set(fs_name)) & CLIMATE_FEATURE_COLS)
            df = load_and_prepare(fs_name, join_external=needs_ext)
            df = df[df[TARGET].notna()].copy()
            X_tr, y_tr, X_va, y_va, X_te, y_te = prepare_split(df, fs_name, seed=args.seed)
            clip_max = float(np.max(y_tr) * 2)
            cached[fs_name] = (X_tr, y_tr, X_va, y_va, X_te, y_te, clip_max)
            print(f"[{fs_name}] loaded | train={len(X_tr):,} val={len(X_va):,} test={len(X_te):,}")
        X_tr, y_tr, X_va, y_va, X_te, y_te, clip_max = cached[fs_name]

        print(f"  [{model_name}] refit + bootstrap ...", flush=True)
        y_pred = refit_and_predict(
            model_name, best_cfg, X_tr, y_tr, X_va, y_va, X_te, y_te, clip_max
        )
        ci = bootstrap_ci(np.asarray(y_te, dtype=float), np.asarray(y_pred, dtype=float),
                          n_boot=args.n_boot, seed=args.seed + i)

        row = {
            "feature_set": fs_name,
            "model": model_name,
            "n_test": int(len(y_te)),
            "test_r2_point":      r["test_r2_tuned"],
            "test_mae_point":     r["test_mae_tuned"],
            "test_log_mae_point": r["test_log_mae_tuned"],
            "r2_mean": ci["r2"][0],       "r2_lo": ci["r2"][1],       "r2_hi": ci["r2"][2],
            "mae_mean": ci["mae"][0],     "mae_lo": ci["mae"][1],     "mae_hi": ci["mae"][2],
            "log_mae_mean": ci["log_mae"][0], "log_mae_lo": ci["log_mae"][1], "log_mae_hi": ci["log_mae"][2],
        }
        print(f"    R²={row['r2_mean']:+.3f} [{row['r2_lo']:+.3f}, {row['r2_hi']:+.3f}]"
              f" | MAE={row['mae_mean']:.1f} [{row['mae_lo']:.1f}, {row['mae_hi']:.1f}]"
              f" | log_mae={row['log_mae_mean']:.3f} [{row['log_mae_lo']:.3f}, {row['log_mae_hi']:.3f}]",
              flush=True)
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    print(f"\nDone → {OUT_CSV}")


if __name__ == "__main__":
    main()
