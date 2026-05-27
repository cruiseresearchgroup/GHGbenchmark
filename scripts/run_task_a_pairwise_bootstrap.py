"""
Task A paired bootstrap — for each (feature_set, model_pair), refit both
tuned models on the SAME grouped split, get per-row test predictions, then
bootstrap-resample the test rows 1000× to compute paired ΔR² / ΔMAE /
Δlog_mae with 95% CI and two-sided p-value.

Why paired: single-model bootstrap CI (already in task_a_bootstrap_ci.csv)
answers "how noisy is model A's point estimate"; paired CI answers "is
model A meaningfully better than model B on these specific test rows",
which is what a reviewer asks when they see "RF 0.437 > LGBM 0.421".

Inputs:
  results/clean_building/task_a_hpsearch_summary.csv

Outputs:
  results/clean_building/task_a_pairwise_bootstrap.csv
    columns: feature_set, model_a, model_b, n_test,
             delta_{r2,mae,log_mae}_{mean,lo,hi},
             p_{r2,mae,log_mae}_two_sided
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
DEFAULT_OUT_CSV = OUT_DIR / "task_a_pairwise_bootstrap.csv"

BUILDERS = {
    "LightGBM":     lambda **kw: LightGBMBaseline(n_jobs=8, early_stopping_rounds=30, **kw),
    "XGBoost":      lambda **kw: XGBoostBaseline(n_jobs=8, early_stopping_rounds=30, **kw),
    "RandomForest": lambda **kw: RandomForestBaseline(n_jobs=8, **kw),
}


def prepare_split(df, fs_name, seed=42):
    splits = grouped_split(df, test_size=0.2, val_size=0.1, seed=seed)
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
    X_tr, y_tr, *_ , encoders, medians, _ = prepare_features(
        train_df, fs_name, encoders=None
    )
    X_va, y_va, *_ = prepare_features(val_df,  fs_name, encoders=encoders, medians=medians)
    X_te, y_te, *_ = prepare_features(test_df, fs_name, encoders=encoders, medians=medians)
    return X_tr, y_tr, X_va, y_va, X_te, y_te


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


def paired_bootstrap(y_true, y_a, y_b, n_boot=1000, seed=0):
    """Return dict of paired Δ statistics on R² / MAE / log_mae.
    Lower-is-better metrics (mae, log_mae): Δ = a − b; a_wins_frac = P(Δ<0).
    Higher-is-better (r2): Δ = a − b; a_wins_frac = P(Δ>0).
    """
    n = len(y_true)
    rng = np.random.default_rng(seed)
    dr2 = np.empty(n_boot); dmae = np.empty(n_boot); dlm = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]; ya = y_a[idx]; yb = y_b[idx]
        dr2[b]  = r2(yt, ya)       - r2(yt, yb)
        dmae[b] = mae(yt, ya)      - mae(yt, yb)
        dlm[b]  = log_mae(yt, ya)  - log_mae(yt, yb)

    def q(a, higher_better: bool):
        lo, hi = np.quantile(a, [0.025, 0.975])
        p = 2.0 * min((a <= 0).mean(), (a >= 0).mean())
        win = (a > 0).mean() if higher_better else (a < 0).mean()
        return float(a.mean()), float(lo), float(hi), float(p), float(win)

    r2_stats  = q(dr2,  True)
    mae_stats = q(dmae, False)
    lm_stats  = q(dlm,  False)
    return {
        "n_test": n,
        "delta_r2_mean": r2_stats[0],  "delta_r2_lo": r2_stats[1],  "delta_r2_hi": r2_stats[2],
        "p_r2_two_sided": r2_stats[3], "a_wins_r2_frac": r2_stats[4],
        "delta_mae_mean": mae_stats[0], "delta_mae_lo": mae_stats[1], "delta_mae_hi": mae_stats[2],
        "p_mae_two_sided": mae_stats[3], "a_wins_mae_frac": mae_stats[4],
        "delta_log_mae_mean": lm_stats[0], "delta_log_mae_lo": lm_stats[1], "delta_log_mae_hi": lm_stats[2],
        "p_log_mae_two_sided": lm_stats[3], "a_wins_log_mae_frac": lm_stats[4],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--feature_sets", type=str, default="")  # default: all in summary
    p.add_argument("--out_path", type=str, default=str(DEFAULT_OUT_CSV),
                   help="Output CSV (default: task_a_pairwise_bootstrap.csv).")
    args = p.parse_args()
    out_csv = Path(args.out_path)

    summary = pd.read_csv(SUMMARY_CSV)
    if args.feature_sets:
        wanted = set(args.feature_sets.split(","))
        summary = summary[summary["feature_set"].isin(wanted)].reset_index(drop=True)

    rows = []
    fs_groups = summary.groupby("feature_set")
    for fs_name, g in fs_groups:
        g = g.reset_index(drop=True)
        if len(g) < 2:
            continue

        needs_ext = bool(set(get_feature_set(fs_name)) & CLIMATE_FEATURE_COLS)
        df = load_and_prepare(fs_name, join_external=needs_ext)
        df = df[df[TARGET].notna()].copy()
        X_tr, y_tr, X_va, y_va, X_te, y_te = prepare_split(df, fs_name, seed=args.seed)
        clip_max = float(np.max(y_tr) * 2)
        y_te_arr = np.asarray(y_te, dtype=float)
        print(f"\n=== {fs_name} | test={len(y_te_arr):,} ===", flush=True)

        # Refit each tuned model once and cache predictions
        preds = {}
        for _, r in g.iterrows():
            model_name = r["model"]
            cfg = json.loads(r["best_cfg"])
            print(f"  [refit] {model_name} ...", flush=True)
            preds[model_name] = np.asarray(
                refit_and_predict(model_name, cfg, X_tr, y_tr, X_va, y_va, X_te, clip_max),
                dtype=float,
            )

        # Pairwise
        models = list(preds.keys())
        for i, a in enumerate(models):
            for b in models[i + 1:]:
                stats = paired_bootstrap(y_te_arr, preds[a], preds[b],
                                         n_boot=args.n_boot, seed=args.seed)
                sig = "*" if stats["p_r2_two_sided"] < 0.05 else " "
                print(f"  {sig} {a:14s} vs {b:14s} | "
                      f"ΔR²={stats['delta_r2_mean']:+.3f} "
                      f"[{stats['delta_r2_lo']:+.3f}, {stats['delta_r2_hi']:+.3f}]  "
                      f"p={stats['p_r2_two_sided']:.3f}",
                      flush=True)
                rows.append({"feature_set": fs_name, "model_a": a, "model_b": b, **stats})
                pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"\nDone → {out_csv}")


if __name__ == "__main__":
    main()
