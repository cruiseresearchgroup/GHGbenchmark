"""
Task A HP search — val-based random search for RF / XGBoost / LightGBM on
A2 pooled, grouped split (the setting we report as headline in §Task A).

Design (matches scripts/run_t1_a_ext.py::run_hpsearch):
  * One HP search per (feature_set, model)
  * 15 random trials per (feature_set, model); grouped split, seed=42
  * Selection metric: val log_mae (minimize) — robust because training is in log1p space
  * For each trial, log val + test metrics so we can inspect overfit-to-val
  * Best-by-val config is refit on train+val → reported as *_tuned test metrics

Outputs (all under results/clean_building/):
  task_a_hpsearch_trials.csv        — every trial (feature_set, model, trial, cfg, val/test metrics)
  task_a_hpsearch_summary.csv       — one row per (feature_set, model): best cfg + final test metrics

Usage:
  python scripts/run_task_a_hpsearch.py \
      --feature_sets core_all_cities,core_all_cities_climate_plus \
      --n_trials 15
"""
from __future__ import annotations
import sys, os, argparse, json, time, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.data.preprocessing import load_and_prepare, prepare_features
from src.data.feature_sets import TARGET, get_feature_set
from src.data.splitters import grouped_split
from src.evaluation.metrics import compute_all_metrics
from src.models.tree import RandomForestBaseline, XGBoostBaseline, LightGBMBaseline

warnings.filterwarnings("ignore")

CLIMATE_FEATURE_COLS = {
    "hdd", "cdd",
    "annual_mean_temp_c", "annual_rh_mean",
    "annual_ssrd_mj_m2_day", "annual_wind_ms",
}

OUT_DIR = Path("results/clean_building")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRIALS_CSV = OUT_DIR / "task_a_hpsearch_trials.csv"
SUMMARY_CSV = OUT_DIR / "task_a_hpsearch_summary.csv"


# ─────────────────────────────────────────────────────────────────────────
# HP grids
# ─────────────────────────────────────────────────────────────────────────
def sample_lgbm(rng):
    return dict(
        n_estimators=800,  # upper bound; early_stopping will cut it
        num_leaves=int(rng.choice([31, 63, 127, 255])),
        learning_rate=float(rng.choice([0.02, 0.05, 0.08, 0.1])),
        min_child_samples=int(rng.choice([5, 10, 20, 50])),
        reg_alpha=float(rng.choice([0.0, 0.1, 1.0])),
        reg_lambda=float(rng.choice([0.0, 0.1, 1.0, 5.0])),
        colsample_bytree=float(rng.choice([0.7, 0.8, 0.9, 1.0])),
        subsample=float(rng.choice([0.7, 0.8, 0.9, 1.0])),
        subsample_freq=1,
    )


def sample_xgb(rng):
    return dict(
        n_estimators=800,
        max_depth=int(rng.choice([4, 6, 8, 10])),
        learning_rate=float(rng.choice([0.02, 0.05, 0.08, 0.1])),
        min_child_weight=float(rng.choice([1, 3, 5, 10])),
        subsample=float(rng.choice([0.7, 0.8, 0.9, 1.0])),
        colsample_bytree=float(rng.choice([0.7, 0.8, 0.9, 1.0])),
        reg_alpha=float(rng.choice([0.0, 0.1, 1.0])),
        reg_lambda=float(rng.choice([0.1, 1.0, 5.0])),
    )


def sample_rf(rng):
    return dict(
        n_estimators=int(rng.choice([200, 300, 500])),
        max_depth=int(rng.choice([10, 15, 20, 25])),
        min_samples_leaf=int(rng.choice([1, 2, 5, 10])),
        max_features=rng.choice([0.5, 0.7, 1.0]),
    )


SAMPLERS = {"LightGBM": sample_lgbm, "XGBoost": sample_xgb, "RandomForest": sample_rf}
BUILDERS = {
    "LightGBM":     lambda **kw: LightGBMBaseline(n_jobs=8, early_stopping_rounds=30, **kw),
    "XGBoost":      lambda **kw: XGBoostBaseline(n_jobs=8, early_stopping_rounds=30, **kw),
    "RandomForest": lambda **kw: RandomForestBaseline(n_jobs=8, **kw),
}


# ─────────────────────────────────────────────────────────────────────────
def prepare_split(df, fs_name, seed=42):
    splits = grouped_split(df, test_size=0.2, val_size=0.1, seed=seed)
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
    X_tr, y_tr, feat_names, encoders, medians, _ = prepare_features(
        train_df, fs_name, encoders=None
    )
    X_va, y_va, *_ = prepare_features(val_df,  fs_name, encoders=encoders, medians=medians)
    X_te, y_te, *_ = prepare_features(test_df, fs_name, encoders=encoders, medians=medians)
    return X_tr, y_tr, X_va, y_va, X_te, y_te


def eval_pair(model, X, y, clip_max):
    y_pred = np.clip(model.predict(X), 0, clip_max)
    return compute_all_metrics(y, y_pred)


def run_one_trial(model_name, cfg, X_tr, y_tr, X_va, y_va, X_te, y_te, clip_max):
    builder = BUILDERS[model_name]
    m = builder(**cfg)
    if model_name in ("LightGBM", "XGBoost"):
        m.fit(X_tr, y_tr, X_val=X_va, y_val=y_va)
    else:
        m.fit(X_tr, y_tr)
    val  = eval_pair(m, X_va, y_va, clip_max)
    test = eval_pair(m, X_te, y_te, clip_max)
    return m, val, test


def refit_on_trainval(model_name, cfg, X_tr, y_tr, X_va, y_va, X_te, y_te, clip_max):
    """Refit best-by-val config on train+val and evaluate on test."""
    builder = BUILDERS[model_name]
    m = builder(**cfg)
    X_tv = np.concatenate([X_tr, X_va], axis=0)
    y_tv = np.concatenate([y_tr, y_va], axis=0)
    if model_name in ("LightGBM", "XGBoost"):
        # For refit, we no longer have a val split; use a tiny holdout off train+val
        # to keep early stopping meaningful.
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(y_tv))
        cut = int(0.92 * len(idx))
        tr_idx, es_idx = idx[:cut], idx[cut:]
        m.fit(X_tv[tr_idx], y_tv[tr_idx], X_val=X_tv[es_idx], y_val=y_tv[es_idx])
    else:
        m.fit(X_tv, y_tv)
    test = eval_pair(m, X_te, y_te, clip_max)
    return test


# ─────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feature_sets", type=str,
                   default="core_all_cities,core_all_cities_climate_plus,"
                           "us_core,us_metadata,us_leaky_eui,us_leaky_full")
    p.add_argument("--models", type=str, default="LightGBM,XGBoost,RandomForest")
    p.add_argument("--n_trials", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    feature_sets = args.feature_sets.split(",")
    models = args.models.split(",")
    trial_rows, summary_rows = [], []
    rng = np.random.default_rng(args.seed)

    # Resume support: if partial CSV exists, skip (fs, model) pairs already done
    done_pairs = set()
    if SUMMARY_CSV.exists():
        prev = pd.read_csv(SUMMARY_CSV)
        done_pairs = set(zip(prev["feature_set"], prev["model"]))
        summary_rows = prev.to_dict("records")
        if TRIALS_CSV.exists():
            trial_rows = pd.read_csv(TRIALS_CSV).to_dict("records")
        print(f"[resume] skipping {len(done_pairs)} already-done (fs, model) pairs")

    for fs_name in feature_sets:
        needs_external = bool(set(get_feature_set(fs_name)) & CLIMATE_FEATURE_COLS)
        print(f"\n{'='*70}\nFeature set: {fs_name}  (climate_join={needs_external})\n{'='*70}")
        df = load_and_prepare(fs_name, join_external=needs_external)
        df = df[df[TARGET].notna()].copy()
        print(f"  rows: {len(df):,}  |  cities: {df['city'].nunique()}")

        X_tr, y_tr, X_va, y_va, X_te, y_te = prepare_split(df, fs_name, seed=args.seed)
        clip_max = float(np.max(y_tr) * 2)
        print(f"  train={len(X_tr):,} | val={len(X_va):,} | test={len(X_te):,}")

        for model_name in models:
            if (fs_name, model_name) in done_pairs:
                print(f"  ({fs_name}, {model_name}) already done — skip")
                continue
            print(f"\n  [{model_name}] sampling {args.n_trials} configs")
            sampler = SAMPLERS[model_name]
            best_val_logmae, best_cfg, best_trial_test = None, None, None
            for t in range(args.n_trials):
                cfg = sampler(rng)
                t0 = time.time()
                try:
                    _, val, test = run_one_trial(
                        model_name, cfg, X_tr, y_tr, X_va, y_va, X_te, y_te, clip_max
                    )
                    row = {
                        "feature_set": fs_name, "model": model_name, "trial": t,
                        "elapsed_s": round(time.time() - t0, 1),
                        "val_r2": val["r2"], "val_mae": val["mae"], "val_log_mae": val["log_mae"],
                        "test_r2": test["r2"], "test_mae": test["mae"], "test_log_mae": test["log_mae"],
                        **{f"cfg_{k}": v for k, v in cfg.items()},
                    }
                    trial_rows.append(row)
                    pd.DataFrame(trial_rows).to_csv(TRIALS_CSV, index=False)
                    print(f"    trial {t:2d} | val R²={val['r2']:+.3f} log_mae={val['log_mae']:.3f}"
                          f" | test R²={test['r2']:+.3f} log_mae={test['log_mae']:.3f}"
                          f" | {row['elapsed_s']:.0f}s")
                    if best_val_logmae is None or val["log_mae"] < best_val_logmae:
                        best_val_logmae, best_cfg, best_trial_test = val["log_mae"], cfg, test
                except Exception as e:
                    print(f"    trial {t:2d} FAILED: {type(e).__name__}: {e}")
                    continue

            if best_cfg is None:
                print(f"    [{model_name}] all trials failed — skipping")
                continue

            # Refit best on train+val
            print(f"  [{model_name}] best val log_mae={best_val_logmae:.3f} "
                  f"→ refit on train+val")
            t0 = time.time()
            try:
                final_test = refit_on_trainval(
                    model_name, best_cfg, X_tr, y_tr, X_va, y_va, X_te, y_te, clip_max
                )
                print(f"    final test R²={final_test['r2']:+.3f} "
                      f"MAE={final_test['mae']:.1f} log_mae={final_test['log_mae']:.3f} "
                      f"({time.time() - t0:.0f}s)")
            except Exception as e:
                print(f"    refit FAILED: {e}  — using best-trial test instead")
                final_test = best_trial_test

            summary_rows.append({
                "feature_set": fs_name, "model": model_name,
                "n_trials": args.n_trials,
                "best_val_log_mae": best_val_logmae,
                "test_r2_tuned": final_test["r2"],
                "test_mae_tuned": final_test["mae"],
                "test_log_mae_tuned": final_test["log_mae"],
                "best_cfg": json.dumps(best_cfg),
            })
            pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)

    # final print
    print("\n" + "=" * 70)
    print("HP search complete. Summary:")
    print("=" * 70)
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        cols = ["feature_set", "model", "test_r2_tuned", "test_mae_tuned", "test_log_mae_tuned"]
        print(summary_df[cols].to_string(index=False,
              float_format=lambda x: f"{x:.3f}"))
    print(f"\nTrials  → {TRIALS_CSV}")
    print(f"Summary → {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
