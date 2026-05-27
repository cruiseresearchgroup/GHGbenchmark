"""
T1-A HP search extension — val-based random search for RandomForest and
HistGradientBoosting on the strict T1 split, matching the protocol used by
run_t1_a_ext.py::run_hpsearch for LightGBM and XGBoost.

Rationale: Codex review flagged LGBM/XGB as the only tuned T1 baselines.
Adding RF and HGB (both scikit-learn, no external deps) rounds out the
classical ensemble family. CatBoost is not installed; skipped.

Selection: val R²_log (maximize). Best config is refit on train+val.

Outputs:
  results/t1a_ext_hpsearch_ext.csv           — every trial
  results/t1a_ext_hpsearch_summary_ext.csv   — one row per (model): best cfg + test metrics

Usage:
  python scripts/run_t1_a_hpsearch_ext.py --n_trials 40
"""
from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).parent))
from t1_common import load_scope12, make_features, metrics, align, SEED

OUT_DIR = Path("results"); OUT_DIR.mkdir(exist_ok=True)
TRIALS_CSV = OUT_DIR / "t1a_ext_hpsearch_ext.csv"
SUMMARY_CSV = OUT_DIR / "t1a_ext_hpsearch_summary_ext.csv"


def sample_rf(rng):
    return dict(
        n_estimators=int(rng.choice([200, 400, 600, 800])),
        max_depth=int(rng.choice([8, 12, 16, 20, 25])),
        min_samples_leaf=int(rng.choice([1, 2, 5, 10])),
        max_features=float(rng.choice([0.5, 0.7, 1.0])),
    )


def sample_hgb(rng):
    return dict(
        max_iter=int(rng.choice([300, 500, 800, 1200])),
        learning_rate=float(rng.choice([0.02, 0.05, 0.08, 0.1])),
        max_depth=int(rng.choice([4, 6, 8, 10])),
        max_leaf_nodes=int(rng.choice([15, 31, 63, 127])),
        min_samples_leaf=int(rng.choice([10, 20, 50])),
        l2_regularization=float(rng.choice([0.0, 0.1, 1.0, 10.0])),
    )


def build_rf(cfg):
    return RandomForestRegressor(random_state=SEED, n_jobs=8, **cfg)


def build_hgb(cfg):
    return HistGradientBoostingRegressor(random_state=SEED, **cfg)


MODELS = {
    "random_forest": (sample_rf, build_rf),
    "hist_gbm":      (sample_hgb, build_hgb),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_trials", type=int, default=40)
    ap.add_argument("--models", type=str, default="random_forest,hist_gbm")
    args = ap.parse_args()

    df = load_scope12()
    df = df[df["subset_t1strict"]].copy()
    train = df[df["split"] == "train"].reset_index(drop=True)
    val   = df[df["split"] == "val"].reset_index(drop=True)
    test  = df[df["split"] == "test"].reset_index(drop=True)
    X_tr = make_features(train, "structured_strict")
    X_va = make_features(val,   "structured_strict")
    X_te = make_features(test,  "structured_strict")
    X_tr, X_va = align(X_tr, X_va); X_tr, X_te = align(X_tr, X_te)
    X_va = X_va.reindex(columns=X_tr.columns, fill_value=0)
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)
    y_tr = train["y"].values; y_va = val["y"].values; y_te = test["y"].values

    # RF / HGB do not tolerate NaN the way LGBM/XGB do; impute with train median
    # on the feature frames (same treatment would apply to sklearn models in
    # run_t1_a_ext.py::run_dl). We do it column-wise so feature alignment is
    # preserved.
    med = X_tr.median(numeric_only=True)
    X_tr = X_tr.fillna(med); X_va = X_va.fillna(med); X_te = X_te.fillna(med)

    rng = np.random.default_rng(SEED)
    trial_rows, summary_rows = [], []

    for model_name in args.models.split(","):
        if model_name not in MODELS:
            print(f"[skip] unknown model: {model_name}"); continue
        sampler, builder = MODELS[model_name]
        print(f"\n=== {model_name} | {args.n_trials} trials ===")

        best_val, best_cfg = None, None
        for t in range(args.n_trials):
            cfg = sampler(rng)
            t0 = time.time()
            try:
                m = builder(cfg)
                m.fit(X_tr.values, y_tr)
                mv = metrics(y_va, m.predict(X_va.values))
                mt = metrics(y_te, m.predict(X_te.values))
            except Exception as e:
                print(f"  trial {t:2d} FAILED: {type(e).__name__}: {e}")
                continue
            row = {
                "model": model_name, "trial": t, "elapsed_s": round(time.time() - t0, 1),
                "val_r2": mv["r2_log"], "val_mae": mv["mae_log"],
                "test_r2": mt["r2_log"], "test_mae": mt["mae_log"],
                **{f"cfg_{k}": v for k, v in cfg.items()},
            }
            trial_rows.append(row)
            pd.DataFrame(trial_rows).to_csv(TRIALS_CSV, index=False)
            print(f"  trial {t:2d} | val R²={mv['r2_log']:+.3f} | test R²={mt['r2_log']:+.3f} | {row['elapsed_s']:.0f}s",
                  flush=True)
            if best_val is None or mv["r2_log"] > best_val:
                best_val, best_cfg = mv["r2_log"], cfg

        if best_cfg is None:
            print(f"  [{model_name}] all trials failed"); continue

        # Refit on train+val
        print(f"  [{model_name}] best val R²={best_val:+.3f}  →  refit on train+val")
        X_tv = pd.concat([X_tr, X_va]).reset_index(drop=True)
        y_tv = np.concatenate([y_tr, y_va])
        final = builder(best_cfg)
        final.fit(X_tv.values, y_tv)
        final_m = metrics(y_te, final.predict(X_te.values))
        print(f"    final test R²={final_m['r2_log']:+.3f}  MAE={final_m['mae_log']:.3f}")
        summary_rows.append({
            "model": f"{model_name}_tuned",
            "mae_log": final_m["mae_log"],
            "rmse_log": final_m["rmse_log"],
            "r2_log": final_m["r2_log"],
            "pearson_r": final_m["pearson_r"],
            "mape": final_m["mape"],
            "median_ape": final_m["median_ape"],
            "n_train": len(y_tv), "n_test": len(y_te),
            "best_cfg": json.dumps(best_cfg),
        })
        pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)

    print(f"\nTrials  → {TRIALS_CSV}")
    print(f"Summary → {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
