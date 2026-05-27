"""
Task A feature importance — SHAP (TreeExplainer) + permutation importance
on the tuned LightGBM best_cfg for each feature_set.

Why LightGBM only: (a) it's the headline model for Task A, (b) SHAP
TreeExplainer is fast for tree ensembles, (c) matching findings across
3 models would triple cost for little added insight.

Methodology:
  * Refit LGBM(best_cfg) on train+val (same split as hpsearch, seed=42).
  * SHAP: TreeExplainer over a random 5000-row sample of test rows.
          Report mean(|SHAP|) per feature = global importance in log-emissions.
  * Permutation importance: sklearn's permutation_importance on the full
          test set with n_repeats=5, scoring="r2" (in linear tCO2e space,
          i.e. after expm1). This measures how much test R² drops when
          a feature is shuffled, which is the importance a reviewer cares about.

Inputs:
  results/clean_building/task_a_hpsearch_summary.csv

Outputs:
  results/clean_building/task_a_shap_importance.csv      — rows: (fs, feature, mean_abs_shap)
  results/clean_building/task_a_perm_importance.csv      — rows: (fs, feature, r2_drop_mean, r2_drop_std)
"""
from __future__ import annotations
import sys, json, argparse, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.data.preprocessing import load_and_prepare, prepare_features
from src.data.feature_sets import TARGET, get_feature_set
from src.data.splitters import grouped_split
from src.models.tree import LightGBMBaseline

warnings.filterwarnings("ignore")

CLIMATE_FEATURE_COLS = {
    "hdd", "cdd",
    "annual_mean_temp_c", "annual_rh_mean",
    "annual_ssrd_mj_m2_day", "annual_wind_ms",
}

OUT_DIR = Path("results/clean_building")
SUMMARY_CSV = OUT_DIR / "task_a_hpsearch_summary.csv"
SHAP_CSV = OUT_DIR / "task_a_shap_importance.csv"
PERM_CSV = OUT_DIR / "task_a_perm_importance.csv"


def prepare_split(df, fs_name, seed=42):
    splits = grouped_split(df, test_size=0.2, val_size=0.1, seed=seed)
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
    X_tr, y_tr, feat_names, encoders, medians, _ = prepare_features(
        train_df, fs_name, encoders=None
    )
    X_va, y_va, *_ = prepare_features(val_df,  fs_name, encoders=encoders, medians=medians)
    X_te, y_te, *_ = prepare_features(test_df, fs_name, encoders=encoders, medians=medians)
    return X_tr, y_tr, X_va, y_va, X_te, y_te, feat_names


def refit_lgbm(cfg, X_tr, y_tr, X_va, y_va):
    m = LightGBMBaseline(n_jobs=8, early_stopping_rounds=30, **cfg)
    X_tv = np.concatenate([X_tr, X_va], axis=0)
    y_tv = np.concatenate([y_tr, y_va], axis=0)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y_tv))
    cut = int(0.92 * len(idx))
    m.fit(X_tv[idx[:cut]], y_tv[idx[:cut]],
          X_val=X_tv[idx[cut:]], y_val=y_tv[idx[cut:]])
    return m


def compute_shap(model, X_sample, feat_names):
    """SHAP TreeExplainer on LightGBM booster; returns mean(|SHAP|) per feature
    in log-emission units (the model's native output space)."""
    import shap
    booster = model._model.booster_
    expl = shap.TreeExplainer(booster)
    sv = expl.shap_values(X_sample)
    return pd.DataFrame({
        "feature": feat_names,
        "mean_abs_shap": np.mean(np.abs(sv), axis=0),
    })


class _LGBMWrapper:
    """sklearn-compatible wrapper so permutation_importance can score
    r2 in linear (tCO2e) space via our expm1 predict."""
    def __init__(self, m): self.m = m
    def fit(self, X, y): return self
    def predict(self, X): return self.m.predict(X)
    def score(self, X, y):
        from sklearn.metrics import r2_score
        return r2_score(y, self.m.predict(X))
    def get_params(self, deep=True): return {}
    def set_params(self, **kw): return self


def compute_perm(model, X_te, y_te, feat_names, n_repeats=5, seed=42):
    wrapped = _LGBMWrapper(model)
    r = permutation_importance(
        wrapped, X_te, y_te,
        n_repeats=n_repeats, random_state=seed, n_jobs=1,
        scoring=None,  # uses wrapper.score → r2 in tCO2e
    )
    return pd.DataFrame({
        "feature": feat_names,
        "r2_drop_mean": r.importances_mean,
        "r2_drop_std":  r.importances_std,
    })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feature_sets", type=str,
                   default="core_all_cities,us_core,us_metadata,us_leaky_eui,us_leaky_full")
    p.add_argument("--shap_n", type=int, default=5000,
                   help="test rows subsampled for SHAP (0 = all)")
    p.add_argument("--perm_repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    summary = pd.read_csv(SUMMARY_CSV)
    shap_rows, perm_rows = [], []

    for fs_name in args.feature_sets.split(","):
        row = summary[(summary["feature_set"] == fs_name) & (summary["model"] == "LightGBM")]
        if row.empty:
            print(f"[skip] no LightGBM tuned cfg for {fs_name}")
            continue
        best_cfg = json.loads(row.iloc[0]["best_cfg"])

        print(f"\n=== {fs_name} ===")
        needs_ext = bool(set(get_feature_set(fs_name)) & CLIMATE_FEATURE_COLS)
        df = load_and_prepare(fs_name, join_external=needs_ext)
        df = df[df[TARGET].notna()].copy()

        X_tr, y_tr, X_va, y_va, X_te, y_te, feat_names = prepare_split(df, fs_name, seed=args.seed)
        print(f"  train+val={len(X_tr)+len(X_va):,} test={len(X_te):,} features={feat_names}")

        print("  refit LGBM ...", flush=True)
        m = refit_lgbm(best_cfg, X_tr, y_tr, X_va, y_va)

        # SHAP
        n_sub = min(args.shap_n, len(X_te)) if args.shap_n > 0 else len(X_te)
        rng = np.random.default_rng(args.seed)
        idx_sub = rng.choice(len(X_te), size=n_sub, replace=False)
        X_sub = X_te[idx_sub]
        print(f"  SHAP on {n_sub} test rows ...", flush=True)
        shap_df = compute_shap(m, X_sub, feat_names)
        shap_df.insert(0, "feature_set", fs_name)
        shap_rows.append(shap_df)
        print(shap_df.sort_values("mean_abs_shap", ascending=False).to_string(index=False))

        # Permutation importance
        print(f"  permutation importance (n_repeats={args.perm_repeats}) ...", flush=True)
        perm_df = compute_perm(m, X_te, y_te, feat_names,
                               n_repeats=args.perm_repeats, seed=args.seed)
        perm_df.insert(0, "feature_set", fs_name)
        perm_rows.append(perm_df)
        print(perm_df.sort_values("r2_drop_mean", ascending=False).to_string(index=False))

        # incremental save
        pd.concat(shap_rows, ignore_index=True).to_csv(SHAP_CSV, index=False)
        pd.concat(perm_rows, ignore_index=True).to_csv(PERM_CSV, index=False)

    print(f"\nSHAP → {SHAP_CSV}")
    print(f"Perm → {PERM_CSV}")


if __name__ == "__main__":
    main()
