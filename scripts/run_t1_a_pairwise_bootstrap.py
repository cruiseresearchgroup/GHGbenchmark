"""
T1-A paired bootstrap — for each pair of headline models, resample the same
test rows with replacement and compute the signed difference
Δ = metric(A) − metric(B) on each resample. This is more rigorous than
comparing independent CIs because it accounts for correlated errors: if A
and B agree on the same rows (common in regression), their independent CIs
can overlap while the paired Δ CI clearly excludes zero.

Inputs:
  results/t1a_ext_hpsearch_summary.csv       — lightgbm_tuned, xgboost_tuned
  results/t1a_ext_hpsearch_summary_ext.csv   — random_forest_tuned, hist_gbm_tuned (if exists)

For LLMs we load per-row prediction CSVs already on disk:
  results/t1a_llm_{mode}_{model_tag}_predictions.csv

We refit the tuned tree models once to obtain their per-row predictions,
then bootstrap over the shared test-row index.

Output:
  results/t1a_pairwise_bootstrap.csv
    columns: model_a, model_b, metric, delta_mean, delta_lo, delta_hi,
             p_two_sided, a_wins_frac
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from t1_common import load_scope12, make_features, metrics, align, SEED

warnings.filterwarnings("ignore")

OUT_DIR = Path("results"); OUT_DIR.mkdir(exist_ok=True)
OUT_CSV = OUT_DIR / "t1a_pairwise_bootstrap.csv"
HP_SUMMARY_CSV     = OUT_DIR / "t1a_ext_hpsearch_summary.csv"
HP_SUMMARY_EXT_CSV = OUT_DIR / "t1a_ext_hpsearch_summary_ext.csv"


def refit_tuned_tree(model_name: str, cfg: dict, X_tv, y_tv, X_te):
    """Return test-set predictions (log10) for a tuned tree config."""
    if model_name == "lightgbm":
        from lightgbm import LGBMRegressor
        m = LGBMRegressor(random_state=SEED, n_jobs=8, verbose=-1,
                          force_col_wise=True, **cfg)
        m.fit(X_tv, y_tv)
    elif model_name == "xgboost":
        from xgboost import XGBRegressor
        m = XGBRegressor(random_state=SEED, n_jobs=8, verbosity=0, **cfg)
        m.fit(X_tv, y_tv)
    elif model_name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(random_state=SEED, n_jobs=8, **cfg)
        m.fit(X_tv.values, y_tv)
    elif model_name == "hist_gbm":
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(random_state=SEED, **cfg)
        m.fit(X_tv.values, y_tv)
    else:
        raise ValueError(model_name)
    return m.predict(X_te if isinstance(X_te, np.ndarray) else X_te.values)


def load_predictions():
    """Return (pred_df: pd.DataFrame indexed by test row, y_true: np.ndarray,
    per-model predictions on log10)."""
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
    X_tv = pd.concat([X_tr, X_va]).reset_index(drop=True)
    y_tv = np.concatenate([y_tr, y_va])

    # RF/HGB path needs imputation (LGBM/XGB handle NaN natively)
    med = X_tr.median(numeric_only=True)
    X_tv_imp = X_tv.fillna(med); X_te_imp = X_te.fillna(med)

    preds = {}
    # LightGBM (default) — the "untuned" baseline used in §1.2 seed/CI table.
    # Refitting it here gives us per-row predictions so we can paired-bootstrap
    # "tuned vs default" and "default vs TabPFN" pairs the narrative cares about.
    print("[refit] lightgbm_default ...", flush=True)
    from lightgbm import LGBMRegressor
    m = LGBMRegressor(random_state=SEED, n_jobs=8, verbose=-1, force_col_wise=True)
    m.fit(X_tv, y_tv)
    preds["lightgbm_default"] = m.predict(X_te)

    # TabPFN v2 — the second reference point for §1.2.
    try:
        print("[refit] tabpfn_v2 ...", flush=True)
        from tabpfn import TabPFNRegressor
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # TabPFN handles NaN; feed raw training fold (no median impute) for a
        # like-for-like refit with scripts/run_t1_tabpfn.py::run_tabpfn.
        reg = TabPFNRegressor(device=device, ignore_pretraining_limits=True,
                              random_state=SEED)
        reg.fit(X_tv.values, y_tv)
        preds["tabpfn_v2"] = reg.predict(X_te.values)
    except Exception as e:
        print(f"[refit] tabpfn_v2 FAILED: {type(e).__name__}: {e}")

    # Tuned trees from summary CSV
    if HP_SUMMARY_CSV.exists():
        summ = pd.read_csv(HP_SUMMARY_CSV)
        for _, r in summ.iterrows():
            name = r["model"].replace("_tuned", "")
            print(f"[refit] {name} ...", flush=True)
            cfg = json.loads(r["best_cfg"])
            preds[name + "_tuned"] = refit_tuned_tree(name, cfg, X_tv, y_tv, X_te)

    if HP_SUMMARY_EXT_CSV.exists():
        summ = pd.read_csv(HP_SUMMARY_EXT_CSV)
        for _, r in summ.iterrows():
            name = r["model"].replace("_tuned", "")
            print(f"[refit] {name} ...", flush=True)
            cfg = json.loads(r["best_cfg"])
            preds[name + "_tuned"] = refit_tuned_tree(name, cfg, X_tv_imp, y_tv, X_te_imp)

    # LLMs: match by (nz_id, reporting_year) against test rows
    test_keys = list(zip(test["nz_id"].astype(int), test["reporting_year"].astype(int)))
    for csv_path in sorted(OUT_DIR.glob("t1a_llm_*_predictions.csv")):
        # Format: t1a_llm_{mode}_{modeltag}_predictions.csv
        stem = csv_path.stem.replace("t1a_llm_", "").replace("_predictions", "")
        p = pd.read_csv(csv_path)
        if len(p) < 100:  # incomplete (e.g., Gemini zero_shot aborted)
            continue
        lut = {(int(r["nz_id"]), int(r["reporting_year"])):
               (float(r["pred_scope12"]) if pd.notna(r.get("pred_scope12")) else np.nan)
               for _, r in p.iterrows()}
        aligned = np.array([lut.get(k, np.nan) for k in test_keys], dtype=float)
        # LLMs return raw tCO2e → log10 (with floor at 1e-6 to avoid −inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            aligned_log = np.where(aligned > 0, np.log10(np.maximum(aligned, 1e-6)), np.nan)
        preds["llm_" + stem] = aligned_log

    return y_te, preds


def paired_bootstrap(y_true, y_a, y_b, n_boot=1000, seed=0):
    """Return dict of paired Δ statistics on three metrics (r2, mae, rmse)
    for rows where BOTH a and b have a valid prediction."""
    valid = np.isfinite(y_a) & np.isfinite(y_b) & np.isfinite(y_true)
    y_true, y_a, y_b = y_true[valid], y_a[valid], y_b[valid]
    n = len(y_true)
    if n < 20:
        return None

    rng = np.random.default_rng(seed)
    deltas = {"r2": np.empty(n_boot), "mae": np.empty(n_boot), "rmse": np.empty(n_boot)}
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, ya, yb = y_true[idx], y_a[idx], y_b[idx]
        ybar = yt.mean()
        ss_tot = ((yt - ybar) ** 2).sum()
        if ss_tot > 0:
            r2a = 1 - ((yt - ya) ** 2).sum() / ss_tot
            r2b = 1 - ((yt - yb) ** 2).sum() / ss_tot
        else:
            r2a = r2b = 0.0
        deltas["r2"][b]   = r2a - r2b
        deltas["mae"][b]  = np.mean(np.abs(yt - ya)) - np.mean(np.abs(yt - yb))
        deltas["rmse"][b] = np.sqrt(np.mean((yt - ya) ** 2)) - np.sqrt(np.mean((yt - yb) ** 2))

    out = {"n_pairs": n}
    for key, arr in deltas.items():
        lo, hi = np.quantile(arr, [0.025, 0.975])
        # Two-sided p-value: 2 × min(P(Δ≤0), P(Δ≥0))
        p = 2.0 * min((arr <= 0).mean(), (arr >= 0).mean())
        # For MAE/RMSE, lower is better, so a_wins = frac where Δ < 0
        a_wins = (arr < 0).mean() if key in ("mae", "rmse") else (arr > 0).mean()
        out[f"delta_{key}_mean"] = float(arr.mean())
        out[f"delta_{key}_lo"]   = float(lo)
        out[f"delta_{key}_hi"]   = float(hi)
        out[f"p_{key}_two_sided"] = float(p)
        out[f"a_wins_{key}_frac"] = float(a_wins)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_path", type=str, default=str(OUT_CSV))
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    y_te, preds = load_predictions()
    print(f"\n[bootstrap] loaded {len(preds)} model predictions "
          f"on {len(y_te)} test rows")

    names = list(preds.keys())
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            res = paired_bootstrap(y_te, preds[a], preds[b],
                                   n_boot=args.n_boot, seed=args.seed)
            if res is None:
                continue
            row = {"model_a": a, "model_b": b, **res}
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_path, index=False)

    # Console headline
    print(f"\nTop-5 paired deltas by |ΔR²| (rows with p<0.05 flagged):")
    df_sorted = df.reindex(df["delta_r2_mean"].abs().sort_values(ascending=False).index)
    for _, r in df_sorted.head(20).iterrows():
        sig = "*" if r["p_r2_two_sided"] < 0.05 else " "
        print(f" {sig} {r['model_a']:30s}  vs  {r['model_b']:30s}  | "
              f"ΔR²={r['delta_r2_mean']:+.3f} [{r['delta_r2_lo']:+.3f}, {r['delta_r2_hi']:+.3f}]  "
              f"p={r['p_r2_two_sided']:.3f}  n={r['n_pairs']}")

    print(f"\nFull pairwise table → {args.out_path}")


if __name__ == "__main__":
    main()
