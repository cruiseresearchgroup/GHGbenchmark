"""
T1-A' (Scope 3) HP search — val-based 40-trial random search for
LightGBM / XGBoost / RandomForest / HistGradientBoosting on the T1-Strict
subset with `structured_strict_ty` features.

Mirrors run_t1_a_ext.py::run_hpsearch (Scope 1+2) + run_t1_a_hpsearch_ext.py
(RF + HGB extension) so the Scope 3 table can reach the §1.1 format.

Data prep replicates run_t1_task_a_prime.py: same NZDPU S3 loader, same
FX correction, same T1-Strict subset, same stratified 80/10/10 split with
SEED=42. Target: `log10(total_s3_emissions_ghg)`.

Outputs:
  results/t1a_prime_hpsearch.csv           — every trial
  results/t1a_prime_hpsearch_summary.csv   — best cfg + refit-on-train+val test metrics

Usage:
  python scripts/run_t1_prime_hpsearch.py --n_trials 40
"""
from __future__ import annotations
import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from sklearn.linear_model import Ridge  # noqa: F401 — kept for future ridge tuning
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

# ── Constants (verbatim from run_t1_task_a_prime.py) ────────────────────
BASELINE = Path("data/company-level/nzdpu_enriched/factor_baseline_v2.csv")
NZDPU    = Path("data/company-level/nzdpu/nzdpu_emissions.csv")
SEED     = 42

UPPER_S3 = 2e9
LOWER_S3 = 100.0

NUMERIC_STRICT_TY = ["reporting_year", "factor_tco2e_per_musd",
                     "revenue_best_musd", "employees", "market_cap_musd",
                     "ebitda_best_musd", "net_income_musd_ty"]
NUMERIC_NO_LOG = {"factor_tco2e_per_musd", "reporting_year"}
CATEGORICAL = ["gics_11", "country_iso2"]

FX_PER_USD = {
    "USD": 1.0,    "JPY": 150.0,  "EUR": 0.92,  "GBP": 0.78,  "CNY": 7.2,
    "KRW": 1370,   "INR": 84,     "CAD": 1.35,  "AUD": 1.50,  "BRL": 5.2,
    "MXN": 17,     "SEK": 10.5,   "TWD": 32,    "CHF": 0.88,  "HKD": 7.8,
    "SGD": 1.35,   "NOK": 10.9,   "DKK": 6.9,   "ZAR": 18,    "THB": 36,
    "IDR": 15800,  "TRY": 34,     "PLN": 4.0,   "HUF": 370,   "ILS": 3.7,
    "AED": 3.67,   "SAR": 3.75,   "CZK": 23,    "PHP": 57,    "MYR": 4.7,
    "RON": 4.6,    "COP": 4200,   "CLP": 950,   "PEN": 3.8,   "EGP": 49,
    "PKR": 280,    "NZD": 1.65,   "QAR": 3.64,  "VND": 25000, "BHD": 0.377,
    "KWD": 0.307,  "OMR": 0.385,  "ARS": 950,   "RUB": 95,    "UAH": 41,
    "NGN": 1600,   "KES": 130,
}

OUT_DIR = Path("results"); OUT_DIR.mkdir(exist_ok=True)
TRIALS_CSV  = OUT_DIR / "t1a_prime_hpsearch.csv"
SUMMARY_CSV = OUT_DIR / "t1a_prime_hpsearch_summary.csv"


def metrics(y_true_log, y_pred_log):
    return dict(
        mae_log   = float(mean_absolute_error(y_true_log, y_pred_log)),
        rmse_log  = float(mean_squared_error(y_true_log, y_pred_log) ** 0.5),
        r2_log    = float(r2_score(y_true_log, y_pred_log)),
        pearson_r = float(np.corrcoef(y_true_log, y_pred_log)[0, 1]),
    )


def load_prime_df():
    print("[prime] loading data...", flush=True)
    bl = pd.read_csv(BASELINE)
    nz = pd.read_csv(NZDPU, low_memory=False)

    DASH = "—"
    nz["s3_raw"] = pd.to_numeric(nz["total_s3_emissions_ghg"].replace(DASH, np.nan),
                                 errors="coerce")
    s3_map = (nz.sort_values(["nz_id", "reporting_year"])
                .drop_duplicates(subset=["nz_id", "reporting_year"], keep="first")
                .set_index(["nz_id", "reporting_year"])["s3_raw"])

    df = bl.set_index(["nz_id", "reporting_year"])
    df["scope3_actual_tco2e"] = s3_map
    df = df.reset_index()

    with open("data/company-level/nzdpu_enriched/ticker_cache.json") as f:
        tc = json.load(f)
    with open("data/company-level/nzdpu_enriched/financials_cache.json") as f:
        fc = json.load(f)
    nz_to_ticker = {int(k): v["ticker"] for k, v in tc.items()
                    if isinstance(v, dict) and v.get("ticker")}

    def finlookup(key):
        return df["nz_id"].map(lambda i: (fc.get(nz_to_ticker.get(i), {}) or {}).get(key))

    def _currency(nz_id):
        t = nz_to_ticker.get(nz_id)
        cur = (fc.get(t, {}) or {}).get("currency") if t else None
        return str(cur).upper() if cur else None

    fx_rate = df["nz_id"].map(lambda i: FX_PER_USD.get(_currency(i)))
    df["employees"]       = finlookup("employees")
    df["market_cap_musd"] = finlookup("market_cap_usd").astype(float) / 1e6 / fx_rate
    df["ebitda_musd"]     = finlookup("ebitda_usd").astype(float) / 1e6 / fx_rate
    df["revenue_best_musd"] = df["revenue_musd_ty"].fillna(df["revenue_musd"])
    df["ebitda_best_musd"]  = df["ebitda_musd_ty"].fillna(df["ebitda_musd"])

    df = df.dropna(subset=["scope3_actual_tco2e", "country_iso2"])
    df = df[(df["scope3_actual_tco2e"] >= LOWER_S3) &
            (df["scope3_actual_tco2e"] <= UPPER_S3)]
    df["gics_11"] = df["gics_11"].fillna("Unknown")
    df["y"] = np.log10(df["scope3_actual_tco2e"])

    df["subset_t1strict"] = (df["ticker"].notna() &
                             df["revenue_best_musd"].notna() &
                             (df["revenue_best_musd"] > 0))
    df = df[df["subset_t1strict"]].copy()
    print(f"[prime] T1-Strict S3 rows: {len(df):,}")
    return df


def make_features_prime(df_part):
    X_cat = pd.get_dummies(df_part[CATEGORICAL], drop_first=False, dtype=float)
    X_num = df_part[NUMERIC_STRICT_TY].copy()
    for c in NUMERIC_STRICT_TY:
        med = X_num[c].median()
        X_num[c] = X_num[c].fillna(med)
        if c not in NUMERIC_NO_LOG:
            X_num[c] = np.log1p(X_num[c].clip(lower=0))
    return pd.concat([X_cat, X_num], axis=1)


def build_prime_splits(df):
    strat = df["gics_11"].astype(str) + "|" + df["country_iso2"].astype(str)
    counts = strat.value_counts()
    strat = strat.where(~strat.isin(counts[counts < 5].index),
                        df["gics_11"].astype(str) + "|_tinyRoW")
    counts = strat.value_counts()
    strat = strat.where(~strat.isin(counts[counts < 5].index), "_global")

    idx = df.index.to_numpy()
    idx_tv, idx_test = train_test_split(idx, test_size=0.10,
                                        stratify=strat.loc[idx], random_state=SEED)
    strat_tv = strat.loc[idx_tv]
    vc = strat_tv.value_counts()
    strat_tv_safe = strat_tv.where(strat_tv.isin(set(vc[vc >= 2].index)), "_global")
    idx_train, idx_val = train_test_split(idx_tv, test_size=1 / 9,
                                          stratify=strat_tv_safe, random_state=SEED)

    train = df.loc[idx_train].reset_index(drop=True)
    val   = df.loc[idx_val].reset_index(drop=True)
    test  = df.loc[idx_test].reset_index(drop=True)

    X_tr = make_features_prime(train)
    X_va = make_features_prime(val)
    X_te = make_features_prime(test)
    X_tr, X_va = X_tr.align(X_va, join="outer", axis=1, fill_value=0)
    X_tr, X_te = X_tr.align(X_te, join="outer", axis=1, fill_value=0)
    X_va = X_va.reindex(columns=X_tr.columns, fill_value=0)
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)
    y_tr = train["y"].values; y_va = val["y"].values; y_te = test["y"].values
    print(f"[prime] train={len(y_tr):,}  val={len(y_va):,}  test={len(y_te):,}"
          f"  | features={X_tr.shape[1]}")
    return X_tr, y_tr, X_va, y_va, X_te, y_te


# ── HP samplers ─────────────────────────────────────────────────────────
def sample_lgbm(rng):
    return dict(
        n_estimators=int(rng.choice([300, 400, 600, 800, 1200])),
        num_leaves=int(rng.choice([15, 31, 63, 127, 255])),
        learning_rate=float(rng.choice([0.01, 0.02, 0.05, 0.08, 0.1])),
        min_child_samples=int(rng.choice([5, 10, 20, 50])),
        reg_alpha=float(rng.choice([0.0, 0.01, 0.1, 1.0])),
        reg_lambda=float(rng.choice([0.0, 0.01, 0.1, 1.0])),
        feature_fraction=float(rng.choice([0.7, 0.8, 0.9, 1.0])),
        bagging_fraction=float(rng.choice([0.7, 0.8, 0.9, 1.0])),
        bagging_freq=1,
    )


def sample_xgb(rng):
    return dict(
        n_estimators=int(rng.choice([300, 500, 800, 1200])),
        max_depth=int(rng.choice([4, 6, 8, 10])),
        learning_rate=float(rng.choice([0.01, 0.02, 0.05, 0.08, 0.1])),
        min_child_weight=float(rng.choice([1, 3, 5, 10])),
        subsample=float(rng.choice([0.7, 0.8, 0.9, 1.0])),
        colsample_bytree=float(rng.choice([0.7, 0.8, 0.9, 1.0])),
        reg_alpha=float(rng.choice([0.0, 0.01, 0.1, 1.0])),
        reg_lambda=float(rng.choice([0.0, 0.1, 1.0, 5.0])),
    )


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


def build_lgbm(cfg): return LGBMRegressor(random_state=SEED, n_jobs=8, verbose=-1,
                                          force_col_wise=True, **cfg)
def build_xgb(cfg):  return XGBRegressor(random_state=SEED, n_jobs=8, verbosity=0, **cfg)
def build_rf(cfg):   return RandomForestRegressor(random_state=SEED, n_jobs=8, **cfg)
def build_hgb(cfg):  return HistGradientBoostingRegressor(random_state=SEED, **cfg)


MODELS = {
    "lightgbm":      (sample_lgbm, build_lgbm, False),  # accepts NaN
    "xgboost":       (sample_xgb,  build_xgb,  False),  # accepts NaN
    "random_forest": (sample_rf,   build_rf,   True),   # needs impute
    "hist_gbm":      (sample_hgb,  build_hgb,  False),  # HGB accepts NaN
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_trials", type=int, default=40)
    ap.add_argument("--models", type=str,
                    default="lightgbm,xgboost,random_forest,hist_gbm")
    args = ap.parse_args()

    df = load_prime_df()
    X_tr, y_tr, X_va, y_va, X_te, y_te = build_prime_splits(df)

    # Imputed copies for RF (sklearn forest does not accept NaN).
    med = X_tr.median(numeric_only=True)
    X_tr_imp = X_tr.fillna(med); X_va_imp = X_va.fillna(med); X_te_imp = X_te.fillna(med)

    rng = np.random.default_rng(SEED)
    trial_rows, summary_rows = [], []

    for model_name in args.models.split(","):
        if model_name not in MODELS:
            print(f"[skip] unknown model: {model_name}"); continue
        sampler, builder, needs_impute = MODELS[model_name]
        X_tr_use, X_va_use, X_te_use = (X_tr_imp, X_va_imp, X_te_imp) if needs_impute \
                                      else (X_tr, X_va, X_te)
        print(f"\n=== {model_name} | {args.n_trials} trials"
              f" | impute={needs_impute} ===")

        best_val, best_cfg = None, None
        for t in range(args.n_trials):
            cfg = sampler(rng)
            t0 = time.time()
            try:
                m = builder(cfg)
                # xgb / lgbm accept DataFrame; RF/HGB prefer numpy
                fit_X = X_tr_use.values if needs_impute else X_tr_use
                pred_va = X_va_use.values if needs_impute else X_va_use
                pred_te = X_te_use.values if needs_impute else X_te_use
                m.fit(fit_X, y_tr)
                mv = metrics(y_va, m.predict(pred_va))
                mt = metrics(y_te, m.predict(pred_te))
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
            print(f"  trial {t:2d} | val R²={mv['r2_log']:+.3f}"
                  f" | test R²={mt['r2_log']:+.3f} | {row['elapsed_s']:.0f}s",
                  flush=True)
            if best_val is None or mv["r2_log"] > best_val:
                best_val, best_cfg = mv["r2_log"], cfg

        if best_cfg is None:
            print(f"  [{model_name}] all trials failed"); continue

        print(f"  [{model_name}] best val R²={best_val:+.3f}  →  refit on train+val")
        if needs_impute:
            X_tv = pd.concat([X_tr_imp, X_va_imp]).reset_index(drop=True)
        else:
            X_tv = pd.concat([X_tr, X_va]).reset_index(drop=True)
        y_tv = np.concatenate([y_tr, y_va])
        final = builder(best_cfg)
        fit_X = X_tv.values if needs_impute else X_tv
        pred_te = X_te_use.values if needs_impute else X_te_use
        final.fit(fit_X, y_tv)
        final_m = metrics(y_te, final.predict(pred_te))
        print(f"    final test R²={final_m['r2_log']:+.3f}  MAE={final_m['mae_log']:.3f}")
        summary_rows.append({
            "model": f"{model_name}_tuned",
            **final_m,
            "n_train": len(y_tv), "n_test": len(y_te),
            "best_cfg": json.dumps(best_cfg),
        })
        pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)

    print(f"\nTrials  → {TRIALS_CSV}")
    print(f"Summary → {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
