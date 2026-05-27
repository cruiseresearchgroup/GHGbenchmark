"""
Task T1-A: predict Scope 1 + Scope 2 (log₁₀ tCO₂e) from company features.

Runs all (feature_set × model) combinations on both T1-Wide and T1-Strict, using
the shared test set produced by `build_task_a_splits.py`.

Feature sets
  structured_wide:   gics_11, country_iso2, reporting_year  (categorical only)
  structured_strict: + revenue_musd, employees, market_cap_musd, ebitda_musd,
                       factor_tco2e_per_musd   (financial features, only for ticker-matched rows)

Models
  baseline_factor:   revenue × ExioML factor (zero-ML reference line)
  sector_mean:       predict log target = sector-wise train mean
  ridge:             Ridge regression on one-hot + standardized numeric
  xgboost:           XGBRegressor default
  lightgbm:          LGBMRegressor default

Output: results/t1a_results.csv
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

BASELINE = Path("data/company-level/nzdpu_enriched/factor_baseline_v2.csv")
SPLIT    = Path("data/company-level/splits/task_a_split_v2.csv")
OUT      = Path("results/t1a_results.csv")
SEED     = 42

CATEGORICAL    = ["gics_11", "country_iso2"]
# Wide-available numerics (no yfinance dependency).
NUMERIC_WIDE   = ["reporting_year", "factor_tco2e_per_musd"]
# Snapshot financials (yfinance .info, latest year)
NUMERIC_STRICT = NUMERIC_WIDE + ["revenue_musd", "employees", "market_cap_musd",
                                 "ebitda_musd"]
# Time-aligned financials (yfinance .income_stmt, matched to reporting_year)
#   revenue_best = revenue_musd_ty.fillna(revenue_musd)   # use panel if available
#   ebitda_best  = ebitda_musd_ty.fillna(ebitda_musd)
NUMERIC_STRICT_TY = NUMERIC_WIDE + ["revenue_best_musd", "employees", "market_cap_musd",
                                    "ebitda_best_musd", "net_income_musd_ty"]
NUMERIC_NO_LOG = {"factor_tco2e_per_musd", "reporting_year"}


# ── Load + join ─────────────────────────────────────────────────────────────
print("Loading data...")
bl = pd.read_csv(BASELINE)
sp = pd.read_csv(SPLIT)

with open("data/company-level/nzdpu_enriched/ticker_cache.json") as f:
    tc = json.load(f)
with open("data/company-level/nzdpu_enriched/financials_cache.json") as f:
    fc = json.load(f)
nz_to_ticker = {int(k): v["ticker"] for k, v in tc.items()
                if isinstance(v, dict) and v.get("ticker")}

# yfinance _usd fields are actually in local reporting currency. Apply FX.
# Keep in sync with scripts/apply_fx_fix.py.
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

df = sp.merge(bl[["nz_id","reporting_year","ticker","revenue_musd",
                  "revenue_musd_ty","ebitda_musd_ty","net_income_musd_ty",
                  "factor_tco2e_per_musd",
                  "scope12_actual_tco2e","scope12_pred_tco2e"]],
              on=["nz_id","reporting_year"], how="left")

# per-row employees / market_cap via ticker (snapshot)
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

# "Best" financials = time-aligned where available, else snapshot
df["revenue_best_musd"] = df["revenue_musd_ty"].fillna(df["revenue_musd"])
df["ebitda_best_musd"]  = df["ebitda_musd_ty"].fillna(df["ebitda_musd"])

df["y"] = np.log10(df["scope12_actual_tco2e"])
df["gics_11"]      = df["gics_11"].fillna("Unknown")
df["country_iso2"] = df["country_iso2"].fillna("XX")

print(f"  joined: {len(df):,} rows")
print(f"  train/val/test: "
      f"{(df['split']=='train').sum():,} / "
      f"{(df['split']=='val').sum():,} / "
      f"{(df['split']=='test').sum():,}")


# ── Feature pipeline ────────────────────────────────────────────────────────
def make_features(df_part, feature_set, train_cols=None):
    X_cat = pd.get_dummies(df_part[CATEGORICAL], drop_first=False, dtype=float)
    if feature_set == "structured_wide":
        cols = NUMERIC_WIDE
    elif feature_set == "structured_strict":
        cols = NUMERIC_STRICT
    elif feature_set == "structured_strict_ty":
        cols = NUMERIC_STRICT_TY
    else:
        return X_cat
    X_num = df_part[cols].copy()
    for c in cols:
        med = X_num[c].median()
        X_num[c] = X_num[c].fillna(med)
        if c not in NUMERIC_NO_LOG:
            X_num[c] = np.log1p(X_num[c].clip(lower=0))
    return pd.concat([X_cat, X_num], axis=1)


def metrics(y_true_log, y_pred_log):
    y_true = 10 ** y_true_log
    y_pred = 10 ** y_pred_log
    ape = np.abs(y_true - y_pred) / np.abs(y_true)
    return dict(
        mae_log   = mean_absolute_error(y_true_log, y_pred_log),
        rmse_log  = mean_squared_error(y_true_log, y_pred_log) ** 0.5,
        r2_log    = r2_score(y_true_log, y_pred_log),
        pearson_r = float(np.corrcoef(y_true_log, y_pred_log)[0, 1]),
        mape      = float(ape.mean()),
        median_ape= float(np.median(ape)),
    )


# ── Experiment loop ─────────────────────────────────────────────────────────
results = []

SUBSETS = {
    "T1-Wide":   lambda d: d,
    "T1-Strict": lambda d: d[d["subset_t1strict"]],
}

FEATURE_SETS = ["structured_wide", "structured_strict", "structured_strict_ty"]

MODELS = {
    "ridge":    lambda: Ridge(alpha=1.0, random_state=SEED),
    "xgboost":  lambda: XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                                     random_state=SEED, n_jobs=8, verbosity=0),
    "lightgbm": lambda: LGBMRegressor(n_estimators=400, num_leaves=63, learning_rate=0.05,
                                      random_state=SEED, n_jobs=8, verbose=-1,
                                      force_col_wise=True),
}


for subset_name, subset_fn in SUBSETS.items():
    sub = subset_fn(df).copy()
    if len(sub) == 0:
        continue

    train = sub[sub["split"] == "train"]
    test  = sub[sub["split"] == "test"]

    # ── Baseline 1: factor × revenue ────────────────────────────────────────
    evalable = test.dropna(subset=["scope12_pred_tco2e"])
    evalable = evalable[evalable["scope12_pred_tco2e"] > 0]
    if len(evalable):
        m = metrics(np.log10(evalable["scope12_actual_tco2e"].values),
                    np.log10(evalable["scope12_pred_tco2e"].values))
        results.append({"subset": subset_name, "feature_set": "-",
                        "model": "baseline_factor", "n_train": 0,
                        "n_test": len(evalable), **m})
        print(f"  {subset_name:10s} | -                  | baseline_factor | "
              f"R²_log={m['r2_log']:+.3f}  MAE_log={m['mae_log']:.3f}  n={len(evalable)}")

    # ── Baseline 2: sector-wise mean ────────────────────────────────────────
    sector_mean = train.groupby("gics_11")["y"].mean()
    global_mean = train["y"].mean()
    y_pred = test["gics_11"].map(sector_mean).fillna(global_mean).values
    m = metrics(test["y"].values, y_pred)
    results.append({"subset": subset_name, "feature_set": "-",
                    "model": "sector_mean", "n_train": len(train),
                    "n_test": len(test), **m})
    print(f"  {subset_name:10s} | -                  | sector_mean     | "
          f"R²_log={m['r2_log']:+.3f}  MAE_log={m['mae_log']:.3f}")

    # ── ML grid ─────────────────────────────────────────────────────────────
    for fs in FEATURE_SETS:
        if subset_name == "T1-Wide" and fs.startswith("structured_strict"):
            continue
        X_train = make_features(train, fs)
        X_test  = make_features(test,  fs)
        X_train, X_test = X_train.align(X_test, join="outer", axis=1, fill_value=0)
        y_train = train["y"].values
        y_test  = test["y"].values

        for name, make in MODELS.items():
            model = make()
            if name == "ridge":
                sc = StandardScaler()
                Xtr = sc.fit_transform(X_train)
                Xte = sc.transform(X_test)
                model.fit(Xtr, y_train)
                y_pred = model.predict(Xte)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            m = metrics(y_test, y_pred)
            results.append({"subset": subset_name, "feature_set": fs, "model": name,
                            "n_train": len(train), "n_test": len(test), **m})
            print(f"  {subset_name:10s} | {fs:18s} | {name:15s} | "
                  f"R²_log={m['r2_log']:+.3f}  MAE_log={m['mae_log']:.3f}  r={m['pearson_r']:.3f}")


OUT.parent.mkdir(parents=True, exist_ok=True)
res = pd.DataFrame(results)
res.to_csv(OUT, index=False)
print(f"\n{len(res)} results → {OUT}")
print()
# Pretty summary
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
print(res[["subset","feature_set","model","n_train","n_test",
           "mae_log","r2_log","pearson_r","median_ape"]]
      .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
