"""
Task T1-C: cross-region generalization for Scope 1+2.

Setup: split T1-Strict rows by region (US / EU / APAC / Other) and run three
"leave-one-region-out" experiments:

  LORO(APAC)  train = US + EU,   test = APAC
  LORO(US)    train = EU + APAC, test = US
  LORO(EU)    train = US + APAC, test = EU

Question: how much does R² drop when the model has to predict companies from
a region it has never seen? If ~= T1-A random-split R², the model has learned
physics (revenue → emissions) that transfers. If it plummets, it was relying on
region-specific shortcuts (country_iso2 = latent FX/regulation/disclosure
style).

Uses the postFX + outlier-cleaned factor_baseline_v2.csv — same data as T1-A.
Only structured_strict features are tested (the other tiers are not
region-informative enough to be interesting).

Output:
  results/t1c_results.csv
  results/t1c_run.log
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

BASELINE = Path("data/company-level/nzdpu_enriched/factor_baseline_v2.csv")
SPLIT    = Path("data/company-level/splits/task_a_split_v2.csv")
OUT      = Path("results/t1c_results.csv")
SEED     = 42

NUMERIC_STRICT = ["reporting_year", "factor_tco2e_per_musd",
                  "revenue_musd", "employees", "market_cap_musd",
                  "ebitda_musd"]
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

REGIONS = {
    "US":   {"US"},
    "EU":   {"DE","FR","IT","ES","NL","BE","SE","FI","DK","AT","IE","PT","GR",
             "LU","CZ","PL","HU","RO","BG","HR","SI","SK","LT","LV","EE","CY",
             "MT","GB","CH","NO","IS"},
    "APAC": {"JP","KR","CN","IN","AU","NZ","SG","HK","TW","TH","MY","ID","PH",
             "VN","BD","PK","LK"},
}

def region_of(iso):
    for r, members in REGIONS.items():
        if iso in members:
            return r
    return "Other"


# ── Load + join (mirror run_t1_task_a.py) ──────────────────────────────
print("Loading data...")
bl = pd.read_csv(BASELINE)
sp = pd.read_csv(SPLIT)

with open("data/company-level/nzdpu_enriched/ticker_cache.json") as f:
    tc = json.load(f)
with open("data/company-level/nzdpu_enriched/financials_cache.json") as f:
    fc = json.load(f)
nz_to_ticker = {int(k): v["ticker"] for k, v in tc.items()
                if isinstance(v, dict) and v.get("ticker")}

df = sp.merge(bl[["nz_id","reporting_year","ticker","revenue_musd",
                  "revenue_musd_ty","ebitda_musd_ty","net_income_musd_ty",
                  "factor_tco2e_per_musd","scope12_actual_tco2e"]],
              on=["nz_id","reporting_year"], how="left")

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

df["y"] = np.log10(df["scope12_actual_tco2e"])
df["gics_11"]      = df["gics_11"].fillna("Unknown")
df["country_iso2"] = df["country_iso2"].fillna("XX")
df["region"] = df["country_iso2"].map(region_of)

# T1-Strict only: needs financials for structured_strict
df = df[df["subset_t1strict"]].copy()

print("\nRegion sizes (T1-Strict):")
print(df["region"].value_counts().to_string())
print()


def make_features(df_part):
    X_cat = pd.get_dummies(df_part[CATEGORICAL], drop_first=False, dtype=float)
    X_num = df_part[NUMERIC_STRICT].copy()
    for c in NUMERIC_STRICT:
        med = X_num[c].median()
        X_num[c] = X_num[c].fillna(med)
        if c not in NUMERIC_NO_LOG:
            X_num[c] = np.log1p(X_num[c].clip(lower=0))
    return pd.concat([X_cat, X_num], axis=1)


def metrics(y_true_log, y_pred_log):
    y_true = 10 ** y_true_log; y_pred = 10 ** y_pred_log
    ape = np.abs(y_true - y_pred) / np.abs(y_true)
    return dict(
        mae_log    = mean_absolute_error(y_true_log, y_pred_log),
        rmse_log   = mean_squared_error(y_true_log, y_pred_log) ** 0.5,
        r2_log     = r2_score(y_true_log, y_pred_log),
        pearson_r  = float(np.corrcoef(y_true_log, y_pred_log)[0, 1]),
        mape       = float(ape.mean()),
        median_ape = float(np.median(ape)),
    )


MODELS = {
    "ridge":    lambda: Ridge(alpha=1.0, random_state=SEED),
    "xgboost":  lambda: XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                                     random_state=SEED, n_jobs=8, verbosity=0),
    "lightgbm": lambda: LGBMRegressor(n_estimators=400, num_leaves=63, learning_rate=0.05,
                                      random_state=SEED, n_jobs=8, verbose=-1,
                                      force_col_wise=True),
}

results = []
for held_out in ["APAC", "US", "EU"]:
    train_df = df[df["region"].isin([r for r in ["US","EU","APAC"] if r != held_out])]
    test_df  = df[df["region"] == held_out]
    print(f"── LORO({held_out}): train={len(train_df):,}  test={len(test_df):,} ─────────")

    X_tr_full = make_features(train_df)
    X_te_full = make_features(test_df)
    X_tr, X_te = X_tr_full.align(X_te_full, join="outer", axis=1, fill_value=0)
    y_tr = train_df["y"].values
    y_te = test_df["y"].values

    for name, make in MODELS.items():
        model = make()
        if name == "ridge":
            sc = StandardScaler()
            Xa = sc.fit_transform(X_tr); Xb = sc.transform(X_te)
            model.fit(Xa, y_tr); y_pred = model.predict(Xb)
        else:
            model.fit(X_tr, y_tr); y_pred = model.predict(X_te)
        m = metrics(y_te, y_pred)
        results.append({"held_out": held_out, "model": name,
                        "n_train": len(train_df), "n_test": len(test_df), **m})
        print(f"  {name:10s}  R²_log={m['r2_log']:+.3f}  MAE_log={m['mae_log']:.3f}  "
              f"r={m['pearson_r']:.3f}  MedAPE={m['median_ape']:.3f}")
    print()


OUT.parent.mkdir(parents=True, exist_ok=True)
res = pd.DataFrame(results)
res.to_csv(OUT, index=False)
print(f"{len(res)} results → {OUT}")
print()
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
print(res[["held_out","model","n_train","n_test","mae_log","r2_log","pearson_r","median_ape"]]
      .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# ── Compare to T1-A random-split R² on same LightGBM ──────────────────
print("\n=== Transfer-gap summary (LightGBM, structured_strict) ===")
t1a_r2 = 0.826  # from t1a_run_postFX_clean.log
lgbm = res[res["model"] == "lightgbm"].set_index("held_out")
for r in ["APAC","US","EU"]:
    gap = t1a_r2 - lgbm.loc[r, "r2_log"]
    print(f"  T1-A (random) R² = {t1a_r2:.3f}   "
          f"T1-C LORO({r}) R² = {lgbm.loc[r,'r2_log']:+.3f}   "
          f"gap = {gap:+.3f}")
