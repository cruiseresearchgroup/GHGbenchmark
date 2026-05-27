"""
Task T1-A′: predict Scope 3 (log₁₀ tCO₂e) from company features.

Same pipeline as T1-A; only the target changes. Scope 3 covers upstream +
downstream indirect emissions (15 GHGP categories); typically 10-100× larger
than Scope 1+2 and much harder to predict because (a) coverage varies by
category, (b) self-reporting inconsistency is well-documented in the ESG
literature.

Outlier thresholds (looser than S1+2, because Cat 11 use-of-sold-products
can legitimately exceed 1 B tCO2e for fossil-fuel majors):
  upper:  2×10^9 tCO2e    (removes 19 clearly impossible rows, e.g. Northern
                           Trust 49 B tCO2e — the US total is ~5 B/yr)
  lower:  100 tCO2e       (one commercial airliner trip ~ 100 t; anything
                           below is almost certainly under-reporting/NaN)

Features + models: identical to T1-A.

Output:
  results/t1a_prime_results.csv   (same schema as t1a_results.csv)
  results/t1a_prime_run.log
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
NZDPU    = Path("data/company-level/nzdpu/nzdpu_emissions.csv")
OUT      = Path("results/t1a_prime_results.csv")
SEED     = 42

UPPER_S3 = 2e9
LOWER_S3 = 100.0

NUMERIC_STRICT = ["reporting_year", "factor_tco2e_per_musd",
                  "revenue_musd", "employees", "market_cap_musd",
                  "ebitda_musd"]
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


print("Loading data...")
bl = pd.read_csv(BASELINE)
nz = pd.read_csv(NZDPU, low_memory=False)

# Build scope 3 target (raw NZDPU, joined by nz_id × reporting_year, dedup)
DASH = "\u2014"
nz["s3_raw"] = pd.to_numeric(nz["total_s3_emissions_ghg"].replace(DASH, np.nan), errors="coerce")
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

# Filter rows: valid scope3 within outlier bounds + country
df = df.dropna(subset=["scope3_actual_tco2e", "country_iso2"])
df = df[(df["scope3_actual_tco2e"] >= LOWER_S3) &
        (df["scope3_actual_tco2e"] <= UPPER_S3)]
df["gics_11"] = df["gics_11"].fillna("Unknown")
df["y"] = np.log10(df["scope3_actual_tco2e"])

print(f"  T1-Wide S3 rows: {len(df):,}")
print(f"  Scope 3 target log10: min={df['y'].min():.2f}  "
      f"median={df['y'].median():.2f}  max={df['y'].max():.2f}")

# T1-Strict = has ticker + revenue
df["subset_t1strict"] = (df["ticker"].notna() &
                         df["revenue_best_musd"].notna() &
                         (df["revenue_best_musd"] > 0))
print(f"  T1-Strict S3 rows: {df['subset_t1strict'].sum():,}")

# Stratified 80/10/10 split — rebuild because row set changes with target
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
idx_train, idx_val = train_test_split(idx_tv, test_size=1/9,
                                      stratify=strat_tv_safe, random_state=SEED)
df["split"] = "train"
df.loc[idx_val, "split"] = "val"
df.loc[idx_test, "split"] = "test"

print(f"  train/val/test: {(df['split']=='train').sum():,} / "
      f"{(df['split']=='val').sum():,} / {(df['split']=='test').sum():,}")


# ── Features + models (copied from run_t1_task_a.py) ─────────────────────
def make_features(df_part, feature_set):
    X_cat = pd.get_dummies(df_part[CATEGORICAL], drop_first=False, dtype=float)
    if feature_set in ("structured_strict", "structured_strict_ty"):
        cols = NUMERIC_STRICT_TY if feature_set == "structured_strict_ty" else NUMERIC_STRICT
        X_num = df_part[cols].copy()
        for c in cols:
            med = X_num[c].median()
            X_num[c] = X_num[c].fillna(med)
            if c not in NUMERIC_NO_LOG:
                X_num[c] = np.log1p(X_num[c].clip(lower=0))
        X = pd.concat([X_cat, X_num], axis=1)
    else:
        X = X_cat
    return X


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

    # Sector-mean baseline
    sector_mean = train.groupby("gics_11")["y"].mean()
    global_mean = train["y"].mean()
    y_pred = test["gics_11"].map(sector_mean).fillna(global_mean).values
    m = metrics(test["y"].values, y_pred)
    results.append({"subset": subset_name, "feature_set": "-",
                    "model": "sector_mean", "n_train": len(train),
                    "n_test": len(test), **m})
    print(f"  {subset_name:10s} | -                  | sector_mean     | "
          f"R²_log={m['r2_log']:+.3f}  MAE_log={m['mae_log']:.3f}")

    for fs in FEATURE_SETS:
        if subset_name == "T1-Wide" and fs.startswith("structured_strict"):
            continue
        X_train = make_features(train, fs)
        X_test  = make_features(test, fs)
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
            print(f"  {subset_name:10s} | {fs:20s} | {name:15s} | "
                  f"R²_log={m['r2_log']:+.3f}  MAE_log={m['mae_log']:.3f}  r={m['pearson_r']:.3f}")


OUT.parent.mkdir(parents=True, exist_ok=True)
res = pd.DataFrame(results)
res.to_csv(OUT, index=False)
print(f"\n{len(res)} results → {OUT}")
print()
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
print(res[["subset","feature_set","model","n_train","n_test",
           "mae_log","r2_log","pearson_r","median_ape"]]
      .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
