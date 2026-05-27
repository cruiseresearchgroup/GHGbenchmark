"""Quick fill: default Ridge / XGBoost / LightGBM on Scope 3 + SICS sub_sector TE.
Mirrors run_t1_phase3_subsector.py but on the Scope 3 panel.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import sys
sys.path.insert(0, str(Path(__file__).parent))
from t1_common import (
    BASELINE, SPLIT, NZDPU, TICKER_CACHE, FIN_CACHE, FX_PER_USD,
    CATEGORICAL, NUMERIC_STRICT, NUMERIC_NO_LOG, SEED, metrics, align,
)

OUT = Path("results/t1a_s3_subte_defaults.csv")
SMOOTH_M = 20.0

# Load Scope 3 panel
bl = pd.read_csv(BASELINE); sp = pd.read_csv(SPLIT)
nz = pd.read_csv(NZDPU, low_memory=False,
                 usecols=["nz_id","reporting_year","sics_sub_sector","total_s3_emissions_ghg"])
nz = nz.drop_duplicates(["nz_id","reporting_year"], keep="first")

with open(TICKER_CACHE) as f: tc = json.load(f)
with open(FIN_CACHE) as f: fc = json.load(f)
nz_to_ticker = {int(k): v["ticker"] for k, v in tc.items()
                if isinstance(v, dict) and v.get("ticker")}

df = sp.merge(bl[["nz_id","reporting_year","ticker","revenue_musd","factor_tco2e_per_musd",
                  "scope12_actual_tco2e","scope12_pred_tco2e"]],
              on=["nz_id","reporting_year"], how="left")
df = df.merge(nz, on=["nz_id","reporting_year"], how="left")

def fl(key):
    return df["nz_id"].map(lambda i: (fc.get(nz_to_ticker.get(i), {}) or {}).get(key))
def _cur(i):
    t = nz_to_ticker.get(i)
    c = (fc.get(t, {}) or {}).get("currency") if t else None
    return str(c).upper() if c else None
fx = df["nz_id"].map(lambda i: FX_PER_USD.get(_cur(i)))
df["employees"]       = fl("employees")
df["market_cap_musd"] = fl("market_cap_usd").astype(float) / 1e6 / fx
df["ebitda_musd"]     = fl("ebitda_usd").astype(float) / 1e6 / fx

DASH = "—"
s3 = pd.to_numeric(df["total_s3_emissions_ghg"].replace(DASH, np.nan), errors="coerce")
df["s3"] = s3
df = df.dropna(subset=["s3"])
df = df[(df["s3"] >= 100) & (df["s3"] <= 2e9)].reset_index(drop=True)
df["y"] = np.log10(df["s3"])
df["gics_11"]      = df["gics_11"].fillna("Unknown")
df["country_iso2"] = df["country_iso2"].fillna("XX")
df["sics_sub_sector"] = df["sics_sub_sector"].fillna("Unknown").astype(str)

strict = df[df["subset_t1strict"]].copy()
train = strict[strict["split"] == "train"].reset_index(drop=True)
test  = strict[strict["split"] == "test"].reset_index(drop=True)
print(f"S3 Strict train={len(train):,} test={len(test):,}")

def fit_te(train_df, col, m=SMOOTH_M):
    g = train_df.groupby(col)["y"].agg(["mean","count"])
    gm = train_df["y"].mean()
    smoothed = (g["count"] * g["mean"] + m * gm) / (g["count"] + m)
    return smoothed.to_dict(), gm

def make_features(df_part, train_df=None):
    X_cat = pd.get_dummies(df_part[CATEGORICAL], drop_first=False, dtype=float)
    X_num = df_part[NUMERIC_STRICT].copy()
    for c in NUMERIC_STRICT:
        med = X_num[c].median()
        X_num[c] = X_num[c].fillna(med)
        if c not in NUMERIC_NO_LOG:
            X_num[c] = np.log1p(X_num[c].clip(lower=0))
    if train_df is not None:
        te_map, gm = fit_te(train_df, "sics_sub_sector")
        sub_te = df_part["sics_sub_sector"].map(te_map).fillna(gm).astype(float).values
        X_num = X_num.assign(sub_sector_te=sub_te)
    return pd.concat([X_cat, X_num], axis=1)

X_train = make_features(train, train_df=train)
X_test  = make_features(test,  train_df=train)
X_train, X_test = align(X_train, X_test)
y_train = train["y"].values; y_test = test["y"].values

MODELS = {
    "ridge":    lambda: Ridge(alpha=1.0, random_state=SEED),
    "xgboost":  lambda: XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                                     random_state=SEED, n_jobs=8, verbosity=0),
    "lightgbm": lambda: LGBMRegressor(n_estimators=400, num_leaves=63, learning_rate=0.05,
                                      random_state=SEED, n_jobs=8, verbose=-1, force_col_wise=True),
}

results = []
for name, make in MODELS.items():
    t0 = time.time()
    model = make()
    if name == "ridge":
        sc = StandardScaler()
        Xtr = sc.fit_transform(X_train.fillna(0)); Xte = sc.transform(X_test.fillna(0))
        model.fit(Xtr, y_train); y_pred = model.predict(Xte)
    else:
        model.fit(X_train, y_train); y_pred = model.predict(X_test)
    m = metrics(y_test, y_pred)
    print(f"  {name:10s} R²={m['r2_log']:+.4f}  ({time.time()-t0:.1f}s)")
    results.append({"model": name, **m})

OUT.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(results).to_csv(OUT, index=False)
print(f"→ {OUT}")
