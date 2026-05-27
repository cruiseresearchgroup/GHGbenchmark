"""Phase 3a + 3b: add SICS sub_sector / industry target encoding to T1-Strict.

Compares:
  baseline:     Strict (gics_11, country_iso2 + numerics)            — Phase 1 result
  +subTE:       Strict + sics_sub_sector target-encoding             — Phase 3a
  +subTE+indTE: Strict + sub_sector + industry target-encoding       — Phase 3b

Target encoding is computed from TRAIN ONLY (no leak) with Bayesian smoothing
toward the global train mean (m=20).  Unseen categories at val/test → global mean.

Outputs:
  results/t1a_phase3_subsector.csv
  results/t1a_phase3_subsector_summary.csv
"""
from __future__ import annotations
import json
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
    CATEGORICAL, NUMERIC_WIDE, NUMERIC_STRICT, NUMERIC_NO_LOG, SEED,
    metrics, align,
)

OUT       = Path("results/t1a_phase3_subsector.csv")
SUMMARY   = Path("results/t1a_phase3_subsector_summary.csv")
SMOOTH_M  = 20.0  # Bayesian shrinkage prior strength

# ── Load data (mirror run_t1_task_a.py, then join SICS hierarchy) ───────────
print("Loading data...")
bl = pd.read_csv(BASELINE)
sp = pd.read_csv(SPLIT)
nz = pd.read_csv(NZDPU, low_memory=False,
                 usecols=["nz_id", "reporting_year",
                          "sics_sub_sector", "sics_industry"])
# de-dup nzdpu rows (some duplicates exist on (nz_id, year))
nz = nz.drop_duplicates(["nz_id", "reporting_year"], keep="first")

with open(TICKER_CACHE) as f:
    tc = json.load(f)
with open(FIN_CACHE) as f:
    fc = json.load(f)
nz_to_ticker = {int(k): v["ticker"] for k, v in tc.items()
                if isinstance(v, dict) and v.get("ticker")}

df = sp.merge(bl[["nz_id", "reporting_year", "ticker", "revenue_musd",
                  "factor_tco2e_per_musd",
                  "scope12_actual_tco2e", "scope12_pred_tco2e"]],
              on=["nz_id", "reporting_year"], how="left")
df = df.merge(nz, on=["nz_id", "reporting_year"], how="left")

def finlookup(key):
    return df["nz_id"].map(lambda i: (fc.get(nz_to_ticker.get(i), {}) or {}).get(key))
def _cur(i):
    t = nz_to_ticker.get(i)
    c = (fc.get(t, {}) or {}).get("currency") if t else None
    return str(c).upper() if c else None
fx = df["nz_id"].map(lambda i: FX_PER_USD.get(_cur(i)))

df["employees"]       = finlookup("employees")
df["market_cap_musd"] = finlookup("market_cap_usd").astype(float) / 1e6 / fx
df["ebitda_musd"]     = finlookup("ebitda_usd").astype(float) / 1e6 / fx

df["y"] = np.log10(df["scope12_actual_tco2e"])
df["gics_11"]      = df["gics_11"].fillna("Unknown")
df["country_iso2"] = df["country_iso2"].fillna("XX")
df["sics_sub_sector"] = df["sics_sub_sector"].fillna("Unknown").astype(str)
df["sics_industry"]   = df["sics_industry"].fillna("Unknown").astype(str)

print(f"  joined: {len(df):,} rows  ({df['sics_sub_sector'].nunique()} sub_sectors, "
      f"{df['sics_industry'].nunique()} industries)")


# ── Target encoding (train-only, smoothed) ─────────────────────────────────
def fit_target_encoding(train_df, col, m=SMOOTH_M):
    """Return dict: category -> smoothed mean(y).  Smoothing: (n*mean + m*global) / (n+m)."""
    g = train_df.groupby(col)["y"].agg(["mean", "count"])
    global_mean = train_df["y"].mean()
    smoothed = (g["count"] * g["mean"] + m * global_mean) / (g["count"] + m)
    return smoothed.to_dict(), global_mean


def apply_target_encoding(df_part, col, te_map, global_mean):
    return df_part[col].map(te_map).fillna(global_mean).astype(float).values


# ── Feature pipeline ────────────────────────────────────────────────────────
def make_features(df_part, feature_set, te_train=None):
    """feature_set ∈ {'strict', 'strict_subTE', 'strict_subTE_indTE'}"""
    X_cat = pd.get_dummies(df_part[CATEGORICAL], drop_first=False, dtype=float)
    cols = NUMERIC_STRICT.copy()
    X_num = df_part[cols].copy()
    for c in cols:
        med = X_num[c].median()
        X_num[c] = X_num[c].fillna(med)
        if c not in NUMERIC_NO_LOG:
            X_num[c] = np.log1p(X_num[c].clip(lower=0))

    extra = {}
    if feature_set in ("strict_subTE", "strict_subTE_indTE"):
        te_map, gm = te_train["sub"]
        extra["sub_sector_te"] = apply_target_encoding(
            df_part, "sics_sub_sector", te_map, gm)
    if feature_set == "strict_subTE_indTE":
        te_map, gm = te_train["ind"]
        extra["industry_te"] = apply_target_encoding(
            df_part, "sics_industry", te_map, gm)

    if extra:
        X_extra = pd.DataFrame(extra, index=df_part.index)
        return pd.concat([X_cat, X_num, X_extra], axis=1)
    return pd.concat([X_cat, X_num], axis=1)


# ── Run on Strict subset ───────────────────────────────────────────────────
strict = df[df["subset_t1strict"]].copy()
train  = strict[strict["split"] == "train"].reset_index(drop=True)
test   = strict[strict["split"] == "test"].reset_index(drop=True)

print(f"  Strict: train={len(train):,}  test={len(test):,}")
print(f"  Train sub_sector counts (top 5):")
print(train["sics_sub_sector"].value_counts().head().to_string())

# Fit target encoders on TRAIN only
te_train = {
    "sub": fit_target_encoding(train, "sics_sub_sector"),
    "ind": fit_target_encoding(train, "sics_industry"),
}
print(f"\n  Sub-sector TE: {len(te_train['sub'][0])} categories, global_mean={te_train['sub'][1]:.3f}")
print(f"  Industry  TE: {len(te_train['ind'][0])} categories, global_mean={te_train['ind'][1]:.3f}")

# ── Models (defaults match run_t1_task_a.py for apples-to-apples Phase 1 compare) ──
MODELS = {
    "ridge":    lambda: Ridge(alpha=1.0, random_state=SEED),
    "xgboost":  lambda: XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                                     random_state=SEED, n_jobs=8, verbosity=0),
    "lightgbm": lambda: LGBMRegressor(n_estimators=400, num_leaves=63, learning_rate=0.05,
                                      random_state=SEED, n_jobs=8, verbose=-1,
                                      force_col_wise=True),
}

FEATURE_SETS = ["strict", "strict_subTE", "strict_subTE_indTE"]
results = []

for fs in FEATURE_SETS:
    X_train = make_features(train, fs, te_train)
    X_test  = make_features(test,  fs, te_train)
    X_train, X_test = align(X_train, X_test)
    y_train = train["y"].values
    y_test  = test["y"].values

    for name, make in MODELS.items():
        model = make()
        if name == "ridge":
            sc = StandardScaler()
            Xtr = sc.fit_transform(X_train.fillna(0))
            Xte = sc.transform(X_test.fillna(0))
            model.fit(Xtr, y_train)
            y_pred = model.predict(Xte)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        m = metrics(y_test, y_pred)
        row = {"feature_set": fs, "model": name,
               "n_train": len(train), "n_test": len(test),
               "n_features": X_train.shape[1], **m}
        results.append(row)
        print(f"  {fs:20s} | {name:10s} | "
              f"R²_log={m['r2_log']:+.4f}  MAE_log={m['mae_log']:.4f}  "
              f"r={m['pearson_r']:.4f}  n_feat={X_train.shape[1]}")

OUT.parent.mkdir(parents=True, exist_ok=True)
res = pd.DataFrame(results)
res.to_csv(OUT, index=False)

# Pivot: Δ relative to baseline
piv = res.pivot(index="model", columns="feature_set", values="r2_log")
piv["Δ_subTE"]      = piv["strict_subTE"]      - piv["strict"]
piv["Δ_subTE_indTE"] = piv["strict_subTE_indTE"] - piv["strict"]
piv.to_csv(SUMMARY)

print(f"\n{len(res)} results → {OUT}")
print(f"Summary → {SUMMARY}")
print()
print("=== Δ R²_log vs Strict baseline ===")
print(piv.to_string(float_format=lambda x: f"{x:+.4f}"))
