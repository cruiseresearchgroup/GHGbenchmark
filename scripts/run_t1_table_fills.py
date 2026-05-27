"""Fill blank cells in tables/t1_main_summary.tex by running the missing
tuned-tree configurations: tuned-trees with SICS sub_sector TE on Scope 1+2,
tuned-trees on cross-region transfer, HistGradientBoosting tuned on Scope 3,
and TabPFN with SICS sub_sector TE on Scope 1+2.

Reuses existing best_cfg from the hpsearch summaries — does NOT re-tune.

Outputs:
  results/t1_table_fills.csv   (all rows, one per (cell_label, model))
"""
from __future__ import annotations
import json, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge  # noqa
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import sys
sys.path.insert(0, str(Path(__file__).parent))
from t1_common import (
    BASELINE, SPLIT, NZDPU, TICKER_CACHE, FIN_CACHE, FX_PER_USD,
    CATEGORICAL, NUMERIC_STRICT, NUMERIC_NO_LOG, SEED,
    metrics, align,
)

OUT = Path("results/t1_table_fills.csv")
SMOOTH_M = 20.0  # same as Phase 3a TE smoothing

# ── Cfgs from existing hpsearch summaries ──────────────────────────────
S12_CFGS = {
    "xgboost_tuned":   {"n_estimators": 800, "max_depth": 10, "learning_rate": 0.08,
                        "min_child_weight": 1.0, "subsample": 1.0, "colsample_bytree": 0.9,
                        "reg_alpha": 0.01, "reg_lambda": 1.0},
    "lightgbm_tuned":  {"n_estimators": 800, "num_leaves": 127, "learning_rate": 0.08,
                        "min_child_samples": 5, "reg_alpha": 0.01, "reg_lambda": 0.01,
                        "feature_fraction": 0.8, "bagging_fraction": 0.7, "bagging_freq": 1},
    "rf_tuned":        {"n_estimators": 400, "max_depth": 25, "min_samples_leaf": 1, "max_features": 1.0},
    "hgb_tuned":       {"max_iter": 800, "learning_rate": 0.08, "max_depth": 10,
                        "max_leaf_nodes": 127, "min_samples_leaf": 10, "l2_regularization": 0.0},
}
S3_CFGS = {
    # Scope 3 best_cfg from t1a_prime_hpsearch_summary.csv; HGB borrows the S1+2 cfg
    # since prime hpsearch never finished HGB (single-thread too slow at the time).
    "xgboost_tuned":  {"n_estimators": 500, "max_depth": 10, "learning_rate": 0.08,
                       "min_child_weight": 1.0, "subsample": 1.0, "colsample_bytree": 1.0,
                       "reg_alpha": 1.0, "reg_lambda": 5.0},
    "lightgbm_tuned": {"n_estimators": 1200, "num_leaves": 255, "learning_rate": 0.01,
                       "min_child_samples": 5, "reg_alpha": 1.0, "reg_lambda": 0.0,
                       "feature_fraction": 1.0, "bagging_fraction": 0.8, "bagging_freq": 1},
    "rf_tuned":       {"n_estimators": 800, "max_depth": 25, "min_samples_leaf": 1, "max_features": 0.5},
    # HGB S3: borrow S1+2 cfg as a sensible default (we are filling a single cell, not retuning).
    "hgb_tuned":      S12_CFGS["hgb_tuned"],
}


def build_model(name, cfg):
    if name == "xgboost_tuned":
        return XGBRegressor(random_state=SEED, n_jobs=8, verbosity=0, **cfg)
    if name == "lightgbm_tuned":
        return LGBMRegressor(random_state=SEED, n_jobs=8, verbose=-1, force_col_wise=True, **cfg)
    if name == "rf_tuned":
        return RandomForestRegressor(random_state=SEED, n_jobs=8, **cfg)
    if name == "hgb_tuned":
        return HistGradientBoostingRegressor(random_state=SEED, **cfg)
    raise ValueError(name)


# ── Data loaders (Scope 1+2 enriched panel; Scope 3 from raw) ─────────────
def load_panel():
    bl = pd.read_csv(BASELINE)
    sp = pd.read_csv(SPLIT)
    nz = pd.read_csv(NZDPU, low_memory=False,
                     usecols=["nz_id", "reporting_year", "sics_sub_sector"])
    nz = nz.drop_duplicates(["nz_id", "reporting_year"], keep="first")

    with open(TICKER_CACHE) as f: tc = json.load(f)
    with open(FIN_CACHE) as f: fc = json.load(f)
    nz_to_ticker = {int(k): v["ticker"] for k, v in tc.items()
                    if isinstance(v, dict) and v.get("ticker")}

    df = sp.merge(bl[["nz_id", "reporting_year", "ticker", "revenue_musd",
                      "factor_tco2e_per_musd",
                      "scope12_actual_tco2e", "scope12_pred_tco2e"]],
                  on=["nz_id", "reporting_year"], how="left")
    df = df.merge(nz, on=["nz_id", "reporting_year"], how="left")

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
    df["y"] = np.log10(df["scope12_actual_tco2e"])
    df["gics_11"]      = df["gics_11"].fillna("Unknown")
    df["country_iso2"] = df["country_iso2"].fillna("XX")
    df["sics_sub_sector"] = df["sics_sub_sector"].fillna("Unknown").astype(str)
    return df


def load_panel_s3():
    """Scope 3 panel: same as run_t1_task_a_prime.py — bring NZDPU s3 column,
    apply [100, 2e9] outlier filter, rebuild stratified split."""
    df = load_panel()
    nz = pd.read_csv(NZDPU, low_memory=False)
    DASH = "—"
    s3 = pd.to_numeric(nz["total_s3_emissions_ghg"].replace(DASH, np.nan), errors="coerce")
    nz_s3 = nz[["nz_id", "reporting_year"]].copy()
    nz_s3["s3"] = s3
    nz_s3 = (nz_s3.dropna(subset=["s3"])
                  .drop_duplicates(["nz_id","reporting_year"], keep="first"))
    df = df.drop(columns=["y"]).merge(nz_s3, on=["nz_id","reporting_year"], how="inner")
    df = df[(df["s3"] >= 100) & (df["s3"] <= 2e9)].reset_index(drop=True)
    df["y"] = np.log10(df["s3"])
    # Use existing split if subset_t1strict; otherwise rebuild stratified
    df = df[df["subset_t1strict"]].copy()
    return df


def fit_target_encoding(train_df, col, m=SMOOTH_M):
    g = train_df.groupby(col)["y"].agg(["mean","count"])
    gm = train_df["y"].mean()
    smoothed = (g["count"] * g["mean"] + m * gm) / (g["count"] + m)
    return smoothed.to_dict(), gm


def make_features(df_part, train_df=None, with_subte=False):
    X_cat = pd.get_dummies(df_part[CATEGORICAL], drop_first=False, dtype=float)
    X_num = df_part[NUMERIC_STRICT].copy()
    for c in NUMERIC_STRICT:
        med = X_num[c].median()
        X_num[c] = X_num[c].fillna(med)
        if c not in NUMERIC_NO_LOG:
            X_num[c] = np.log1p(X_num[c].clip(lower=0))
    if with_subte and train_df is not None:
        te_map, gm = fit_target_encoding(train_df, "sics_sub_sector")
        sub_te = df_part["sics_sub_sector"].map(te_map).fillna(gm).astype(float).values
        X_num = X_num.assign(sub_sector_te=sub_te)
    return pd.concat([X_cat, X_num], axis=1)


# ── Experiments ──────────────────────────────────────────────────────────
print("Loading data...")
df = load_panel()
df_strict = df[df["subset_t1strict"]].copy()
print(f"S1+2 Strict: {len(df_strict):,} rows")

results = []

# ─── Experiment 1: tuned trees with SICS sub_sector TE on S1+2 ────────────
print("\n=== (1) Tuned trees + SICS sub_sector TE on S1+2 ===")
train = df_strict[df_strict["split"] == "train"].reset_index(drop=True)
val   = df_strict[df_strict["split"] == "val"].reset_index(drop=True)
test  = df_strict[df_strict["split"] == "test"].reset_index(drop=True)

# refit on train+val (matches hpsearch protocol)
trval = pd.concat([train, val]).reset_index(drop=True)
X_trval = make_features(trval, train_df=trval, with_subte=True)
X_test  = make_features(test,  train_df=trval, with_subte=True)
X_trval, X_test = align(X_trval, X_test)
# imputation parity: median fill any cols that ended up with NaN after align
med = X_trval.median()
X_trval = X_trval.fillna(med); X_test = X_test.fillna(med)
y_trval = trval["y"].values; y_test = test["y"].values

for name, cfg in S12_CFGS.items():
    t0 = time.time()
    model = build_model(name, cfg)
    model.fit(X_trval.values if name == "rf_tuned" else X_trval,
              y_trval) if name in ("xgboost_tuned","lightgbm_tuned") else \
        model.fit(X_trval.values, y_trval)
    y_pred = model.predict(X_test.values if name in ("rf_tuned","hgb_tuned") else X_test)
    m = metrics(y_test, y_pred)
    row = {"cell": "S12_+SICS", "model": name, "elapsed_s": round(time.time()-t0, 1),
           "n_train": len(y_trval), "n_test": len(y_test), **m}
    results.append(row)
    print(f"  {name:20s} R²={m['r2_log']:+.4f}  MAE={m['mae_log']:.4f}  ({row['elapsed_s']}s)")
    pd.DataFrame(results).to_csv(OUT, index=False)


# ─── Experiment 2: cross-region for tuned trees ─────────────────────────
print("\n=== (2) Cross-region (LORO) for tuned trees ===")
REGIONS = {
    "US":   {"US"},
    "EU":   {"DE","FR","IT","ES","NL","BE","SE","FI","DK","AT","IE","PT","GR",
             "LU","CZ","PL","HU","RO","BG","HR","SI","SK","LT","LV","EE","CY",
             "MT","GB","CH","NO","IS"},
    "APAC": {"JP","KR","CN","IN","AU","NZ","SG","HK","TW","TH","MY","ID","PH",
             "VN","BD","PK","LK"},
}
def region_of(iso):
    for r, mem in REGIONS.items():
        if iso in mem: return r
    return "Other"

df_strict = df_strict.assign(region=df_strict["country_iso2"].map(region_of))
print(f"Region sizes: {df_strict['region'].value_counts().to_dict()}")

for held_out in ["APAC", "US", "EU"]:
    tr = df_strict[df_strict["region"].isin([r for r in ["US","EU","APAC"] if r != held_out])].reset_index(drop=True)
    te = df_strict[df_strict["region"] == held_out].reset_index(drop=True)
    X_tr = make_features(tr, with_subte=False)
    X_te = make_features(te, with_subte=False)
    X_tr, X_te = align(X_tr, X_te)
    med = X_tr.median(); X_tr = X_tr.fillna(med); X_te = X_te.fillna(med)
    y_tr = tr["y"].values; y_te = te["y"].values
    print(f"\nLORO({held_out}): train={len(tr)}, test={len(te)}")
    for name, cfg in S12_CFGS.items():
        t0 = time.time()
        model = build_model(name, cfg)
        # rf/hgb need numpy; xgb/lgbm accept df
        if name in ("rf_tuned", "hgb_tuned"):
            model.fit(X_tr.values, y_tr); y_pred = model.predict(X_te.values)
        else:
            model.fit(X_tr, y_tr); y_pred = model.predict(X_te)
        m = metrics(y_te, y_pred)
        row = {"cell": f"CR_{held_out}", "model": name,
               "elapsed_s": round(time.time()-t0, 1),
               "n_train": len(y_tr), "n_test": len(y_te), **m}
        results.append(row)
        print(f"  {name:20s} R²={m['r2_log']:+.4f}  MAE={m['mae_log']:.4f}  ({row['elapsed_s']}s)")
        pd.DataFrame(results).to_csv(OUT, index=False)


# ─── Experiment 3: HistGB tuned on Scope 3 ─────────────────────────────
print("\n=== (3) HistGB tuned on Scope 3 +firm ===")
df_s3 = load_panel_s3()
print(f"S3 Strict: {len(df_s3):,} rows")
train_s3 = df_s3[df_s3["split"] == "train"].reset_index(drop=True)
val_s3   = df_s3[df_s3["split"] == "val"].reset_index(drop=True)
test_s3  = df_s3[df_s3["split"] == "test"].reset_index(drop=True)
trval_s3 = pd.concat([train_s3, val_s3]).reset_index(drop=True)

X_trval_s3 = make_features(trval_s3, with_subte=False)
X_test_s3  = make_features(test_s3,  with_subte=False)
X_trval_s3, X_test_s3 = align(X_trval_s3, X_test_s3)
med = X_trval_s3.median()
X_trval_s3 = X_trval_s3.fillna(med); X_test_s3 = X_test_s3.fillna(med)
y_trval_s3 = trval_s3["y"].values; y_test_s3 = test_s3["y"].values

t0 = time.time()
hgb = build_model("hgb_tuned", S3_CFGS["hgb_tuned"])
hgb.fit(X_trval_s3.values, y_trval_s3)
y_pred = hgb.predict(X_test_s3.values)
m = metrics(y_test_s3, y_pred)
row = {"cell": "S3_+firm", "model": "hgb_tuned",
       "elapsed_s": round(time.time()-t0, 1),
       "n_train": len(y_trval_s3), "n_test": len(y_test_s3), **m}
results.append(row)
print(f"  hgb_tuned            R²={m['r2_log']:+.4f}  MAE={m['mae_log']:.4f}  ({row['elapsed_s']}s)")
pd.DataFrame(results).to_csv(OUT, index=False)


# ─── Experiment 4: TabPFN + SICS sub_sector TE on S1+2 ─────────────────
print("\n=== (4) TabPFN v2 + SICS sub_sector TE on S1+2 ===")
try:
    import torch
    from tabpfn import TabPFNRegressor

    X_tr = make_features(train, train_df=train, with_subte=True)
    X_te = make_features(test,  train_df=train, with_subte=True)
    X_tr, X_te = align(X_tr, X_te)
    med = X_tr.median(); X_tr = X_tr.fillna(med); X_te = X_te.fillna(med)

    # Subsample to 10K like other TabPFN runs
    rng = np.random.default_rng(SEED)
    if len(X_tr) > 10000:
        idx = rng.choice(len(X_tr), size=10000, replace=False)
        X_fit = X_tr.iloc[idx].values; y_fit = train["y"].values[idx]
    else:
        X_fit = X_tr.values; y_fit = train["y"].values

    cat_idx = [i for i, c in enumerate(X_tr.columns) if c.startswith("gics_11_") or c.startswith("country_iso2_")]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    model = TabPFNRegressor(device=device, ignore_pretraining_limits=True,
                            random_state=SEED, categorical_features_indices=cat_idx)
    model.fit(X_fit, y_fit)
    chunk = 4096
    parts = [model.predict(X_te.values[i:i+chunk]) for i in range(0, len(X_te), chunk)]
    y_pred = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
    y_test = test["y"].values
    m = metrics(y_test, y_pred)
    row = {"cell": "S12_+SICS", "model": "tabpfn_v2",
           "elapsed_s": round(time.time()-t0, 1),
           "n_train": len(y_fit), "n_test": len(y_test), **m}
    results.append(row)
    print(f"  tabpfn_v2            R²={m['r2_log']:+.4f}  MAE={m['mae_log']:.4f}  ({row['elapsed_s']}s)")
    pd.DataFrame(results).to_csv(OUT, index=False)
except Exception as e:
    print(f"  TabPFN failed: {e}")

# ─── Final summary ─────────────────────────────────────────────────────
print(f"\n{len(results)} runs → {OUT}")
res = pd.DataFrame(results)
piv = res.pivot_table(index="model", columns="cell", values="r2_log").round(4)
print(piv.to_string())
