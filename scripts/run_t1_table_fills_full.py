"""Comprehensive fill: every blank cell in tables/t1_main_summary.tex.

Cells filled (~41 fits):

  TUNED TREES (XGB/LGBM/RF/HGB) × 5 cells = 20 fits
    S1+2 Open, S1+2 Open matched, S3 Open, S3 Open matched, S3 +SICS

  TABPFN v2 × 6 cells = 6 fits
    S1+2 Open, S1+2 Open matched, S3 Open, S3 Open matched, S3 +SICS, Cross-region

  MLP × 7 cells = 7 fits
    S1+2 Open, S1+2 Open matched, S1+2 +SICS, S3 Open, S3 Open matched,
    S3 +firm, Cross-region

  FT-TRANSFORMER × 7 cells = 7 fits   (same cells as MLP)

  SECTOR MEAN Cross-region = 1 fit  (other empty cells equal to "matched" by definition)

Outputs:
  results/t1_table_fills_full.csv
"""
from __future__ import annotations
import json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tabpfn import TabPFNRegressor

import sys
sys.path.insert(0, str(Path(__file__).parent))
from t1_common import (
    BASELINE, SPLIT, NZDPU, TICKER_CACHE, FIN_CACHE, FX_PER_USD,
    CATEGORICAL, NUMERIC_WIDE, NUMERIC_STRICT, NUMERIC_NO_LOG, SEED,
    metrics, align,
)

OUT = Path("results/t1_table_fills_full.csv")
SMOOTH_M = 20.0

# ── Cfgs from existing hpsearch summaries ──────────────────────────────────
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
    "xgboost_tuned":  {"n_estimators": 500, "max_depth": 10, "learning_rate": 0.08,
                       "min_child_weight": 1.0, "subsample": 1.0, "colsample_bytree": 1.0,
                       "reg_alpha": 1.0, "reg_lambda": 5.0},
    "lightgbm_tuned": {"n_estimators": 1200, "num_leaves": 255, "learning_rate": 0.01,
                       "min_child_samples": 5, "reg_alpha": 1.0, "reg_lambda": 0.0,
                       "feature_fraction": 1.0, "bagging_fraction": 0.8, "bagging_freq": 1},
    "rf_tuned":       {"n_estimators": 800, "max_depth": 25, "min_samples_leaf": 1, "max_features": 0.5},
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


# ── Data loading ───────────────────────────────────────────────────────────
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
    return df


def fit_te(train_df, col, m=SMOOTH_M):
    g = train_df.groupby(col)["y"].agg(["mean","count"])
    gm = train_df["y"].mean()
    smoothed = (g["count"] * g["mean"] + m * gm) / (g["count"] + m)
    return smoothed.to_dict(), gm


def make_features(df_part, regime, train_df=None):
    """regime ∈ {'wide', 'firm', 'firm_sics'}"""
    X_cat = pd.get_dummies(df_part[CATEGORICAL], drop_first=False, dtype=float)
    if regime == "wide":
        cols = NUMERIC_WIDE
    else:
        cols = NUMERIC_STRICT  # = NUMERIC_WIDE + 4 firm columns
    X_num = df_part[cols].copy()
    for c in cols:
        med = X_num[c].median()
        X_num[c] = X_num[c].fillna(med)
        if c not in NUMERIC_NO_LOG:
            X_num[c] = np.log1p(X_num[c].clip(lower=0))
    if regime == "firm_sics" and train_df is not None:
        te_map, gm = fit_te(train_df, "sics_sub_sector")
        sub_te = df_part["sics_sub_sector"].map(te_map).fillna(gm).astype(float).values
        X_num = X_num.assign(sub_sector_te=sub_te)
    return pd.concat([X_cat, X_num], axis=1)


# ── DL baselines (MLP, FT-Transformer) — copied from run_t1_a_ext.py ──────
class MLP(nn.Module):
    def __init__(self, d_in, hidden=256, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden//2), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden//2, 1),
        )
    def forward(self, x): return self.net(x)


class FTTransformer(nn.Module):
    def __init__(self, d_in, d_tok=32, n_heads=4, n_layers=3, drop=0.1):
        super().__init__()
        self.embed = nn.Parameter(torch.randn(d_in, d_tok) * 0.02)
        self.bias  = nn.Parameter(torch.randn(d_in, d_tok) * 0.02)
        self.cls   = nn.Parameter(torch.randn(1, 1, d_tok) * 0.02)
        enc = nn.TransformerEncoderLayer(d_tok, n_heads, dim_feedforward=d_tok*2,
                                         dropout=drop, batch_first=True, activation="gelu")
        self.tx = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_tok), nn.Linear(d_tok, 1))
    def forward(self, x):
        tok = x.unsqueeze(-1) * self.embed + self.bias
        cls = self.cls.expand(x.size(0), -1, -1)
        tok = torch.cat([cls, tok], dim=1)
        h = self.tx(tok)
        return self.head(h[:, 0])


def fit_dl(model_cls, X_tr, y_tr, X_te, y_te, n_epoch=80, lr=1e-3, bs=256, wd=1e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sc = StandardScaler(); Xtr = sc.fit_transform(X_tr); Xte = sc.transform(X_te)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = model_cls(Xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    ds = TensorDataset(Xtr_t, ytr_t); dl = DataLoader(ds, batch_size=bs, shuffle=True)
    for ep in range(n_epoch):
        model.train()
        for xb, yb in dl:
            opt.zero_grad(); pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        y_pred = model(Xte_t).squeeze(-1).cpu().numpy()
    return metrics(y_te, y_pred)


# ── TabPFN baseline ──────────────────────────────────────────────────────
def fit_tabpfn(X_tr, y_tr, X_te, y_te, max_train=10000, seed=SEED):
    rng = np.random.default_rng(seed)
    if len(X_tr) > max_train:
        idx = rng.choice(len(X_tr), size=max_train, replace=False)
        X_fit = X_tr.values[idx] if hasattr(X_tr, 'values') else X_tr[idx]
        y_fit = y_tr[idx]
    else:
        X_fit = X_tr.values if hasattr(X_tr, 'values') else X_tr
        y_fit = y_tr
    cat_idx = [i for i, c in enumerate(X_tr.columns)
               if c.startswith("gics_11_") or c.startswith("country_iso2_")] if hasattr(X_tr, 'columns') else []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TabPFNRegressor(device=device, ignore_pretraining_limits=True,
                            random_state=seed, categorical_features_indices=cat_idx)
    model.fit(X_fit, y_fit)
    Xte_arr = X_te.values if hasattr(X_te, 'values') else X_te
    chunk = 4096
    parts = [model.predict(Xte_arr[i:i+chunk]) for i in range(0, len(Xte_arr), chunk)]
    y_pred = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
    return metrics(y_te, y_pred)


# ── Region map (cross-region) ─────────────────────────────────────────────
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


# ── Main loop ──────────────────────────────────────────────────────────────
print("Loading data...")
panel_s12 = load_panel()
panel_s3  = load_panel_s3()
panel_s12 = panel_s12.assign(region=panel_s12["country_iso2"].map(region_of))
panel_s3  = panel_s3.assign(region=panel_s3["country_iso2"].map(region_of))
print(f"S1+2 panel: {len(panel_s12):,}")
print(f"S3 panel:   {len(panel_s3):,}")

results = []

def add(row):
    results.append(row)
    pd.DataFrame(results).to_csv(OUT, index=False)


def run_cell_models(df, cell_label, regime, models_to_run):
    """Run all requested models on given panel slice + feature regime.
    df: the panel for this cell (already filtered to subset_t1strict if matched/+firm/+SICS).
    regime: 'wide' / 'firm' / 'firm_sics'
    """
    train = df[df["split"] == "train"].reset_index(drop=True)
    val   = df[df["split"] == "val"].reset_index(drop=True)
    test  = df[df["split"] == "test"].reset_index(drop=True)
    trval = pd.concat([train, val]).reset_index(drop=True)
    X_trval = make_features(trval, regime, train_df=trval)
    X_test  = make_features(test,  regime, train_df=trval)
    X_trval, X_test = align(X_trval, X_test)
    med = X_trval.median()
    X_trval = X_trval.fillna(med); X_test = X_test.fillna(med)
    y_trval = trval["y"].values; y_test = test["y"].values

    # Tuned trees
    for model_name in [m for m in models_to_run if m in S12_CFGS]:
        cfg = S3_CFGS[model_name] if "S3" in cell_label else S12_CFGS[model_name]
        t0 = time.time()
        model = build_model(model_name, cfg)
        if model_name in ("rf_tuned", "hgb_tuned"):
            model.fit(X_trval.values, y_trval); y_pred = model.predict(X_test.values)
        else:
            model.fit(X_trval, y_trval); y_pred = model.predict(X_test)
        m = metrics(y_test, y_pred)
        add({"cell": cell_label, "model": model_name, "regime": regime,
             "n_train": len(y_trval), "n_test": len(y_test),
             "elapsed_s": round(time.time()-t0, 1), **m})
        print(f"  {cell_label:20s} {model_name:18s} R²={m['r2_log']:+.4f}")

    # MLP / FT-T (use train only, no val concat — matches existing run_dl)
    X_tr_only = make_features(train, regime, train_df=train)
    X_te_only = make_features(test, regime, train_df=train)
    X_tr_only, X_te_only = align(X_tr_only, X_te_only)
    med = X_tr_only.median()
    X_tr_only = X_tr_only.fillna(med).values.astype(float)
    X_te_only = X_te_only.fillna(med).values.astype(float)
    y_tr_only = train["y"].values

    if "mlp" in models_to_run:
        t0 = time.time()
        try:
            m = fit_dl(MLP, X_tr_only, y_tr_only, X_te_only, y_test, n_epoch=80, lr=1e-3, bs=256)
            add({"cell": cell_label, "model": "mlp", "regime": regime,
                 "n_train": len(y_tr_only), "n_test": len(y_test),
                 "elapsed_s": round(time.time()-t0,1), **m})
            print(f"  {cell_label:20s} {'mlp':18s} R²={m['r2_log']:+.4f}")
        except Exception as e:
            print(f"  [mlp failed] {e}")
    if "ft_transformer" in models_to_run:
        t0 = time.time()
        try:
            m = fit_dl(FTTransformer, X_tr_only, y_tr_only, X_te_only, y_test, n_epoch=60, lr=5e-4, bs=128)
            add({"cell": cell_label, "model": "ft_transformer", "regime": regime,
                 "n_train": len(y_tr_only), "n_test": len(y_test),
                 "elapsed_s": round(time.time()-t0,1), **m})
            print(f"  {cell_label:20s} {'ft_transformer':18s} R²={m['r2_log']:+.4f}")
        except Exception as e:
            print(f"  [ft_t failed] {e}")
    if "tabpfn_v2" in models_to_run:
        t0 = time.time()
        try:
            X_tr_df = make_features(train, regime, train_df=train)
            X_te_df = make_features(test,  regime, train_df=train)
            X_tr_df, X_te_df = align(X_tr_df, X_te_df)
            med = X_tr_df.median()
            X_tr_df = X_tr_df.fillna(med); X_te_df = X_te_df.fillna(med)
            m = fit_tabpfn(X_tr_df, y_tr_only, X_te_df, y_test)
            add({"cell": cell_label, "model": "tabpfn_v2", "regime": regime,
                 "n_train": len(y_tr_only), "n_test": len(y_test),
                 "elapsed_s": round(time.time()-t0,1), **m})
            print(f"  {cell_label:20s} {'tabpfn_v2':18s} R²={m['r2_log']:+.4f}")
        except Exception as e:
            print(f"  [tabpfn failed] {e}")


# ─────────────────────────────────────────────────────────────────────────
# Run all empty cells
# ─────────────────────────────────────────────────────────────────────────

ALL_TUNED = ["xgboost_tuned", "lightgbm_tuned", "rf_tuned", "hgb_tuned"]
ALL_TUNED_PLUS_TABPFN = ALL_TUNED + ["tabpfn_v2"]
ALL_TUNED_PLUS_NEURAL = ALL_TUNED + ["mlp", "ft_transformer", "tabpfn_v2"]
ALL_NEURAL_AND_TABPFN = ["mlp", "ft_transformer", "tabpfn_v2"]

# === Scope 1+2 ===
print("\n=== S1+2 Open (full panel, wide features) ===")
run_cell_models(panel_s12, "S12_Open", "wide", ALL_TUNED_PLUS_NEURAL)

print("\n=== S1+2 Open matched (subset_t1strict, wide features) ===")
run_cell_models(panel_s12[panel_s12["subset_t1strict"]].reset_index(drop=True),
                "S12_Open_matched", "wide", ALL_TUNED_PLUS_NEURAL)

# +SICS for MLP / FT-T (tuned trees + TabPFN already filled in earlier run)
print("\n=== S1+2 +SICS (MLP / FT-T only — others done earlier) ===")
run_cell_models(panel_s12[panel_s12["subset_t1strict"]].reset_index(drop=True),
                "S12_+SICS", "firm_sics", ["mlp", "ft_transformer"])

# === Scope 3 ===
print("\n=== S3 Open (full S3 panel, wide features) ===")
run_cell_models(panel_s3, "S3_Open", "wide", ALL_TUNED_PLUS_NEURAL)

print("\n=== S3 Open matched (subset_t1strict ∩ S3, wide features) ===")
run_cell_models(panel_s3[panel_s3["subset_t1strict"]].reset_index(drop=True),
                "S3_Open_matched", "wide", ALL_TUNED_PLUS_NEURAL)

# +firm S3 — MLP / FT-T (tuned trees done earlier; TabPFN done earlier; HGB done earlier)
print("\n=== S3 +firm (MLP / FT-T only) ===")
run_cell_models(panel_s3[panel_s3["subset_t1strict"]].reset_index(drop=True),
                "S3_+firm", "firm", ["mlp", "ft_transformer"])

# +SICS S3 — tuned trees + TabPFN (MLP / FT-T to be added later if desired)
print("\n=== S3 +SICS (tuned + TabPFN; MLP/FT-T can be added) ===")
run_cell_models(panel_s3[panel_s3["subset_t1strict"]].reset_index(drop=True),
                "S3_+SICS", "firm_sics", ALL_TUNED_PLUS_TABPFN + ["mlp", "ft_transformer"])

# === Cross-region ===
# For tuned trees: already done in run_t1_table_fills.py — skip
# For MLP/FT-T/TabPFN: run here
print("\n=== Cross-region (MLP, FT-T, TabPFN, sector_mean) ===")
df_strict = panel_s12[panel_s12["subset_t1strict"]].copy()
for held_out in ["APAC", "US", "EU"]:
    tr = df_strict[df_strict["region"].isin([r for r in ["US","EU","APAC"] if r != held_out])].reset_index(drop=True)
    te = df_strict[df_strict["region"] == held_out].reset_index(drop=True)
    X_tr_df = make_features(tr, "firm")
    X_te_df = make_features(te, "firm")
    X_tr_df, X_te_df = align(X_tr_df, X_te_df)
    med = X_tr_df.median()
    X_tr_df = X_tr_df.fillna(med); X_te_df = X_te_df.fillna(med)
    y_tr = tr["y"].values; y_te = te["y"].values
    print(f"\nLORO({held_out}): train={len(tr)}, test={len(te)}")

    # sector_mean
    train_smean = tr.groupby("gics_11")["y"].mean(); global_mean = tr["y"].mean()
    y_pred = te["gics_11"].map(train_smean).fillna(global_mean).values
    m = metrics(y_te, y_pred)
    add({"cell": f"CR_{held_out}", "model": "sector_mean", "regime": "firm",
         "n_train": len(y_tr), "n_test": len(y_te), "elapsed_s": 0.0, **m})
    print(f"  sector_mean        R²={m['r2_log']:+.4f}")

    # MLP
    t0 = time.time()
    try:
        m = fit_dl(MLP, X_tr_df.values.astype(float), y_tr, X_te_df.values.astype(float), y_te,
                   n_epoch=80, lr=1e-3, bs=256)
        add({"cell": f"CR_{held_out}", "model": "mlp", "regime": "firm",
             "n_train": len(y_tr), "n_test": len(y_te), "elapsed_s": round(time.time()-t0,1), **m})
        print(f"  mlp                R²={m['r2_log']:+.4f}")
    except Exception as e:
        print(f"  [mlp failed] {e}")

    # FT-T
    t0 = time.time()
    try:
        m = fit_dl(FTTransformer, X_tr_df.values.astype(float), y_tr, X_te_df.values.astype(float), y_te,
                   n_epoch=60, lr=5e-4, bs=128)
        add({"cell": f"CR_{held_out}", "model": "ft_transformer", "regime": "firm",
             "n_train": len(y_tr), "n_test": len(y_te), "elapsed_s": round(time.time()-t0,1), **m})
        print(f"  ft_transformer     R²={m['r2_log']:+.4f}")
    except Exception as e:
        print(f"  [ft_t failed] {e}")

    # TabPFN
    t0 = time.time()
    try:
        m = fit_tabpfn(X_tr_df, y_tr, X_te_df, y_te)
        add({"cell": f"CR_{held_out}", "model": "tabpfn_v2", "regime": "firm",
             "n_train": len(y_tr), "n_test": len(y_te), "elapsed_s": round(time.time()-t0,1), **m})
        print(f"  tabpfn_v2          R²={m['r2_log']:+.4f}")
    except Exception as e:
        print(f"  [tabpfn failed] {e}")


# ── Final summary: aggregate cross-region into single mean per model ─────
print(f"\n=== {len(results)} runs → {OUT} ===")
res = pd.DataFrame(results)
print("\nSummary (R²_log) — table-ready means:")
piv = res.pivot_table(index="model", columns="cell", values="r2_log").round(4)
# Aggregate CR_APAC + CR_US + CR_EU into single CR_mean
cr_cols = [c for c in piv.columns if c.startswith("CR_")]
if cr_cols:
    piv["CR_mean"] = piv[cr_cols].mean(axis=1)
print(piv.to_string())
