"""
Tabular foundation-model baseline (TabPFN v2) on T1-A (Scope 1+2) and
T1-A' (Scope 3), T1-Strict / structured_strict only.

TabPFN is a pretrained transformer for tabular regression/classification
(PriorLabs, Nature 2024). Zero/low-shot: fit in seconds, predict on GPU.

Run in ghg conda env (GPU auto-detected). Output row appended to the
existing t1a_results.csv / t1a_prime_results.csv for consolidated reporting.
"""

import numpy as np
import pandas as pd
import json, time
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import torch

from tabpfn import TabPFNRegressor

BASELINE = Path("data/company-level/nzdpu_enriched/factor_baseline_v2.csv")
SPLIT    = Path("data/company-level/splits/task_a_split_v2.csv")
NZDPU    = Path("data/company-level/nzdpu/nzdpu_emissions.csv")
SEED     = 42

NUMERIC_STRICT = ["reporting_year", "factor_tco2e_per_musd",
                  "revenue_musd", "employees", "market_cap_musd",
                  "ebitda_musd"]
NUMERIC_NO_LOG = {"factor_tco2e_per_musd", "reporting_year"}
CATEGORICAL    = ["gics_11", "country_iso2"]

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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── Load + enrich (shared with run_t1_task_a.py) ────────────────────────
print("Loading data...")
bl = pd.read_csv(BASELINE)
sp = pd.read_csv(SPLIT)

with open("data/company-level/nzdpu_enriched/ticker_cache.json") as f:
    tc = json.load(f)
with open("data/company-level/nzdpu_enriched/financials_cache.json") as f:
    fc = json.load(f)
nz_to_ticker = {int(k): v["ticker"] for k, v in tc.items()
                if isinstance(v, dict) and v.get("ticker")}

def enrich(df):
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
    df["gics_11"]      = df["gics_11"].fillna("Unknown")
    df["country_iso2"] = df["country_iso2"].fillna("XX")
    return df


def make_features(df_part):
    X_cat = pd.get_dummies(df_part[CATEGORICAL], drop_first=False, dtype=float)
    X_num = df_part[NUMERIC_STRICT].copy()
    for c in NUMERIC_STRICT:
        X_num[c] = X_num[c].fillna(X_num[c].median())
        if c not in NUMERIC_NO_LOG:
            X_num[c] = np.log1p(X_num[c].clip(lower=0))
    return pd.concat([X_cat, X_num], axis=1)


def metrics(y_true_log, y_pred_log):
    y_true = 10 ** y_true_log; y_pred = 10 ** y_pred_log
    ape = np.abs(y_true - y_pred) / np.abs(y_true)
    return dict(
        mae_log    = float(mean_absolute_error(y_true_log, y_pred_log)),
        rmse_log   = float(mean_squared_error(y_true_log, y_pred_log) ** 0.5),
        r2_log     = float(r2_score(y_true_log, y_pred_log)),
        pearson_r  = float(np.corrcoef(y_true_log, y_pred_log)[0, 1]),
        mape       = float(ape.mean()),
        median_ape = float(np.median(ape)),
    )


def run_tabpfn(X_train, y_train, X_test, y_test, tag):
    # TabPFN v2 handles ~10K train rows comfortably; subsample if larger.
    MAX_N = 10000
    rng = np.random.default_rng(SEED)
    if len(X_train) > MAX_N:
        idx = rng.choice(len(X_train), MAX_N, replace=False)
        X_tr = X_train.iloc[idx]; y_tr = y_train[idx]
        print(f"  [{tag}] subsampled train {len(X_train)} → {MAX_N}")
    else:
        X_tr, y_tr = X_train, y_train

    t0 = time.time()
    reg = TabPFNRegressor(device=device, ignore_pretraining_limits=True,
                          random_state=SEED)
    reg.fit(X_tr.values, y_tr)
    y_pred = reg.predict(X_test.values)
    dt = time.time() - t0
    m = metrics(y_test, y_pred)
    print(f"  [{tag}] TabPFN  R²={m['r2_log']:+.3f}  MAE={m['mae_log']:.3f}  "
          f"r={m['pearson_r']:.3f}  ({dt:.1f}s, n_train={len(X_tr)}, n_test={len(X_test)})")
    return m, len(X_tr), len(X_test)


# ── T1-A: Scope 1+2, T1-Strict, structured_strict ──────────────────────
print("\n=== T1-A Scope 1+2 ===")
df = sp.merge(bl[["nz_id","reporting_year","ticker","revenue_musd",
                  "factor_tco2e_per_musd","scope12_actual_tco2e"]],
              on=["nz_id","reporting_year"], how="left")
df = enrich(df)
df["y"] = np.log10(df["scope12_actual_tco2e"])
df = df[df["subset_t1strict"]].copy()
train = df[df["split"] == "train"]
test  = df[df["split"] == "test"]
X_tr = make_features(train); X_te = make_features(test)
X_tr, X_te = X_tr.align(X_te, join="outer", axis=1, fill_value=0)
m_a, n_tr_a, n_te_a = run_tabpfn(X_tr, train["y"].values, X_te, test["y"].values,
                                  "T1-A/T1-Strict/structured_strict")

# Append to existing CSV
out_a = Path("results/t1a_results.csv")
res_a = pd.read_csv(out_a)
new = {"subset":"T1-Strict","feature_set":"structured_strict","model":"tabpfn",
       "n_train":n_tr_a,"n_test":n_te_a, **m_a}
res_a = pd.concat([res_a, pd.DataFrame([new])], ignore_index=True)
res_a.to_csv(out_a, index=False)
print(f"  → appended to {out_a}")


# ── T1-A': Scope 3, T1-Strict, structured_strict ───────────────────────
print("\n=== T1-A' Scope 3 ===")
nz = pd.read_csv(NZDPU, low_memory=False)
DASH = "\u2014"
nz["s3_raw"] = pd.to_numeric(nz["total_s3_emissions_ghg"].replace(DASH, np.nan),
                              errors="coerce")
s3 = (nz.sort_values(["nz_id","reporting_year"])
        .drop_duplicates(subset=["nz_id","reporting_year"], keep="first")
        .set_index(["nz_id","reporting_year"])["s3_raw"])

df3 = bl.set_index(["nz_id","reporting_year"]).copy()
df3["scope3_actual_tco2e"] = s3
df3 = df3.reset_index()
df3 = enrich(df3)
df3 = df3.dropna(subset=["scope3_actual_tco2e","country_iso2"])
df3 = df3[(df3["scope3_actual_tco2e"] >= 100) &
          (df3["scope3_actual_tco2e"] <= 2e9)]
df3["y"] = np.log10(df3["scope3_actual_tco2e"])
df3["subset_t1strict"] = (df3["ticker"].notna() &
                          df3["revenue_musd"].notna() &
                          (df3["revenue_musd"] > 0))
df3 = df3[df3["subset_t1strict"]].copy()

# Stratified 80/10/10 matching run_t1_task_a_prime.py
strat = df3["gics_11"].astype(str) + "|" + df3["country_iso2"].astype(str)
counts = strat.value_counts()
strat = strat.where(~strat.isin(counts[counts < 5].index),
                    df3["gics_11"].astype(str) + "|_tinyRoW")
counts = strat.value_counts()
strat = strat.where(~strat.isin(counts[counts < 5].index), "_global")
idx = df3.index.to_numpy()
idx_tv, idx_test = train_test_split(idx, test_size=0.10,
                                    stratify=strat.loc[idx], random_state=SEED)
strat_tv = strat.loc[idx_tv]
vc = strat_tv.value_counts()
strat_tv_safe = strat_tv.where(strat_tv.isin(set(vc[vc >= 2].index)), "_global")
idx_train, idx_val = train_test_split(idx_tv, test_size=1/9,
                                      stratify=strat_tv_safe, random_state=SEED)
df3["split"] = "train"
df3.loc[idx_val, "split"] = "val"
df3.loc[idx_test, "split"] = "test"

train = df3[df3["split"] == "train"]
test  = df3[df3["split"] == "test"]
X_tr = make_features(train); X_te = make_features(test)
X_tr, X_te = X_tr.align(X_te, join="outer", axis=1, fill_value=0)
m_p, n_tr_p, n_te_p = run_tabpfn(X_tr, train["y"].values, X_te, test["y"].values,
                                  "T1-A'/T1-Strict/structured_strict")

out_p = Path("results/t1a_prime_results.csv")
res_p = pd.read_csv(out_p)
new = {"subset":"T1-Strict","feature_set":"structured_strict","model":"tabpfn",
       "n_train":n_tr_p,"n_test":n_te_p, **m_p}
res_p = pd.concat([res_p, pd.DataFrame([new])], ignore_index=True)
res_p.to_csv(out_p, index=False)
print(f"  → appended to {out_p}")

print("\nDone.")
