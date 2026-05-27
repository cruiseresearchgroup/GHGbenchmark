"""Shared data loader / feature pipeline / metrics for T1 extended experiments.

Keeps the exact same preprocessing as run_t1_task_a.py so that new experiments
are comparable to the already-logged main results.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "data/company-level/nzdpu_enriched/factor_baseline_v2.csv"
SPLIT    = ROOT / "data/company-level/splits/task_a_split_v2.csv"
NZDPU    = ROOT / "data/company-level/nzdpu/nzdpu_emissions.csv"
TICKER_CACHE = ROOT / "data/company-level/nzdpu_enriched/ticker_cache.json"
FIN_CACHE    = ROOT / "data/company-level/nzdpu_enriched/financials_cache.json"
SEED = 42

CATEGORICAL    = ["gics_11", "country_iso2"]
# Wide-available numerics: do NOT require yfinance match. reporting_year
# captures panel-effect (2018-2023 disclosure-standard / energy-mix drift);
# factor_tco2e_per_musd is the ExioML (sector × exio_region × year) intensity
# lookup — it depends only on sector + region + year, no yfinance needed.
NUMERIC_WIDE   = ["reporting_year", "factor_tco2e_per_musd"]
# Strict-only numerics: require yfinance ticker match.
NUMERIC_STRICT_ONLY = ["revenue_musd", "employees", "market_cap_musd", "ebitda_musd"]
NUMERIC_STRICT = NUMERIC_WIDE + NUMERIC_STRICT_ONLY
# Numeric features that already live on a sensible scale and should NOT be
# log1p-transformed: factor (tCO2e per MUSD, naturally O(1-10^3)) and year.
NUMERIC_NO_LOG = {"factor_tco2e_per_musd", "reporting_year"}

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


def _load_caches():
    with open(TICKER_CACHE) as f:
        tc = json.load(f)
    with open(FIN_CACHE) as f:
        fc = json.load(f)
    nz_to_ticker = {int(k): v["ticker"] for k, v in tc.items()
                    if isinstance(v, dict) and v.get("ticker")}
    return nz_to_ticker, fc


def load_scope12(enrich_financials: bool = True) -> pd.DataFrame:
    """Returns joined baseline + split DataFrame with y=log10(scope1+2) and enriched features."""
    bl = pd.read_csv(BASELINE)
    sp = pd.read_csv(SPLIT)
    nz_to_ticker, fc = _load_caches()

    df = sp.merge(bl[["nz_id","reporting_year","ticker","revenue_musd",
                      "revenue_musd_ty","ebitda_musd_ty","net_income_musd_ty",
                      "factor_tco2e_per_musd",
                      "scope12_actual_tco2e","scope12_pred_tco2e",
                      "business_summary"]],
                  on=["nz_id","reporting_year"], how="left")

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
    df["revenue_best_musd"] = df["revenue_musd_ty"].fillna(df["revenue_musd"])
    df["ebitda_best_musd"]  = df["ebitda_musd_ty"].fillna(df["ebitda_musd"])

    df["y"] = np.log10(df["scope12_actual_tco2e"])
    df["gics_11"]      = df["gics_11"].fillna("Unknown")
    df["country_iso2"] = df["country_iso2"].fillna("XX")
    return df


def load_scope_split(which: str = "s1") -> pd.DataFrame:
    """Load a specific scope target from raw NZDPU joined with baseline features.
    which ∈ {'s1','s2','s12'}  (s2 uses lb preferentially, falls back to mb)."""
    bl = pd.read_csv(BASELINE)
    sp = pd.read_csv(SPLIT)
    nz = pd.read_csv(NZDPU, low_memory=False)
    DASH = "\u2014"
    def col(c):
        return pd.to_numeric(nz[c].replace(DASH, np.nan), errors="coerce")
    s1  = col("total_s1_emissions_ghg")
    s2  = col("total_s2_lb_emissions_ghg").fillna(col("total_s2_mb_emissions_ghg"))
    if which == "s1":
        tgt = s1
    elif which == "s2":
        tgt = s2
    elif which == "s12":
        tgt = s1.fillna(0) + s2.fillna(0)
        tgt = tgt.where((s1.notna()) & (s2.notna()))
    else:
        raise ValueError(which)
    nz_clean = nz[["nz_id","reporting_year"]].copy()
    nz_clean["target"] = tgt
    nz_clean = (nz_clean.dropna(subset=["target"])
                        .drop_duplicates(["nz_id","reporting_year"], keep="first"))
    # Outlier filter
    nz_clean = nz_clean[(nz_clean["target"] >= 1) & (nz_clean["target"] <= 5e8)]
    df = load_scope12(enrich_financials=True)
    df = df.drop(columns=["y"]).merge(nz_clean, on=["nz_id","reporting_year"], how="inner")
    df["y"] = np.log10(df["target"])
    return df


def make_features(df_part: pd.DataFrame, feature_set: str = "structured_strict") -> pd.DataFrame:
    X_cat = pd.get_dummies(df_part[CATEGORICAL], drop_first=False, dtype=float)
    if feature_set == "structured_wide":
        cols = NUMERIC_WIDE
    elif feature_set == "structured_strict":
        cols = NUMERIC_STRICT
    else:
        raise ValueError(f"unknown feature_set: {feature_set}")
    X_num = df_part[cols].copy()
    for c in cols:
        med = X_num[c].median()
        X_num[c] = X_num[c].fillna(med)
        if c not in NUMERIC_NO_LOG:
            X_num[c] = np.log1p(X_num[c].clip(lower=0))
    return pd.concat([X_cat, X_num], axis=1)


def metrics(y_true_log, y_pred_log) -> dict:
    y_true_log = np.asarray(y_true_log); y_pred_log = np.asarray(y_pred_log)
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


def align(Xa, Xb):
    """outer-align two feature frames, filling with 0."""
    return Xa.align(Xb, join="outer", axis=1, fill_value=0)
