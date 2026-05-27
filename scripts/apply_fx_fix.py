"""
Fix the currency bug: yfinance `totalRevenue`, `ebitda`, `marketCap`, and the
panel fields from `.income_stmt` are in the company's *local reporting currency*
(e.g. JPY, EUR, KRW), not USD. Our earlier code just divided by 1e6 and labeled
the column `_musd`, so Japanese companies were stored as 150× too large.

This script:
  1. Reads `financials_cache.json` for ticker → currency map.
  2. Applies FX to the snapshot and panel fields inside
     - factor_baseline.csv            (revenue_musd, scope12_pred_tco2e)
     - factor_baseline_v2.csv         (revenue_musd + 4 panel cols + pred)
  3. Adds a `currency` column for traceability.

Tickers with unknown currency → financial columns set to NaN (treated as
missing downstream). This drops ~8% of T1-Strict rows but keeps the rest clean.

Usage:  python scripts/apply_fx_fix.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

FC_PATH   = Path("data/company-level/nzdpu_enriched/financials_cache.json")
EXTRA_PATH = Path("data/company-level/nzdpu_enriched/extra_cache.json")
V1_PATH   = Path("data/company-level/nzdpu_enriched/factor_baseline.csv")
V2_PATH   = Path("data/company-level/nzdpu_enriched/factor_baseline_v2.csv")

# Mid-2024 spot rates, LOCAL_PER_USD. For a benchmark baseline this is OK;
# year-specific rates would tighten pre-2020 data marginally.
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


def build_ticker_currency_map():
    with open(FC_PATH) as f:
        fc = json.load(f)
    m = {}
    missing = 0
    unknown = set()
    for t, v in fc.items():
        if not isinstance(v, dict):
            continue
        cur = v.get("currency")
        if not cur:
            missing += 1
            continue
        cur = str(cur).upper()
        if cur not in FX_PER_USD:
            unknown.add(cur)
            continue
        m[t] = cur
    print(f"  ticker → currency map: {len(m):,} / {len(fc):,} "
          f"({missing} missing field, {len(unknown)} unsupported)")
    if unknown:
        print(f"  unsupported currencies (dropped): {sorted(unknown)}")
    return m


def fx_correct(df, ticker_col, cols_local, ticker_to_cur):
    """Divide each local-currency _musd column by its FX rate. In-place."""
    cur = df[ticker_col].map(ticker_to_cur)
    rate = cur.map(FX_PER_USD)  # NaN where currency unknown
    for c in cols_local:
        if c in df.columns:
            df[c] = df[c] / rate
    df["currency"] = cur
    return df


# ─────────────────────────────────────────────────────────────────────────────
print("Building ticker → currency map...")
tk_cur = build_ticker_currency_map()

# ── V1: factor_baseline.csv ──────────────────────────────────────────────
print(f"\nFixing {V1_PATH.name} ...")
v1 = pd.read_csv(V1_PATH)
before = v1["revenue_musd"].describe()[["50%", "max"]]
v1 = fx_correct(v1, "ticker", ["revenue_musd"], tk_cur)
# recompute factor prediction on corrected revenue
v1["scope12_pred_tco2e"] = v1["revenue_musd"] * v1["factor_tco2e_per_musd"]
after = v1["revenue_musd"].describe()[["50%", "max"]]
print(f"  revenue_musd: median {before['50%']:>13,.0f} → {after['50%']:>13,.0f}  "
      f"max {before['max']:>13,.0f} → {after['max']:>13,.0f}")
v1.to_csv(V1_PATH, index=False)
print(f"  {len(v1):,} rows → {V1_PATH}")

# ── V2: factor_baseline_v2.csv ───────────────────────────────────────────
print(f"\nFixing {V2_PATH.name} ...")
v2 = pd.read_csv(V2_PATH)
panel_cols = ["revenue_musd", "revenue_musd_ty", "ebitda_musd_ty",
              "net_income_musd_ty", "operating_income_musd_ty"]
before = v2["revenue_musd"].describe()[["50%", "max"]]
v2 = fx_correct(v2, "ticker", panel_cols, tk_cur)
v2["scope12_pred_tco2e"] = v2["revenue_musd"] * v2["factor_tco2e_per_musd"]
after = v2["revenue_musd"].describe()[["50%", "max"]]
print(f"  revenue_musd: median {before['50%']:>13,.0f} → {after['50%']:>13,.0f}  "
      f"max {before['max']:>13,.0f} → {after['max']:>13,.0f}")

# Sanity: per-currency medians after correction
print("\n  Median revenue_musd by currency (should be similar across groups):")
med = v2.groupby("currency")["revenue_musd"].median().sort_values(ascending=False)
for cur, m in med.head(10).items():
    n = (v2["currency"] == cur).sum()
    print(f"    {cur:4s}  median={m:>10,.0f} MUSD   n={n:,}")

v2.to_csv(V2_PATH, index=False)
print(f"\n  {len(v2):,} rows × {len(v2.columns)} cols → {V2_PATH}")

# ── Spot check ───────────────────────────────────────────────────────────
print("\n=== Spot check: known non-USD companies ===")
for tk, name in [("7911.T", "Toppan (JPY)"),
                 ("005930.KS", "Samsung (KRW)"),
                 ("AAPL", "Apple (USD)"),
                 ("SAP", "SAP (EUR)")]:
    row = v2[v2["ticker"] == tk].head(1)
    if len(row):
        r = row.iloc[0]
        print(f"  {name:20s}  revenue_musd = {r['revenue_musd']:>12,.0f}  "
              f"currency={r['currency']}  pred/actual ratio = "
              f"{r['scope12_pred_tco2e']/max(r['scope12_actual_tco2e'],1):.2f}")
