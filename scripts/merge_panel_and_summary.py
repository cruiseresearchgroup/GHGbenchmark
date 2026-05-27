"""
P2: Merge time-aligned financials (P1a) and business summaries (P1b) into the
T1 working table.

Inputs:
  data/company-level/nzdpu_enriched/factor_baseline.csv  (NZDPU × row, 33,684)
  data/company-level/nzdpu_enriched/financials_panel.csv (ticker × year)
  data/company-level/nzdpu_enriched/business_summaries.csv
  data/company-level/nzdpu_enriched/ticker_cache.json

Output:
  data/company-level/nzdpu_enriched/factor_baseline_v2.csv
    → factor_baseline.csv with:
      - duplicates on (nz_id, reporting_year) removed
      - revenue_musd_ty / ebitda_musd_ty / net_income_musd_ty (TIME-ALIGNED)
      - financials_timevary (bool: True when panel match found)
      - business_summary (str)

  data/company-level/splits/task_a_split_v2.csv
    → same as task_a_split.csv but regenerated on the deduped table
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

IN_BASE  = Path("data/company-level/nzdpu_enriched/factor_baseline.csv")
IN_PANEL = Path("data/company-level/nzdpu_enriched/financials_panel.csv")
IN_SUMS  = Path("data/company-level/nzdpu_enriched/business_summaries.csv")
IN_TC    = Path("data/company-level/nzdpu_enriched/ticker_cache.json")
OUT_BASE = Path("data/company-level/nzdpu_enriched/factor_baseline_v2.csv")
OUT_SPLIT = Path("data/company-level/splits/task_a_split_v2.csv")
SEED = 42


# ── 1. Load + dedupe factor baseline ──────────────────────────────────────
print("Loading factor_baseline...")
bl = pd.read_csv(IN_BASE)
print(f"  raw: {len(bl):,} rows")
before = len(bl)
bl = bl.sort_values(["nz_id", "reporting_year"]).drop_duplicates(
    subset=["nz_id", "reporting_year"], keep="first"
)
print(f"  deduped: {len(bl):,} rows (dropped {before - len(bl)} dupes)")


# ── 2. Attach ticker (in case not already in bl) ──────────────────────────
with open(IN_TC) as f:
    tc = json.load(f)
nz_to_ticker = {int(k): v["ticker"] for k, v in tc.items()
                if isinstance(v, dict) and v.get("ticker")}
# If factor_baseline.csv already has ticker (it does), use it; else fill
if "ticker" not in bl.columns:
    bl["ticker"] = bl["nz_id"].map(nz_to_ticker)


# ── 3. Time-aligned financials (P1a) ─────────────────────────────────────
print("Loading panel...")
panel = pd.read_csv(IN_PANEL)
# Rename for clarity: _ty = "this year" (time-aligned)
panel = panel.rename(columns={
    "year": "reporting_year",
    "revenue_usd": "revenue_musd_ty",
    "ebitda_usd":  "ebitda_musd_ty",
    "net_income_usd": "net_income_musd_ty",
    "operating_income_usd": "operating_income_musd_ty",
})
for c in ["revenue_musd_ty", "ebitda_musd_ty", "net_income_musd_ty", "operating_income_musd_ty"]:
    panel[c] = panel[c] / 1e6

bl = bl.merge(panel, on=["ticker", "reporting_year"], how="left")
bl["financials_timevary"] = bl["revenue_musd_ty"].notna()
print(f"  rows with time-aligned financials: {bl['financials_timevary'].sum():,} "
      f"({bl['financials_timevary'].mean():.1%})")


# ── 4. Business summaries (P1b) ──────────────────────────────────────────
print("Loading business summaries...")
bs = pd.read_csv(IN_SUMS)
bl = bl.merge(bs, on="ticker", how="left")
print(f"  rows with summary: {bl['business_summary'].notna().sum():,} "
      f"({bl['business_summary'].notna().mean():.1%})")


# ── 5. Save enriched factor_baseline ─────────────────────────────────────
bl.to_csv(OUT_BASE, index=False)
print(f"\n{OUT_BASE} ({len(bl):,} rows, {len(bl.columns)} cols)")


# ── 6. Rebuild splits on deduped table ───────────────────────────────────
print("\nRebuilding splits...")
wide = bl.dropna(subset=["scope12_actual_tco2e", "country_iso2"]).copy()
wide = wide[wide["scope12_actual_tco2e"] > 0]
wide["gics_11"] = wide["gics_11"].fillna("Unknown")
wide["has_factor_baseline"] = wide["factor_tco2e_per_musd"].notna()
print(f"  T1-Wide eligible: {len(wide):,}")

strat = wide["gics_11"].astype(str) + "|" + wide["country_iso2"].astype(str)
counts = strat.value_counts()
strat = strat.where(~strat.isin(counts[counts < 5].index),
                    wide["gics_11"].astype(str) + "|_tinyRoW")
counts = strat.value_counts()
strat = strat.where(~strat.isin(counts[counts < 5].index), "_global")

idx = wide.index.to_numpy()
idx_tv, idx_test = train_test_split(idx, test_size=0.10,
                                    stratify=strat.loc[idx], random_state=SEED)
strat_tv = strat.loc[idx_tv]
vc = strat_tv.value_counts()
strat_tv_safe = strat_tv.where(strat_tv.isin(set(vc[vc >= 2].index)), "_global")
idx_train, idx_val = train_test_split(idx_tv, test_size=1/9,
                                      stratify=strat_tv_safe, random_state=SEED)

sm = {i: "train" for i in idx_train}
sm.update({i: "val" for i in idx_val})
sm.update({i: "test" for i in idx_test})
wide["split"] = wide.index.map(sm)

# T1-Strict = has ticker AND has revenue (snapshot or time-aligned)
wide["revenue_best_musd"] = wide["revenue_musd_ty"].fillna(wide["revenue_musd"])
wide["subset_t1strict"] = (
    wide["ticker"].notna() &
    wide["revenue_best_musd"].notna() &
    (wide["revenue_best_musd"] > 0)
)

OUT_SPLIT.parent.mkdir(parents=True, exist_ok=True)
out = wide[["nz_id", "reporting_year", "gics_11", "country_iso2",
            "split", "subset_t1strict", "has_factor_baseline",
            "financials_timevary"]].copy()
out.to_csv(OUT_SPLIT, index=False)

print("\n=== T1-Wide split breakdown ===")
print(out["split"].value_counts().to_string())
print(f"\nT1-Strict total: {out['subset_t1strict'].sum():,}")
print(f"  of which time-aligned (panel match): "
      f"{(out['subset_t1strict'] & out['financials_timevary']).sum():,}")
print(f"Factor-baseline-evaluable: {out['has_factor_baseline'].sum():,}")
