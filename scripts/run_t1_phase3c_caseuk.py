"""Phase 3c: re-run the UK Financial 121× case study at SICS sub_sector granularity.

Current paper claim (experiments.tex §5.2):
  UK Financial co., $8.5B revenue, GICS-11 'Financials' sector
  ExioML factor (Financials) = 10.7 tCO2e / MUSD
  Predicted = 90,841  tCO2e   (factor × revenue)
  Actual    = 750     tCO2e
  Ratio     = 121×    over-prediction

This script:
  1. Locates the case-study company in the test panel
     (subset_t1strict, country UK, GICS Financials, revenue ≈ 8.5e9, scope12 ≈ 750)
  2. Reads its sics_sub_sector (Banks vs Insurance vs Asset Management vs ...)
  3. Computes a sub_sector-level intensity factor from TRAIN ONLY:
        f_sub = sum(scope12_actual) / sum(revenue_musd)   over train rows in that sub_sector
  4. Re-predicts the company's emissions and reports the new over-prediction ratio.
  5. Also reports the GICS-11 'Financials' factor recomputed the same way for context,
     and the same statistic for the whole UK-Financials test cohort.

Output: results/t1a_phase3c_caseuk.csv
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from t1_common import BASELINE, SPLIT, NZDPU

OUT = Path("results/t1a_phase3c_caseuk.csv")

# ── Load + join SICS hierarchy ─────────────────────────────────────────────
bl = pd.read_csv(BASELINE)
sp = pd.read_csv(SPLIT)
nz = pd.read_csv(NZDPU, low_memory=False,
                 usecols=["nz_id", "reporting_year",
                          "sics_sector", "sics_sub_sector", "sics_industry",
                          "company_name"])
nz = nz.drop_duplicates(["nz_id", "reporting_year"], keep="first")

df = sp.merge(bl[["nz_id", "reporting_year", "revenue_musd",
                  "factor_tco2e_per_musd",
                  "scope12_actual_tco2e", "scope12_pred_tco2e"]],
              on=["nz_id", "reporting_year"], how="left")
df = df.merge(nz, on=["nz_id", "reporting_year"], how="left")

# ── Locate the case study company (paper §5.2: nz_id=11513, year=2018) ───
# Found globally: revenue=8487 MUSD, actual=750.4 tCO2e, ExioML pred=90841 (121.06x)
# Note: this row is in TRAIN split, so the sub_sector aggregation below treats
# it as a population statistic (not an out-of-sample prediction).
target = df[(df["nz_id"] == 11513) & (df["reporting_year"] == 2018)].copy()
target = target.assign(
    pred_actual_ratio=target["scope12_pred_tco2e"] / target["scope12_actual_tco2e"]
)
print(f"\n=== Case-study row (paper §5.2 UK Financial 121x) ===")
print(target[["nz_id", "reporting_year", "split", "company_name",
              "sics_sector", "sics_sub_sector", "sics_industry",
              "revenue_musd", "scope12_actual_tco2e",
              "scope12_pred_tco2e", "pred_actual_ratio"]].to_string(index=False))

case_row = target.iloc[0]
case_subsector = case_row["sics_sub_sector"]
case_industry  = case_row["sics_industry"]
case_revenue   = case_row["revenue_musd"]
case_actual    = case_row["scope12_actual_tco2e"]
case_pred_gics = case_row["scope12_pred_tco2e"]
case_factor_gics = case_row["factor_tco2e_per_musd"]

# ── Compute sub_sector-level intensity factor from TRAIN ONLY ──────────────
train = df[(df["split"] == "train") & df["subset_t1strict"]].copy()
train = train.dropna(subset=["scope12_actual_tco2e", "revenue_musd"])
train = train[train["revenue_musd"] > 0]

def intensity(sub_df):
    """Aggregate (sum scope12 / sum revenue) → tCO2e per MUSD."""
    if len(sub_df) == 0:
        return np.nan, 0
    return sub_df["scope12_actual_tco2e"].sum() / sub_df["revenue_musd"].sum(), len(sub_df)

# Comparison ladder
ladders = {
    "GICS-11 Financials":          train[train["gics_11"] == "Financial Services"],
    f"sics_sector ({case_row['sics_sector']})":   train[train["sics_sector"] == case_row["sics_sector"]],
    f"sics_sub_sector ({case_subsector})":        train[train["sics_sub_sector"] == case_subsector],
    f"sics_industry ({case_industry})":           train[train["sics_industry"] == case_industry],
}

rows = []
print(f"\n=== Intensity factors at increasing granularity ===")
print(f"  {'level':50s} {'factor':>10s} {'n_train':>10s} {'pred':>12s} {'×actual':>10s}")
for label, sub_df in ladders.items():
    f, n = intensity(sub_df)
    pred = f * case_revenue if not np.isnan(f) else np.nan
    ratio = pred / case_actual if not np.isnan(pred) else np.nan
    print(f"  {label:50s} {f:>10.3f} {n:>10d} {pred:>12.0f} {ratio:>9.1f}×")
    rows.append({"granularity": label, "factor": f, "n_train": n,
                 "pred_tco2e": pred, "over_ratio": ratio})

# Also: original GICS-11 ExioML factor (joined column, not retrained)
exio_pred = case_factor_gics * case_revenue
print(f"\n  {'paper baseline (ExioML GICS-11 joined)':50s} {case_factor_gics:>10.3f} "
      f"{'-':>10s} {exio_pred:>12.0f} {exio_pred/case_actual:>9.1f}×")
rows.append({"granularity": "paper baseline (ExioML GICS-11 joined)",
             "factor": case_factor_gics, "n_train": np.nan,
             "pred_tco2e": exio_pred, "over_ratio": exio_pred / case_actual})

# Cohort-level: how does sub_sector-level factor do across ALL UK Financial Services rows in the panel?
uk_fin = df[(df["country_iso2"] == "GB") & (df["gics_11"] == "Financial Services")
            & df["subset_t1strict"]].copy()
uk_fin = uk_fin.dropna(subset=["scope12_actual_tco2e", "revenue_musd"])
uk_fin = uk_fin[uk_fin["revenue_musd"] > 0]
print(f"\n=== Cohort: UK Financials in test ({len(uk_fin)} rows) ===")
for label, sub_df in ladders.items():
    f, n = intensity(sub_df)
    if np.isnan(f):
        continue
    preds = f * uk_fin["revenue_musd"]
    log_ratio_med = np.median(np.log10(preds / uk_fin["scope12_actual_tco2e"]))
    abs_ratio_geomean = 10 ** np.mean(np.abs(np.log10(preds / uk_fin["scope12_actual_tco2e"])))
    print(f"  {label:50s}  median log10(pred/actual)={log_ratio_med:+.2f}  "
          f"geo-mean |×|={abs_ratio_geomean:.1f}×")

OUT.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"\n→ {OUT}")
