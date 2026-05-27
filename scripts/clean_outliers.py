"""
Drop physically implausible Scope 1+2 rows.

Rules (see T1.md Section "Data QC"):
  - Upper cap: 5×10^8 tCO2e. Rationale: Saudi Aramco, the largest single
    corporate emitter on Earth, is ~500 M tCO2e/yr. Anything above is a
    unit-confusion error (kgCO2e mislabeled as tCO2e, Scope 2 double-unit, etc.).
  - Lower cap: 10 tCO2e. A single delivery truck emits >10 tCO2e/yr — any
    reporting organization smaller than that is essentially dormant; most are
    NaN mis-coded as 0 and then bumped up by rounding.

Inputs:
  data/company-level/nzdpu_enriched/factor_baseline_v2.csv
  data/company-level/splits/task_a_split_v2.csv

Outputs (overwritten in place, preFX / postFX versions already backed up):
  same paths, minus outliers
  + data/company-level/nzdpu_enriched/outliers_removed.csv  (audit trail)
"""

import pandas as pd
import numpy as np
from pathlib import Path

V2_PATH    = Path("data/company-level/nzdpu_enriched/factor_baseline_v2.csv")
SPLIT_PATH = Path("data/company-level/splits/task_a_split_v2.csv")
AUDIT_PATH = Path("data/company-level/nzdpu_enriched/outliers_removed.csv")

UPPER = 5e8   # 500 M tCO2e
LOWER = 10.0  # 10 tCO2e

print(f"Loading {V2_PATH.name} ...")
v2 = pd.read_csv(V2_PATH)
n0 = len(v2)
print(f"  {n0:,} rows")

y = v2["scope12_actual_tco2e"]
upper_mask = y > UPPER
lower_mask = (y > 0) & (y < LOWER)
zero_mask  = y == 0
drop_mask  = upper_mask | lower_mask | zero_mask

print(f"\nOutlier counts (before drop):")
print(f"  > {UPPER:.0e} tCO2e : {upper_mask.sum():>4d}")
print(f"  == 0 tCO2e       : {zero_mask.sum():>4d}")
print(f"  (0, {LOWER}) tCO2e : {lower_mask.sum():>4d}")
print(f"  TOTAL to drop    : {drop_mask.sum():>4d}  ({drop_mask.mean():.2%})")

# Audit trail
audit_cols = ["nz_id", "reporting_year", "company_name", "country_iso2",
              "gics_11", "ticker", "scope12_actual_tco2e"]
audit = v2.loc[drop_mask, audit_cols].copy()
audit["outlier_reason"] = np.where(
    v2.loc[drop_mask, "scope12_actual_tco2e"] > UPPER, "too_high",
    np.where(v2.loc[drop_mask, "scope12_actual_tco2e"] == 0, "zero", "too_low")
)
audit.to_csv(AUDIT_PATH, index=False)
print(f"\nAudit trail → {AUDIT_PATH} ({len(audit):,} rows)")

v2_clean = v2.loc[~drop_mask].copy()
print(f"\n{n0:,} → {len(v2_clean):,} rows ({n0 - len(v2_clean)} dropped)")
v2_clean.to_csv(V2_PATH, index=False)
print(f"Overwrote {V2_PATH}")

# ── Apply same drop to split csv ───────────────────────────────────────────
print(f"\nUpdating {SPLIT_PATH.name} ...")
sp = pd.read_csv(SPLIT_PATH)
n_sp0 = len(sp)
keep_keys = set(zip(v2_clean["nz_id"], v2_clean["reporting_year"]))
sp_keep = sp[sp.apply(lambda r: (r["nz_id"], r["reporting_year"]) in keep_keys, axis=1)]
sp_keep.to_csv(SPLIT_PATH, index=False)
print(f"  {n_sp0:,} → {len(sp_keep):,} rows")
print(f"  test size: {(sp_keep['split']=='test').sum():,}  "
      f"(before: {(sp['split']=='test').sum():,})")
print(f"  T1-Strict: {sp_keep['subset_t1strict'].sum():,}  "
      f"(before: {sp['subset_t1strict'].sum():,})")
