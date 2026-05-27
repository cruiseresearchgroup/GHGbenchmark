"""
Build stratified train/val/test splits for T1-A (in-distribution regression).

Design:
  - Stratify T1-Wide (~33 K rows) by (gics_11 × country_iso2)
  - 80 / 10 / 10 random split, deterministic (seed=42)
  - T1-Strict (~13 K rows with ticker+revenue) INHERITS the same split labels
    by subsetting → same test rows, apples-to-apples comparison of
    "how much do financials help?"
  - Unit of split = (nz_id, reporting_year)  i.e. one company-year row
    (Later: if we want company-level split to avoid same-company leakage
     across train/test, can swap to GroupShuffleSplit on nz_id.)

Inputs:
  data/company-level/nzdpu_enriched/factor_baseline.csv
    (has nz_id, reporting_year, gics_11, country_iso2, ticker, revenue_musd,
     scope12_actual_tco2e)

Output:
  data/company-level/splits/task_a_split.csv
    columns: nz_id, reporting_year, split, subset_t1strict (bool)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

IN   = Path("data/company-level/nzdpu_enriched/factor_baseline.csv")
OUT  = Path("data/company-level/splits/task_a_split.csv")
SEED = 42


df = pd.read_csv(IN)
print(f"Loaded {len(df):,} rows from {IN}")

# ── 1. Eligible for T1-Wide: has target (Scope 1+2 > 0) and country ───────
# NOTE: sector is NOT required — rows with no sector get gics_11="Unknown"
# so ML can still learn from them. Factor baseline is only reported on the
# subset where gics_11 is real (the baseline-evaluable subset).
wide = df.dropna(subset=["scope12_actual_tco2e", "country_iso2"]).copy()
wide = wide[wide["scope12_actual_tco2e"] > 0]
wide["gics_11"] = wide["gics_11"].fillna("Unknown")
wide["has_factor_baseline"] = wide["factor_tco2e_per_musd"].notna()
print(f"T1-Wide eligible (target>0, country): {len(wide):,}")
print(f"  of which factor-baseline-evaluable: {wide['has_factor_baseline'].sum():,}")

# ── 2. Stratify by (gics_11, country_iso2) ──────────────────────────────────
# Build stratum label; collapse tiny strata (n<5) to a country-level fallback
strat = wide["gics_11"].astype(str) + "|" + wide["country_iso2"].astype(str)
counts = strat.value_counts()
tiny = set(counts[counts < 5].index)
print(f"  tiny strata (n<5): {len(tiny)} — fallback to gics_11 only")
strat_final = strat.where(~strat.isin(tiny), wide["gics_11"].astype(str) + "|_tinyRoW")
# If even that is too small, fallback to global
counts2 = strat_final.value_counts()
tiny2 = set(counts2[counts2 < 5].index)
strat_final = strat_final.where(~strat_final.isin(tiny2), "_global")
print(f"  final strata: {strat_final.nunique()}")

# ── 3. 80/10/10 split (train/val/test) ──────────────────────────────────────
idx = wide.index.to_numpy()

# Step 1: test = 10%
idx_tv, idx_test = train_test_split(
    idx, test_size=0.10, stratify=strat_final.loc[idx], random_state=SEED
)
# Step 2: of remaining 90%, take 1/9 = 10% of total as val
strat_tv = strat_final.loc[idx_tv]
# Filter out any strata that dropped below 2 members after test split
vc = strat_tv.value_counts()
safe = set(vc[vc >= 2].index)
strat_tv_safe = strat_tv.where(strat_tv.isin(safe), "_global")
idx_train, idx_val = train_test_split(
    idx_tv, test_size=1/9, stratify=strat_tv_safe, random_state=SEED
)

split_map = {}
for i in idx_train: split_map[i] = "train"
for i in idx_val:   split_map[i] = "val"
for i in idx_test:  split_map[i] = "test"
wide["split"] = wide.index.map(split_map)

# ── 4. T1-Strict flag (ticker + revenue_musd > 0) ───────────────────────────
wide["subset_t1strict"] = (
    wide["ticker"].notna() & wide["revenue_musd"].notna() & (wide["revenue_musd"] > 0)
)

# ── 5. Save ─────────────────────────────────────────────────────────────────
OUT.parent.mkdir(parents=True, exist_ok=True)
out = wide[["nz_id", "reporting_year", "gics_11", "country_iso2",
            "split", "subset_t1strict", "has_factor_baseline"]].copy()
out.to_csv(OUT, index=False)
print(f"\nSaved → {OUT}")

# ── Report ──────────────────────────────────────────────────────────────────
print("\n=== T1-Wide split breakdown ===")
print(out["split"].value_counts())
print(f"\n=== T1-Strict (ticker+revenue) within each split ===")
strict = out[out["subset_t1strict"]]
print(strict["split"].value_counts())
print(f"\nT1-Strict total: {len(strict):,} ({len(strict)/len(out):.1%} of T1-Wide)")

print("\n=== sector distribution in test set ===")
print(out[out["split"]=="test"]["gics_11"].value_counts())
