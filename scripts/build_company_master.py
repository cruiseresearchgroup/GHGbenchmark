"""
Build unified company-level GHG dataset from all available sources.

Sources:
  1. NZDPU (primary): 33,684 rows, Scope 1/2/3 + C1-C15 categories, multi-year global
  2. Forbes Marginal:  208 rows, large companies with revenue+profit, 2022
  3. danielrosehill:   78 rows, large companies with EBITDA, 2022
  4. ExioNAICS:       20,535 rows, US companies with Value Added+Employment, total GHG, 2022

Schema (unified columns):
  - source, company_name, reporting_year, country, industry_raw, industry_naics2
  - scope1_tco2e, scope2_lb_tco2e, scope3_tco2e, ghg_total_tco2e
  - scope3_c1..c15_tco2e  (NZDPU only)
  - revenue_musd, ebitda_musd, profit_musd, employees, market_cap_musd
  - ticker, lei, data_provider

Output: data/company-level/company_master.csv
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

OUTPUT = Path("data/company-level/company_master.csv")
DASH   = "\u2014"

S3_CATS = [f"total_s3_ghgp_c{i}_emissions_ghg" for i in range(1, 16)]
S3_OUT  = [f"scope3_c{i}_tco2e" for i in range(1, 16)]


def clean_num(s):
    """Convert dash-placeholder and coerce to float."""
    if isinstance(s, pd.Series):
        return pd.to_numeric(s.replace(DASH, np.nan), errors="coerce")
    return np.nan if s == DASH else s


# ── 1. NZDPU (base) ────────────────────────────────────────────────────────
print("Loading NZDPU...")
nz = pd.read_csv("data/company-level/nzdpu/nzdpu_emissions.csv", low_memory=False)

# Clean numeric S3 category columns
for col in S3_CATS:
    if col in nz.columns:
        nz[col] = clean_num(nz[col])

# Load financial data from Yahoo Finance cache (built by enrich_nzdpu_financials.py)
ticker_cache = {}
fin_cache    = {}
tc_path = Path("data/company-level/nzdpu_enriched/ticker_cache.json")
fc_path = Path("data/company-level/nzdpu_enriched/financials_cache.json")
if tc_path.exists():
    with open(tc_path) as f:
        ticker_cache = json.load(f)
if fc_path.exists():
    with open(fc_path) as f:
        fin_cache = json.load(f)

# Build nz_id → financials
fin_rows = []
for nz_id, v in ticker_cache.items():
    ticker = v.get("ticker") if isinstance(v, dict) else v
    if ticker and ticker in fin_cache:
        fd = fin_cache[ticker]
        fin_rows.append({
            "nz_id": int(nz_id),
            "ticker": ticker,
            "revenue_musd":    fd.get("revenue_usd", np.nan) and fd["revenue_usd"] / 1e6 if fd.get("revenue_usd") else np.nan,
            "ebitda_musd":     fd.get("ebitda_usd",  np.nan) and fd["ebitda_usd"]  / 1e6 if fd.get("ebitda_usd")  else np.nan,
            "market_cap_musd": fd.get("market_cap_usd", np.nan) and fd["market_cap_usd"] / 1e6 if fd.get("market_cap_usd") else np.nan,
            "employees":       fd.get("employees"),
            "yf_sector":       fd.get("yf_sector"),
            "yf_industry":     fd.get("yf_industry"),
            "yf_country":      fd.get("country"),
        })

fin_df = pd.DataFrame(fin_rows)
if not fin_df.empty:
    nz = nz.merge(fin_df, on="nz_id", how="left")

nzdpu_block = pd.DataFrame({
    "source":          "nzdpu",
    "company_name":    nz["company_name"],
    "reporting_year":  nz["reporting_year"],
    "country":         nz.get("jurisdiction", pd.Series(dtype=str)),
    "industry_raw":    nz["sics_sector"],
    "industry_naics2": np.nan,
    "scope1_tco2e":    clean_num(nz["total_s1_emissions_ghg"]),
    "scope2_lb_tco2e": clean_num(nz["total_s2_lb_emissions_ghg"]),
    "scope3_tco2e":    clean_num(nz["total_s3_emissions_ghg"]),
    "ghg_total_tco2e": np.nan,
    "revenue_musd":    nz.get("revenue_musd", np.nan),
    "ebitda_musd":     nz.get("ebitda_musd",  np.nan),
    "profit_musd":     np.nan,
    "employees":       nz.get("employees", np.nan),
    "market_cap_musd": nz.get("market_cap_musd", np.nan),
    "ticker":          nz.get("ticker", np.nan),
    "lei":             clean_num(nz.get("legal_entity_identifier", pd.Series(dtype=str))),
    "data_provider":   nz.get("data_provider", np.nan),
    "nz_id":           nz["nz_id"],
})

# Attach S3 category columns
for in_col, out_col in zip(S3_CATS, S3_OUT):
    nzdpu_block[out_col] = nz[in_col].values if in_col in nz.columns else np.nan

print(f"  NZDPU block: {len(nzdpu_block):,} rows")


# ── 2. Forbes Marginal ─────────────────────────────────────────────────────
print("Loading Forbes Marginal...")
fb = pd.read_csv("data/company-level/forbes_marginal/forbes_marginal_2022.csv")

forbes_block = pd.DataFrame({
    "source":          "forbes",
    "company_name":    fb["company_name"],
    "reporting_year":  fb["reporting_year"],
    "country":         np.nan,
    "industry_raw":    fb["industry"],
    "industry_naics2": np.nan,
    "scope1_tco2e":    fb["scope1_tco2e"],
    "scope2_lb_tco2e": fb["scope2_tco2e"],
    "scope3_tco2e":    fb["scope3_tco2e"],
    "ghg_total_tco2e": np.nan,
    "revenue_musd":    fb["revenue_musd"],
    "ebitda_musd":     np.nan,
    "profit_musd":     fb["profit_musd"],
    "employees":       np.nan,
    "market_cap_musd": np.nan,
    "ticker":          np.nan,
    "lei":             np.nan,
    "data_provider":   "Forbes Global 2000",
    "nz_id":           np.nan,
})
for out_col in S3_OUT:
    forbes_block[out_col] = np.nan

# Keep only companies NOT already in NZDPU (avoid duplicates)
nzdpu_names = set(nzdpu_block["company_name"].str.lower().str.strip())
forbes_block = forbes_block[~forbes_block["company_name"].str.lower().str.strip().isin(nzdpu_names)]
print(f"  Forbes block (non-NZDPU): {len(forbes_block):,} rows")


# ── 3. danielrosehill ──────────────────────────────────────────────────────
print("Loading danielrosehill...")
dr = pd.read_csv("data/company-level/danielrosehill/company_data.csv")

# Emissions: "million tonnes CO2e" → tCO2e (* 1e6); some rows have numeric unit already
def dr_emission_to_tco2e(series, unit_col):
    result = pd.to_numeric(series, errors="coerce")
    # multiply by 1e6 where unit is "million tonnes CO2e"
    is_mt = unit_col.astype(str).str.contains("million", case=False, na=False)
    result = result.where(~is_mt, result * 1e6)
    return result

# EBITDA: billions USD/EUR → USD millions (* 1000; ignore EUR/USD difference for now, flag non-USD)
dr["ebitda_musd"] = pd.to_numeric(dr["ebitda_2022"], errors="coerce") * 1000
dr.loc[dr["ebitda_currency"] == "EUR", "ebitda_musd"] *= 1.08  # rough EUR→USD

dr_block = pd.DataFrame({
    "source":          "danielrosehill",
    "company_name":    dr["company_name"],
    "reporting_year":  2022,
    "country":         dr.get("headquarters_country", np.nan),
    "industry_raw":    dr.get("sector", np.nan),
    "industry_naics2": np.nan,
    "scope1_tco2e":    dr_emission_to_tco2e(dr["scope_1_emissions"], dr["emissions_reporting_unit"]),
    "scope2_lb_tco2e": dr_emission_to_tco2e(dr["scope_2_emissions"], dr["emissions_reporting_unit"]),
    "scope3_tco2e":    dr_emission_to_tco2e(dr["scope_3_emissions"], dr["emissions_reporting_unit"]),
    "ghg_total_tco2e": np.nan,
    "revenue_musd":    np.nan,
    "ebitda_musd":     dr["ebitda_musd"],
    "profit_musd":     np.nan,
    "employees":       np.nan,
    "market_cap_musd": np.nan,
    "ticker":          dr.get("stock_ticker", np.nan),
    "lei":             np.nan,
    "data_provider":   "danielrosehill",
    "nz_id":           np.nan,
})
for out_col in S3_OUT:
    dr_block[out_col] = np.nan

dr_block = dr_block[~dr_block["company_name"].str.lower().str.strip().isin(nzdpu_names)]
print(f"  danielrosehill block (non-NZDPU): {len(dr_block):,} rows")


# ── 4. ExioNAICS ───────────────────────────────────────────────────────────
print("Loading ExioNAICS...")
ex = pd.read_csv("data/company-level/exionaics/ExioNAICS.csv")

exio_block = pd.DataFrame({
    "source":          "exionaics",
    "company_name":    ex["Company Name"],
    "reporting_year":  ex["Year"],
    "country":         "United States",
    "industry_raw":    ex["NAICS_2 Title"],
    "industry_naics2": ex["NAICS_2 Code"],
    "scope1_tco2e":    np.nan,
    "scope2_lb_tco2e": np.nan,
    "scope3_tco2e":    np.nan,
    # GHG total: kg CO2eq → tCO2e
    "ghg_total_tco2e": pd.to_numeric(ex["GHG emissions [kg CO2 eq.]"], errors="coerce") / 1000,
    # Value Added M.EUR → USD millions (~1.08 EUR/USD)
    "revenue_musd":    pd.to_numeric(ex["Value Added [M.EUR]"], errors="coerce") * 1.08,
    "ebitda_musd":     np.nan,
    "profit_musd":     np.nan,
    # Employment: 1000 persons → persons
    "employees":       pd.to_numeric(ex["Employment [1000 p.]"], errors="coerce") * 1000,
    "market_cap_musd": np.nan,
    "ticker":          np.nan,
    "lei":             np.nan,
    "data_provider":   "ExioNAICS/Exiobase",
    "nz_id":           np.nan,
})
for out_col in S3_OUT:
    exio_block[out_col] = np.nan

print(f"  ExioNAICS block: {len(exio_block):,} rows")


# ── 5. Concatenate all ────────────────────────────────────────────────────
print("\nConcatenating all sources...")
master = pd.concat([nzdpu_block, forbes_block, dr_block, exio_block],
                   ignore_index=True, sort=False)

master.to_csv(OUTPUT, index=False)
print(f"Saved: {OUTPUT}")
print(f"Shape: {master.shape}")

# ── Coverage report ───────────────────────────────────────────────────────
print("\n=== Coverage by source ===")
for src in ["nzdpu", "forbes", "danielrosehill", "exionaics"]:
    sub = master[master["source"] == src]
    s1  = sub["scope1_tco2e"].notna().sum()
    rev = sub["revenue_musd"].notna().sum()
    emp = sub["employees"].notna().sum()
    ghg = sub["ghg_total_tco2e"].notna().sum()
    print(f"  {src:15s}: {len(sub):6,} rows | scope1={s1:6,} | revenue={rev:6,} | employees={emp:6,} | ghg_total={ghg:6,}")

print("\n=== Global coverage ===")
for col in ["scope1_tco2e","scope2_lb_tco2e","scope3_tco2e","ghg_total_tco2e",
            "revenue_musd","ebitda_musd","employees","market_cap_musd"]:
    n = master[col].notna().sum()
    print(f"  {col:25s}: {n:7,} ({n/len(master):.0%})")

s3_any = master[[c for c in S3_OUT if c in master.columns]].notna().any(axis=1).sum()
print(f"  {'≥1 S3 category':25s}: {s3_any:7,} ({s3_any/len(master):.0%})")

complete = (master["scope1_tco2e"].notna() & master["revenue_musd"].notna()).sum()
print(f"\n  scope1 + revenue (ML-ready): {complete:,} ({complete/len(master):.0%})")
