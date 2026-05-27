"""
Build an industry-factor baseline estimate for NZDPU companies.

Approach:
  1. Aggregate ExioML (162 Exiobase sectors × 49 regions × 28 years) up to
     GICS-11 sectors, weighted by Value Added → (iso2, gics_11, year) table
     of GHG intensity per EUR of value added.
  2. Map NZDPU country → ExioML region (iso2 + Rest-of-World buckets).
  3. Map NZDPU sics_sector (11 SASB) and yf_sector (11 GICS) → GICS-11.
     Use sics primary, yf as fallback.
  4. For each NZDPU row: baseline_scope12 = revenue × factor.

Outputs:
  data/company-level/exioml/factor_lookup_gics11.csv
    → (iso2, gics_11, year, ghg_per_musd_eur, n_exio_sectors, total_va_meur)
  data/company-level/nzdpu_enriched/factor_baseline.csv
    → (nz_id, reporting_year, sector_resolved, sector_source,
       factor_ghg_per_musd_eur, scope12_pred_tco2e)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

EXIO_PATH    = Path("data/company-level/exioml/ExioML_factor_accounting_IxI.csv")
NZDPU_PATH   = Path("data/company-level/nzdpu/nzdpu_emissions.csv")
MASTER_PATH  = Path("data/company-level/company_master.csv")
TC_PATH      = Path("data/company-level/nzdpu_enriched/ticker_cache.json")
FC_PATH      = Path("data/company-level/nzdpu_enriched/financials_cache.json")
OUT_LOOKUP   = Path("data/company-level/exioml/factor_lookup_gics11.csv")
OUT_BASELINE = Path("data/company-level/nzdpu_enriched/factor_baseline.csv")


# ── Crosswalks ────────────────────────────────────────────────────────────────

# Exiobase 162 sectors → GICS-11
# Sources: NACE codes embedded in sector names + Exiobase supplementary docs.
# Where Exiobase is more granular than GICS, we pick the dominant GICS bucket.
EXIO_TO_GICS = {
    # ── Basic Materials (raw inputs, metals, chemicals, paper/wood) ──────────
    "Aluminium production": "Basic Materials",
    "Casting of metals": "Basic Materials",
    "Chemicals nec": "Basic Materials",
    "Copper production": "Basic Materials",
    "Lead, zinc and tin production": "Basic Materials",
    "Manufacture of basic iron and steel and of ferro-alloys and first products thereof": "Basic Materials",
    "Manufacture of bricks, tiles and construction products, in baked clay": "Basic Materials",
    "Manufacture of cement, lime and plaster": "Basic Materials",
    "Manufacture of ceramic goods": "Basic Materials",
    "Manufacture of coke oven products": "Basic Materials",
    "Manufacture of glass and glass products": "Basic Materials",
    "Manufacture of other non-metallic mineral products n.e.c.": "Basic Materials",
    "Manufacture of rubber and plastic products (25)": "Basic Materials",
    "Manufacture of wood and of products of wood and cork, except furniture; manufacture of articles of straw and plaiting materials (20)": "Basic Materials",
    "Mining of aluminium ores and concentrates": "Basic Materials",
    "Mining of chemical and fertilizer minerals, production of salt, other mining and quarrying n.e.c.": "Basic Materials",
    "Mining of copper ores and concentrates": "Basic Materials",
    "Mining of iron ores": "Basic Materials",
    "Mining of lead, zinc and tin ores and concentrates": "Basic Materials",
    "Mining of nickel ores and concentrates": "Basic Materials",
    "Mining of other non-ferrous metal ores and concentrates": "Basic Materials",
    "Mining of precious metal ores and concentrates": "Basic Materials",
    "N-fertiliser": "Basic Materials",
    "Other non-ferrous metal production": "Basic Materials",
    "P- and other fertiliser": "Basic Materials",
    "Paper": "Basic Materials",
    "Plastics, basic": "Basic Materials",
    "Precious metals production": "Basic Materials",
    "Pulp": "Basic Materials",
    "Quarrying of sand and clay": "Basic Materials",
    "Quarrying of stone": "Basic Materials",
    "Re-processing of ash into clinker": "Basic Materials",
    "Re-processing of secondary aluminium into new aluminium": "Basic Materials",
    "Re-processing of secondary construction material into aggregates": "Basic Materials",
    "Re-processing of secondary copper into new copper": "Basic Materials",
    "Re-processing of secondary glass into new glass": "Basic Materials",
    "Re-processing of secondary lead into new lead, zinc and tin": "Basic Materials",
    "Re-processing of secondary other non-ferrous metals into new other non-ferrous metals": "Basic Materials",
    "Re-processing of secondary paper into new pulp": "Basic Materials",
    "Re-processing of secondary plastic into new plastic": "Basic Materials",
    "Re-processing of secondary preciuos metals into new preciuos metals": "Basic Materials",
    "Re-processing of secondary steel into new steel": "Basic Materials",
    "Re-processing of secondary wood material into new wood material": "Basic Materials",
    "Forestry, logging and related service activities (02)": "Basic Materials",

    # ── Energy (oil/gas/coal extraction, refining, fuel retail) ──────────────
    "Extraction of crude petroleum and services related to crude oil extraction, excluding surveying": "Energy",
    "Extraction of natural gas and services related to natural gas extraction, excluding surveying": "Energy",
    "Extraction, liquefaction, and regasification of other petroleum and gaseous materials": "Energy",
    "Mining of coal and lignite; extraction of peat (10)": "Energy",
    "Mining of uranium and thorium ores (12)": "Energy",
    "Petroleum Refinery": "Energy",
    "Processing of nuclear fuel": "Energy",
    "Retail sale of automotive fuel": "Energy",

    # ── Utilities (electricity, water, gas distribution, steam) ──────────────
    "Collection, purification and distribution of water (41)": "Utilities",
    "Distribution and trade of electricity": "Utilities",
    "Manufacture of gas; distribution of gaseous fuels through mains": "Utilities",
    "Production of electricity by Geothermal": "Utilities",
    "Production of electricity by biomass and waste": "Utilities",
    "Production of electricity by coal": "Utilities",
    "Production of electricity by gas": "Utilities",
    "Production of electricity by hydro": "Utilities",
    "Production of electricity by nuclear": "Utilities",
    "Production of electricity by petroleum and other oil derivatives": "Utilities",
    "Production of electricity by solar photovoltaic": "Utilities",
    "Production of electricity by solar thermal": "Utilities",
    "Production of electricity by tide, wave, ocean": "Utilities",
    "Production of electricity by wind": "Utilities",
    "Production of electricity nec": "Utilities",
    "Steam and hot water supply": "Utilities",
    "Transmission of electricity": "Utilities",

    # ── Consumer Defensive (food, beverage, tobacco, agriculture) ────────────
    "Animal products nec": "Consumer Defensive",
    "Cattle farming": "Consumer Defensive",
    "Cultivation of cereal grains nec": "Consumer Defensive",
    "Cultivation of crops nec": "Consumer Defensive",
    "Cultivation of oil seeds": "Consumer Defensive",
    "Cultivation of paddy rice": "Consumer Defensive",
    "Cultivation of plant-based fibers": "Consumer Defensive",
    "Cultivation of sugar cane, sugar beet": "Consumer Defensive",
    "Cultivation of vegetables, fruit, nuts": "Consumer Defensive",
    "Cultivation of wheat": "Consumer Defensive",
    "Fishing, operating of fish hatcheries and fish farms; service activities incidental to fishing (05)": "Consumer Defensive",
    "Manufacture of beverages": "Consumer Defensive",
    "Manufacture of fish products": "Consumer Defensive",
    "Manufacture of tobacco products (16)": "Consumer Defensive",
    "Manure treatment (biogas), storage and land application": "Consumer Defensive",
    "Manure treatment (conventional), storage and land application": "Consumer Defensive",
    "Meat animals nec": "Consumer Defensive",
    "Pigs farming": "Consumer Defensive",
    "Poultry farming": "Consumer Defensive",
    "Processed rice": "Consumer Defensive",
    "Processing of Food products nec": "Consumer Defensive",
    "Processing of dairy products": "Consumer Defensive",
    "Processing of meat cattle": "Consumer Defensive",
    "Processing of meat pigs": "Consumer Defensive",
    "Processing of meat poultry": "Consumer Defensive",
    "Processing vegetable oils and fats": "Consumer Defensive",
    "Production of meat products nec": "Consumer Defensive",
    "Raw milk": "Consumer Defensive",
    "Sugar refining": "Consumer Defensive",
    "Wool, silk-worm cocoons": "Consumer Defensive",

    # ── Consumer Cyclical (apparel, autos, hotels, retail, recreation) ───────
    "Hotels and restaurants (55)": "Consumer Cyclical",
    "Manufacture of furniture; manufacturing n.e.c. (36)": "Consumer Cyclical",
    "Manufacture of motor vehicles, trailers and semi-trailers (34)": "Consumer Cyclical",
    "Manufacture of textiles (17)": "Consumer Cyclical",
    "Manufacture of wearing apparel; dressing and dyeing of fur (18)": "Consumer Cyclical",
    "Recreational, cultural and sporting activities (92)": "Consumer Cyclical",
    "Renting of machinery and equipment without operator and of personal and household goods (71)": "Consumer Cyclical",
    "Retail trade, except of motor vehicles and motorcycles; repair of personal and household goods (52)": "Consumer Cyclical",
    "Sale, maintenance, repair of motor vehicles, motor vehicles parts, motorcycles, motor cycles parts and accessoiries": "Consumer Cyclical",
    "Tanning and dressing of leather; manufacture of luggage, handbags, saddlery, harness and footwear (19)": "Consumer Cyclical",
    "Wholesale trade and commission trade, except of motor vehicles and motorcycles (51)": "Consumer Cyclical",

    # ── Industrials (construction, machinery, transport, waste, business svc) ─
    "Air transport (62)": "Industrials",
    "Construction (45)": "Industrials",
    "Inland water transport": "Industrials",
    "Manufacture of electrical machinery and apparatus n.e.c. (31)": "Industrials",
    "Manufacture of fabricated metal products, except machinery and equipment (28)": "Industrials",
    "Manufacture of machinery and equipment n.e.c. (29)": "Industrials",
    "Manufacture of other transport equipment (35)": "Industrials",
    "Other business activities (74)": "Industrials",
    "Other land transport": "Industrials",
    "Publishing, printing and reproduction of recorded media (22)": "Industrials",
    "Sea and coastal water transport": "Industrials",
    "Supporting and auxiliary transport activities; activities of travel agencies (63)": "Industrials",
    "Transport via pipelines": "Industrials",
    "Transport via railways": "Industrials",
    "Biogasification of food waste, incl. land application": "Industrials",
    "Biogasification of paper, incl. land application": "Industrials",
    "Biogasification of sewage slugde, incl. land application": "Industrials",
    "Composting of food waste, incl. land application": "Industrials",
    "Composting of paper and wood, incl. land application": "Industrials",
    "Incineration of waste: Food": "Industrials",
    "Incineration of waste: Metals and Inert materials": "Industrials",
    "Incineration of waste: Oil/Hazardous waste": "Industrials",
    "Incineration of waste: Paper": "Industrials",
    "Incineration of waste: Plastic": "Industrials",
    "Incineration of waste: Textiles": "Industrials",
    "Incineration of waste: Wood": "Industrials",
    "Landfill of waste: Food": "Industrials",
    "Landfill of waste: Inert/metal/hazardous": "Industrials",
    "Landfill of waste: Paper": "Industrials",
    "Landfill of waste: Plastic": "Industrials",
    "Landfill of waste: Textiles": "Industrials",
    "Landfill of waste: Wood": "Industrials",
    "Recycling of bottles by direct reuse": "Industrials",
    "Recycling of waste and scrap": "Industrials",
    "Waste water treatment, food": "Industrials",
    "Waste water treatment, other": "Industrials",

    # ── Technology (computers, instruments, office machinery, comms equip) ───
    "Computer and related activities (72)": "Technology",
    "Manufacture of medical, precision and optical instruments, watches and clocks (33)": "Technology",
    "Manufacture of office machinery and computers (30)": "Technology",
    "Manufacture of radio, television and communication equipment and apparatus (32)": "Technology",
    "Research and development (73)": "Technology",

    # ── Communication Services (telecom, post, membership orgs) ──────────────
    "Post and telecommunications (64)": "Communication Services",
    "Activities of membership organisation n.e.c. (91)": "Communication Services",

    # ── Financial Services (banking, insurance, auxiliary finance) ───────────
    "Activities auxiliary to financial intermediation (67)": "Financial Services",
    "Financial intermediation, except insurance and pension funding (65)": "Financial Services",
    "Insurance and pension funding, except compulsory social security (66)": "Financial Services",

    # ── Real Estate ──────────────────────────────────────────────────────────
    "Real estate activities (70)": "Real Estate",

    # ── Healthcare ───────────────────────────────────────────────────────────
    "Health and social work (85)": "Healthcare",

    # ── Other (public admin, education, households) — excluded from baseline ─
    "Education (80)": "Other",
    "Other service activities (93)": "Other",
    "Private households with employed persons (95)": "Other",
    "Public administration and defence; compulsory social security (75)": "Other",
}

# SICS sics_sector (SASB 11) → GICS-11
SICS_TO_GICS = {
    "Consumer Goods":                           "Consumer Cyclical",
    "Extractives & Minerals Processing":        "Energy",        # mostly oil/gas/mining
    "Financials":                               "Financial Services",
    "Food & Beverage":                           "Consumer Defensive",
    "Health Care":                               "Healthcare",
    "Infrastructure":                           "Industrials",   # EPC, real estate, electric utilities mix
    "Renewable Resources & Alternative Energy": "Utilities",
    "Resource Transformation":                  "Industrials",   # chemicals, industrial machinery
    "Services":                                 "Industrials",
    "Technology & Communications":              "Technology",
    "Transportation":                           "Industrials",
    "Information Not Available":                None,
}

# ExioML region non-ISO aggregates (Rest-of-World buckets)
# Used to match NZDPU rows whose iso2 is NOT one of the 44 ISO-coded regions.
ROW_REGION_MAP = {
    # These 5 are aggregates; we distribute by geography.
    # See Exiobase 3 documentation for WA/WE/WF/WL/WM definitions.
    "Asia-Pacific": "WA",  # South/SE Asia, Oceania excl AU/JP/KR/CN/TW/ID/IN
    "Europe":       "WE",  # non-EU Europe
    "Africa":       "WF",
    "Americas":     "WL",
    "Middle East":  "WM",
}

# iso2 → RoW bucket for the 60 countries NOT in the 44 ISO-coded ExioML regions
ISO_TO_ROW = {
    # Asia-Pacific (WA)
    "TH":"WA","SG":"WA","MY":"WA","NZ":"WA","VN":"WA","PH":"WA","BD":"WA","LK":"WA",
    "KH":"WA","PK":"WA","MN":"WA","FJ":"WA","MH":"WA","AF":"WA",
    # Europe (WE)
    "IS":"WE","RS":"WE","UA":"WE","BY":"WE","MK":"WE","SM":"WE","MC":"WE","LI":"WE",
    # Africa (WF)
    "EG":"WF","NG":"WF","KE":"WF","MU":"WF","TN":"WF","MA":"WF","MZ":"WF","MG":"WF",
    "CM":"WF","GQ":"WF","LY":"WF",
    # Americas (WL)
    "CO":"WL","AR":"WL","CL":"WL","PE":"WL","EC":"WL","CR":"WL","GT":"WL","PA":"WL",
    "UY":"WL","DO":"WL","HN":"WL","BO":"WL","VE":"WL","GY":"WL","PY":"WL","TT":"WL",
    # Middle East (WM)
    "AE":"WM","SA":"WM","IL":"WM","KW":"WM","QA":"WM","JO":"WM","BH":"WM","OM":"WM",
    "IQ":"WM","AZ":"WM","KZ":"WM",  # KZ is Central Asia but Exiobase routes it with WM
}


def map_region(iso2):
    if pd.isna(iso2) or iso2 is None:
        return None
    if iso2 in EXIO_REGIONS:
        return iso2
    return ISO_TO_ROW.get(iso2)


def resolve_gics(row):
    """Primary: sics_sector. Fallback: yf_sector. Return (gics_11, source)."""
    sics = row.get("sics_sector")
    if sics and sics != "Information Not Available":
        gics = SICS_TO_GICS.get(sics)
        if gics:
            return gics, "sics"
    yf = row.get("yf_sector")
    if isinstance(yf, str) and yf.strip():
        return yf.strip(), "yf"
    return None, None


# ── 1. Load ExioML, aggregate to GICS-11 × region × year ─────────────────────
print("Loading ExioML...")
exio = pd.read_csv(EXIO_PATH)
EXIO_REGIONS = set(exio["region"].unique())
print(f"  {len(exio):,} rows, {len(EXIO_REGIONS)} regions, {exio['sector'].nunique()} sectors")

# Verify crosswalk covers all sectors
exio_sectors = set(exio["sector"].unique())
missing = exio_sectors - set(EXIO_TO_GICS)
if missing:
    print(f"  ⚠ Exio sectors NOT in crosswalk (will be 'Other'): {sorted(missing)}")
    for s in missing:
        EXIO_TO_GICS[s] = "Other"

exio["gics_11"] = exio["sector"].map(EXIO_TO_GICS)

# Aggregate GHG and Value Added by (region, gics_11, year)
agg = (exio.groupby(["region", "gics_11", "Year"], as_index=False)
           .agg(ghg_kgco2e=("GHG emissions [kg CO2 eq.]", "sum"),
                va_meur=("Value Added [M.EUR]", "sum"),
                n_sectors=("sector", "nunique")))

# Drop "Other" bucket (public admin, education — not real corporate sectors)
agg = agg[agg["gics_11"] != "Other"].copy()

# Intensity: kg CO2e per M.EUR of value added → tCO2e per MUSD (approx, 1 EUR ≈ 1.08 USD)
# (kg / 1000 = tonnes; /M.EUR * 1/1.08 = per MUSD)
EUR_PER_USD = 1.08
agg["tco2e_per_musd"] = (agg["ghg_kgco2e"] / 1000) / (agg["va_meur"] * EUR_PER_USD)
agg = agg.rename(columns={"region": "iso2_or_row", "Year": "year"})

agg.to_csv(OUT_LOOKUP, index=False)
print(f"Lookup table: {len(agg):,} rows → {OUT_LOOKUP}")
print(f"  GICS sectors covered: {sorted(agg['gics_11'].unique())}")


# ── 2. Load NZDPU + yf_sector from ticker cache ──────────────────────────────
print("\nLoading NZDPU + yfinance sector...")
nz = pd.read_csv(NZDPU_PATH, low_memory=False)
with open(TC_PATH) as f:
    tc = json.load(f)
with open(FC_PATH) as f:
    fc = json.load(f)

nz_to_ticker = {int(k): v["ticker"] for k, v in tc.items()
                if isinstance(v, dict) and v.get("ticker")}
nz["ticker"]    = nz["nz_id"].map(nz_to_ticker)
nz["yf_sector"] = nz["ticker"].map(lambda t: fc.get(t, {}).get("yf_sector") if t else None)

# Pull ISO2 from master (already normalized)
master = pd.read_csv(MASTER_PATH, low_memory=False, usecols=["nz_id", "country_iso2"])
master = master.dropna(subset=["nz_id"]).drop_duplicates("nz_id")
nz = nz.merge(master, on="nz_id", how="left")

# Map region
nz["exio_region"] = nz["country_iso2"].map(map_region)

# Resolve sector
resolved = nz.apply(resolve_gics, axis=1, result_type="expand")
nz["gics_11"]       = resolved[0]
nz["sector_source"] = resolved[1]

print(f"  NZDPU rows: {len(nz):,}")
print(f"  With exio_region: {nz['exio_region'].notna().sum():,}")
print(f"  With gics_11:     {nz['gics_11'].notna().sum():,}")
print(f"    from sics:       {(nz['sector_source']=='sics').sum():,}")
print(f"    from yf_sector:  {(nz['sector_source']=='yf').sum():,}")


# ── 3. Join factors + apply to revenue ───────────────────────────────────────
print("\nComputing factor baseline...")

# Build lookup dict: (iso2_or_row, gics_11, year) → tco2e_per_musd
factor_map = dict(zip(
    zip(agg["iso2_or_row"], agg["gics_11"], agg["year"]),
    agg["tco2e_per_musd"]
))

# yfinance revenue (static latest snapshot — fine for baseline)
nz["revenue_musd"] = nz["ticker"].map(
    lambda t: (fc.get(t, {}).get("revenue_usd") or 0) / 1e6 if t else None
)
nz["revenue_musd"] = nz["revenue_musd"].replace(0, np.nan)

# Factor lookup — match exact year, else clip to 2022 (latest ExioML year)
def lookup_factor(row):
    region, sector, year = row["exio_region"], row["gics_11"], row["reporting_year"]
    if not region or not sector or pd.isna(year):
        return np.nan
    year = int(min(year, 2022))
    return factor_map.get((region, sector, year), np.nan)

nz["factor_tco2e_per_musd"] = nz.apply(lookup_factor, axis=1)

# Predicted Scope 1+2
nz["scope12_pred_tco2e"] = nz["revenue_musd"] * nz["factor_tco2e_per_musd"]

# Actual Scope 1+2
DASH = "\u2014"
for c in ["total_s1_emissions_ghg", "total_s2_lb_emissions_ghg"]:
    nz[c] = pd.to_numeric(nz[c].replace(DASH, np.nan), errors="coerce")
nz["scope12_actual_tco2e"] = nz["total_s1_emissions_ghg"].fillna(0) + nz["total_s2_lb_emissions_ghg"].fillna(0)
# If BOTH s1 and s2 were NaN, drop it (not genuine zero)
nz.loc[nz["total_s1_emissions_ghg"].isna() & nz["total_s2_lb_emissions_ghg"].isna(), "scope12_actual_tco2e"] = np.nan

# Save baseline table
out = nz[["nz_id", "reporting_year", "company_name", "country_iso2", "exio_region",
          "sics_sector", "yf_sector", "gics_11", "sector_source",
          "ticker", "revenue_musd",
          "factor_tco2e_per_musd", "scope12_pred_tco2e", "scope12_actual_tco2e"]]
out.to_csv(OUT_BASELINE, index=False)
print(f"Baseline table: {len(out):,} rows → {OUT_BASELINE}")


# ── 4. Report coverage + baseline accuracy ──────────────────────────────────
print("\n=== Coverage ===")
print(f"  rows with factor:          {nz['factor_tco2e_per_musd'].notna().sum():,} ({nz['factor_tco2e_per_musd'].notna().mean():.0%})")
print(f"  rows with revenue:         {nz['revenue_musd'].notna().sum():,}")
print(f"  rows with factor + rev:    {(nz['factor_tco2e_per_musd'].notna() & nz['revenue_musd'].notna()).sum():,}")
print(f"  rows with factor+rev+act:  {(nz['factor_tco2e_per_musd'].notna() & nz['revenue_musd'].notna() & nz['scope12_actual_tco2e'].notna()).sum():,}")

ok = nz.dropna(subset=["scope12_pred_tco2e", "scope12_actual_tco2e"])
ok = ok[(ok["scope12_pred_tco2e"] > 0) & (ok["scope12_actual_tco2e"] > 0)]
print(f"\n=== Baseline accuracy (factor × revenue vs reported, n={len(ok):,}) ===")
if len(ok):
    log_pred = np.log10(ok["scope12_pred_tco2e"])
    log_act  = np.log10(ok["scope12_actual_tco2e"])
    diff = log_pred - log_act
    mape = (np.abs(ok["scope12_pred_tco2e"] - ok["scope12_actual_tco2e"]) / ok["scope12_actual_tco2e"]).median()
    corr = np.corrcoef(log_pred, log_act)[0, 1]
    print(f"  median |log10 ratio|: {np.abs(diff).median():.2f}  (1.0 = 10× off, 2.0 = 100× off)")
    print(f"  median MAPE:         {mape:.0%}")
    print(f"  Pearson r (log):     {corr:.3f}")
