"""
P0: Normalize country names in company_master.csv to ISO-3166 alpha-2 / alpha-3.

Adds columns: country_iso2, country_iso3

Enables downstream joins against:
  - ExioML factor accounting (uses ISO-alpha-2: AT, AU, US, ...)
  - OWID CO2 country panel (uses iso_code = ISO-alpha-3: AUT, AUS, USA, ...)
  - GLEIF LEI registry
"""

import pandas as pd
import country_converter as coco
from pathlib import Path

SRC = Path("data/company-level/company_master.csv")

df = pd.read_csv(SRC, low_memory=False)
print(f"Loaded {len(df):,} rows from {SRC}")

cc = coco.CountryConverter()
unique_names = sorted(df["country"].dropna().unique())
print(f"Unique country values: {len(unique_names)}")

iso2_map = dict(zip(unique_names, cc.convert(names=unique_names, to="ISO2", not_found=None)))
iso3_map = dict(zip(unique_names, cc.convert(names=unique_names, to="ISO3", not_found=None)))

df["country_iso2"] = df["country"].map(iso2_map)
df["country_iso3"] = df["country"].map(iso3_map)

n_iso2 = df["country_iso2"].notna().sum()
n_iso3 = df["country_iso3"].notna().sum()
n_country = df["country"].notna().sum()
print(f"country_iso2: {n_iso2:,} / {n_country:,} non-null ({n_iso2 / n_country:.1%} of rows with country)")
print(f"country_iso3: {n_iso3:,} / {n_country:,} non-null ({n_iso3 / n_country:.1%} of rows with country)")

df.to_csv(SRC, index=False)
print(f"Saved → {SRC}")
print(f"Shape: {df.shape}")

print("\nTop 10 ISO2 codes by row count:")
print(df["country_iso2"].value_counts().head(10))
