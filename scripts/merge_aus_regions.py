"""
Merge Australian BEEC locality labels into cleaner metro-level city labels.

This script reads the already-processed building table plus the original
Australian BEEC raw file, reconstructs the BEEC `building_id`, infers a
metro-level city from suburb/state, and rewrites Australian rows only.

It does not rerun the full standardization pipeline.

Usage:
    python scripts/merge_aus_regions.py
"""

import argparse
import os
import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RAW_BEEC_PATH = os.path.join(BASE_DIR, "data", "raw", "melbourne_cbd", "cbd_beec_2026.xlsx")
GEOCODED_CSV = os.path.join(PROCESSED_DIR, "building_geocoded_latlon.csv")

CORE_NON_AUS_CITIES = {
    "boston", "chicago", "dc", "denver", "la", "nyc",
    "philadelphia", "portland", "seattle", "sf", "singapore",
}

# Sanity-filter radius for Nominatim backfill. Nominatim occasionally returns
# matches from the wrong state/country for ambiguous street names; we discard
# any geocoded point more than ±2° from the city centre before trusting it.
GEOCODE_SANITY_CENTRES = {
    "la": (34.05, -118.24),
    "denver": (39.74, -104.99),
    "boston": (42.36, -71.06),
    "portland": (45.52, -122.68),
    "philadelphia": (39.95, -75.17),
    "singapore": (1.35, 103.82),
    "sydney": (-33.87, 151.21),
    "melbourne": (-37.81, 144.96),
    "brisbane": (-27.47, 153.03),
    "perth": (-31.95, 115.86),
    "adelaide": (-34.93, 138.60),
    "canberra": (-35.28, 149.13),
    "hobart": (-42.88, 147.33),
    "darwin": (-12.46, 130.84),
    "gold_coast": (-28.02, 153.40),
    "townsville": (-19.26, 146.82),
    "cairns": (-16.92, 145.77),
    "newcastle": (-32.93, 151.78),
    "wollongong": (-34.43, 150.89),
    "geelong": (-38.15, 144.36),
    "port_macquarie": (-31.43, 152.91),
}
GEOCODE_SANITY_RADIUS_DEG = 2.0


def merge_geocoded_latlon(df: pd.DataFrame) -> pd.DataFrame:
    """Coalesce raw lat/lon with the offline Nominatim backfill.

    Several city disclosure files do not ship lat/lon, so we geocode street
    addresses via Nominatim once in a separate offline step
    (`scripts/build_building_geocodes.py`). The resulting side table is merged
    here so that the canonical processed CSV carries the enriched coordinates
    directly — downstream scripts only ever need to read the main file.

    Rules:
      - preserve any existing `coord_source` provenance already present in the
        processed table (for example `history_backfill`)
      - if no `coord_source` exists yet, initialize
        `coord_source ∈ {city_disclosure, missing}` from raw lat/lon presence
      - geocoded lat/lon fills NaN slots, subject to a ±2° sanity filter
        around known city centroids (drops Nominatim mis-matches)
      - newly filled rows get `coord_source='geocoded_address'`
    """
    if "coord_source" not in df.columns:
        raw_has = df["latitude"].notna() & df["longitude"].notna()
        df["coord_source"] = np.where(raw_has, "city_disclosure", "missing")
    else:
        # Keep existing provenance labels. Only rows with missing coords and a
        # blank source are initialized as `missing` so later geocode fills can
        # be tracked without rewriting prior sources.
        df["coord_source"] = df["coord_source"].where(
            df["coord_source"].notna() & (df["coord_source"].astype(str).str.strip() != ""),
            other=np.where(
                df["latitude"].notna() & df["longitude"].notna(),
                "city_disclosure",
                "missing",
            ),
        )

    if not os.path.exists(GEOCODED_CSV):
        print(f"  (no geocoded side table at {GEOCODED_CSV}; skipping backfill)")
        return df

    geo = pd.read_csv(
        GEOCODED_CSV,
        usecols=["city", "building_id", "latitude", "longitude"],
    )
    geo = geo.dropna(subset=["latitude", "longitude"])
    if geo.empty:
        print("  (geocoded side table has no usable rows; skipping backfill)")
        return df

    # Drop Nominatim mismatches outside the per-city sanity radius.
    def _in_radius(row):
        centre = GEOCODE_SANITY_CENTRES.get(row["city"])
        if centre is None:
            return True
        return (
            abs(row["latitude"] - centre[0]) <= GEOCODE_SANITY_RADIUS_DEG
            and abs(row["longitude"] - centre[1]) <= GEOCODE_SANITY_RADIUS_DEG
        )

    before_geo = len(geo)
    geo = geo[geo.apply(_in_radius, axis=1)].copy()
    dropped = before_geo - len(geo)
    if dropped:
        print(f"  geocode sanity filter dropped {dropped} rows outside ±{GEOCODE_SANITY_RADIUS_DEG}°")

    # Dedup by (city, building_id) — one geocoded coordinate per building.
    geo = geo.drop_duplicates(subset=["city", "building_id"], keep="first")
    geo = geo.rename(columns={"latitude": "_geo_lat", "longitude": "_geo_lon"})

    n_before = len(df)
    df = df.merge(geo, on=["city", "building_id"], how="left")
    assert len(df) == n_before, "geocoded merge changed row count"

    fill_mask = df["latitude"].isna() & df["_geo_lat"].notna()
    n_filled = int(fill_mask.sum())
    df.loc[fill_mask, "latitude"] = df.loc[fill_mask, "_geo_lat"]
    df.loc[fill_mask, "longitude"] = df.loc[fill_mask, "_geo_lon"]
    df.loc[fill_mask, "coord_source"] = "geocoded_address"
    df = df.drop(columns=["_geo_lat", "_geo_lon"])

    print(f"  geocoded backfill filled {n_filled:,} row-level lat/lon slots")
    for city in sorted(GEOCODE_SANITY_CENTRES.keys()):
        city_df = df[df["city"] == city]
        if len(city_df) == 0:
            continue
        pct = city_df["latitude"].notna().mean() * 100
        print(f"    {city:10s}: n={len(city_df):6d}, lat non-null = {pct:5.1f}%")
    return df


def _clean_token(value: str) -> str:
    if pd.isna(value):
        return ""
    value = str(value).strip().lower()
    value = value.replace("/", " ").replace("-", " ")
    value = "_".join(value.split())
    return value


def normalize_aus_metro(suburb: str, state: str) -> str:
    suburb = _clean_token(suburb)
    state = "" if pd.isna(state) else str(state).strip().upper()

    metro_maps = {
        "melbourne": {
            "melbourne", "docklands", "southbank", "south_melbourne",
            "west_melbourne", "east_melbourne", "north_melbourne",
            "port_melbourne", "south_wharf", "carlton", "carlton_south",
            "abbotsford", "box_hill", "richmond", "hawthorn", "burnley",
            "burwood", "camberwell", "mount_waverley", "burwood_east",
            "dandenong", "footscray", "mulgrave", "notting_hill",
            "melbourne_airport",
        },
        "sydney": {
            "sydney", "north_sydney", "parramatta", "st_leonards",
            "north_ryde", "macquarie_park", "surry_hills", "barangaroo",
            "pyrmont", "chatswood", "rhodes", "haymarket", "cremorne",
            "mascot", "alexandria", "eveleigh", "ultimo", "millers_point",
            "greenwich", "frenchs_forest", "the_rocks",
            "sydney_olympic_park", "sydney_city", "hurstville", "liverpool",
        },
        "brisbane": {
            "brisbane", "brisbane_city", "south_brisbane", "milton",
            "newstead", "fortitude_valley", "spring_hill", "kangaroo_point",
            "eight_mile_plains", "bowen_hills", "upper_mount_gravatt",
            "toowong", "cannon_hill", "hamilton", "port_of_brisbane",
            "brisbane_airport", "ipswich",
        },
        "perth": {
            "perth", "west_perth", "east_perth", "osborne_park",
            "subiaco", "north_perth", "south_perth", "perth_airport",
        },
        "adelaide": {
            "adelaide", "north_adelaide", "port_adelaide",
            "adelaide_city", "mawson_lakes",
        },
        "canberra": {
            "canberra", "city", "civic", "barton", "braddon",
            "phillip", "forrest", "majura", "belconnen", "greenway",
            "canberra_airport",
        },
        "gold_coast": {
            "southport", "bundall", "robina", "varsity_lakes",
        },
        "hobart": {"hobart"},
        "darwin": {"darwin", "darwin_city"},
        "townsville": {"townsville", "townsville_city"},
        "cairns": {"cairns", "cairns_city"},
        "newcastle": {"newcastle"},
        "wollongong": {"wollongong"},
        "geelong": {"geelong"},
        "port_macquarie": {"port_macquarie"},
    }

    for metro, localities in metro_maps.items():
        if suburb in localities:
            return metro

    state_defaults = {
        "VIC": "melbourne",
        "NSW": "sydney",
        "QLD": "brisbane",
        "WA": "perth",
        "SA": "adelaide",
        "ACT": "canberra",
        "TAS": "hobart",
        "NT": "darwin",
    }
    return state_defaults.get(state, suburb or "australia_unknown")


def build_aus_id_map(raw_beec_path: str) -> dict:
    df = pd.read_excel(raw_beec_path, sheet_name=0)

    suburb_col = "Suburb/City (Address / Building) (Building)"
    state_col = "State (Address / Building) (Building)"
    geocode_col = "Geocode (Address / Building) (Building)"
    street_col = "Street Address Line 1 (Address / Building) (Building)"

    if geocode_col not in df.columns or street_col not in df.columns:
        raise ValueError("Expected BEEC ID columns not found in raw file.")

    geocode = df[geocode_col].astype(str).str.strip()
    street = (
        df[street_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
    )

    building_id = np.where(
        geocode.notna() & (geocode != "") & (geocode.str.lower() != "nan"),
        "aus_" + geocode,
        "aus_" + street,
    )

    metro_city = [
        normalize_aus_metro(suburb, state)
        for suburb, state in zip(df[suburb_col], df[state_col])
    ]

    out = pd.DataFrame({"building_id": building_id, "metro_city": metro_city})
    out = out.drop_duplicates(subset=["building_id"], keep="first")
    return dict(zip(out["building_id"], out["metro_city"]))


def recompute_validation(df: pd.DataFrame):
    summary_rows = []
    availability_rows = []
    for city, city_df in df.groupby("city"):
        summary_rows.append({
            "city": city,
            "rows": len(city_df),
            "unique_buildings": city_df["building_id"].nunique(),
            "year_min": city_df["year"].min(),
            "year_max": city_df["year"].max(),
            "n_years": city_df["year"].nunique(dropna=True),
            "target_non_null": int(city_df["total_ghg_emissions_mtco2e"].notna().sum()),
            "site_eui_non_null": int(city_df["site_eui_kbtu_sqft"].notna().sum()),
            "duplicate_building_year_rows": int(city_df.duplicated(["building_id", "year"]).sum()),
            "fallback_row_ids": int(city_df["building_id"].astype(str).str.contains("_row_").sum()),
        })
        for col in df.columns:
            if col == "city":
                continue
            availability_rows.append({
                "city": city,
                "column": col,
                "non_null_count": int(city_df[col].notna().sum()),
                "missing_rate_pct": float(city_df[col].isna().mean() * 100.0),
                "all_null": bool(city_df[col].notna().sum() == 0),
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(availability_rows)


def main():
    parser = argparse.ArgumentParser(description="Merge Australian locality labels into metro-level city labels.")
    parser.add_argument("--in_path", default=os.path.join(PROCESSED_DIR, "building_all.csv"))
    parser.add_argument("--out_path", default=os.path.join(PROCESSED_DIR, "building_all_aus_merged.csv"))
    parser.add_argument("--summary_out", default=os.path.join(PROCESSED_DIR, "building_validation_summary_aus_merged.csv"))
    parser.add_argument("--availability_out", default=os.path.join(PROCESSED_DIR, "city_feature_availability_aus_merged.csv"))
    args = parser.parse_args()

    df = pd.read_csv(args.in_path, low_memory=False)
    id_to_city = build_aus_id_map(RAW_BEEC_PATH)

    aus_mask = df["building_id"].astype(str).str.startswith("aus_")
    before_n = int(df.loc[aus_mask, "city"].nunique())
    df.loc[aus_mask, "city"] = df.loc[aus_mask, "building_id"].map(id_to_city).fillna(df.loc[aus_mask, "city"])
    after_n = int(df.loc[aus_mask, "city"].nunique())

    # Geocode backfill must run AFTER the AU metro relabel so that the
    # (city, building_id) key is in its final form.
    print("\nMerging geocoded lat/lon backfill:")
    df = merge_geocoded_latlon(df)

    df.to_csv(args.out_path, index=False)

    summary_df, availability_df = recompute_validation(df)
    summary_df.to_csv(args.summary_out, index=False)
    availability_df.to_csv(args.availability_out, index=False)

    print(f"Australian city labels: {before_n} -> {after_n}")
    print(f"Wrote merged table to {args.out_path}")
    print(f"Wrote summary to {args.summary_out}")
    print(f"Wrote availability to {args.availability_out}")


if __name__ == "__main__":
    main()
