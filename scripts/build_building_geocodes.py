"""Direct building-level geocoder for selected cities without raw lat/lon.

This script geocodes building coordinates directly from the raw source files,
without relying on any intermediate address-level CSV. It is building-level in
its output, but internally de-duplicates identical geocoding queries so the
same address is only requested once and then fanned out to all matching
buildings.

Primary query pattern:
    street/address + city + state + ZIP

Fallback query pattern (only if primary returns no_match):
    street/address + city + state

Output:
    data/processed/building_geocoded_latlon.csv

Columns:
    city, building_id, latitude, longitude, coord_source, query,
    resolved_query, status

Design notes:
  - Resume-safe: only prior rows with status == "ok" are treated as complete.
  - Nominatim policy: max ~1 req/sec with a descriptive User-Agent.
  - One row per building_id in the final CSV, but one request per unique query.
  - Building IDs intentionally mirror `standardize_buildings.py` so the
    resulting coordinates can be merged back into the canonical processed CSV.
"""

from pathlib import Path
import argparse
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import requests

from standardize_buildings import read_excel_best_effort
from merge_aus_regions import normalize_aus_metro


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUT_PATH = PROCESSED_DIR / "building_geocoded_latlon.csv"
DEFAULT_S2_META_PATH = PROCESSED_DIR / "building_year_s2_metadata.parquet"

USER_AGENT = "GHGbench/0.2 (anonymous submission; anonymous-submission@example.com)"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
RATE_LIMIT_SEC = 1.05
FLUSH_EVERY = 25
MAX_RETRIES = 2
RETRY_SLEEP_SEC = 5.0


def _clean_text(value):
    if pd.isna(value):
        return ""
    value = str(value).strip()
    value = re.sub(r"\s+", " ", value)
    return value


def _build_query(street, city_name, state_code, zip_code="", include_zip=True):
    parts = [street, city_name, state_code]
    query = ", ".join([p for p in parts if p])
    if include_zip and zip_code:
        query = f"{query} {zip_code}"
    return query.strip()


def _mode_or_first(series):
    series = series.dropna()
    if series.empty:
        return ""
    mode = series.mode(dropna=True)
    if not mode.empty:
        return mode.iloc[0]
    return series.iloc[0]


def _find_first_col(columns, patterns, exclude=()):
    originals = list(columns)
    lowered = [str(c).strip().lower() for c in originals]
    for pattern in patterns:
        for orig, lower in zip(originals, lowered):
            if pattern in lower and "unnamed" not in lower:
                if any(ex in lower for ex in exclude):
                    continue
                return orig
    return None


def _pack_tasks(df, city, city_name, state_code, country_code, building_id_col, street_col, zip_col=None):
    df = df.dropna(subset=[building_id_col, street_col]).copy()
    df[building_id_col] = df[building_id_col].astype(str).str.strip()
    df[street_col] = df[street_col].map(_clean_text)
    if zip_col is None:
        df["_zip"] = ""
    else:
        df["_zip"] = df[zip_col].map(_clean_text)

    rows = []
    for building_id, g in df.groupby(building_id_col):
        street = _mode_or_first(g[street_col])
        zip_ = _mode_or_first(g["_zip"])
        query = _build_query(street, city_name, state_code, zip_, include_zip=bool(zip_))
        fallback_query = _build_query(street, city_name, state_code, zip_, include_zip=False)
        rows.append(
            {
                "city": city,
                "building_id": building_id,
                "street": street,
                "zip": zip_,
                "country_code": country_code,
                "query": query,
                "fallback_query": fallback_query,
            }
        )
    return pd.DataFrame(rows)


def _load_s2_cityfallback_targets(s2_meta_path: Path):
    cols = ["city", "building_id", "status", "img_source"]
    meta = pd.read_parquet(s2_meta_path, columns=cols)
    mask = meta["img_source"].eq("city_fallback") & meta["status"].eq("ok")
    targets = meta.loc[mask, ["city", "building_id"]].drop_duplicates().copy()
    return targets


def _load_la_buildings():
    df = pd.read_csv(
        RAW_DIR / "la_benchmarking" / "la_ebewe_all.csv",
        usecols=["BUILDING ID", "BUILDING ADDRESS", "POSTAL CODE"],
        low_memory=False,
    )
    df = df.dropna(subset=["BUILDING ID", "BUILDING ADDRESS"]).copy()
    df["building_id"] = "la_" + df["BUILDING ID"].astype(str).str.strip()
    df["street"] = df["BUILDING ADDRESS"].map(_clean_text)
    df["zip"] = df["POSTAL CODE"].map(_clean_text)

    return _pack_tasks(
        df, city="la", city_name="Los Angeles", state_code="CA", country_code="us",
        building_id_col="building_id", street_col="street", zip_col="zip"
    )


def _load_denver_buildings():
    df = pd.read_csv(
        RAW_DIR / "denver_benchmarking" / "denver_2024.csv",
        usecols=["Building_ID", "Street", "Zipcode"],
        low_memory=False,
    )
    df = df.dropna(subset=["Building_ID", "Street"]).copy()
    df["building_id"] = "denver_" + df["Building_ID"].astype(str).str.strip()
    df["street"] = df["Street"].map(_clean_text)
    df["zip"] = df["Zipcode"].map(_clean_text)

    return _pack_tasks(
        df, city="denver", city_name="Denver", state_code="CO", country_code="us",
        building_id_col="building_id", street_col="street", zip_col="zip"
    )


def _load_boston_buildings():
    rows = []

    csv_files = sorted((RAW_DIR / "boston_berdo").glob("*.csv"))
    for f in csv_files:
        df = pd.read_csv(f, encoding="latin1", low_memory=False)
        tmp = pd.DataFrame()
        tmp["building_id"] = "boston_" + df["Tax Parcel"].astype(str).str.strip()
        tmp["street"] = df["Address"].map(_clean_text)
        tmp["zip"] = df["ZIP"].map(_clean_text)
        rows.append(tmp)

    xlsx_files = sorted((RAW_DIR / "boston_berdo").glob("*.xlsx"))
    for f in xlsx_files:
        try:
            df = read_excel_best_effort(
                f,
                sheet_names=["Data Disclosure", "2020 Data", "Final Form", "Sheet1",
                             "Data Disclosure - Campuses", "2024 Reported Energy and Water"],
                header_candidates=(0, 1, 2),
            )
        except Exception:
            try:
                df = read_excel_best_effort(f, header_candidates=(0, 1, 2))
            except Exception:
                continue

        tmp = pd.DataFrame()
        bid_col = _find_first_col(df.columns, ["tax parcel", "berdo id"])
        if not bid_col:
            continue
        tmp["building_id"] = "boston_" + df[bid_col].astype(str).str.strip()

        street = None
        zip_code = None
        street_cols = [
            c for c in [df.get(_find_first_col(df.columns, ["building address"], exclude=["zip", "city"])),
                        df.get(_find_first_col(df.columns, ["parcel address"], exclude=["zip", "city"])),
                        df.get(_find_first_col(df.columns, ["address"], exclude=["zip", "city"]))]
            if c is not None
        ]
        zip_cols = [
            c for c in [df.get(_find_first_col(df.columns, ["building address zip"])),
                        df.get(_find_first_col(df.columns, ["parcel address zip"])),
                        df.get(_find_first_col(df.columns, ["zip"]))]
            if c is not None
        ]
        for candidate in street_cols:
            if street is None:
                street = candidate
            else:
                street = street.where(street.notna() & (street.astype(str).str.strip() != ""), candidate)
        for candidate in zip_cols:
            if zip_code is None:
                zip_code = candidate
            else:
                zip_code = zip_code.where(zip_code.notna() & (zip_code.astype(str).str.strip() != ""), candidate)

        if street is None:
            continue
        if zip_code is None:
            zip_code = ""

        tmp["street"] = street.map(_clean_text)
        if isinstance(zip_code, str):
            tmp["zip"] = zip_code
        else:
            tmp["zip"] = zip_code.map(_clean_text)
        rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=["city", "building_id", "street", "zip", "country_code", "query", "fallback_query"])

    df = pd.concat(rows, ignore_index=True)
    return _pack_tasks(
        df, city="boston", city_name="Boston", state_code="MA", country_code="us",
        building_id_col="building_id", street_col="street", zip_col="zip"
    )


def _load_portland_buildings():
    rows = []
    files = sorted((RAW_DIR / "portland_benchmarking").glob("*.xlsx"))
    for f in files:
        xls = pd.ExcelFile(f)
        sheet = [s for s in xls.sheet_names if "energy performance" in s.lower()][0]
        df = pd.read_excel(f, sheet_name=sheet)
        df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]
        tmp = pd.DataFrame()
        bid_col = _find_first_col(df.columns, ["building id"])
        street_col = _find_first_col(df.columns, ["site address", "address"])
        if not bid_col or not street_col:
            continue
        tmp["building_id"] = "portland_" + df[bid_col].astype(str).str.strip()
        tmp["street"] = df[street_col].map(_clean_text)
        tmp["zip"] = ""
        rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=["city", "building_id", "street", "zip", "country_code", "query", "fallback_query"])

    df = pd.concat(rows, ignore_index=True)
    return _pack_tasks(
        df, city="portland", city_name="Portland", state_code="OR", country_code="us",
        building_id_col="building_id", street_col="street", zip_col="zip"
    )


def _load_singapore_buildings():
    rows = []

    paths = [
        RAW_DIR / "singapore_bca" / "Listing of Building Energy Performance Data 2020.csv",
        RAW_DIR / "singapore_bca" / "Listing of Building Energy Performance Data for Commercial Buildings.csv",
    ]
    for path in paths:
        if path.suffix == ".csv":
            df = pd.read_csv(path, low_memory=False)
        else:
            df = pd.read_excel(path)
        tmp = pd.DataFrame()
        tmp["building_id"] = (
            "singapore_" + df["buildingname"].astype(str).str.strip().str.upper().str.replace(" ", "_")
        )
        tmp["street"] = df["buildingaddress"].map(_clean_text)
        tmp["zip"] = ""
        rows.append(tmp)

    df = pd.concat(rows, ignore_index=True)
    df = df[df["street"] != ""].copy()
    return _pack_tasks(
        df, city="singapore", city_name="Singapore", state_code="", country_code="sg",
        building_id_col="building_id", street_col="street", zip_col="zip"
    )


def _load_philadelphia_buildings():
    df = pd.read_csv(
        RAW_DIR / "philadelphia_benchmarking" / "philly_2023_reported.csv",
        usecols=["philadelphia_building_id", "street_address", "postal_code"],
        low_memory=False,
    )
    df = df.dropna(subset=["philadelphia_building_id", "street_address"]).copy()
    df["building_id"] = "philadelphia_" + df["philadelphia_building_id"].astype(str).str.strip()
    df["street"] = df["street_address"].map(_clean_text)
    df["zip"] = df["postal_code"].map(_clean_text)

    return _pack_tasks(
        df, city="philadelphia", city_name="Philadelphia", state_code="PA", country_code="us",
        building_id_col="building_id", street_col="street", zip_col="zip"
    )


def _load_aus_buildings():
    path = RAW_DIR / "melbourne_cbd" / "cbd_beec_2026.xlsx"
    df = pd.read_excel(path, sheet_name=0)

    geocode_col = "Geocode (Address / Building) (Building)"
    street_col = "Street Address Line 1 (Address / Building) (Building)"
    suburb_col = "Suburb/City (Address / Building) (Building)"
    state_col = "State (Address / Building) (Building)"
    postcode_col = "Post Code (Address / Building) (Building)"

    for col in [geocode_col, street_col, suburb_col, state_col]:
        if col not in df.columns:
            raise ValueError(f"Expected AU BEEC column not found: {col}")

    geocode = df[geocode_col].astype(str).str.strip()
    street = df[street_col].map(_clean_text)
    suburb = df[suburb_col].map(_clean_text)
    state = df[state_col].map(_clean_text).str.upper()
    postcode = df[postcode_col].map(_clean_text) if postcode_col in df.columns else ""

    fallback = (
        df[street_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
    )
    building_id = np.where(
        geocode.notna() & (geocode != "") & (geocode.str.lower() != "nan"),
        "aus_" + geocode,
        "aus_" + fallback,
    )
    metro_city = [
        normalize_aus_metro(suburb_raw, state_raw)
        for suburb_raw, state_raw in zip(df[suburb_col], df[state_col])
    ]

    rows = []
    for bid, street_val, suburb_val, state_val, zip_val, city in zip(
        building_id, street, suburb, state, postcode, metro_city
    ):
        if not street_val:
            continue
        primary_parts = [street_val, suburb_val, state_val]
        primary_query = ", ".join([p for p in primary_parts if p])
        if zip_val:
            primary_query = f"{primary_query} {zip_val}".strip()
        fallback_query = ", ".join([p for p in [street_val, suburb_val, state_val] if p]).strip()
        rows.append(
            {
                "city": city,
                "building_id": bid,
                "street": street_val,
                "zip": zip_val,
                "country_code": "au",
                "query": primary_query,
                "fallback_query": fallback_query,
            }
        )
    return pd.DataFrame(rows)

def _geocode_one(query, country_code):
    params = {
        "q": query,
        "format": "json",
        "limit": 1,
        "addressdetails": 0,
        "countrycodes": country_code,
    }
    headers = {"User-Agent": USER_AGENT}
    try:
        r = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=20)
        if r.status_code != 200:
            return None, None, f"http_{r.status_code}"
        data = r.json()
        if not data:
            return None, None, "no_match"
        return float(data[0]["lat"]), float(data[0]["lon"]), "ok"
    except Exception as e:
        return None, None, f"err:{type(e).__name__}"


def _should_retry(status):
    if status.startswith("err:"):
        return True
    if status.startswith("http_"):
        try:
            code = int(status.split("_", 1)[1])
            return 500 <= code < 600
        except Exception:
            return False
    return False


def _geocode_with_retry(query, country_code):
    lat, lon, status = _geocode_one(query, country_code)
    attempts = 0
    while attempts < MAX_RETRIES and _should_retry(status):
        attempts += 1
        time.sleep(RETRY_SLEEP_SEC)
        lat, lon, status = _geocode_one(query, country_code)
    return lat, lon, status


def _geocode_with_fallbacks(primary_query, fallback_query, country_code):
    lat, lon, status = _geocode_with_retry(primary_query, country_code)
    resolved_query = primary_query
    if status == "no_match" and fallback_query and fallback_query != primary_query:
        lat, lon, status = _geocode_with_retry(fallback_query, country_code)
        resolved_query = fallback_query
    return lat, lon, status, resolved_query


def main():
    parser = argparse.ArgumentParser(description="Build building-level geocodes for selected cities.")
    parser.add_argument(
        "--cities",
        type=str,
        default="la,denver,boston,portland,singapore,philadelphia",
        help="Comma-separated city list to process.",
    )
    parser.add_argument(
        "--s2_fallback_only",
        action="store_true",
        help="Only geocode building_ids that currently appear as S2 city_fallback (status==ok).",
    )
    parser.add_argument(
        "--s2_meta_path",
        type=str,
        default=str(DEFAULT_S2_META_PATH),
        help="Canonical S2 metadata parquet used to derive city_fallback building_ids.",
    )
    args = parser.parse_args()

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    loaders = {
        "la": _load_la_buildings,
        "denver": _load_denver_buildings,
        "boston": _load_boston_buildings,
        "portland": _load_portland_buildings,
        "singapore": _load_singapore_buildings,
        "philadelphia": _load_philadelphia_buildings,
        "australia": _load_aus_buildings,
    }
    selected = [c.strip() for c in args.cities.split(",") if c.strip()]
    unknown = [c for c in selected if c not in loaders]
    if unknown:
        raise ValueError(f"Unsupported cities: {unknown}. Supported: {sorted(loaders)}")

    tasks = pd.concat([loaders[c]() for c in selected], ignore_index=True)

    if args.s2_fallback_only:
        s2_meta_path = Path(args.s2_meta_path)
        if not s2_meta_path.exists():
            raise FileNotFoundError(f"S2 metadata not found: {s2_meta_path}")

        targets = _load_s2_cityfallback_targets(s2_meta_path)
        target_cities = set(targets["city"].unique())
        unsupported = sorted(target_cities - set(selected))
        if unsupported:
            print(
                "[info] S2 city_fallback buildings exist for unsupported/unselected cities: "
                + ", ".join(unsupported)
            )

        target_keys = set(zip(targets["city"].astype(str), targets["building_id"].astype(str)))
        before = len(tasks)
        tasks = tasks[
            [
                (city, building_id) in target_keys
                for city, building_id in zip(tasks["city"].astype(str), tasks["building_id"].astype(str))
            ]
        ].copy()
        print(
            f"[filter] s2_fallback_only kept {len(tasks)}/{before} building geocode tasks "
            f"from {s2_meta_path}"
        )

    if OUT_PATH.exists():
        existing = pd.read_csv(OUT_PATH)
        expected_cols = {
            "city", "building_id", "latitude", "longitude", "coord_source",
            "query", "resolved_query", "status"
        }
        if expected_cols.issubset(existing.columns):
            done = set(existing.loc[existing["status"] == "ok", "building_id"].astype(str))
            rows = existing.to_dict("records")
            print(
                f"[resume] loaded {len(existing)} prior rows from {OUT_PATH}; "
                f"{len(done)} successful building_ids will be skipped"
            )
        else:
            print(f"[reset] {OUT_PATH} has an older schema; rebuilding from scratch")
            done = set()
            rows = []
    else:
        done = set()
        rows = []

    todo = tasks[~tasks["building_id"].isin(done)].copy()
    print(f"Total building tasks: {len(tasks)}  pending: {len(todo)}")
    if todo.empty:
        print("nothing to do.")
        return

    key_cols = ["city", "query", "fallback_query"]
    grouped = {}
    for row in todo.itertuples(index=False):
        key = (row.city, row.query, row.fallback_query, row.country_code)
        grouped.setdefault(key, []).append(row)

    unique_queries = list(grouped.items())
    print(f"Unique geocoding queries to issue: {len(unique_queries)}")

    try:
        processed_buildings = 0
        for i, (key, buildings) in enumerate(unique_queries, start=1):
            city, query, fallback_query, country_code = key
            lat, lon, status, resolved_query = _geocode_with_fallbacks(query, fallback_query, country_code)

            for building in buildings:
                rows.append(
                    {
                        "city": building.city,
                        "building_id": building.building_id,
                        "latitude": lat,
                        "longitude": lon,
                        "coord_source": "geocoded_address" if status == "ok" else "missing",
                        "query": building.query,
                        "resolved_query": resolved_query,
                        "status": status,
                    }
                )
            processed_buildings += len(buildings)

            if i % FLUSH_EVERY == 0 or i == len(unique_queries):
                pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
                ok = sum(1 for r in rows if r["status"] == "ok")
                print(
                    f"  [queries {i}/{len(unique_queries)} | buildings {processed_buildings}/{len(todo)}] "
                    f"ok={ok} last='{query[:70]}' -> {status}"
                )
                sys.stdout.flush()

            time.sleep(RATE_LIMIT_SEC)
    except KeyboardInterrupt:
        print("\n[interrupt] flushing partial building-level geocodes...")
        pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
        print(f"  saved {len(rows)} rows to {OUT_PATH}; rerun to resume.")
        return

    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    out = pd.DataFrame(rows)
    ok = int((out["status"] == "ok").sum())
    print(f"\nDone. {ok}/{len(out)} building IDs geocoded. -> {OUT_PATH}")
    print("\nBreakdown by city:")
    print(out.groupby(["city", "status"]).size().to_string())


if __name__ == "__main__":
    main()
