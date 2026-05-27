"""Build building-year climate features via NASA POWER daily API.

This is the one-step replacement for the older ERA5 two-phase climate path
(`download_era5.py` + `extract_era5_features.py`) and the later
Open-Meteo prototype.

Protocol
--------
- Source: NASA POWER daily point API
- Granularity: building-year first, city fallback second
- Input table: data/processed/building_all_aus_merged.csv
- Output table: data/processed/building_year_climate.csv

For each (building_id, year):
1. If the row has sane latitude/longitude -> query that building location
   (`climate_source = building_grid`)
2. Otherwise -> query the city centre (`climate_source = city_fallback`)

The API is queried once per unique (rounded_lat, rounded_lon, year) key, then
the derived annual features are fanned back out to all matching building-year
rows. This keeps the request count manageable while still remaining
building-aligned whenever valid coordinates exist.

Derived annual features (6)
---------------------------
- hdd_18c:              sum(max(18 - daily_mean_temp_c, 0))
- cdd_18c:              sum(max(daily_mean_temp_c - 18, 0))
- annual_mean_temp_c:   mean(daily temperature_2m_mean)
- annual_rh_mean:       mean(daily RH2M)
- annual_ssrd_mj_m2_day:
    mean(daily ALLSKY_SFC_SW_DWN), converted from W/m^2 (daily mean) to
    MJ/m^2/day via ×86400/1e6.
- annual_wind_ms:       mean(daily WS10M)

Notes
-----
- NASA POWER is a point-based historical weather service backed by NASA
  analysis-ready meteorology and solar products, not raw CDS ERA5 download.
- The script is resume-safe at the building-year row level: existing output
  rows are skipped on rerun.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
BUILDING_CSV = PROCESSED_DIR / "building_all_aus_merged.csv"
OUT_CSV = PROCESSED_DIR / "building_year_climate.csv"
CHECKPOINT_PATH = PROCESSED_DIR / ".nasa_power_feature_cache.json"
CHECKPOINT_EVERY = 25

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
ROUND_DECIMALS = 4
YEAR_MIN = 2011
YEAR_MAX = 2025  # ERA5 has ~3 month lag; today=2026-04 → full 2025 is available.
                 # derive_features also guards partial years (<360 days) as a safety net.
SANITY_RADIUS_DEG = 2.0
BASE_TEMP_C = 18.0
POWER_GRID_DEG = 0.5
# NASA POWER SB community returns ALLSKY_SFC_SW_DWN in W m-2 (a daily-mean
# irradiance), NOT in kWh/m^2/day as we initially assumed. Convert to
# MJ/m^2/day by multiplying by 86400 s/day / 1e6 J/MJ = 0.0864.
SOLAR_W_PER_M2_TO_MJ_PER_DAY = 86400.0 / 1.0e6


CITY_CENTRES = {
    "nyc": (40.71, -74.01),
    "la": (34.05, -118.24),
    "seattle": (47.61, -122.33),
    "dc": (38.91, -77.04),
    "chicago": (41.88, -87.63),
    "sf": (37.77, -122.42),
    "boston": (42.36, -71.06),
    "denver": (39.74, -104.99),
    "portland": (45.52, -122.68),
    "philadelphia": (39.95, -75.17),
    "sydney": (-33.87, 151.21),
    "melbourne": (-37.81, 144.96),
    "brisbane": (-27.47, 153.03),
    "perth": (-31.95, 115.86),
    "adelaide": (-34.93, 138.60),
    "canberra": (-35.28, 149.13),
    "hobart": (-42.88, 147.33),
    "darwin": (-12.46, 130.84),
    "newcastle": (-32.93, 151.78),
    "wollongong": (-34.43, 150.89),
    "gold_coast": (-28.02, 153.40),
    "townsville": (-19.26, 146.82),
    "cairns": (-16.92, 145.77),
    "geelong": (-38.15, 144.36),
    "port_macquarie": (-31.43, 152.91),
    "singapore": (1.35, 103.82),
}


def is_valid_coord(lat: float, lon: float, city_lat: float, city_lon: float) -> bool:
    if pd.isna(lat) or pd.isna(lon):
        return False
    if not (np.isfinite(lat) and np.isfinite(lon)):
        return False
    if lat == 0.0 or lon == 0.0:
        return False
    if abs(lat - city_lat) > SANITY_RADIUS_DEG:
        return False
    if abs(lon - city_lon) > SANITY_RADIUS_DEG:
        return False
    return True


def parse_years(s: str) -> List[int]:
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def quantize_to_era5_grid(value: float) -> float:
    """Snap a coordinate to the nearest NASA POWER daily 0.5° grid centre."""
    return round(round(value / POWER_GRID_DEG) * POWER_GRID_DEG, 4)


def fetch_json(lat: float, lon: float, year: int, timeout_s: int, max_retries: int) -> Dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start": f"{year}0101",
        "end": f"{year}1231",
        "community": "SB",
        "format": "JSON",
        "time-standard": "UTC",
        "parameters": "T2M,RH2M,WS10M,ALLSKY_SFC_SW_DWN",
    }
    url = f"{NASA_POWER_URL}?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": "GHGbench/1.0 (research use)"})

    total_attempts = max_retries + 6
    for attempt in range(total_attempts):
        try:
            with urlopen(req, timeout=timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            if e.code == 429:
                if attempt >= total_attempts - 1:
                    raise
                backoff = 60 + 30 * attempt
                print(f"    [rate-limit 429] sleeping {backoff}s before retry {attempt + 1}")
                time.sleep(backoff)
                continue
            if attempt >= total_attempts - 1:
                raise
            time.sleep(2 + 3 * attempt)
        except Exception:
            if attempt >= total_attempts - 1:
                raise
            time.sleep(2 + 3 * attempt)
    raise RuntimeError("unreachable")


def derive_features(payload: Dict) -> Dict[str, float]:
    params = payload.get("properties", {}).get("parameter", {})

    def series(name: str) -> np.ndarray:
        values = params.get(name, {})
        if isinstance(values, dict):
            vals = []
            for _, v in sorted(values.items()):
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    vals.append(np.nan)
            return np.asarray(vals, dtype=float)
        return np.asarray([], dtype=float)

    t_daily = series("T2M")
    rh_daily = series("RH2M")
    wind_daily = series("WS10M")
    ssrd_daily = series("ALLSKY_SFC_SW_DWN")

    if t_daily.size == 0:
        raise ValueError("NASA POWER response missing T2M")

    # HDD/CDD are annual sums — a partial year would silently under-count, so
    # only emit them when the response covers a full calendar year (≥360 days).
    if t_daily.size >= 360:
        hdd = float(np.maximum(0.0, BASE_TEMP_C - t_daily).sum())
        cdd = float(np.maximum(0.0, t_daily - BASE_TEMP_C).sum())
    else:
        hdd = np.nan
        cdd = np.nan

    annual_mean_temp = float(np.nanmean(t_daily))
    annual_rh = float(np.nanmean(rh_daily)) if rh_daily.size else np.nan
    annual_ssrd = float(np.nanmean(ssrd_daily) * SOLAR_W_PER_M2_TO_MJ_PER_DAY) if ssrd_daily.size else np.nan
    annual_wind = float(np.nanmean(wind_daily)) if wind_daily.size else np.nan

    return {
        "hdd_18c": float(hdd),
        "cdd_18c": float(cdd),
        "annual_mean_temp_c": annual_mean_temp,
        "annual_rh_mean": annual_rh,
        "annual_ssrd_mj_m2_day": annual_ssrd,
        "annual_wind_ms": annual_wind,
    }


def load_existing_done(path: Path) -> set[Tuple[str, int]]:
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, usecols=["building_id", "year"])
    except Exception:
        return set()
    return set(zip(df["building_id"].astype(str), df["year"].astype(int)))


def build_query_plan(df: pd.DataFrame, years: Iterable[int]) -> pd.DataFrame:
    df = df[df["year"].between(min(years), max(years))].copy()
    df = df[df["year"].isin(list(years))].copy()

    plan_rows = []
    for row in df.itertuples(index=False):
        city = row.city
        year = int(row.year)
        city_lat, city_lon = CITY_CENTRES[city]
        if is_valid_coord(row.latitude, row.longitude, city_lat, city_lon):
            qlat = quantize_to_era5_grid(float(row.latitude))
            qlon = quantize_to_era5_grid(float(row.longitude))
            source = "building_grid"
        else:
            qlat = quantize_to_era5_grid(float(city_lat))
            qlon = quantize_to_era5_grid(float(city_lon))
            source = "city_fallback"

        plan_rows.append(
            {
                "building_id": str(row.building_id),
                "city": city,
                "year": year,
                "climate_source": source,
                "query_lat": qlat,
                "query_lon": qlon,
                "query_key": f"{qlat:.{ROUND_DECIMALS}f}|{qlon:.{ROUND_DECIMALS}f}|{year}",
            }
        )

    return pd.DataFrame(plan_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", type=str, default=None, help="Comma-separated city names")
    ap.add_argument("--years", type=str, default=f"{YEAR_MIN}-{YEAR_MAX}", help="A-B or comma list")
    ap.add_argument("--timeout_s", type=int, default=120)
    ap.add_argument("--max_retries", type=int, default=2)
    ap.add_argument("--sleep_s", type=float, default=1.1)  # ~55 calls/min; well under docs' 600/min to avoid 429
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    years = parse_years(args.years)
    if args.cities:
        cities = args.cities.split(",")
    else:
        cities = list(CITY_CENTRES.keys())

    df = pd.read_csv(BUILDING_CSV, low_memory=False)
    df = df[df["city"].isin(cities)].copy()
    df = df[df["year"].notna()].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"].notna()].copy()
    df["year"] = df["year"].astype(int)
    # Defensive: avoid string/object lat/lon breaking np.isfinite in is_valid_coord.
    if "latitude" in df.columns:
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    if "longitude" in df.columns:
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    done = set() if args.overwrite else load_existing_done(OUT_CSV)
    plan = build_query_plan(df, years)
    if done:
        plan = plan[~plan.apply(lambda r: (r["building_id"], int(r["year"])) in done, axis=1)].copy()

    total_rows = len(plan)
    unique_queries = (
        plan[["query_key", "query_lat", "query_lon", "year"]]
        .drop_duplicates()
        .sort_values(["year", "query_key"])
        .reset_index(drop=True)
    )

    print("NASA POWER climate build")
    print(f"  cities         : {len(cities)}")
    print(f"  years          : {years[0]}..{years[-1]} ({len(years)} years)")
    print(f"  building-years : {total_rows}")
    print(f"  unique queries : {len(unique_queries)}")
    print(f"  output         : {OUT_CSV}")

    if args.dry_run:
        return

    # Load checkpoint: the in-memory feature_cache is flushed to disk every
    # CHECKPOINT_EVERY successful queries so an interrupted run (rate limit,
    # network hiccup, Ctrl-C) can resume without redoing successful work.
    # Keys with NaN HDD (previous failures) are dropped so we retry them.
    feature_cache: Dict[str, Dict] = {}
    if CHECKPOINT_PATH.exists() and not args.overwrite:
        try:
            with open(CHECKPOINT_PATH, "r") as fh:
                raw_cache = json.load(fh)
            for k, v in raw_cache.items():
                hdd_val = v.get("hdd_18c")
                if hdd_val is not None and not (isinstance(hdd_val, float) and math.isnan(hdd_val)):
                    feature_cache[k] = v
            print(f"  loaded checkpoint: {len(feature_cache)} successful queries resumed")
        except Exception as e:
            print(f"  (could not load checkpoint {CHECKPOINT_PATH}: {e})")

    def flush_checkpoint():
        tmp = CHECKPOINT_PATH.with_suffix(".json.tmp")
        with open(tmp, "w") as fh:
            json.dump(feature_cache, fh)
        tmp.replace(CHECKPOINT_PATH)

    successes_since_flush = 0
    for i, row in enumerate(unique_queries.itertuples(index=False), start=1):
        if row.query_key in feature_cache:
            if i == 1 or i % 100 == 0 or i == len(unique_queries):
                print(f"  [{i}/{len(unique_queries)}] skip (cached)  year={row.year}")
            continue

        try:
            payload = fetch_json(row.query_lat, row.query_lon, int(row.year), args.timeout_s, args.max_retries)
            feats = derive_features(payload)
            feature_cache[row.query_key] = {
                "grid_lat": float(row.query_lat),
                "grid_lon": float(row.query_lon),
                **feats,
            }
            status = "ok"
            successes_since_flush += 1
        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            feature_cache[row.query_key] = {
                "grid_lat": float(row.query_lat),
                "grid_lon": float(row.query_lon),
                "hdd_18c": float("nan"),
                "cdd_18c": float("nan"),
                "annual_mean_temp_c": float("nan"),
                "annual_rh_mean": float("nan"),
                "annual_ssrd_mj_m2_day": float("nan"),
                "annual_wind_ms": float("nan"),
            }
            status = f"error ({err_msg})"

        if i == 1 or i % 100 == 0 or i == len(unique_queries) or status.startswith("error"):
            print(f"  [{i}/{len(unique_queries)}] {status}  year={row.year}  lat={row.query_lat} lon={row.query_lon}")

        if successes_since_flush >= CHECKPOINT_EVERY:
            flush_checkpoint()
            successes_since_flush = 0

        time.sleep(args.sleep_s)

    flush_checkpoint()

    out = plan.copy()
    feat_df = pd.DataFrame(
        [{"query_key": k, **v} for k, v in feature_cache.items()]
    )
    out = out.merge(feat_df, on="query_key", how="left")
    out = out.drop(columns=["query_key"])
    out = out[
        [
            "building_id",
            "year",
            "city",
            "climate_source",
            "grid_lat",
            "grid_lon",
            "hdd_18c",
            "cdd_18c",
            "annual_mean_temp_c",
            "annual_rh_mean",
            "annual_ssrd_mj_m2_day",
            "annual_wind_ms",
        ]
    ].sort_values(["city", "year", "building_id"])

    if OUT_CSV.exists() and not args.overwrite:
        prev = pd.read_csv(OUT_CSV)
        out = pd.concat([prev, out], ignore_index=True)
        out = out.drop_duplicates(subset=["building_id", "year"], keep="last")

    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(out):,} rows to {OUT_CSV}")

    # Successful full-run — drop the checkpoint so the next fresh invocation
    # doesn't pick up stale state. An --overwrite run ignores the checkpoint
    # anyway, but this keeps the processed/ directory clean.
    if CHECKPOINT_PATH.exists():
        try:
            CHECKPOINT_PATH.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
