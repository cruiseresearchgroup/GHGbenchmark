"""One-time geocoder for LA and Denver building addresses.

The feature-availability audit showed LA and Denver both have 0% lat/lon
in the processed file, so they cannot participate in the Core-All-Cities
benchmark's location feature. This script reads the raw benchmarking files,
extracts unique addresses, queries Nominatim (OpenStreetMap's free service),
and writes a lookup CSV that the standardization pipeline can merge in.

Nominatim usage policy: max 1 req/sec, must set a User-Agent identifying
the project and contact email. Respect this or get banned.

Output: data/processed/geocoded_addresses.csv with columns:
    city, raw_address, lat, lon, geocoder, status

Resumability: the script reads any existing geocoded_addresses.csv and
skips addresses already present, so it can be interrupted and resumed.
"""
import os
import time
import sys
import pandas as pd
import requests

USER_AGENT = "GHGbench/0.1 (anonymous submission; anonymous-submission@example.com)"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OUT_PATH = "data/processed/geocoded_addresses.csv"
RATE_LIMIT_SEC = 1.05  # a bit over 1s to be safe


def load_la_addresses():
    df = pd.read_csv("data/raw/la_benchmarking/la_ebewe_all.csv", low_memory=False)
    uniq = df[["BUILDING ADDRESS", "POSTAL CODE"]].dropna(subset=["BUILDING ADDRESS"])
    uniq = uniq.drop_duplicates()
    records = []
    for _, r in uniq.iterrows():
        addr = str(r["BUILDING ADDRESS"]).strip()
        zip_ = str(r["POSTAL CODE"]).strip() if pd.notna(r["POSTAL CODE"]) else ""
        full = f"{addr}, Los Angeles, CA {zip_}".strip()
        records.append(("la", addr, full))
    return records


def load_denver_addresses():
    df = pd.read_csv("data/raw/denver_benchmarking/denver_2024.csv", low_memory=False)
    uniq = df[["Street", "Zipcode"]].dropna(subset=["Street"]).drop_duplicates()
    records = []
    for _, r in uniq.iterrows():
        addr = str(r["Street"]).strip()
        zip_ = str(r["Zipcode"]).strip() if pd.notna(r["Zipcode"]) else ""
        full = f"{addr}, Denver, CO {zip_}".strip()
        records.append(("denver", addr, full))
    return records


def geocode_one(full_address):
    params = {
        "q": full_address,
        "format": "json",
        "limit": 1,
        "addressdetails": 0,
        "countrycodes": "us",
    }
    headers = {"User-Agent": USER_AGENT}
    try:
        r = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=15)
        if r.status_code != 200:
            return None, None, f"http_{r.status_code}"
        data = r.json()
        if not data:
            return None, None, "no_match"
        return float(data[0]["lat"]), float(data[0]["lon"]), "ok"
    except Exception as e:
        return None, None, f"err:{type(e).__name__}"


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    # Load existing results (for resume)
    if os.path.exists(OUT_PATH):
        existing = pd.read_csv(OUT_PATH)
        done = set(zip(existing["city"], existing["raw_address"]))
        rows = existing.to_dict("records")
        print(f"[resume] loaded {len(done)} existing entries from {OUT_PATH}")
    else:
        done = set()
        rows = []

    tasks = load_la_addresses() + load_denver_addresses()
    todo = [t for t in tasks if (t[0], t[1]) not in done]
    print(f"Total unique addresses: {len(tasks)}  pending: {len(todo)}")
    if not todo:
        print("nothing to do.")
        return

    FLUSH_EVERY = 50
    try:
        for i, (city, raw_addr, full_addr) in enumerate(todo, start=1):
            lat, lon, status = geocode_one(full_addr)
            rows.append({
                "city": city,
                "raw_address": raw_addr,
                "lat": lat,
                "lon": lon,
                "geocoder": "nominatim",
                "status": status,
            })
            if i % FLUSH_EVERY == 0 or i == len(todo):
                pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
                ok = sum(1 for r in rows if r["status"] == "ok")
                print(f"  [{i}/{len(todo)}]  ok={ok}  last='{full_addr[:50]}' → {status}")
                sys.stdout.flush()
            time.sleep(RATE_LIMIT_SEC)
    except KeyboardInterrupt:
        print("\n[interrupt] flushing partial results...")
        pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
        print(f"  saved {len(rows)} entries to {OUT_PATH}; rerun to resume.")
        return

    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    ok = sum(1 for r in rows if r["status"] == "ok")
    print(f"\nDone. {ok}/{len(rows)} successfully geocoded. → {OUT_PATH}")


if __name__ == "__main__":
    main()
