"""
Batch download company-level GHG emissions from Climate Data Utility API
(formerly NZDPU, now at climatedatautility.org).

Output: data/company-level/nzdpu/nzdpu_emissions.csv

How to get an API token (free):
  1. Register at https://climatedatautility.org/ (no institutional affiliation
     required; takes ~1 minute).
  2. Go to your account settings -> API tokens -> create a new token.
  3. Export it before running this script:
        export NZDPU_API_TOKEN=<your_token>
  4. Run: python scripts/download_nzdpu.py

The downloader is fully resumable via the checkpoint file, so an interrupted
run can simply be restarted with the same token.
"""

import os
import sys
import requests
import urllib3
import pandas as pd
import time
import json
from pathlib import Path

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_TOKEN = os.environ.get("NZDPU_API_TOKEN")
if not API_TOKEN:
    sys.exit(
        "ERROR: NZDPU_API_TOKEN not set.\n"
        "Get a free token at https://climatedatautility.org/ and run:\n"
        "  export NZDPU_API_TOKEN=<your_token>"
    )
BASE_URL   = "https://climatedatautility.org/wis"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "company-level" / "nzdpu"
OUTPUT_FILE      = OUTPUT_DIR / "nzdpu_emissions.csv"
CHECKPOINT_FILE  = OUTPUT_DIR / "nzdpu_checkpoint.json"

HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json",
}

FIELDS = [
    "company_name", "nz_id", "reporting_year", "jurisdiction",
    "sics_sector", "sics_sub_sector", "sics_industry",
    "total_s1_emissions_ghg", "total_s2_lb_emissions_ghg",
    "total_s2_mb_emissions_ghg", "total_s3_emissions_ghg",
    # Scope 3 GHG Protocol categories C1–C15
    "total_s3_ghgp_c1_emissions_ghg",   # Purchased goods & services
    "total_s3_ghgp_c2_emissions_ghg",   # Capital goods
    "total_s3_ghgp_c3_emissions_ghg",   # Fuel & energy activities
    "total_s3_ghgp_c4_emissions_ghg",   # Upstream transportation
    "total_s3_ghgp_c5_emissions_ghg",   # Waste generated in operations
    "total_s3_ghgp_c6_emissions_ghg",   # Business travel
    "total_s3_ghgp_c7_emissions_ghg",   # Employee commuting
    "total_s3_ghgp_c8_emissions_ghg",   # Upstream leased assets
    "total_s3_ghgp_c9_emissions_ghg",   # Downstream transportation
    "total_s3_ghgp_c10_emissions_ghg",  # Processing of sold products
    "total_s3_ghgp_c11_emissions_ghg",  # Use of sold products
    "total_s3_ghgp_c12_emissions_ghg",  # End-of-life treatment
    "total_s3_ghgp_c13_emissions_ghg",  # Downstream leased assets
    "total_s3_ghgp_c14_emissions_ghg",  # Franchises
    "total_s3_ghgp_c15_emissions_ghg",  # Investments
    "org_boundary", "s1_emissions_method", "data_provider",
    "legal_entity_identifier",
]

BATCH_SIZE = 100
SLEEP_SEC  = 0.3


def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"offset": 0, "rows_saved": 0}


def save_checkpoint(offset, rows_saved):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"offset": offset, "rows_saved": rows_saved}, f)


def clean_value(v):
    if v == "\u2014" or v == "—":
        return None
    return v


def fetch_batch(start: int) -> dict:
    payload = {"query": {}, "fields": FIELDS}
    resp = requests.post(
        f"{BASE_URL}/search",
        headers=HEADERS,
        params={"limit": BATCH_SIZE, "start": start},
        json=payload,
        timeout=30,
        verify=False,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint()
    offset     = checkpoint["offset"]
    rows_saved = checkpoint["rows_saved"]

    # Fetch first batch to get total
    print("Fetching first batch to get total count...")
    first = fetch_batch(0)
    total = first["total_disclosures"]
    print(f"Total disclosures: {total:,} | Companies: {first.get('total_companies', '?'):,}")

    if offset > 0:
        print(f"Resuming from offset={offset} (already saved {rows_saved:,} rows)")

    write_mode   = "a" if offset > 0 else "w"
    write_header = offset == 0

    consecutive_errors = 0

    with open(OUTPUT_FILE, write_mode, encoding="utf-8") as f:
        if write_header:
            pd.DataFrame(columns=FIELDS).to_csv(f, index=False)

        # Process first batch if starting fresh
        if offset == 0:
            items = first.get("items", [])
            rows = [{field: clean_value(item.get(field)) for field in FIELDS} for item in items]
            pd.DataFrame(rows, columns=FIELDS).to_csv(f, index=False, header=False)
            rows_saved += len(rows)
            offset = len(rows)
            print(f"  [{offset:>7,}/{total:,}] saved")
            save_checkpoint(offset, rows_saved)
            time.sleep(SLEEP_SEC)

        while offset < total:
            try:
                data  = fetch_batch(offset)
                items = data.get("items", [])
                if not items:
                    break
                rows = [{field: clean_value(item.get(field)) for field in FIELDS} for item in items]
                pd.DataFrame(rows, columns=FIELDS).to_csv(f, index=False, header=False)
                rows_saved += len(rows)
                offset     += len(rows)
                consecutive_errors = 0
                print(f"  [{offset:>7,}/{total:,}] saved")
                save_checkpoint(offset, rows_saved)
                time.sleep(SLEEP_SEC)

            except requests.HTTPError as e:
                if e.response.status_code == 429:
                    print("Rate limited — sleeping 60s")
                    time.sleep(60)
                else:
                    raise
            except Exception as e:
                consecutive_errors += 1
                wait = min(300, 5 * (2 ** consecutive_errors))
                print(f"  Error at offset={offset}: {e} — retrying in {wait}s")
                time.sleep(wait)

    print(f"\nDone. Total rows saved: {rows_saved:,}")
    print(f"Output: {OUTPUT_FILE}")

    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

    df = pd.read_csv(OUTPUT_FILE)
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Unique companies:    {df['nz_id'].nunique():,}")
    print(f"Year range:          {df['reporting_year'].min()} – {df['reporting_year'].max()}")
    print(f"Scope 1 non-null:    {df['total_s1_emissions_ghg'].notna().sum():,}")
    print(f"Scope 2 (LB) non-null: {df['total_s2_lb_emissions_ghg'].notna().sum():,}")
    print(f"Scope 3 non-null:    {df['total_s3_emissions_ghg'].notna().sum():,}")
    print(f"\nSICS sector distribution:")
    print(df['sics_sector'].value_counts().to_string())


if __name__ == "__main__":
    main()
