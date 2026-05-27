"""
Enrich NZDPU emissions data with financial features (revenue, employees, sector)
via Yahoo Finance (yfinance).

Strategy:
1. Get unique companies from NZDPU
2. Search Yahoo Finance by company name → get ticker symbol
3. Fetch financials for matched tickers
4. Join back to NZDPU emissions

Output: data/company-level/nzdpu_enriched/nzdpu_with_financials.csv
"""

import yfinance as yf
import pandas as pd
import time
import json
from pathlib import Path
from typing import Optional, Tuple
from rapidfuzz import fuzz

SIMILARITY_THRESHOLD = 60  # token_set_ratio minimum to accept a match

NZDPU_FILE   = Path("data/company-level/nzdpu/nzdpu_emissions.csv")
OUTPUT_DIR   = Path("data/company-level/nzdpu_enriched")
TICKER_CACHE = OUTPUT_DIR / "ticker_cache.json"   # company_name → ticker (or None)
FIN_CACHE    = OUTPUT_DIR / "financials_cache.json"  # ticker → {revenue, employees, ...}
OUTPUT_FILE  = OUTPUT_DIR / "nzdpu_with_financials.csv"

SLEEP_SEARCH = 0.5   # between name searches
SLEEP_INFO   = 0.3   # between ticker info fetches


def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False)


def search_ticker(name: str) -> Tuple[Optional[str], str, int]:
    """Search Yahoo Finance for ticker by company name.
    Returns (ticker, matched_name, similarity_score) or (None, '', 0)."""
    try:
        results = yf.Search(name, max_results=3).quotes
        for r in results:
            sym = r.get("symbol", "")
            yf_name = r.get("longname") or r.get("shortname") or ""
            score = fuzz.token_set_ratio(name.lower().strip(), yf_name.lower().strip())
            if score >= SIMILARITY_THRESHOLD and sym and len(sym) <= 12:
                return sym, yf_name, score
    except Exception:
        pass
    return None, "", 0


def fetch_financials(ticker: str) -> dict:
    """Fetch key financial metrics for a ticker."""
    try:
        info = yf.Ticker(ticker).info
        return {
            "ticker": ticker,
            "yf_name": info.get("longName") or info.get("shortName"),
            "revenue_usd": info.get("totalRevenue"),
            "employees": info.get("fullTimeEmployees"),
            "market_cap_usd": info.get("marketCap"),
            "total_assets_usd": info.get("totalAssets"),
            "ebitda_usd": info.get("ebitda"),
            "yf_sector": info.get("sector"),
            "yf_industry": info.get("industry"),
            "country": info.get("country"),
            "currency": info.get("financialCurrency"),
        }
    except Exception:
        return {"ticker": ticker}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    nzdpu = pd.read_csv(NZDPU_FILE)
    # One row per company (use most recent year for deduplication)
    companies = (nzdpu.sort_values("reporting_year", ascending=False)
                 .drop_duplicates("nz_id")[["nz_id", "company_name", "jurisdiction"]]
                 .reset_index(drop=True))
    print(f"Unique companies to enrich: {len(companies):,}")

    # Load caches
    ticker_cache = load_json(TICKER_CACHE)
    fin_cache    = load_json(FIN_CACHE)

    # ── Step 1: Search for tickers ──────────────────────────────────────────
    to_search = [r for _, r in companies.iterrows()
                 if str(r["nz_id"]) not in ticker_cache]
    print(f"Need to search tickers for {len(to_search):,} companies "
          f"({len(ticker_cache):,} already cached)")

    for i, row in enumerate(to_search):
        ticker, yf_name, score = search_ticker(row["company_name"])
        ticker_cache[str(row["nz_id"])] = {"ticker": ticker, "yf_name": yf_name, "score": score}
        if i % 200 == 0:
            save_json(ticker_cache, TICKER_CACHE)
            found = sum(1 for v in ticker_cache.values() if v.get("ticker"))
            print(f"  [{i+1}/{len(to_search)}] searched | matched so far: {found:,} ({found/(i+1):.0%})")
        time.sleep(SLEEP_SEARCH)

    save_json(ticker_cache, TICKER_CACHE)
    found_tickers = {nz_id: v["ticker"] for nz_id, v in ticker_cache.items() if v.get("ticker")}
    print(f"\nTicker match rate: {len(found_tickers):,} / {len(ticker_cache):,} "
          f"({len(found_tickers)/len(ticker_cache):.0%})")

    # ── Step 2: Fetch financials for matched tickers ─────────────────────────
    unique_tickers = list(set(found_tickers.values()))
    to_fetch = [t for t in unique_tickers if t not in fin_cache]
    print(f"\nFetching financials for {len(to_fetch):,} tickers "
          f"({len(fin_cache):,} already cached)")

    for i, ticker in enumerate(to_fetch):
        fin_cache[ticker] = fetch_financials(ticker)
        if i % 100 == 0:
            save_json(fin_cache, FIN_CACHE)
            print(f"  [{i+1}/{len(to_fetch)}] fetched")
        time.sleep(SLEEP_INFO)

    save_json(fin_cache, FIN_CACHE)

    # ── Step 3: Join back to NZDPU ───────────────────────────────────────────
    # Build nz_id → financials mapping
    fin_rows = []
    for nz_id, v in ticker_cache.items():
        ticker = v.get("ticker") if isinstance(v, dict) else v
        if ticker and ticker in fin_cache:
            row = {"nz_id": int(nz_id)}
            row.update(fin_cache[ticker])
            fin_rows.append(row)

    fin_df = pd.DataFrame(fin_rows)
    print(f"\nFinancials joined for {len(fin_df):,} companies")

    # Merge onto full NZDPU (all years)
    merged = nzdpu.merge(fin_df, on="nz_id", how="left")
    merged.to_csv(OUTPUT_FILE, index=False)
    print(f"Output shape: {merged.shape}")
    print(f"Output: {OUTPUT_FILE}")

    # Coverage report
    print("\n=== Coverage report ===")
    has_rev  = merged["revenue_usd"].notna()
    has_emp  = merged["employees"].notna()
    has_s1   = merged["total_s1_emissions_ghg"].notna()
    complete = has_rev & has_emp & has_s1
    print(f"Rows with revenue:   {has_rev.sum():,} ({has_rev.mean():.0%})")
    print(f"Rows with employees: {has_emp.sum():,} ({has_emp.mean():.0%})")
    print(f"Rows with scope1:    {has_s1.sum():,} ({has_s1.mean():.0%})")
    print(f"Rows with all three: {complete.sum():,} ({complete.mean():.0%})")
    print(f"\nComplete rows (revenue+employees+scope1): {complete.sum():,}")


if __name__ == "__main__":
    main()
