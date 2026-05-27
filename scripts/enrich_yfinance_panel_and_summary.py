"""
P1a + P1b: for every ticker already matched to an NZDPU company, fetch

  (A) longBusinessSummary — free-text company description (P1b, for T1-A-LLM)
  (B) income_stmt        — 4-year history of Revenue / EBITDA / NetIncome
                            (P1a, for time-aligned financials)

Inputs:
  data/company-level/nzdpu_enriched/ticker_cache.json
    → set of tickers to fetch (already matched from previous run)

Outputs:
  data/company-level/nzdpu_enriched/extra_cache.json
    → {ticker: {summary: str, panel: {year: {revenue, ebitda, ...}}}}  (resumable)

  data/company-level/nzdpu_enriched/business_summaries.csv
    → (ticker, business_summary)                                (P1b final)

  data/company-level/nzdpu_enriched/financials_panel.csv
    → (ticker, year, revenue_usd, ebitda_usd, net_income_usd, operating_income_usd)
                                                                 (P1a final)
"""

import yfinance as yf
import pandas as pd
import json
import time
import sys
from pathlib import Path

OUTPUT_DIR = Path("data/company-level/nzdpu_enriched")
TICKER_CACHE = OUTPUT_DIR / "ticker_cache.json"
EXTRA_CACHE  = OUTPUT_DIR / "extra_cache.json"
OUT_SUMMARY  = OUTPUT_DIR / "business_summaries.csv"
OUT_PANEL    = OUTPUT_DIR / "financials_panel.csv"

SLEEP = 0.3
SAVE_EVERY = 50

FIELDS = {
    "revenue_usd":          "Total Revenue",
    "ebitda_usd":           "EBITDA",
    "net_income_usd":       "Net Income",
    "operating_income_usd": "Operating Income",
}


def load_json(p):
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def save_json(d, p):
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(d, f, ensure_ascii=False)
    tmp.replace(p)


def fetch_one(ticker: str) -> dict:
    """One Yahoo fetch → extract summary + per-year panel."""
    out = {"summary": None, "panel": {}, "error": None}
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        out["summary"] = info.get("longBusinessSummary")

        ic = t.income_stmt
        if ic is not None and not ic.empty:
            for out_key, idx_name in FIELDS.items():
                if idx_name not in ic.index:
                    continue
                row = ic.loc[idx_name]
                for col, val in row.items():
                    year = str(col.year) if hasattr(col, "year") else str(col)[:4]
                    try:
                        f = float(val)
                        if f != f:  # NaN check
                            continue
                    except (TypeError, ValueError):
                        continue
                    out["panel"].setdefault(year, {})[out_key] = f
    except Exception as e:
        out["error"] = str(e)[:200]
    return out


def main():
    # Load which tickers to hit
    ticker_cache = load_json(TICKER_CACHE)
    tickers = sorted({
        v["ticker"] for v in ticker_cache.values()
        if isinstance(v, dict) and v.get("ticker")
    })
    print(f"Unique tickers to enrich: {len(tickers):,}")

    extra = load_json(EXTRA_CACHE)
    print(f"Already cached: {len(extra):,}")

    todo = [t for t in tickers if t not in extra]
    print(f"To fetch: {len(todo):,}")

    if todo:
        print(f"Est. time: ~{len(todo) * (SLEEP + 0.8) / 60:.1f} min\n", flush=True)

    t0 = time.time()
    for i, ticker in enumerate(todo, 1):
        extra[ticker] = fetch_one(ticker)
        if i % SAVE_EVERY == 0 or i == len(todo):
            save_json(extra, EXTRA_CACHE)
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (len(todo) - i) / rate if rate > 0 else 0
            have_summary = sum(1 for v in extra.values() if v.get("summary"))
            have_panel   = sum(1 for v in extra.values() if v.get("panel"))
            print(f"  [{i}/{len(todo)}] {rate:.2f}/s | summary={have_summary} panel={have_panel} | eta={eta/60:.1f} min",
                  flush=True)
        time.sleep(SLEEP)

    save_json(extra, EXTRA_CACHE)
    print(f"\nFetch done. Total cached: {len(extra):,}")

    # ── Write P1b: summaries ────────────────────────────────────────────────
    sum_rows = [{"ticker": t, "business_summary": v["summary"]}
                for t, v in extra.items() if v.get("summary")]
    pd.DataFrame(sum_rows).to_csv(OUT_SUMMARY, index=False)
    print(f"Summaries: {len(sum_rows):,} rows → {OUT_SUMMARY}")

    # ── Write P1a: panel ────────────────────────────────────────────────────
    panel_rows = []
    for ticker, v in extra.items():
        for year, vals in (v.get("panel") or {}).items():
            panel_rows.append({"ticker": ticker, "year": int(year), **vals})
    panel = pd.DataFrame(panel_rows)
    panel.to_csv(OUT_PANEL, index=False)
    print(f"Panel:     {len(panel):,} rows → {OUT_PANEL}")
    if len(panel):
        print(f"  years covered: {sorted(panel['year'].unique())}")
        print(f"  unique tickers in panel: {panel['ticker'].nunique():,}")


if __name__ == "__main__":
    main()
