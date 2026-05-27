"""
T1-A-LLM ablation: predict Scope 1+2 emissions with an LLM instead of XGBoost.

Three modes:
  zero_shot     — structured features + business_summary, no examples
  few_shot      — add 10 labeled examples in prompt (one per GICS sector)
  no_summary    — structured features only (tests whether summary text helps)

Eval set: data/company-level/splits/t1a_llm_sample.csv (200 rows, stratified)
Few-shot pool: data/company-level/splits/t1a_llm_fewshot.csv (11 rows, one/sector)

Usage:
  export ANTHROPIC_API_KEY=...    # or set OPENAI_API_KEY
  python scripts/run_t1_a_llm.py --mode zero_shot --model claude-haiku-4-5
  python scripts/run_t1_a_llm.py --mode few_shot  --model claude-haiku-4-5
  python scripts/run_t1_a_llm.py --mode no_summary --model claude-haiku-4-5

Output: results/t1a_llm_{mode}_{model}.csv  — per-row predictions
        appended to results/t1a_results.csv with matching metric schema
"""

import os
import re
import json
import time
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

SAMPLE_FILE  = Path("data/company-level/splits/t1a_llm_sample.csv")
FEWSHOT_FILE = Path("data/company-level/splits/t1a_llm_fewshot.csv")
RESULTS_FILE = Path("results/t1a_results.csv")

# Rough FX rates (LOCAL_PER_USD) for converting yfinance revenue to USD.
# Source: mid-2024 spot rates. For benchmark baseline, this is fine; we can
# replace with year-specific rates in a later version.
FX_PER_USD = {
    "JPY": 150.0, "EUR": 0.92, "GBP": 0.78, "CNY": 7.2, "KRW": 1370,
    "INR": 84, "CAD": 1.35, "AUD": 1.50, "BRL": 5.2, "MXN": 17,
    "SEK": 10.5, "TWD": 32, "CHF": 0.88, "HKD": 7.8, "SGD": 1.35,
    "NOK": 10.9, "DKK": 6.9, "ZAR": 18, "THB": 36, "IDR": 15800,
    "TRY": 34, "PLN": 4.0, "HUF": 370, "ILS": 3.7, "AED": 3.67,
    "SAR": 3.75, "CZK": 23, "PHP": 57, "MYR": 4.7, "RON": 4.6,
    "COP": 4200, "CLP": 950, "PEN": 3.8, "EGP": 49, "PKR": 280,
    "USD": 1.0,
}


def fx_convert_to_musd(amount, currency):
    """amount is in local currency 'units' (not millions). Returns millions USD."""
    if pd.isna(amount) or amount is None or pd.isna(currency):
        return np.nan
    rate = FX_PER_USD.get(str(currency).upper())
    if not rate:
        return np.nan
    return (amount / rate) / 1e6


def build_prompt(row, fewshot_rows=None, include_summary=True):
    """Build the prompt for a single company."""
    lines = [
        "You are an expert in corporate greenhouse-gas accounting. Given the",
        "company information below, predict Scope 1 + Scope 2 emissions in",
        "metric tons of CO2-equivalent (tCO2e) for the reporting year.",
        "",
        "Respond with a SINGLE NUMBER ONLY — the predicted tCO2e value.",
        "Do not include units, explanations, or punctuation. Example: 42500",
        "",
    ]

    if fewshot_rows is not None and len(fewshot_rows) > 0:
        lines.append("=== Labeled examples ===\n")
        for _, ex in fewshot_rows.iterrows():
            lines.extend(_format_row(ex, include_summary=include_summary))
            lines.append(f"Scope 1+2 emissions: {int(ex['scope12_actual_tco2e']):d} tCO2e")
            lines.append("")
        lines.append("=== Target company ===\n")

    lines.extend(_format_row(row, include_summary=include_summary))
    lines.append("Scope 1+2 emissions:")
    return "\n".join(lines)


def _format_row(row, include_summary=True):
    sector = row.get("gics_11_b") or row.get("gics_11") or "Unknown"
    revenue = row["revenue_best_musd"]
    # FX correction (if we know the financial currency)
    currency = row.get("currency") or row.get("financialCurrency")
    if pd.notna(currency) and str(currency) != "USD" and currency in FX_PER_USD:
        # revenue_best_musd was stored assuming the field was in USD.
        # The raw value is (amount_local_currency / 1e6). To convert to MUSD
        # we divide by FX rate again.
        revenue_musd = revenue / FX_PER_USD[currency]
    else:
        revenue_musd = revenue

    lines = [
        f"Company: {row['company_name']}",
        f"Country: {row['country_iso2']}",
        f"Sector (GICS): {sector}",
        f"Reporting year: {int(row['reporting_year'])}",
        f"Annual revenue: ~${revenue_musd:,.0f} million USD",
    ]
    if include_summary and pd.notna(row.get("business_summary")):
        summary = str(row["business_summary"])[:800]  # cap at 800 chars
        lines.append(f"Business description: {summary}")
    return lines + [""]


def parse_prediction(text):
    """Extract a number from the LLM response."""
    # Look for the first number (may have commas, scientific notation, etc.)
    m = re.search(r"(-?[\d,]+\.?\d*(?:[eE][-+]?\d+)?)", text)
    if not m:
        return np.nan
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return np.nan


def call_claude(prompt, model="claude-haiku-4-5"):
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model,
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def call_openai(prompt, model="gpt-4o-mini"):
    import openai
    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model,
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


_GEMINI_CLIENT = None
def call_gemini(prompt, model="gemini-2.0-flash"):
    from google import genai
    from google.genai import errors as gerrors
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        _GEMINI_CLIENT = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    for attempt in range(6):
        try:
            resp = _GEMINI_CLIENT.models.generate_content(
                model=model, contents=prompt,
                config={"max_output_tokens": 100, "temperature": 0.0})
            return resp.text or ""
        except gerrors.ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 12 * (attempt + 1)
                print(f"   gemini 429 → sleep {wait}s", flush=True)
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("gemini: exhausted retries")


def metrics(y_true_log, y_pred_log):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y_true_log = np.asarray(y_true_log)
    y_pred_log = np.asarray(y_pred_log)
    mask = np.isfinite(y_pred_log)  # drop rows the LLM failed to parse
    y_true_log, y_pred_log = y_true_log[mask], y_pred_log[mask]
    if len(y_true_log) == 0:
        return dict(n=0)
    y_true = 10 ** y_true_log; y_pred = 10 ** y_pred_log
    ape = np.abs(y_true - y_pred) / np.abs(y_true)
    return dict(
        n         = int(len(y_true_log)),
        mae_log   = mean_absolute_error(y_true_log, y_pred_log),
        rmse_log  = mean_squared_error(y_true_log, y_pred_log) ** 0.5,
        r2_log    = r2_score(y_true_log, y_pred_log),
        pearson_r = float(np.corrcoef(y_true_log, y_pred_log)[0, 1]),
        mape      = float(ape.mean()),
        median_ape= float(np.median(ape)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["zero_shot", "few_shot", "no_summary"], required=True)
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--provider", choices=["claude", "openai", "gemini"], default="claude")
    ap.add_argument("--limit", type=int, default=None, help="Cap eval rows (for smoke testing)")
    ap.add_argument("--resume", action="store_true",
                    help="Skip rows already in the predictions CSV")
    ap.add_argument("--flush-every", type=int, default=20,
                    help="Write predictions CSV every N rows (for crash safety)")
    args = ap.parse_args()

    sample = pd.read_csv(SAMPLE_FILE)
    fewshot = pd.read_csv(FEWSHOT_FILE) if args.mode == "few_shot" else None
    if args.limit:
        sample = sample.head(args.limit)

    caller = {"claude": call_claude, "openai": call_openai,
              "gemini": call_gemini}[args.provider]
    include_summary = args.mode != "no_summary"

    # Resume from prior partial run
    out_pred = Path(f"results/t1a_llm_{args.mode}_{args.model.replace('/','_')}_predictions.csv")
    preds = []
    done_keys = set()
    if args.resume and out_pred.exists():
        prior = pd.read_csv(out_pred)
        for _, r in prior.iterrows():
            preds.append({
                "nz_id": int(r["nz_id"]),
                "reporting_year": int(r["reporting_year"]),
                "actual_scope12": float(r["actual_scope12"]),
                "pred_scope12": r["pred_scope12"],
                "raw_response": r.get("raw_response", ""),
            })
            done_keys.add((int(r["nz_id"]), int(r["reporting_year"])))
        print(f"Resume: loaded {len(done_keys)} prior predictions from {out_pred}")
        sample = sample[~sample.apply(
            lambda r: (int(r["nz_id"]), int(r["reporting_year"])) in done_keys,
            axis=1)].reset_index(drop=True)
        print(f"  remaining rows to run: {len(sample)}")
    t0 = time.time()
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        prompt = build_prompt(row, fewshot_rows=fewshot, include_summary=include_summary)
        try:
            text = caller(prompt, model=args.model)
            pred = parse_prediction(text)
        except Exception as e:
            print(f"[{i}] ERROR: {e}")
            text = ""; pred = np.nan
        preds.append({
            "nz_id": int(row["nz_id"]),
            "reporting_year": int(row["reporting_year"]),
            "actual_scope12": float(row["scope12_actual_tco2e"]),
            "pred_scope12": pred,
            "raw_response": text,
        })
        if i % 10 == 0 or i == len(sample):
            rate = i / (time.time() - t0)
            eta = (len(sample) - i) / rate if rate > 0 else 0
            ok = sum(1 for p in preds if np.isfinite(p["pred_scope12"]))
            print(f"  [{i}/{len(sample)}] {rate:.1f}/s | parsed={ok} | eta={eta/60:.1f}min",
                  flush=True)
        if i % args.flush_every == 0:
            pd.DataFrame(preds).to_csv(out_pred, index=False)

    pd.DataFrame(preds).to_csv(out_pred, index=False)
    print(f"\nPer-row predictions → {out_pred}")

    y_true = np.log10([p["actual_scope12"] for p in preds])
    y_pred = np.array([
        np.log10(p["pred_scope12"]) if p["pred_scope12"] and p["pred_scope12"] > 0 else np.nan
        for p in preds
    ])
    m = metrics(y_true, y_pred)
    print(f"\n=== {args.mode} / {args.model} ===")
    for k, v in m.items():
        print(f"  {k:12s}: {v}")

    # Append to results/t1a_results.csv
    row = {
        "subset": "T1-Strict", "feature_set": args.mode, "model": args.model,
        "n_train": 0 if args.mode != "few_shot" else 11,
        "n_test": m.get("n", 0),
        "mae_log": m.get("mae_log"), "rmse_log": m.get("rmse_log"),
        "r2_log":  m.get("r2_log"),  "pearson_r": m.get("pearson_r"),
        "mape":    m.get("mape"),    "median_ape": m.get("median_ape"),
    }
    if RESULTS_FILE.exists():
        existing = pd.read_csv(RESULTS_FILE)
        pd.concat([existing, pd.DataFrame([row])], ignore_index=True).to_csv(RESULTS_FILE, index=False)
    else:
        pd.DataFrame([row]).to_csv(RESULTS_FILE, index=False)
    print(f"Appended to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
