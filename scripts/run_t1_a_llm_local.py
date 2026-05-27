"""
T1-A-LLM ablation with an open-source LLM, run locally on GPU.

Reuses prompt / parse / metric logic from scripts/run_t1_a_llm.py, but
replaces the API call with HuggingFace transformers batched inference.

Default: Qwen/Qwen2.5-7B-Instruct (fp16, fits on a single A5000 24GB).

Usage:
  python scripts/run_t1_a_llm_local.py --mode zero_shot
  python scripts/run_t1_a_llm_local.py --mode few_shot --model Qwen/Qwen2.5-7B-Instruct
"""

import os
# Keep HF cache out of /home (which is small on many clusters)
os.environ.setdefault("HF_HOME", "${HF_HOME:-~/.cache/huggingface}")
os.environ.setdefault("TRANSFORMERS_CACHE", "${HF_HOME:-~/.cache/huggingface}")

import argparse
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from run_t1_a_llm import (
    SAMPLE_FILE, FEWSHOT_FILE, RESULTS_FILE,
    build_prompt, parse_prediction, metrics,
)

from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["zero_shot", "few_shot", "no_summary"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--flush-every", type=int, default=40)
    args = ap.parse_args()

    sample = pd.read_csv(SAMPLE_FILE)
    fewshot = pd.read_csv(FEWSHOT_FILE) if args.mode == "few_shot" else None
    if args.limit:
        sample = sample.head(args.limit)
    include_summary = args.mode != "no_summary"

    mtag = args.model.replace("/", "_")
    out_pred = Path(f"results/t1a_llm_{args.mode}_{mtag}_predictions.csv")
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
        sample = sample[~sample.apply(
            lambda r: (int(r["nz_id"]), int(r["reporting_year"])) in done_keys,
            axis=1)].reset_index(drop=True)
        print(f"Resume: loaded {len(done_keys)} prior rows, {len(sample)} remaining")

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"Loading {args.model} ...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # needed for causal LM batched generation
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s, device map = "
          f"{set(str(p.device) for p in model.parameters())}")

    # ── Build all prompts once ────────────────────────────────────────────
    rows_list = list(sample.iterrows())
    prompts = []
    for _, row in rows_list:
        p = build_prompt(row, fewshot_rows=fewshot, include_summary=include_summary)
        # Qwen instruct chat template — makes it follow the "respond with a single number"
        # instruction much more reliably than raw prompting.
        msgs = [{"role": "user", "content": p}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    # ── Batched generation ────────────────────────────────────────────────
    t0 = time.time()
    N = len(prompts)
    for batch_start in range(0, N, args.batch_size):
        batch_prompts = prompts[batch_start:batch_start + args.batch_size]
        batch_rows = rows_list[batch_start:batch_start + args.batch_size]
        enc = tok(batch_prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=4096).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tok.pad_token_id,
            )
        # Slice off the prompt portion to get only generated tokens
        gen = out[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(gen, skip_special_tokens=True)

        for (_, row), text in zip(batch_rows, texts):
            pred = parse_prediction(text)
            preds.append({
                "nz_id": int(row["nz_id"]),
                "reporting_year": int(row["reporting_year"]),
                "actual_scope12": float(row["scope12_actual_tco2e"]),
                "pred_scope12": pred,
                "raw_response": text,
            })

        done = batch_start + len(batch_prompts)
        rate = done / (time.time() - t0)
        eta = (N - done) / rate if rate > 0 else 0
        ok = sum(1 for p in preds if np.isfinite(p["pred_scope12"]))
        print(f"  [{done}/{N}]  {rate:.2f}/s | parsed={ok} | eta={eta/60:.1f}min",
              flush=True)
        if done % args.flush_every < args.batch_size:
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
