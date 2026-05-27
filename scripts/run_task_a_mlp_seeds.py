"""
Task A tuned MLP multi-seed refit (GPU).

Takes the best_cfg per feature_set from task_a_mlp_hpsearch_summary.csv and
refits on train+val using 5 seeds, evaluating each on the same held-out
test set. Reports mean ± std + per-seed breakdown so the paper can report
a defensible MLP number rather than a single-seed point estimate that the
hpsearch's val-overfit artefact can collapse to negative R².

Why 5 seeds (not 3): single-seed tuned MLP produced test R² in [-0.45,
+0.47] across feature sets with σ ≈ 0.5 (§1.2 default 3-seed run, matched
by the tuned single-seed summary). 5 seeds gives a tighter mean estimate
without cherry-picking a subset.

Outputs:
  results/clean_building/task_a_mlp_seeds_raw.csv       — one row per (fs, seed)
  results/clean_building/task_a_mlp_seeds_summary.csv   — one row per fs (mean, std, min, max, n)

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/run_task_a_mlp_seeds.py
"""
from __future__ import annotations
import sys, json, time, warnings, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch

from src.data.preprocessing import load_and_prepare, prepare_features
from src.data.feature_sets import TARGET, get_feature_set
from src.data.splitters import grouped_split
from src.evaluation.metrics import compute_all_metrics
from src.models.mlp_gpu import TorchMLPBaseline

warnings.filterwarnings("ignore")

CLIMATE_FEATURE_COLS = {
    "hdd", "cdd",
    "annual_mean_temp_c", "annual_rh_mean",
    "annual_ssrd_mj_m2_day", "annual_wind_ms",
}
SEEDS = [42, 123, 456, 789, 1011]

OUT_DIR = Path("results/clean_building")
HP_SUMMARY = OUT_DIR / "task_a_mlp_hpsearch_summary.csv"
RAW_CSV = OUT_DIR / "task_a_mlp_seeds_raw.csv"
SUMMARY_CSV = OUT_DIR / "task_a_mlp_seeds_summary.csv"


def prepare_split(df, fs_name, split_seed=42):
    """Split seed is kept at 42 so test set is identical to the hpsearch
    test set — we vary only the MODEL seed across runs."""
    splits = grouped_split(df, test_size=0.2, val_size=0.1, seed=split_seed)
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
    X_tr, y_tr, *_, encoders, medians, _ = prepare_features(
        train_df, fs_name, encoders=None
    )
    X_va, y_va, *_ = prepare_features(val_df, fs_name, encoders=encoders, medians=medians)
    X_te, y_te, *_ = prepare_features(test_df, fs_name, encoders=encoders, medians=medians)
    return X_tr, y_tr, X_va, y_va, X_te, y_te


def eval_pair(model, X, y, clip_max):
    y_pred = np.clip(model.predict(X), 0, clip_max)
    return compute_all_metrics(y, y_pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_sets", type=str, default="",
                    help="Comma-separated list; if empty, run all feature_sets in hpsearch summary.")
    ap.add_argument("--out_suffix", type=str, default="",
                    help="Suffix appended to output filenames (e.g. '_au'). Blank uses the default paths.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("MLP multi-seed requires CUDA.")
    device = "cuda"
    print(f"[mlp_seeds] device={device} | gpu={torch.cuda.get_device_name(0)}")

    summ = pd.read_csv(HP_SUMMARY)
    if args.feature_sets:
        wanted = set(args.feature_sets.split(","))
        summ = summ[summ["feature_set"].isin(wanted)].reset_index(drop=True)
    print(f"[mlp_seeds] {len(summ)} tuned configs from {HP_SUMMARY.name}")

    # Allow separate output file per filter (so AU doesn't overwrite US)
    raw_csv = RAW_CSV.with_name(RAW_CSV.stem + args.out_suffix + RAW_CSV.suffix) if args.out_suffix else RAW_CSV
    summary_csv = SUMMARY_CSV.with_name(SUMMARY_CSV.stem + args.out_suffix + SUMMARY_CSV.suffix) if args.out_suffix else SUMMARY_CSV

    raw_rows, summary_rows = [], []
    for _, r in summ.iterrows():
        fs_name = r["feature_set"]
        best_cfg = json.loads(r["best_cfg"])
        # JSON deserialises tuples as lists → coerce hidden_layers back
        if "hidden_layers" in best_cfg and isinstance(best_cfg["hidden_layers"], list):
            best_cfg["hidden_layers"] = tuple(best_cfg["hidden_layers"])

        needs_ext = bool(set(get_feature_set(fs_name)) & CLIMATE_FEATURE_COLS)
        df = load_and_prepare(fs_name, join_external=needs_ext)
        df = df[df[TARGET].notna()].copy()
        X_tr, y_tr, X_va, y_va, X_te, y_te = prepare_split(df, fs_name, split_seed=42)
        clip_max = float(np.max(y_tr) * 2)
        X_tv = np.concatenate([X_tr, X_va], axis=0)
        y_tv = np.concatenate([y_tr, y_va], axis=0)
        print(f"\n=== {fs_name} | train+val={len(y_tv):,} test={len(y_te):,} ===", flush=True)
        print(f"  best_cfg: {best_cfg}", flush=True)

        per_seed_r2, per_seed_mae, per_seed_lm = [], [], []
        for s in SEEDS:
            t0 = time.time()
            # Use a tiny holdout off train+val for early stopping (matches
            # run_task_a_mlp_hpsearch.py's refit protocol).
            rng_refit = np.random.default_rng(s)
            idx = rng_refit.permutation(len(y_tv))
            cut = int(0.92 * len(idx))
            tr_idx, es_idx = idx[:cut], idx[cut:]

            cfg_s = {**best_cfg, "seed": s}
            m = TorchMLPBaseline(device=device, **cfg_s)
            m.fit(X_tv[tr_idx], y_tv[tr_idx], X_val=X_tv[es_idx], y_val=y_tv[es_idx])
            te = eval_pair(m, X_te, y_te, clip_max)
            elapsed = time.time() - t0
            per_seed_r2.append(te["r2"])
            per_seed_mae.append(te["mae"])
            per_seed_lm.append(te["log_mae"])
            raw_rows.append({
                "feature_set": fs_name, "seed": s,
                "test_r2": te["r2"], "test_mae": te["mae"], "test_log_mae": te["log_mae"],
                "elapsed_s": round(elapsed, 1),
            })
            pd.DataFrame(raw_rows).to_csv(raw_csv, index=False)
            print(f"  seed={s:6d} | R²={te['r2']:+.3f} MAE={te['mae']:.1f} "
                  f"log_mae={te['log_mae']:.3f} ({elapsed:.0f}s)", flush=True)

        r2_arr = np.array(per_seed_r2)
        mae_arr = np.array(per_seed_mae)
        lm_arr = np.array(per_seed_lm)
        summary_rows.append({
            "feature_set": fs_name,
            "n_seeds": len(SEEDS),
            "seeds": ",".join(map(str, SEEDS)),
            "test_r2_mean": float(r2_arr.mean()),
            "test_r2_std":  float(r2_arr.std(ddof=1)),
            "test_r2_min":  float(r2_arr.min()),
            "test_r2_max":  float(r2_arr.max()),
            "test_r2_median": float(np.median(r2_arr)),
            "test_mae_mean": float(mae_arr.mean()),
            "test_mae_std":  float(mae_arr.std(ddof=1)),
            "test_log_mae_mean": float(lm_arr.mean()),
            "test_log_mae_std":  float(lm_arr.std(ddof=1)),
            "best_cfg": json.dumps({k: (list(v) if isinstance(v, tuple) else v)
                                    for k, v in best_cfg.items()}),
        })
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
        print(f"  → mean R² = {r2_arr.mean():+.3f} ± {r2_arr.std(ddof=1):.3f}"
              f" (range [{r2_arr.min():+.3f}, {r2_arr.max():+.3f}], median {np.median(r2_arr):+.3f})",
              flush=True)

    print(f"\nRaw     → {raw_csv}")
    print(f"Summary → {summary_csv}")


if __name__ == "__main__":
    main()
