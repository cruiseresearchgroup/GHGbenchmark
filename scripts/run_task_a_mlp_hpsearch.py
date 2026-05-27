"""
Task A MLP HP search — val-based 15-trial random search for TorchMLPBaseline
on the same A2 pooled grouped split used by scripts/run_task_a_hpsearch.py
for the three tree models. Runs on GPU (cuda).

Why: §5.1 of results/building_final_summary_zh.md lists MLP alongside RF/XGB/
LightGBM as a headline tabular baseline, but the tree hpsearch summary does
not include MLP, so currently our MLP numbers are from default HP. This
script closes that gap so the headline table is apples-to-apples across all
four models.

Outputs:
  results/clean_building/task_a_mlp_hpsearch.csv          — every trial
  results/clean_building/task_a_mlp_hpsearch_summary.csv  — one row per feature_set

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/run_task_a_mlp_hpsearch.py --n_trials 15
"""
from __future__ import annotations
import sys, os, argparse, json, time, warnings
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

OUT_DIR = Path("results/clean_building")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TRIALS_CSV = OUT_DIR / "task_a_mlp_hpsearch.csv"
SUMMARY_CSV = OUT_DIR / "task_a_mlp_hpsearch_summary.csv"


def sample_mlp(rng):
    # Hidden layer candidates — always 3 layers for fair comparison with the
    # default (256, 128, 64) baseline; widen / narrow the capacity budget.
    hidden_choices = [
        (512, 256, 128),
        (256, 128, 64),   # default
        (128, 64, 32),
        (512, 512, 256),
        (256, 256, 128),
    ]
    return dict(
        hidden_layers=hidden_choices[int(rng.integers(0, len(hidden_choices)))],
        dropout=float(rng.choice([0.0, 0.1, 0.2, 0.3])),
        lr=float(rng.choice([1e-4, 5e-4, 1e-3, 3e-3])),
        weight_decay=float(rng.choice([0.0, 1e-5, 1e-4, 1e-3])),
        batch_size=int(rng.choice([1024, 2048, 4096])),
        max_epochs=200,
        patience=20,
    )


def prepare_split(df, fs_name, seed=42):
    splits = grouped_split(df, test_size=0.2, val_size=0.1, seed=seed)
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
    X_tr, y_tr, feat_names, encoders, medians, _ = prepare_features(
        train_df, fs_name, encoders=None
    )
    X_va, y_va, *_ = prepare_features(val_df,  fs_name, encoders=encoders, medians=medians)
    X_te, y_te, *_ = prepare_features(test_df, fs_name, encoders=encoders, medians=medians)
    return X_tr, y_tr, X_va, y_va, X_te, y_te


def eval_pair(model, X, y, clip_max):
    y_pred = np.clip(model.predict(X), 0, clip_max)
    return compute_all_metrics(y, y_pred)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feature_sets", type=str,
                   default="core_all_cities,core_all_cities_climate_plus,"
                           "us_core,us_metadata,us_leaky_eui,us_leaky_full")
    p.add_argument("--n_trials", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "MLP hpsearch requires CUDA — neural baselines must run on GPU. "
            "Check CUDA_VISIBLE_DEVICES and torch install."
        )
    device = "cuda"
    print(f"[mlp_hp] device={device} | gpu_name={torch.cuda.get_device_name(0)}")

    feature_sets = args.feature_sets.split(",")
    trial_rows, summary_rows = [], []
    rng = np.random.default_rng(args.seed)

    done_pairs = set()
    if SUMMARY_CSV.exists():
        prev = pd.read_csv(SUMMARY_CSV)
        done_pairs = set(prev["feature_set"])
        summary_rows = prev.to_dict("records")
        if TRIALS_CSV.exists():
            trial_rows = pd.read_csv(TRIALS_CSV).to_dict("records")
        print(f"[resume] skipping {len(done_pairs)} already-done feature_sets")

    for fs_name in feature_sets:
        if fs_name in done_pairs:
            print(f"  ({fs_name}) already done — skip"); continue
        needs_external = bool(set(get_feature_set(fs_name)) & CLIMATE_FEATURE_COLS)
        print(f"\n{'='*70}\nFeature set: {fs_name}  (climate_join={needs_external})\n{'='*70}", flush=True)
        df = load_and_prepare(fs_name, join_external=needs_external)
        df = df[df[TARGET].notna()].copy()
        print(f"  rows: {len(df):,}  |  cities: {df['city'].nunique()}")

        X_tr, y_tr, X_va, y_va, X_te, y_te = prepare_split(df, fs_name, seed=args.seed)
        clip_max = float(np.max(y_tr) * 2)
        print(f"  train={len(X_tr):,} | val={len(X_va):,} | test={len(X_te):,}", flush=True)

        best_val_logmae, best_cfg, best_trial_test = None, None, None
        for t in range(args.n_trials):
            cfg = sample_mlp(rng)
            t0 = time.time()
            try:
                m = TorchMLPBaseline(device=device, **cfg)
                m.fit(X_tr, y_tr, X_val=X_va, y_val=y_va)
                val = eval_pair(m, X_va, y_va, clip_max)
                test = eval_pair(m, X_te, y_te, clip_max)
            except Exception as e:
                print(f"    trial {t:2d} FAILED: {type(e).__name__}: {e}", flush=True)
                continue
            row = {
                "feature_set": fs_name, "model": "MLP", "trial": t,
                "elapsed_s": round(time.time() - t0, 1),
                "val_r2": val["r2"], "val_mae": val["mae"], "val_log_mae": val["log_mae"],
                "test_r2": test["r2"], "test_mae": test["mae"], "test_log_mae": test["log_mae"],
                "cfg_hidden_layers": "-".join(map(str, cfg["hidden_layers"])),
                **{f"cfg_{k}": v for k, v in cfg.items() if k != "hidden_layers"},
            }
            trial_rows.append(row)
            pd.DataFrame(trial_rows).to_csv(TRIALS_CSV, index=False)
            print(f"    trial {t:2d} | val R²={val['r2']:+.3f} log_mae={val['log_mae']:.3f}"
                  f" | test R²={test['r2']:+.3f} log_mae={test['log_mae']:.3f}"
                  f" | h={row['cfg_hidden_layers']} lr={cfg['lr']:.0e} wd={cfg['weight_decay']:.0e}"
                  f" drop={cfg['dropout']:.1f} bs={cfg['batch_size']} | {row['elapsed_s']:.0f}s", flush=True)
            if best_val_logmae is None or val["log_mae"] < best_val_logmae:
                best_val_logmae, best_cfg, best_trial_test = val["log_mae"], cfg, test

        if best_cfg is None:
            print(f"    [MLP] all trials failed — skipping {fs_name}"); continue

        # Refit on train+val with best config; use a tiny holdout off train+val
        # for early stopping (same pattern as tree hpsearch).
        print(f"  [MLP] best val log_mae={best_val_logmae:.3f}  →  refit on train+val", flush=True)
        t0 = time.time()
        try:
            rng_refit = np.random.default_rng(0)
            X_tv = np.concatenate([X_tr, X_va], axis=0)
            y_tv = np.concatenate([y_tr, y_va], axis=0)
            idx = rng_refit.permutation(len(y_tv))
            cut = int(0.92 * len(idx))
            m = TorchMLPBaseline(device=device, **best_cfg)
            m.fit(X_tv[idx[:cut]], y_tv[idx[:cut]],
                  X_val=X_tv[idx[cut:]], y_val=y_tv[idx[cut:]])
            final_test = eval_pair(m, X_te, y_te, clip_max)
            print(f"    final test R²={final_test['r2']:+.3f} "
                  f"MAE={final_test['mae']:.1f} log_mae={final_test['log_mae']:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"    refit FAILED: {e}  — using best-trial test instead", flush=True)
            final_test = best_trial_test

        cfg_flat = {"cfg_hidden_layers": "-".join(map(str, best_cfg["hidden_layers"]))}
        cfg_flat.update({f"cfg_{k}": v for k, v in best_cfg.items() if k != "hidden_layers"})
        summary_rows.append({
            "feature_set": fs_name, "model": "MLP",
            "n_trials": args.n_trials,
            "best_val_log_mae": best_val_logmae,
            "test_r2_tuned": final_test["r2"],
            "test_mae_tuned": final_test["mae"],
            "test_log_mae_tuned": final_test["log_mae"],
            "best_cfg": json.dumps({k: (list(v) if isinstance(v, tuple) else v)
                                    for k, v in best_cfg.items()}),
            **cfg_flat,
        })
        pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)

    print("\n" + "=" * 70)
    print("MLP HP search complete. Summary:")
    print("=" * 70)
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        cols = ["feature_set", "model", "test_r2_tuned", "test_mae_tuned", "test_log_mae_tuned"]
        print(summary_df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nTrials  → {TRIALS_CSV}")
    print(f"Summary → {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
