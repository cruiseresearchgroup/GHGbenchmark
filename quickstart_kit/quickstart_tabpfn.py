"""GHGbench Quickstart: reproduce TabPFN v2 on the 26-city core+climate tier.

Expected output (single seed=42, paper's stratified grouped-building split,
10K TabPFN subsample):

    R^2  ~= 0.451
    MAE  ~= 380.5
    n_test = 94,875

This matches the seed=42 row of the 3-seed average reported in the paper
(R^2 = 0.479 +/- 0.024 on this tier; the other seeds are 123 and 456).

Runtime: ~3 min on a single GPU (tested on RTX A5000), ~25 min on CPU.

Usage:
    pip install -r requirements_quickstart.txt
    python quickstart_tabpfn.py --data_dir <path/to/quickstart_data>

Required data files (download from the dataset host listed in the parent
landing_page.md, then unzip into <data_dir>):
    quickstart_data/building_quickstart.csv     (~47 MB)
    quickstart_data/splits_grouped_seed42.npz   (~0.7 MB)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = ["gross_floor_area_sqft", "latitude", "longitude", "year", "hdd", "cdd"]
TARGET = "total_ghg_emissions_mtco2e"


def median_impute(X_train: np.ndarray, X_test: np.ndarray):
    medians = np.nanmedian(X_train, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    X_train = np.where(np.isnan(X_train), medians, X_train)
    X_test = np.where(np.isnan(X_test), medians, X_test)
    return X_train, X_test


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=Path, default=Path("quickstart_data"),
                   help="Directory containing building_quickstart.csv + "
                        "splits_grouped_seed42.npz")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train", type=int, default=10000,
                   help="TabPFN subsample size (paper uses 10000)")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = p.parse_args()

    data_csv = args.data_dir / "building_quickstart.csv"
    splits_npz = args.data_dir / "splits_grouped_seed42.npz"
    for f in (data_csv, splits_npz):
        if not f.exists():
            raise SystemExit(
                f"Required file not found: {f}\n"
                f"Download the quickstart bundle from the dataset host "
                f"(see landing_page.md)."
            )

    print(f"[1/4] Loading {data_csv} ...", flush=True)
    df = pd.read_csv(data_csv)
    print(f"      {len(df):,} rows, {df['building_id'].nunique():,} buildings, "
          f"{df['city'].nunique()} cities")

    print(f"[2/4] Loading precomputed split (seed=42) ...", flush=True)
    sp = np.load(splits_npz)
    train_idx, val_idx, test_idx = sp["train_idx"], sp["val_idx"], sp["test_idx"]
    print(f"      train={len(train_idx):,}  val={len(val_idx):,}  test={len(test_idx):,}")

    # Paper's TabPFN baseline (run_task_a_tabpfn.py) uses train rows only
    # (not train+val). Match that exactly for bit-faithful reproduction.
    X_tr = df.iloc[train_idx][FEATURES].to_numpy(dtype=float)
    y_tr = df.iloc[train_idx][TARGET].to_numpy(dtype=float)
    X_te = df.iloc[test_idx][FEATURES].to_numpy(dtype=float)
    y_te = df.iloc[test_idx][TARGET].to_numpy(dtype=float)
    X_tr, X_te = median_impute(X_tr, X_te)

    rng = np.random.default_rng(args.seed)
    if len(X_tr) > args.max_train:
        idx = rng.choice(len(X_tr), size=args.max_train, replace=False)
        X_fit, y_fit = X_tr[idx], y_tr[idx]
    else:
        X_fit, y_fit = X_tr, y_tr
    clip_max = float(np.max(y_fit) * 2)

    print(f"[3/4] Fitting TabPFN v2 on {len(X_fit):,} rows "
          f"(features: {FEATURES}) ...", flush=True)
    import torch
    from tabpfn import TabPFNRegressor
    use_dev = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"      device={use_dev}")
    t0 = time.time()
    model = TabPFNRegressor(device=use_dev, ignore_pretraining_limits=True,
                            random_state=args.seed)
    model.fit(X_fit, y_fit)

    print(f"[4/4] Predicting on {len(X_te):,} test rows ...", flush=True)
    chunk = 4096
    parts = [model.predict(X_te[i:i + chunk]) for i in range(0, len(X_te), chunk)]
    y_pred = np.clip(np.concatenate(parts), 0, clip_max)
    elapsed = time.time() - t0

    ss_res = float(np.sum((y_te - y_pred) ** 2))
    ss_tot = float(np.sum((y_te - y_te.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    mae = float(np.mean(np.abs(y_te - y_pred)))

    print()
    print("=" * 56)
    print(f"  TabPFN v2 | core_all_cities (+climate) | seed={args.seed}")
    print("=" * 56)
    print(f"  R^2     = {r2:.4f}    (paper 3-seed mean: 0.479 +/- 0.024)")
    print(f"  MAE     = {mae:.1f}")
    print(f"  n_test  = {len(y_te):,}")
    print(f"  runtime = {elapsed:.0f}s on {use_dev}")
    print("=" * 56)


if __name__ == "__main__":
    main()
