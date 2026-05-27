"""
TabPFN v2 on Task B / C1 / D (fills coverage gap left after Task A TabPFN).

Reuses the Task A TabPFN pattern:
  * subsample training to --max_train (default 10K) because TabPFN's ICL has
    a practical size limit
  * chunked test prediction (chunk=4096) to avoid CUDA OOM on 90K+ test rows
  * single seed per invocation; user launches 3 parallel invocations on
    different GPUs to build the 3-seed table

Subtasks:
  * b1      — train years<=2019, test years>=2021 (per feature_set)
  * b2      — train years<=2018, separate eval on 2019/2020/2021
  * c1      — leave-one-city-out across all cities
  * d1      — leave-one-type-out across all property_type values

Outputs per seed:
  results/clean_building/task_{task}_tabpfn_seed{S}.csv
  (then merge manually across seeds if needed)

Usage:
  CUDA_VISIBLE_DEVICES=5 python scripts/run_tasks_bcd_tabpfn.py --task c1 --seed 42 \
      --feature_sets core_all_cities,core_all_cities_climate_plus
"""
from __future__ import annotations
import sys, os, argparse, time, warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch

from tabpfn import TabPFNRegressor

from src.data.preprocessing import load_and_prepare, prepare_features
from src.data.feature_sets import TARGET, CATEGORICAL_COLS, get_feature_set
from src.data.splitters import forecasting_split, grouped_split
from src.evaluation.metrics import compute_all_metrics

warnings.filterwarnings("ignore")
RESULTS_DIR = Path("results/clean_building")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLIMATE_FEATURE_COLS = {
    "hdd", "cdd", "annual_mean_temp_c", "annual_rh_mean",
    "annual_ssrd_mj_m2_day", "annual_wind_ms",
}
MIN_CITY_ROWS = 50
MIN_TYPE_ROWS = 200


def _categorical_indices(feat_names):
    return [i for i, name in enumerate(feat_names) if name in CATEGORICAL_COLS]


def _subsample(X, y, max_train, seed):
    if len(X) <= max_train:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max_train, replace=False)
    return X[idx], y[idx]


def _fit_predict(train_df, test_df, fs_name, seed, max_train, device):
    """Return metrics dict (or None)."""
    X_train, y_train, feat_names, encoders, medians, dropped = prepare_features(
        train_df, fs_name, encoders=None, drop_all_nan=True
    )
    X_test, y_test, *_ = prepare_features(
        test_df, fs_name, encoders=encoders, medians=medians, drop_cols=dropped,
    )
    if len(X_train) < 2 or len(X_test) == 0:
        return None

    X_fit, y_fit = _subsample(X_train, y_train, max_train, seed)
    clip_max = np.max(y_fit) * 2 if len(y_fit) else 1.0
    cat_idx = _categorical_indices(feat_names)

    model = TabPFNRegressor(
        device=device, ignore_pretraining_limits=True, random_state=seed,
        categorical_features_indices=cat_idx,
    )
    t0 = time.time()
    model.fit(X_fit, y_fit)

    # chunked predict
    chunk = 4096
    preds = []
    for i in range(0, len(X_test), chunk):
        preds.append(model.predict(X_test[i:i + chunk]))
    y_pred = np.clip(np.concatenate(preds, axis=0), 0, clip_max)

    metrics = compute_all_metrics(y_test, y_pred)
    metrics["n_train"] = len(X_fit)
    metrics["n_test"] = len(X_test)
    metrics["full_train_rows"] = len(X_train)
    metrics["elapsed_sec"] = round(time.time() - t0, 1)
    return metrics


def _split_train_val(df, val_frac=0.1, seed=42):
    """Match run_task_c._split_train_val (val split off source cities only)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    cut = int((1 - val_frac) * len(idx))
    return df.iloc[idx[:cut]].reset_index(drop=True), df.iloc[idx[cut:]].reset_index(drop=True)


def run_b(df, fs_name, seed, max_train, device, b2: bool):
    """B1 or B2 forecasting-style split."""
    rows = []
    if not b2:
        # B1: train <= 2019, val = 2020, test >= 2021
        splits = forecasting_split(df, train_end=2019, val_year=2020, test_start=2021, min_years=1)
        train_df, test_df = splits["train"], splits["test"]
        print(f"  [B1 {fs_name}] train={len(train_df):,} test={len(test_df):,}", flush=True)
        m = _fit_predict(train_df, test_df, fs_name, seed, max_train, device)
        if m is not None:
            m.update({"task": "B1_temporal", "feature_set": fs_name, "model": "TabPFN",
                      "seed": seed, "year": "2021+"})
            rows.append(m)
    else:
        # B2: train <= 2018, evaluate separately on 2019, 2020, 2021
        train_df = df[df["year"] <= 2018]
        for yr in [2019, 2020, 2021]:
            test_df = df[df["year"] == yr]
            print(f"  [B2 {fs_name} y={yr}] train={len(train_df):,} test={len(test_df):,}", flush=True)
            m = _fit_predict(train_df, test_df, fs_name, seed, max_train, device)
            if m is not None:
                m.update({"task": "B2_covid", "feature_set": fs_name, "model": "TabPFN",
                          "seed": seed, "year": yr})
                rows.append(m)
    return rows


def run_c1(df, fs_name, seed, max_train, device):
    rows = []
    cities = sorted(df["city"].unique())
    for city in cities:
        target = df[df["city"] == city]
        if len(target) < MIN_CITY_ROWS:
            continue
        source = df[df["city"] != city]
        if len(source) < MIN_CITY_ROWS:
            continue
        train_src, _ = _split_train_val(source, seed=seed)
        t0 = time.time()
        m = _fit_predict(train_src, target, fs_name, seed, max_train, device)
        if m is None:
            continue
        m.update({"task": "C1_LOCO", "feature_set": fs_name, "model": "TabPFN",
                  "seed": seed, "target_city": city})
        rows.append(m)
        print(f"  [{fs_name} LOCO={city}] R²={m['r2']:+.3f}  ({m['elapsed_sec']:.0f}s)", flush=True)
    return rows


def run_d(df, fs_name, seed, max_train, device):
    rows = []
    if "property_type" not in df.columns:
        print(f"  [{fs_name}] no property_type column — skip"); return rows
    types = sorted([t for t in df["property_type"].dropna().unique() if t])
    for ptype in types:
        target = df[df["property_type"] == ptype]
        if len(target) < MIN_TYPE_ROWS:
            continue
        source = df[df["property_type"] != ptype]
        if len(source) < MIN_TYPE_ROWS:
            continue
        train_src, _ = _split_train_val(source, seed=seed)
        m = _fit_predict(train_src, target, fs_name, seed, max_train, device)
        if m is None:
            continue
        m.update({"task": "D1_LOTO", "feature_set": fs_name, "model": "TabPFN",
                  "seed": seed, "target_type": ptype})
        rows.append(m)
        print(f"  [{fs_name} LOTO={ptype}] R²={m['r2']:+.3f}  n_test={m['n_test']}  ({m['elapsed_sec']:.0f}s)", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["b1", "b2", "c1", "d"])
    ap.add_argument("--feature_sets", type=str, default="core_all_cities,core_all_cities_climate_plus")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_train", type=int, default=10000)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out_path", type=str, default=None)
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    feature_sets = [x for x in args.feature_sets.split(",") if x]
    if args.out_path is None:
        args.out_path = str(RESULTS_DIR / f"task_{args.task}_tabpfn_seed{args.seed}.csv")
    out_path = Path(args.out_path)

    all_rows = []
    for fs in feature_sets:
        needs_ext = bool(set(get_feature_set(fs)) & CLIMATE_FEATURE_COLS)
        df = load_and_prepare(fs, join_external=needs_ext)
        df = df[df[TARGET].notna()].copy()
        print(f"\n=== task={args.task} | {fs} | seed={args.seed} | device={device} ===", flush=True)
        print(f"  rows={len(df):,}", flush=True)

        if args.task == "b1":
            rows = run_b(df, fs, args.seed, args.max_train, device, b2=False)
        elif args.task == "b2":
            rows = run_b(df, fs, args.seed, args.max_train, device, b2=True)
        elif args.task == "c1":
            rows = run_c1(df, fs, args.seed, args.max_train, device)
        elif args.task == "d":
            rows = run_d(df, fs, args.seed, args.max_train, device)

        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(out_path, index=False)

    print(f"\n{len(all_rows)} rows → {out_path}")


if __name__ == "__main__":
    main()
