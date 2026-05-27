"""
Task A baseline with TabPFN (tabular foundation model), tabular-only.

Runs the same Task A protocol as the classic tabular benchmark:
- A1: per-city models
- A2: pooled model
- split types: random, grouped

We keep this as a separate runner because TabPFN has different scaling
characteristics from conventional baselines and may subsample large training
folds for practicality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import time
import warnings

import numpy as np
import pandas as pd
import torch

from tabpfn import TabPFNRegressor

from src.data.preprocessing import load_and_prepare, prepare_features
from src.data.feature_sets import TARGET, CATEGORICAL_COLS, get_feature_set
from src.data.splitters import random_split, grouped_split
from src.evaluation.metrics import compute_all_metrics

warnings.filterwarnings("ignore")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CLIMATE_FEATURE_COLS = {
    "hdd", "cdd",
    "annual_mean_temp_c", "annual_rh_mean",
    "annual_ssrd_mj_m2_day", "annual_wind_ms",
}


def _split(df: pd.DataFrame, split_type: str, seed: int):
    if split_type == "grouped":
        return grouped_split(df, test_size=0.2, val_size=0.1, seed=seed)
    return random_split(df, test_size=0.2, val_size=0.1, seed=seed)


def _categorical_indices(feature_names: list[str]) -> list[int]:
    return [i for i, name in enumerate(feature_names) if name in CATEGORICAL_COLS]


def _subsample_if_needed(X, y, max_train, seed):
    if len(X) <= max_train:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max_train, replace=False)
    return X[idx], y[idx]


def _prepare_xy(train_df, test_df, feature_set_name):
    X_train, y_train, feat_names, encoders, medians, dropped_cols = prepare_features(
        train_df, feature_set_name, encoders=None, drop_all_nan=True
    )
    X_test, y_test, _, _, _, _ = prepare_features(
        test_df,
        feature_set_name,
        encoders=encoders,
        medians=medians,
        drop_cols=dropped_cols,
    )
    return X_train, y_train, X_test, y_test, feat_names


def fit_eval(train_df, test_df, feature_set_name, seed, max_train, device):
    X_train, y_train, X_test, y_test, feat_names = _prepare_xy(
        train_df, test_df, feature_set_name
    )

    if len(X_train) < 2 or len(X_test) == 0:
        return None

    X_fit, y_fit = _subsample_if_needed(X_train, y_train, max_train=max_train, seed=seed)
    clip_max = np.max(y_fit) * 2 if len(y_fit) else 1.0

    categorical_idx = _categorical_indices(feat_names)

    def _run(device_name: str):
        model = TabPFNRegressor(
            device=device_name,
            ignore_pretraining_limits=True,
            random_state=seed,
            categorical_features_indices=categorical_idx,
        )
        model.fit(X_fit, y_fit)
        # Chunked predict — TabPFN v2 holds the train set in GPU memory per
        # inference call; feeding 90K+ test rows at once OOMs even on a 24GB
        # GPU. Split test into chunks so memory footprint stays bounded.
        chunk = 4096
        if len(X_test) > chunk:
            parts = []
            for i in range(0, len(X_test), chunk):
                parts.append(model.predict(X_test[i:i + chunk]))
            y_pred = np.concatenate(parts, axis=0)
        else:
            y_pred = model.predict(X_test)
        return np.clip(y_pred, 0, clip_max), device_name

    preferred_device = device
    if preferred_device == "auto":
        preferred_device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        y_pred, used_device = _run(preferred_device)
    except RuntimeError as e:
        if preferred_device == "cuda" and "CUDA" in str(e):
            print(f"    TabPFN CUDA failed, retrying on CPU: {e}")
            y_pred, used_device = _run("cpu")
        else:
            raise

    metrics = compute_all_metrics(y_test, y_pred)
    metrics["n_train"] = len(X_fit)
    metrics["n_test"] = len(X_test)
    metrics["full_train_rows"] = len(X_train)
    metrics["device_used"] = used_device
    return metrics


def run_a1(df, feature_set_name, split_type, seed, max_train, device):
    rows = []
    for city in sorted(df["city"].unique()):
        city_df = df[df["city"] == city].reset_index(drop=True)
        if len(city_df) < 50:
            continue
        splits = _split(city_df, split_type=split_type, seed=seed)
        metrics = fit_eval(
            splits["train"], splits["test"], feature_set_name, seed, max_train, device
        )
        if metrics is None:
            continue
        metrics["city"] = city
        rows.append(metrics)
    return rows


def run_a2(df, feature_set_name, split_type, seed, max_train, device):
    splits = _split(df, split_type=split_type, seed=seed)
    return fit_eval(
        splits["train"], splits["test"], feature_set_name, seed, max_train, device
    )


def flush(results, out_path):
    if results:
        pd.DataFrame(results).to_csv(out_path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_sets", type=str, default="core_all_cities,core_all_cities_climate_plus")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_train", type=int, default=10000)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--split_types", type=str, default="random,grouped")
    ap.add_argument("--mode", type=str, default="both", choices=["pooled", "per_city", "both"])
    ap.add_argument(
        "--out_path",
        type=str,
        default=os.path.join(RESULTS_DIR, "task_a_results_tabpfn.csv"),
    )
    args = ap.parse_args()

    feature_sets = [x for x in args.feature_sets.split(",") if x]
    split_types = [x for x in args.split_types.split(",") if x]
    all_results = []

    print("=" * 70)
    print("Task A: TabPFN baseline")
    print(f"Feature sets: {feature_sets}")
    print(f"Split types: {split_types}")
    print(f"Mode: {args.mode}")
    print(f"Seed: {args.seed}")
    print(f"Max train rows: {args.max_train}")
    resolved_device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {resolved_device}")
    print("=" * 70)

    for fs_name in feature_sets:
        needs_external = any(col in CLIMATE_FEATURE_COLS for col in get_feature_set(fs_name))
        df = load_and_prepare(fs_name, join_external=needs_external)
        df = df[df[TARGET].notna()].copy()

        print(f"\nFeature set: {fs_name}")
        print(f"Total rows with target: {len(df)}")
        print(f"Cities: {sorted(df['city'].unique())}")

        for split_type in split_types:
            if args.mode in {"per_city", "both"}:
                t0 = time.time()
                a1_rows = run_a1(df, fs_name, split_type, args.seed, args.max_train, args.device)
                if a1_rows:
                    macro_r2 = float(np.nanmean([r["r2"] for r in a1_rows]))
                    macro_mae = float(np.nanmean([r["mae"] for r in a1_rows]))
                    avg_n_train = float(np.nanmean([r["n_train"] for r in a1_rows]))
                    all_results.append({
                        "task": "A1_per_city",
                        "feature_set": fs_name,
                        "model": "TabPFN",
                        "split_type": split_type,
                        "macro_r2": macro_r2,
                        "macro_mae": macro_mae,
                        "n_cities": len(a1_rows),
                        "avg_n_train": avg_n_train,
                        "elapsed_sec": time.time() - t0,
                    })
                    flush(all_results, args.out_path)
                    print(f"  [{split_type}] [A1] TabPFN | macro_R²={macro_r2:6.3f} | macro_MAE={macro_mae:8.1f}")

            if args.mode in {"pooled", "both"}:
                t0 = time.time()
                a2 = run_a2(df, fs_name, split_type, args.seed, args.max_train, args.device)
                if a2 is not None:
                    row = {
                        "task": "A2_pooled",
                        "feature_set": fs_name,
                        "model": "TabPFN",
                        "split_type": split_type,
                        "elapsed_sec": time.time() - t0,
                        **a2,
                    }
                    all_results.append(row)
                    flush(all_results, args.out_path)
                    print(f"  [{split_type}] [A2] TabPFN | R²={row['r2']:6.3f} | MAE={row['mae']:8.1f}")

    print(f"\nResults saved to {args.out_path}")


if __name__ == "__main__":
    main()
