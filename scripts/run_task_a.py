"""
Task A: In-Distribution GHG Prediction Experiment.

A1: Per-city models — train/test independently per city
A2: Pooled model — train on all cities, test per city

Default: 7 models × 3 feature sets × 2 split types × 2 modes = 84 experiments.
Reports per-city metrics + macro-averaged (city-equal-weight) summary.

Usage:
    python scripts/run_task_a.py [--feature_sets minimal,standard,full] [--seeds 42,123,456]
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import time
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict

from src.data.preprocessing import load_and_prepare, prepare_features, encode_categoricals
from src.data.feature_sets import TARGET, FEATURE_SETS, get_feature_set


CLIMATE_FEATURE_COLS = {
    'hdd',
    'cdd',
    'annual_mean_temp_c',
    'annual_rh_mean',
    'annual_ssrd_mj_m2_day',
    'annual_wind_ms',
}
from src.data.splitters import random_split, grouped_split, get_city_sample_weights
from src.evaluation.metrics import compute_all_metrics

from src.models.t2_baselines import GlobalMeanBaseline, CityTypeMeanBaseline
from src.models.mlp_gpu import TorchMLPBaseline
from src.models.linear import RidgeBaseline
from src.models.tree import RandomForestBaseline, XGBoostBaseline, LightGBMBaseline

warnings.filterwarnings('ignore')

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_model_zoo(ridge_alpha: float = 1.0):
    """Return dict of model_name -> model_constructor."""
    return {
        'GlobalMean': lambda: GlobalMeanBaseline(),
        'CityTypeMean': lambda: CityTypeMeanBaseline(),
        'Ridge': lambda: RidgeBaseline(alpha=ridge_alpha),
        'RandomForest': lambda: RandomForestBaseline(n_estimators=200, max_depth=10, n_jobs=8),
        'XGBoost': lambda: XGBoostBaseline(n_estimators=200, max_depth=6, n_jobs=8),
        'LightGBM': lambda: LightGBMBaseline(n_estimators=200, max_depth=6, n_jobs=8),
        'MLP': lambda: TorchMLPBaseline(),
    }


def prepare_citytype_features(train_df, test_df, val_df=None):
    """Special feature prep for CityTypeMean baseline.

    This baseline should always use semantic (city, property_type) group keys,
    rather than assuming they are the first two columns of an arbitrary
    feature-set matrix.
    """
    base_cols = ['city', 'property_type', TARGET]
    tr = train_df[base_cols].copy()
    te = test_df[base_cols].copy()
    va = val_df[base_cols].copy() if val_df is not None else None

    tr, encoders = encode_categoricals(tr, columns=['city', 'property_type'], encoders=None)
    te, _ = encode_categoricals(te, columns=['city', 'property_type'], encoders=encoders)
    if va is not None:
        va, _ = encode_categoricals(va, columns=['city', 'property_type'], encoders=encoders)

    X_train = tr[['city', 'property_type']].values.astype(np.float64)
    y_train = tr[TARGET].values.astype(np.float64)
    X_test = te[['city', 'property_type']].values.astype(np.float64)
    y_test = te[TARGET].values.astype(np.float64)
    X_val = y_val = None
    if va is not None:
        X_val = va[['city', 'property_type']].values.astype(np.float64)
        y_val = va[TARGET].values.astype(np.float64)

    train_valid = ~np.isnan(y_train)
    test_valid = ~np.isnan(y_test)
    X_train, y_train = X_train[train_valid], y_train[train_valid]
    X_test, y_test = X_test[test_valid], y_test[test_valid]
    if X_val is not None:
        val_valid = ~np.isnan(y_val)
        X_val, y_val = X_val[val_valid], y_val[val_valid]

    return X_train, y_train, X_test, y_test, X_val, y_val


def run_single_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_set_name: str,
    model_name: str,
    model_fn,
    val_df: pd.DataFrame = None,
) -> dict:
    """
    Train a model and evaluate on test set.

    Returns dict with metrics.
    """
    # Prepare features (fit encoders on train).
    # This path is only invoked by run_per_city_experiment, so drop entirely
    # missing columns using the city's training fold to avoid median=0
    # sentinels polluting the feature matrix.
    if model_name == 'CityTypeMean':
        X_train, y_train, X_test, y_test, X_val, y_val = prepare_citytype_features(
            train_df, test_df, val_df=val_df
        )
        feat_names, encoders, medians, dropped_cols = ['city', 'property_type'], None, None, []
    else:
        X_train, y_train, feat_names, encoders, medians, dropped_cols = prepare_features(
            train_df, feature_set_name, encoders=None, drop_all_nan=True
        )
        X_test, y_test, _, _, _, _ = prepare_features(
            test_df, feature_set_name,
            encoders=encoders, medians=medians, drop_cols=dropped_cols,
        )
        X_val = y_val = None

    if len(X_train) < 2 or len(X_test) == 0:
        return {'mae': np.nan, 'rmse': np.nan, 'r2': np.nan, 'mape': np.nan,
                'log_mae': np.nan, 'nrmse': np.nan,
                'n_train': len(X_train), 'n_test': len(X_test)}

    model = model_fn()

    try:
        # For XGBoost/LightGBM, optionally pass validation data
        if model_name in ('XGBoost', 'LightGBM'):
            if X_val is None and val_df is not None:
                X_val, y_val, _, _, _, _ = prepare_features(
                    val_df, feature_set_name,
                    encoders=encoders, medians=medians, drop_cols=dropped_cols,
                )
            if len(X_val) > 0:
                model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
            else:
                model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        # Clip predictions to reasonable range
        y_pred = np.clip(y_pred, 0, np.max(y_train) * 2)
        metrics = compute_all_metrics(y_test, y_pred)
        metrics['n_train'] = len(X_train)
        metrics['n_test'] = len(X_test)
        return metrics
    except Exception as e:
        print(f"    WARNING: {model_name} failed: {e}")
        return {'mae': np.nan, 'rmse': np.nan, 'r2': np.nan, 'mape': np.nan,
                'log_mae': np.nan, 'nrmse': np.nan,
                'n_train': len(X_train), 'n_test': len(X_test)}


def run_pooled_experiment(
    df: pd.DataFrame,
    feature_set_name: str,
    model_name: str,
    model_fn,
    seed: int = 42,
    split_type: str = 'random',
) -> dict:
    """
    A2: Pooled model — train on all cities, evaluate per city.

    Returns dict with per-city metrics and macro-averaged summary.
    """
    nan_result = {
        'overall': {'mae': np.nan, 'rmse': np.nan, 'r2': np.nan, 'mape': np.nan,
                     'log_mae': np.nan, 'nrmse': np.nan, 'n_test': 0, 'n_train': 0},
        'per_city': {},
        'macro': {},
    }

    # Split (stratified by city + property_type)
    if split_type == 'grouped':
        splits = grouped_split(df, test_size=0.2, val_size=0.1, seed=seed)
    else:
        splits = random_split(df, test_size=0.2, val_size=0.1, seed=seed)
    train_df, val_df, test_df = splits['train'], splits['val'], splits['test']

    # Train on all
    if model_name == 'CityTypeMean':
        X_train, y_train, X_test_overall, y_test_overall, X_val, y_val = prepare_citytype_features(
            train_df, test_df, val_df=val_df
        )
        encoders = medians = None
    else:
        X_train, y_train, feat_names, encoders, medians, _ = prepare_features(
            train_df, feature_set_name, encoders=None
        )
        X_test_overall = y_test_overall = X_val = y_val = None

    if len(X_train) < 2 or len(y_train) == 0:
        print(f"    WARNING: Pooled {model_name} skipped — insufficient training data")
        return nan_result

    model = model_fn()

    try:
        if model_name in ('XGBoost', 'LightGBM'):
            if X_val is None:
                X_val, y_val, _, _, _, _ = prepare_features(
                    val_df, feature_set_name, encoders=encoders, medians=medians
                )
            if len(X_val) > 0:
                model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
            else:
                model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)

        # Clip bound from training
        clip_max = np.max(y_train) * 2

        # Evaluate per city
        cities = test_df['city'].unique()
        city_metrics = {}
        for city in sorted(cities):
            city_test = test_df[test_df['city'] == city]
            if len(city_test) < 5:
                continue
            if model_name == 'CityTypeMean':
                _, _, X_c, y_c, _, _ = prepare_citytype_features(train_df, city_test, val_df=None)
            else:
                X_c, y_c, _, _, _, _ = prepare_features(
                    city_test, feature_set_name, encoders=encoders, medians=medians
                )
            if len(X_c) == 0:
                continue
            y_pred = np.clip(model.predict(X_c), 0, clip_max)
            city_metrics[city] = compute_all_metrics(y_c, y_pred)
            city_metrics[city]['n_test'] = len(X_c)

        # Also compute overall test metrics
        if model_name == 'CityTypeMean':
            X_test, y_test = X_test_overall, y_test_overall
        else:
            X_test, y_test, _, _, _, _ = prepare_features(
                test_df, feature_set_name, encoders=encoders, medians=medians
            )
        y_pred_all = np.clip(model.predict(X_test), 0, clip_max)
        overall = compute_all_metrics(y_test, y_pred_all)
        overall['n_test'] = len(X_test)
        overall['n_train'] = len(X_train)

        # Macro-averaged (city-equal-weight)
        macro = {}
        if city_metrics:
            for metric_name in ['mae', 'rmse', 'r2', 'mape', 'log_mae', 'nrmse']:
                vals = [city_metrics[c][metric_name] for c in city_metrics
                        if not np.isnan(city_metrics[c].get(metric_name, np.nan))]
                macro[f'macro_{metric_name}'] = np.mean(vals) if vals else np.nan

        return {
            'overall': overall,
            'per_city': city_metrics,
            'macro': macro,
        }

    except Exception as e:
        print(f"    WARNING: Pooled {model_name} failed: {e}")
        return nan_result


def run_per_city_experiment(
    df: pd.DataFrame,
    feature_set_name: str,
    model_name: str,
    model_fn,
    seed: int = 42,
    split_type: str = 'random',
) -> dict:
    """
    A1: Per-city models — independent train/test per city.

    Returns dict with per-city metrics and macro-averaged summary.
    """
    cities = sorted(df['city'].unique())
    city_metrics = {}

    for city in cities:
        city_df = df[df['city'] == city]
        if len(city_df) < 50:  # Skip tiny cities
            continue

        # Within-city split
        if split_type == 'grouped':
            splits = grouped_split(city_df, test_size=0.2, val_size=0.1, seed=seed)
        else:
            splits = random_split(city_df, test_size=0.2, val_size=0.1, seed=seed)
        metrics = run_single_experiment(
            splits['train'], splits['test'], feature_set_name,
            model_name, model_fn, val_df=splits['val']
        )
        city_metrics[city] = metrics

    # Macro average
    macro = {}
    for metric_name in ['mae', 'rmse', 'r2', 'mape', 'log_mae', 'nrmse']:
        vals = [city_metrics[c][metric_name] for c in city_metrics
                if not np.isnan(city_metrics[c].get(metric_name, np.nan))]
        macro[f'macro_{metric_name}'] = np.mean(vals) if vals else np.nan

    return {
        'per_city': city_metrics,
        'macro': macro,
    }


def main():
    parser = argparse.ArgumentParser(description='Task A: In-Distribution GHG Prediction')
    parser.add_argument('--feature_sets', type=str,
                        default='core_all_cities,us_core,us_metadata,us_leaky_eui,us_leaky_full',
                        help='Comma-separated feature set names')
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated model names (default: all)')
    parser.add_argument('--seeds', type=str, default='42',
                        help='Comma-separated random seeds')
    parser.add_argument('--ridge_alpha', type=float, default=1.0,
                        help='Regularization strength for Ridge baseline')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['pooled', 'per_city', 'both'],
                        help='Run pooled, per-city, or both')
    parser.add_argument('--split_types', type=str, default='random,grouped',
                        help='Comma-separated split types: random, grouped')
    parser.add_argument('--out_path', type=str, default=None,
                        help='Optional output CSV path (default: results/clean_building/task_a_results.csv; '
                             'the legacy root path results/task_a_results.csv is a historical merge-polluted '
                             'file and is no longer the default — see results/building_final_summary_zh.md §0).')
    args = parser.parse_args()

    feature_sets = args.feature_sets.split(',')
    seeds = [int(s) for s in args.seeds.split(',')]
    split_types = args.split_types.split(',')
    valid_split_types = {'random', 'grouped'}
    invalid_split_types = [s for s in split_types if s not in valid_split_types]
    if invalid_split_types:
        raise ValueError(
            f"Unknown split type(s): {invalid_split_types}. "
            f"Valid options are: {sorted(valid_split_types)}"
        )
    model_zoo = get_model_zoo(ridge_alpha=args.ridge_alpha)
    model_names = args.models.split(',') if args.models else list(model_zoo.keys())

    print("=" * 70)
    print("Task A: In-Distribution GHG Prediction")
    print(f"Feature sets: {feature_sets}")
    print(f"Models: {model_names}")
    print(f"Ridge alpha: {args.ridge_alpha}")
    print(f"Seeds: {seeds}")
    print(f"Mode: {args.mode}")
    print(f"Split types: {split_types}")
    print("=" * 70)

    all_results = []
    # Default to the clean-building bundle (results/task_a_results.csv is a
    # historical merge-polluted file we no longer want to overwrite on rerun).
    out_path = args.out_path if args.out_path else os.path.join(
        RESULTS_DIR, 'clean_building', 'task_a_results.csv'
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def _flush_results():
        """Write results to disk incrementally."""
        if all_results:
            pd.DataFrame(all_results).to_csv(out_path, index=False)

    for fs_name in feature_sets:
        print(f"\n{'='*50}")
        print(f"Feature set: {fs_name}")
        print(f"{'='*50}")

        # Join external climate whenever the requested feature set actually
        # contains any climate-derived columns. This keeps the new
        # core_all_cities/core_all_cities_climate_plus wiring correct without
        # relying on stale hard-coded feature-set names.
        needs_external = bool(set(get_feature_set(fs_name)) & CLIMATE_FEATURE_COLS)
        df = load_and_prepare(fs_name, join_external=needs_external)

        # Filter to rows with non-null target
        df = df[df[TARGET].notna()].copy()
        print(f"Total rows with target: {len(df)}")
        print(f"Cities: {sorted(df['city'].unique())}")

        for model_name in model_names:
            if model_name not in model_zoo:
                print(f"  Skipping unknown model: {model_name}")
                continue

            model_fn = model_zoo[model_name]

            for seed in seeds:
              for split_type in split_types:
                split_label = f"[{split_type}]"

                # A2: Pooled
                if args.mode in ('pooled', 'both'):
                    t0 = time.time()
                    pooled_result = run_pooled_experiment(
                        df, fs_name, model_name, model_fn, seed,
                        split_type=split_type,
                    )
                    elapsed = time.time() - t0

                    macro = pooled_result['macro']
                    print(f"  {split_label:10s} [Pooled] {model_name:15s} | {fs_name:15s} | seed={seed} | "
                          f"macro_MAE={macro.get('macro_mae', np.nan):8.1f} | "
                          f"macro_R²={macro.get('macro_r2', np.nan):6.3f} | "
                          f"{elapsed:.1f}s")

                    row = {
                        'task': 'A2_pooled', 'feature_set': fs_name,
                        'model': model_name, 'seed': seed,
                        'split_type': split_type,
                        **macro,
                        **{f'overall_{k}': v for k, v in pooled_result['overall'].items()},
                    }
                    # Add per-city R²
                    for city, cm in pooled_result['per_city'].items():
                        row[f'r2_{city}'] = cm.get('r2', np.nan)
                        row[f'mae_{city}'] = cm.get('mae', np.nan)
                    all_results.append(row)
                    _flush_results()

                # A1: Per-city
                if args.mode in ('per_city', 'both'):
                    t0 = time.time()
                    percity_result = run_per_city_experiment(
                        df, fs_name, model_name, model_fn, seed,
                        split_type=split_type,
                    )
                    elapsed = time.time() - t0

                    macro = percity_result['macro']
                    print(f"  {split_label:10s} [PerCity] {model_name:15s} | {fs_name:15s} | seed={seed} | "
                          f"macro_MAE={macro.get('macro_mae', np.nan):8.1f} | "
                          f"macro_R²={macro.get('macro_r2', np.nan):6.3f} | "
                          f"{elapsed:.1f}s")

                    row = {
                        'task': 'A1_per_city', 'feature_set': fs_name,
                        'model': model_name, 'seed': seed,
                        'split_type': split_type,
                        **macro,
                    }
                    for city, cm in percity_result['per_city'].items():
                        row[f'r2_{city}'] = cm.get('r2', np.nan)
                        row[f'mae_{city}'] = cm.get('mae', np.nan)
                    all_results.append(row)
                    _flush_results()

    # Save final results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out_path, index=False)
    print(f"\n{'='*70}")
    print(f"Results saved to {out_path}")
    print(f"Total experiments: {len(all_results)}")

    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY: Macro-averaged R² (city-equal-weight)")
    print(f"{'='*70}")
    if not results_df.empty:
        pivot = results_df.pivot_table(
            values='macro_r2', index='model',
            columns=['task', 'feature_set', 'split_type'], aggfunc='mean'
        )
        print(pivot.round(3).to_string())

    print(f"\nSUMMARY: Macro-averaged MAE (tCO2e)")
    if not results_df.empty:
        pivot_mae = results_df.pivot_table(
            values='macro_mae', index='model',
            columns=['task', 'feature_set', 'split_type'], aggfunc='mean'
        )
        print(pivot_mae.round(1).to_string())


if __name__ == '__main__':
    main()
