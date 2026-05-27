"""
Task C: Cross-City Transfer Experiment.

Aligned to the 2026-04-12 two-layer benchmark design:
  Layer 1 (default): core_all_cities, core_all_cities_climate_plus  (26 metros)
  Layer 2A (optional): us_core / us_metadata / us_leaky_eui / us_leaky_full (6 US)
  Layer 2B (optional): au_core / au_eui / au_full (15 AU metros)

Subtasks:
  C1 — Leave-one-city-out: train on N-1 cities, test on held-out city
  C2 — Directed transfer pairs (source cities → target city)
  C3 — Few-shot city adaptation (k=10/50/100/500 target samples)

All models run on GPU where applicable (MLP via TorchMLPBaseline).
Results are written incrementally to `--out_path` so a crash mid-run does not
lose finished experiments.

Usage:
  python scripts/run_task_c.py \
    --feature_sets core_all_cities,core_all_cities_climate_plus \
    --subtasks C1 \
    --out_path results/task_c_core_all_cities.csv
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import time
import warnings
import numpy as np
import pandas as pd

from src.data.preprocessing import load_and_prepare, prepare_features, encode_categoricals
from src.data.feature_sets import TARGET, get_feature_set, get_city_set
from src.data.splitters import grouped_split, random_split
from src.evaluation.metrics import compute_all_metrics

from src.models.t2_baselines import GlobalMeanBaseline, CityTypeMeanBaseline
from src.models.mlp_gpu import TorchMLPBaseline
from src.models.linear import RidgeBaseline
from src.models.tree import RandomForestBaseline, XGBoostBaseline, LightGBMBaseline

warnings.filterwarnings('ignore')

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Feature sets that require the external climate join (hdd/cdd/temp/etc.)
CLIMATE_FEATURE_COLS = {
    'hdd', 'cdd',
    'annual_mean_temp_c', 'annual_rh_mean',
    'annual_ssrd_mj_m2_day', 'annual_wind_ms',
}

MIN_CITY_ROWS = 50  # skip target cities with fewer than this many rows
FEW_SHOT_K = [10, 50, 100, 500]


def get_model_zoo(ridge_alpha: float = 1.0):
    return {
        'GlobalMean': lambda: GlobalMeanBaseline(),
        'CityTypeMean': lambda: CityTypeMeanBaseline(),
        'Ridge': lambda: RidgeBaseline(alpha=ridge_alpha),
        'RandomForest': lambda: RandomForestBaseline(n_estimators=200, max_depth=10, n_jobs=8),
        'XGBoost': lambda: XGBoostBaseline(n_estimators=300, max_depth=6, n_jobs=8),
        'LightGBM': lambda: LightGBMBaseline(n_estimators=300, max_depth=6, n_jobs=8),
        'MLP': lambda: TorchMLPBaseline(),
    }


# C2 transfer pairs. `source` = None means "all other cities in fs".
# Each pair is silently skipped if the target or all sources are absent from
# the current feature set's city list.
TRANSFER_PAIRS = [
    ('nyc',       ['chicago', 'dc', 'boston', 'philadelphia']),
    ('seattle',   ['nyc', 'chicago', 'dc', 'sf', 'la']),
    ('sf',        ['la']),
    ('la',        ['sf']),
    ('melbourne', None),
    ('singapore', None),
    ('sydney',    ['melbourne', 'brisbane']),
]

# C3 few-shot target cities — we only evaluate targets that are large enough
# to support k=500 sampling.
FEW_SHOT_TARGETS = ['seattle', 'sf', 'chicago', 'melbourne', 'sydney']


def _prepare_citytype_features(train_df, test_df, val_df=None):
    """Special feature prep for CityTypeMean baseline (mirrors run_task_a.py).

    CityTypeMean always uses semantic (city, property_type) group keys, not
    the current feature set matrix.
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

    tv = ~np.isnan(y_train)
    ev = ~np.isnan(y_test)
    X_train, y_train = X_train[tv], y_train[tv]
    X_test, y_test = X_test[ev], y_test[ev]
    if X_val is not None:
        vv = ~np.isnan(y_val)
        X_val, y_val = X_val[vv], y_val[vv]
    return X_train, y_train, X_test, y_test, X_val, y_val


def _split_train_val(train_df, seed=42, val_frac=0.15):
    """Carve a validation fold out of the source training set, grouping by
    building so a single building never appears in both folds."""
    bids = train_df['building_id'].dropna().unique()
    if len(bids) < 10:
        # Not enough buildings to carve a meaningful val fold
        return train_df, train_df.iloc[0:0]
    rng = np.random.default_rng(seed)
    bids_shuffled = rng.permutation(bids)
    n_val = max(1, int(round(len(bids) * val_frac)))
    val_ids = set(bids_shuffled[:n_val])
    mask = train_df['building_id'].isin(val_ids)
    return train_df[~mask].reset_index(drop=True), train_df[mask].reset_index(drop=True)


def train_and_evaluate(
    train_df, test_df, feature_set_name, model_name, model_fn,
    val_df=None, seed=42,
):
    """Fit model on train_df, evaluate on test_df. Returns metrics dict."""
    nan_result = {
        'mae': np.nan, 'rmse': np.nan, 'r2': np.nan, 'mape': np.nan,
        'log_mae': np.nan, 'nrmse': np.nan,
        'n_train': 0, 'n_test': 0,
    }

    if len(train_df) < 2 or len(test_df) == 0:
        return nan_result

    # CityTypeMean has its own feature path
    if model_name == 'CityTypeMean':
        X_train, y_train, X_test, y_test, X_val, y_val = _prepare_citytype_features(
            train_df, test_df, val_df=val_df
        )
    else:
        X_train, y_train, _, encoders, medians, dropped_cols = prepare_features(
            train_df, feature_set_name, encoders=None, drop_all_nan=False,
        )
        X_test, y_test, _, _, _, _ = prepare_features(
            test_df, feature_set_name,
            encoders=encoders, medians=medians, drop_cols=dropped_cols,
        )
        X_val = y_val = None
        if val_df is not None and len(val_df) > 0:
            X_val, y_val, _, _, _, _ = prepare_features(
                val_df, feature_set_name,
                encoders=encoders, medians=medians, drop_cols=dropped_cols,
            )

    if len(X_train) < 2 or len(X_test) == 0:
        return nan_result

    model = model_fn()
    try:
        supports_val = model_name in ('XGBoost', 'LightGBM', 'MLP')
        if supports_val and X_val is not None and len(X_val) > 0:
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        else:
            model.fit(X_train, y_train)

        y_pred = np.clip(model.predict(X_test), 0, np.max(y_train) * 2)
        metrics = compute_all_metrics(y_test, y_pred)
        metrics['n_train'] = len(X_train)
        metrics['n_test'] = len(X_test)
        return metrics
    except Exception as e:
        print(f"    WARNING: {model_name} failed: {e}")
        return {**nan_result, 'n_train': len(X_train), 'n_test': len(X_test)}


# ---------------------------------------------------------------------------
# Subtask runners
# ---------------------------------------------------------------------------

def run_c1(df, feature_set_name, model_name, model_fn, seed=42):
    """C1: Leave-one-city-out across all cities in the current feature set."""
    results = []
    cities = sorted(df['city'].unique())

    for target_city in cities:
        target = df[df['city'] == target_city]
        if len(target) < MIN_CITY_ROWS:
            continue
        source = df[df['city'] != target_city]
        if len(source) < MIN_CITY_ROWS:
            continue

        train_src, val_src = _split_train_val(source, seed=seed)
        metrics = train_and_evaluate(
            train_src, target, feature_set_name, model_name, model_fn,
            val_df=val_src, seed=seed,
        )
        metrics['target_city'] = target_city
        results.append(metrics)
    return results


def run_c2(df, feature_set_name, model_name, model_fn, seed=42):
    """C2: Directed transfer pairs."""
    results = []
    available = set(df['city'].unique())
    for target_city, source_list in TRANSFER_PAIRS:
        if target_city not in available:
            continue

        if source_list is None:
            sources = sorted(available - {target_city})
            src_label = 'all_others'
        else:
            sources = [c for c in source_list if c in available]
            if not sources:
                continue
            src_label = ','.join(sources)

        train = df[df['city'].isin(sources)]
        test = df[df['city'] == target_city]
        if len(train) < MIN_CITY_ROWS or len(test) < MIN_CITY_ROWS:
            continue

        train_src, val_src = _split_train_val(train, seed=seed)
        metrics = train_and_evaluate(
            train_src, test, feature_set_name, model_name, model_fn,
            val_df=val_src, seed=seed,
        )
        metrics['target_city'] = target_city
        metrics['source_cities'] = src_label
        results.append(metrics)
    return results


def run_c3(df, feature_set_name, model_name, model_fn, seed=42):
    """C3: Few-shot city adaptation.

    Train on all non-target cities + k samples from target; test on the rest
    of the target city. Samples are drawn at the building level (not the row
    level) so a single building doesn't leak across train/test.
    """
    results = []
    available = set(df['city'].unique())
    rng = np.random.default_rng(seed)

    for target_city in FEW_SHOT_TARGETS:
        if target_city not in available:
            continue

        source = df[df['city'] != target_city]
        target = df[df['city'] == target_city]
        if len(target) < MIN_CITY_ROWS:
            continue

        target_bids = target['building_id'].unique()
        if len(target_bids) < max(FEW_SHOT_K) + 10:
            # Fall back to row-level sampling when the target has too few
            # distinct buildings for building-level few-shot.
            sampler = 'row'
        else:
            sampler = 'building'

        for k in FEW_SHOT_K:
            if sampler == 'building':
                bids_shuffled = rng.permutation(target_bids)
                shot_bids = set(bids_shuffled[:k])
                shot = target[target['building_id'].isin(shot_bids)]
                rest = target[~target['building_id'].isin(shot_bids)]
            else:
                if len(target) <= k:
                    continue
                shot = target.sample(n=k, random_state=seed)
                rest = target.drop(shot.index)

            if len(rest) < MIN_CITY_ROWS:
                continue

            train = pd.concat([source, shot], ignore_index=True)
            train_src, val_src = _split_train_val(train, seed=seed)
            metrics = train_and_evaluate(
                train_src, rest, feature_set_name, model_name, model_fn,
                val_df=val_src, seed=seed,
            )
            metrics['target_city'] = target_city
            metrics['k'] = k
            results.append(metrics)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Task C: Cross-City Transfer')
    parser.add_argument(
        '--feature_sets', type=str,
        default='core_all_cities,core_all_cities_climate_plus',
        help='Comma-separated feature set names',
    )
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated model names (default: all)')
    parser.add_argument('--subtasks', type=str, default='C1',
                        help='Comma-separated subtasks: C1,C2,C3 or "all"')
    parser.add_argument('--ridge_alpha', type=float, default=1.0,
                        help='Alpha regularization strength for Ridge baseline')
    parser.add_argument('--seeds', type=str, default='42',
                        help='Comma-separated random seeds')
    parser.add_argument('--out_path', type=str, default=None,
                        help='Output CSV path (default: results/task_c_results.csv)')
    args = parser.parse_args()

    feature_sets = [fs.strip() for fs in args.feature_sets.split(',') if fs.strip()]
    if args.subtasks.lower() == 'all':
        subtasks = ['C1', 'C2', 'C3']
    else:
        subtasks = [s.strip().upper() for s in args.subtasks.split(',') if s.strip()]

    model_zoo = get_model_zoo(ridge_alpha=args.ridge_alpha)
    model_names = args.models.split(',') if args.models else list(model_zoo.keys())
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]

    out_path = args.out_path or os.path.join(RESULTS_DIR, 'task_c_results.csv')

    print('=' * 70)
    print('Task C: Cross-City Transfer')
    print(f'Subtasks: {subtasks}')
    print(f'Feature sets: {feature_sets}')
    print(f'Models: {model_names}')
    print(f'Ridge alpha: {args.ridge_alpha}')
    print(f'Seeds: {seeds}')
    print(f'Output: {out_path}')
    print('=' * 70)

    all_results = []

    def _flush():
        if all_results:
            pd.DataFrame(all_results).to_csv(out_path, index=False)

    for fs_name in feature_sets:
        print(f"\n{'=' * 50}")
        print(f'Feature set: {fs_name}')
        print(f"{'=' * 50}")

        needs_external = bool(set(get_feature_set(fs_name)) & CLIMATE_FEATURE_COLS)
        df = load_and_prepare(fs_name, join_external=needs_external)
        df = df[df[TARGET].notna()].copy()

        city_set = set(get_city_set(fs_name))
        df = df[df['city'].isin(city_set)].copy()

        print(f'Total rows: {len(df)}')
        print(f'Cities ({df["city"].nunique()}): {sorted(df["city"].unique())}')

        for model_name in model_names:
            if model_name not in model_zoo:
                print(f'  Skipping unknown model: {model_name}')
                continue
            model_fn = model_zoo[model_name]

            for seed in seeds:
                # C1
                if 'C1' in subtasks:
                    t0 = time.time()
                    c1_results = run_c1(df, fs_name, model_name, model_fn, seed=seed)
                    dt = time.time() - t0
                    r2s = [r['r2'] for r in c1_results if not np.isnan(r.get('r2', np.nan))]
                    maes = [r['mae'] for r in c1_results if not np.isnan(r.get('mae', np.nan))]
                    print(f'  [C1] {model_name:15s} | {fs_name:28s} | seed={seed} | '
                          f'cities={len(c1_results):2d} | '
                          f'macro_R²={np.mean(r2s) if r2s else np.nan:6.3f} | '
                          f'macro_MAE={np.mean(maes) if maes else np.nan:8.1f} | '
                          f'{dt:.1f}s')
                    for r in c1_results:
                        r.update({'task': 'C1_loco', 'feature_set': fs_name,
                                  'model': model_name, 'seed': seed})
                    all_results.extend(c1_results)
                    _flush()

                # C2
                if 'C2' in subtasks:
                    t0 = time.time()
                    c2_results = run_c2(df, fs_name, model_name, model_fn, seed=seed)
                    dt = time.time() - t0
                    print(f'  [C2] {model_name:15s} | {fs_name:28s} | seed={seed} | '
                          f'pairs={len(c2_results):2d} | {dt:.1f}s')
                    for r in c2_results:
                        r.update({'task': 'C2_directed', 'feature_set': fs_name,
                                  'model': model_name, 'seed': seed})
                    all_results.extend(c2_results)
                    _flush()

                # C3
                if 'C3' in subtasks:
                    t0 = time.time()
                    c3_results = run_c3(df, fs_name, model_name, model_fn, seed=seed)
                    dt = time.time() - t0
                    print(f'  [C3] {model_name:15s} | {fs_name:28s} | seed={seed} | '
                          f'runs={len(c3_results):2d} | {dt:.1f}s')
                    for r in c3_results:
                        r.update({'task': 'C3_fewshot', 'feature_set': fs_name,
                                  'model': model_name, 'seed': seed})
                    all_results.extend(c3_results)
                    _flush()

    # Final flush
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(out_path, index=False)
    print(f"\n{'=' * 70}")
    print(f'Results saved to {out_path}')
    print(f'Total experiments: {len(all_results)}')

    # C1 summary
    if 'C1' in subtasks and not results_df.empty:
        c1 = results_df[results_df['task'] == 'C1_loco']
        if not c1.empty:
            print(f"\n{'=' * 70}")
            print('C1: Leave-One-City-Out macro R² (city-equal-weight)')
            print('=' * 70)
            pivot = c1.pivot_table(
                values='r2', index='model',
                columns=['feature_set'], aggfunc='mean',
            )
            print(pivot.round(3).to_string())


if __name__ == '__main__':
    main()
