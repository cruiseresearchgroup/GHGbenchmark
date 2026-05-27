"""
Task B: Temporal Generalization (appendix-level, secondary to Task E).

B1: Standard temporal split — train ≤2019, val 2020, test ≥2021
B2: COVID natural experiment — train ≤2018, test on 2019/2020/2021 separately

Aligned to the 2026-04-12 two-layer benchmark design.

Usage:
    python scripts/run_task_b.py \
        --feature_sets core_all_cities,core_all_cities_climate_plus \
        --out_path results/task_b_results_core.csv
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import time
import warnings
import numpy as np
import pandas as pd

from src.data.preprocessing import load_and_prepare, prepare_features, encode_categoricals
from src.data.feature_sets import TARGET, get_feature_set, CATEGORICAL_COLS
from src.data.splitters import temporal_split_t2, covid_split
from src.evaluation.metrics import compute_all_metrics

from src.models.t2_baselines import GlobalMeanBaseline, CityTypeMeanBaseline
from src.models.mlp_gpu import TorchMLPBaseline
from src.models.linear import RidgeBaseline
from src.models.tree import RandomForestBaseline, XGBoostBaseline, LightGBMBaseline

warnings.filterwarnings('ignore')

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

CLIMATE_FEATURE_COLS = {
    'hdd', 'cdd',
    'annual_mean_temp_c', 'annual_rh_mean',
    'annual_ssrd_mj_m2_day', 'annual_wind_ms',
}


def get_model_zoo():
    return {
        'GlobalMean': lambda: GlobalMeanBaseline(),
        'CityTypeMean': lambda: CityTypeMeanBaseline(),
        'Ridge': lambda: RidgeBaseline(alpha=1.0),
        'RandomForest': lambda: RandomForestBaseline(n_estimators=200, max_depth=10, n_jobs=8),
        'XGBoost': lambda: XGBoostBaseline(n_estimators=300, max_depth=6, n_jobs=8),
        'LightGBM': lambda: LightGBMBaseline(n_estimators=300, max_depth=6, n_jobs=8),
        'MLP': lambda: TorchMLPBaseline(),
    }


def _flush(results, out_path):
    if not results:
        return
    pd.DataFrame(results).to_csv(out_path, index=False)


def _prepare_citytype_features(train_df, test_df, val_df=None):
    base_cols = ['city', 'property_type', TARGET]
    tr = train_df[base_cols].copy()
    te = test_df[base_cols].copy()
    va = val_df[base_cols].copy() if val_df is not None and len(val_df) > 0 else None

    tr, encoders = encode_categoricals(tr, columns=['city', 'property_type'], encoders=None)
    te, _ = encode_categoricals(te, columns=['city', 'property_type'], encoders=encoders)
    if va is not None:
        va, _ = encode_categoricals(va, columns=['city', 'property_type'], encoders=encoders)

    X_train = tr[['city', 'property_type']].values.astype(np.float64)
    y_train = tr[TARGET].values.astype(np.float64)
    X_test = te[['city', 'property_type']].values.astype(np.float64)
    y_test = te[TARGET].values.astype(np.float64)

    valid_tr = ~np.isnan(y_train); X_train, y_train = X_train[valid_tr], y_train[valid_tr]
    valid_te = ~np.isnan(y_test); X_test, y_test = X_test[valid_te], y_test[valid_te]

    X_val = y_val = None
    if va is not None:
        X_val = va[['city', 'property_type']].values.astype(np.float64)
        y_val = va[TARGET].values.astype(np.float64)
        valid_v = ~np.isnan(y_val); X_val, y_val = X_val[valid_v], y_val[valid_v]

    return X_train, y_train, X_test, y_test, X_val, y_val


def _fit_and_eval(model_name, model_fn, feature_set_name, train_df, test_df, val_df=None):
    """Fit on train, optionally use val for early stopping, return overall + per-city metrics."""
    if model_name == 'CityTypeMean':
        X_train, y_train, X_test, y_test, X_val, y_val = _prepare_citytype_features(
            train_df, test_df, val_df=val_df
        )
    else:
        X_train, y_train, _, encoders, medians, _ = prepare_features(train_df, feature_set_name)
        X_test, y_test, _, _, _, _ = prepare_features(
            test_df, feature_set_name, encoders=encoders, medians=medians
        )
        X_val = y_val = None
        if val_df is not None and len(val_df) > 0:
            X_val, y_val, _, _, _, _ = prepare_features(
                val_df, feature_set_name, encoders=encoders, medians=medians
            )

    if len(X_train) < 2 or len(X_test) == 0:
        return None

    model = model_fn()
    supports_val = model_name in ('XGBoost', 'LightGBM', 'MLP')
    if supports_val and X_val is not None and len(X_val) > 0:
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    else:
        model.fit(X_train, y_train)

    y_pred = np.clip(model.predict(X_test), 0, np.nanmax(y_train) * 2)
    overall = compute_all_metrics(y_test, y_pred)

    # Per-city metrics via test_df alignment
    test_valid = test_df.iloc[:len(y_test)].copy() if model_name == 'CityTypeMean' else test_df.copy()
    city_metrics = {}
    if 'city' in test_valid.columns and len(test_valid) == len(y_test):
        test_valid = test_valid.reset_index(drop=True)
        test_valid['__y_test'] = y_test
        test_valid['__y_pred'] = y_pred
        for city in sorted(test_valid['city'].unique()):
            sub = test_valid[test_valid['city'] == city]
            if len(sub) < 5:
                continue
            city_metrics[city] = compute_all_metrics(sub['__y_test'].values, sub['__y_pred'].values)

    macro = {}
    for m in ['mae', 'rmse', 'r2', 'mape', 'log_mae']:
        vals = [city_metrics[c][m] for c in city_metrics
                if not np.isnan(city_metrics[c].get(m, np.nan))]
        macro[f'macro_{m}'] = np.mean(vals) if vals else np.nan

    return {
        'overall': overall,
        'city_metrics': city_metrics,
        'macro': macro,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'model_obj': model,
    }


def run_task_b1(df, feature_set_name, model_name, model_fn):
    splits = temporal_split_t2(df)
    result = _fit_and_eval(
        model_name, model_fn, feature_set_name,
        splits['train'], splits['test'], val_df=splits['val'],
    )
    if result is None:
        return {'task': 'B1_temporal', 'error': 'insufficient data'}

    row = {
        'task': 'B1_temporal',
        'n_train': result['n_train'],
        'n_test': result['n_test'],
        **{f'overall_{k}': v for k, v in result['overall'].items()},
        **result['macro'],
    }
    for city, cm in result['city_metrics'].items():
        row[f'r2_{city}'] = cm.get('r2', np.nan)
    return row


def run_task_b2(df, feature_set_name, model_name, model_fn):
    splits = covid_split(df)
    train_df = splits['train']

    if model_name == 'CityTypeMean':
        X_train, y_train, _, _, _, _ = _prepare_citytype_features(train_df, train_df.iloc[:0])
        model = model_fn()
        model.fit(X_train, y_train)
        encoders = medians = None
    else:
        X_train, y_train, _, encoders, medians, _ = prepare_features(train_df, feature_set_name)
        model = model_fn()
        model.fit(X_train, y_train)

    y_max = np.nanmax(y_train) if len(y_train) > 0 else 1.0

    row = {'task': 'B2_covid', 'n_train': len(X_train)}

    for year_key in ['test_2019', 'test_2020', 'test_2021']:
        test_df = splits[year_key]
        if len(test_df) == 0:
            continue
        year_label = year_key.split('_')[1]

        if model_name == 'CityTypeMean':
            _, _, X_test, y_test, _, _ = _prepare_citytype_features(train_df, test_df)
        else:
            X_test, y_test, _, _, _, _ = prepare_features(
                test_df, feature_set_name, encoders=encoders, medians=medians
            )

        if len(X_test) < 5:
            continue
        y_pred = np.clip(model.predict(X_test), 0, y_max * 2)
        metrics = compute_all_metrics(y_test, y_pred)
        row[f'mae_{year_label}'] = metrics.get('mae', np.nan)
        row[f'r2_{year_label}'] = metrics.get('r2', np.nan)
        row[f'n_test_{year_label}'] = len(X_test)

    return row


def main():
    parser = argparse.ArgumentParser(description='Task B: Temporal Generalization')
    parser.add_argument(
        '--feature_sets', type=str,
        default='core_all_cities,core_all_cities_climate_plus',
    )
    parser.add_argument('--models', type=str, default=None)
    parser.add_argument('--subtasks', type=str, default='all',
                        help='B1,B2 or all')
    parser.add_argument('--out_path', type=str,
                        default=os.path.join(RESULTS_DIR, 'task_b_results_core.csv'))
    args = parser.parse_args()

    feature_sets = [x for x in args.feature_sets.split(',') if x]
    model_zoo = get_model_zoo()
    model_names = args.models.split(',') if args.models else list(model_zoo.keys())
    subtasks = ['B1', 'B2'] if args.subtasks == 'all' else [x for x in args.subtasks.split(',') if x]

    print("=" * 70)
    print("Task B: Temporal Generalization")
    print(f"Feature sets: {feature_sets}")
    print(f"Models: {model_names}")
    print(f"Subtasks: {subtasks}")
    print(f"Output: {args.out_path}")
    print("=" * 70)

    all_results = []

    for fs_name in feature_sets:
        needs_external = any(col in CLIMATE_FEATURE_COLS for col in get_feature_set(fs_name))
        df = load_and_prepare(fs_name, join_external=needs_external)
        df = df[df[TARGET].notna()].copy()

        print(f"\n{'='*50}")
        print(f"Feature set: {fs_name}")
        print(f"{'='*50}")
        print(f"Total rows: {len(df)}, Cities: {df['city'].nunique()}")
        print(f"Year range: {int(df['year'].min())}-{int(df['year'].max())}")

        for model_name in model_names:
            model_fn = model_zoo[model_name]

            if 'B1' in subtasks:
                t0 = time.time()
                b1 = run_task_b1(df, fs_name, model_name, model_fn)
                b1['feature_set'] = fs_name
                b1['model'] = model_name
                all_results.append(b1)
                _flush(all_results, args.out_path)
                print(f"  [B1] {model_name:15s} | {fs_name:28s} | "
                      f"overall_R²={b1.get('overall_r2', np.nan):6.3f} | "
                      f"macro_R²={b1.get('macro_r2', np.nan):6.3f} | "
                      f"{time.time()-t0:.1f}s")

            if 'B2' in subtasks:
                t0 = time.time()
                b2 = run_task_b2(df, fs_name, model_name, model_fn)
                b2['feature_set'] = fs_name
                b2['model'] = model_name
                all_results.append(b2)
                _flush(all_results, args.out_path)
                print(f"  [B2] {model_name:15s} | {fs_name:28s} | "
                      f"R²_2019={b2.get('r2_2019', np.nan):6.3f} | "
                      f"R²_2020={b2.get('r2_2020', np.nan):6.3f} | "
                      f"R²_2021={b2.get('r2_2021', np.nan):6.3f} | "
                      f"{time.time()-t0:.1f}s")

    print(f"\nResults saved to {args.out_path}")


if __name__ == '__main__':
    main()
