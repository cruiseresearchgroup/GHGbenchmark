"""
Task D: Cross-Building-Type Transfer (appendix-level).

D1: Leave-one-type-out — train on all types except one, test on held-out type.

Only runs on feature sets that include `property_type` (US subset tiers).
Layer 1 core_all_cities does NOT have property_type (AU = 0% coverage),
so D defaults to us_core.

Usage:
    python scripts/run_task_d.py \
        --feature_sets us_core,us_metadata \
        --out_path results/task_d_results_core.csv
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
from src.data.splitters import cross_type_split
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

# Major building types to test
TARGET_TYPES = [
    'Office', 'Multifamily Housing', 'Retail', 'Hotel',
    'K-12 School', 'Hospital/Medical', 'Warehouse/Distribution',
]

MIN_TYPE_ROWS = 50


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


def _prepare_citytype_features(train_df, test_df):
    base_cols = ['city', 'property_type', TARGET]
    tr = train_df[base_cols].copy()
    te = test_df[base_cols].copy()

    tr, encoders = encode_categoricals(tr, columns=['city', 'property_type'], encoders=None)
    te, _ = encode_categoricals(te, columns=['city', 'property_type'], encoders=encoders)

    X_train = tr[['city', 'property_type']].values.astype(np.float64)
    y_train = tr[TARGET].values.astype(np.float64)
    X_test = te[['city', 'property_type']].values.astype(np.float64)
    y_test = te[TARGET].values.astype(np.float64)

    valid_tr = ~np.isnan(y_train); X_train, y_train = X_train[valid_tr], y_train[valid_tr]
    valid_te = ~np.isnan(y_test); X_test, y_test = X_test[valid_te], y_test[valid_te]
    return X_train, y_train, X_test, y_test


def main():
    parser = argparse.ArgumentParser(description='Task D: Cross-Building-Type Transfer')
    parser.add_argument(
        '--feature_sets', type=str, default='us_core,us_metadata',
        help='Must be feature sets that include property_type',
    )
    parser.add_argument('--models', type=str, default=None)
    parser.add_argument('--out_path', type=str,
                        default=os.path.join(RESULTS_DIR, 'task_d_results_core.csv'))
    args = parser.parse_args()

    feature_sets = [x for x in args.feature_sets.split(',') if x]
    model_zoo = get_model_zoo()
    model_names = args.models.split(',') if args.models else list(model_zoo.keys())

    print("=" * 70)
    print("Task D: Cross-Building-Type Transfer")
    print(f"Feature sets: {feature_sets}")
    print(f"Models: {model_names}")
    print(f"Target types: {TARGET_TYPES}")
    print(f"Output: {args.out_path}")
    print("=" * 70)

    all_results = []

    for fs_name in feature_sets:
        feat_cols = get_feature_set(fs_name)
        if 'property_type' not in feat_cols:
            print(f"\n  SKIP {fs_name} — no property_type column")
            continue

        needs_external = any(col in CLIMATE_FEATURE_COLS for col in feat_cols)
        df = load_and_prepare(fs_name, join_external=needs_external)
        df = df[df[TARGET].notna()].copy()

        type_counts = df['property_type'].value_counts()
        available_types = [t for t in TARGET_TYPES
                          if t in type_counts.index and type_counts[t] >= MIN_TYPE_ROWS]

        print(f"\n{'='*50}")
        print(f"Feature set: {fs_name}")
        print(f"{'='*50}")
        print(f"Total rows: {len(df)}, Cities: {df['city'].nunique()}")
        print(f"Available types ({len(available_types)}): {available_types}")

        for model_name in model_names:
            model_fn = model_zoo[model_name]

            for target_type in available_types:
                t0 = time.time()
                splits = cross_type_split(df, target_type)
                train_df, test_df = splits['train'], splits['test']

                if model_name == 'CityTypeMean':
                    X_train, y_train, X_test, y_test = _prepare_citytype_features(
                        train_df, test_df
                    )
                else:
                    X_train, y_train, _, encoders, medians, _ = prepare_features(
                        train_df, fs_name
                    )
                    X_test, y_test, _, _, _, _ = prepare_features(
                        test_df, fs_name, encoders=encoders, medians=medians
                    )

                if len(X_train) < 2 or len(X_test) == 0:
                    continue

                model = model_fn()
                model.fit(X_train, y_train)
                y_pred = np.clip(model.predict(X_test), 0, np.nanmax(y_train) * 2)
                metrics = compute_all_metrics(y_test, y_pred)

                elapsed = time.time() - t0
                print(f"  {model_name:15s} | {fs_name:12s} | {target_type:25s} | "
                      f"R²={metrics['r2']:6.3f} | MAE={metrics['mae']:8.1f} | "
                      f"n_test={len(X_test)} | {elapsed:.1f}s")

                row = {
                    'task': 'D1_leave_type_out',
                    'feature_set': fs_name,
                    'model': model_name,
                    'target_type': target_type,
                    'n_train': len(X_train),
                    'n_test': len(X_test),
                    **metrics,
                }
                all_results.append(row)
                _flush(all_results, args.out_path)

    print(f"\nResults saved to {args.out_path}")
    print(f"Total experiments: {len(all_results)}")

    if all_results:
        results_df = pd.DataFrame(all_results)
        print(f"\n{'='*70}")
        print("Summary: R² by model × target type")
        for fs in feature_sets:
            sub = results_df[results_df['feature_set'] == fs]
            if sub.empty:
                continue
            print(f"\n  Feature set: {fs}")
            pivot = sub.pivot_table(values='r2', index='model', columns='target_type')
            pivot['macro_avg'] = pivot.mean(axis=1)
            print(pivot.round(3).to_string())


if __name__ == '__main__':
    main()
