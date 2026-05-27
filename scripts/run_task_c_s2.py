"""Task C1 multimodal benchmark: cross-city transfer on the S2-valid subset."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import time
import warnings
import numpy as np
import pandas as pd

from src.data.preprocessing import load_and_prepare, prepare_features
from src.data.feature_sets import get_feature_set
from src.data.s2_multimodal import (
    load_s2_index, filter_to_s2_subset, prepare_s2_arrays, concat_tabular_and_s2,
)
from src.evaluation.metrics import compute_all_metrics
from src.models.linear import RidgeBaseline
from src.models.tree import XGBoostBaseline, LightGBMBaseline
from src.models.mlp_gpu import TorchMLPBaseline
from src.models.mlp_multimodal import TorchMultimodalMLPBaseline

warnings.filterwarnings("ignore")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

CLIMATE_FEATURE_COLS = {
    'hdd', 'cdd',
    'annual_mean_temp_c', 'annual_rh_mean',
    'annual_ssrd_mj_m2_day', 'annual_wind_ms',
}

MIN_CITY_ROWS = 50


def get_model_zoo():
    return {
        'Ridge': lambda **kwargs: RidgeBaseline(alpha=1.0, **kwargs),
        'XGBoost': lambda **kwargs: XGBoostBaseline(n_estimators=300, max_depth=6, n_jobs=8, **kwargs),
        'LightGBM': lambda **kwargs: LightGBMBaseline(n_estimators=300, max_depth=6, n_jobs=8, **kwargs),
        'MLP': lambda **kwargs: TorchMLPBaseline(**kwargs),
        'MLPGated': lambda **kwargs: TorchMultimodalMLPBaseline(fusion='gated', **kwargs),
    }


def allowed_variant(model_name: str, input_variant: str) -> bool:
    if model_name == 'MLPGated':
        return input_variant in {'tab_s2_raw', 'tab_s2_pca64', 'tab_s2_pca128'}
    if model_name == 'MLP':
        return input_variant in {'tab', 'tab_s2_raw', 'tab_s2_pca64', 'tab_s2_pca128'}
    return input_variant in {'tab', 'tab_s2_raw', 'tab_s2_pca64', 'tab_s2_pca128'}


def s2_variant_from_input(input_variant: str) -> str:
    return {
        'tab_s2_raw': 'raw',
        'tab_s2_pca64': 'pca64',
        'tab_s2_pca128': 'pca128',
    }[input_variant]


def _split_train_val(train_df: pd.DataFrame, seed=42, val_frac=0.15):
    bids = train_df['building_id'].dropna().unique()
    if len(bids) < 10:
        return train_df, train_df.iloc[0:0]
    rng = np.random.default_rng(seed)
    bids = rng.permutation(bids)
    n_val = max(1, int(round(len(bids) * val_frac)))
    val_ids = set(bids[:n_val])
    mask = train_df['building_id'].isin(val_ids)
    return train_df[~mask].reset_index(drop=True), train_df[mask].reset_index(drop=True)


def _prepare_inputs(train_df, test_df, val_df, feature_set_name, input_variant):
    X_train_tab, y_train, _, encoders, medians, dropped_cols = prepare_features(
        train_df, feature_set_name, encoders=None, drop_all_nan=False
    )
    X_test_tab, y_test, _, _, _, _ = prepare_features(
        test_df, feature_set_name, encoders=encoders, medians=medians, drop_cols=dropped_cols
    )
    X_val_tab = y_val = None
    if val_df is not None and len(val_df) > 0:
        X_val_tab, y_val, _, _, _, _ = prepare_features(
            val_df, feature_set_name, encoders=encoders, medians=medians, drop_cols=dropped_cols
        )

    if input_variant == 'tab':
        return X_train_tab, y_train, X_test_tab, y_test, X_val_tab, y_val, X_train_tab.shape[1], 0

    train_s2, test_s2, val_s2, bundle = prepare_s2_arrays(
        train_df, test_df, variant=s2_variant_from_input(input_variant), val_df=val_df
    )
    X_train = concat_tabular_and_s2(X_train_tab, train_s2)
    X_test = concat_tabular_and_s2(X_test_tab, test_s2)
    X_val = concat_tabular_and_s2(X_val_tab, val_s2) if X_val_tab is not None and val_s2 is not None else None
    return X_train, y_train, X_test, y_test, X_val, y_val, X_train_tab.shape[1], bundle.out_dim


def fit_eval(train_df, test_df, feature_set_name, model_name, input_variant, seed):
    train_src, val_src = _split_train_val(train_df, seed=seed)
    X_train, y_train, X_test, y_test, X_val, y_val, tab_dim, s2_dim = _prepare_inputs(
        train_src, test_df, val_src, feature_set_name, input_variant
    )
    if len(X_train) < 2 or len(X_test) == 0:
        return None

    zoo = get_model_zoo()
    if model_name == 'MLP' and input_variant != 'tab':
        model = TorchMultimodalMLPBaseline(tab_dim=tab_dim, s2_dim=s2_dim, fusion='two_tower')
    elif model_name == 'MLPGated':
        model = zoo[model_name](tab_dim=tab_dim, s2_dim=s2_dim)
    else:
        model = zoo[model_name]()

    if model_name in {'XGBoost', 'LightGBM', 'MLP', 'MLPGated'} and X_val is not None and len(X_val) > 0:
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    else:
        model.fit(X_train, y_train)

    y_pred = np.clip(model.predict(X_test), 0, np.max(y_train) * 2)
    metrics = compute_all_metrics(y_test, y_pred)
    metrics['n_train'] = len(X_train)
    metrics['n_test'] = len(X_test)
    metrics['tab_dim'] = tab_dim
    metrics['s2_dim'] = s2_dim
    return metrics


def flush(results, out_path):
    if results:
        pd.DataFrame(results).to_csv(out_path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feature_sets', type=str, default='core_all_cities,core_all_cities_climate_plus')
    ap.add_argument('--models', type=str, default='Ridge,XGBoost,LightGBM,MLP,MLPGated')
    ap.add_argument('--input_variants', type=str, default='tab,tab_s2_raw,tab_s2_pca64')
    ap.add_argument('--seeds', type=str, default='42',
                    help='Comma-separated random seeds')
    ap.add_argument('--out_path', type=str, default=os.path.join(RESULTS_DIR, 'task_c_results_s2.csv'))
    args = ap.parse_args()

    feature_sets = [x for x in args.feature_sets.split(',') if x]
    model_names = [x for x in args.models.split(',') if x]
    input_variants = [x for x in args.input_variants.split(',') if x]
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    s2_index = load_s2_index()
    all_results = []

    for fs_name in feature_sets:
        needs_external = any(col in CLIMATE_FEATURE_COLS for col in get_feature_set(fs_name))
        df = load_and_prepare(fs_name, join_external=needs_external)
        mm_df = filter_to_s2_subset(df, s2_index)
        print(f"\nFeature set: {fs_name} | rows={len(df)} | s2_subset={len(mm_df)}")

        for target_city in sorted(mm_df['city'].unique()):
            test_df = mm_df[mm_df['city'] == target_city].reset_index(drop=True)
            train_df = mm_df[mm_df['city'] != target_city].reset_index(drop=True)
            if len(test_df) < MIN_CITY_ROWS or len(train_df) < MIN_CITY_ROWS:
                continue

            for model_name in model_names:
                for input_variant in input_variants:
                    if not allowed_variant(model_name, input_variant):
                        continue
                    for seed in seeds:
                        t0 = time.time()
                        metrics = fit_eval(train_df, test_df, fs_name, model_name, input_variant, seed)
                        if metrics is None:
                            continue
                        row = {
                            'task': 'C1_loco_s2',
                            'feature_set': fs_name,
                            'model': model_name,
                            'input_variant': input_variant,
                            'seed': seed,
                            'target_city': target_city,
                            'subset_rows': len(mm_df),
                            'elapsed_sec': time.time() - t0,
                            **metrics,
                        }
                        all_results.append(row)
                        flush(all_results, args.out_path)
                        print(f"  [C1] {target_city:14s} | {model_name:9s} | {input_variant:13s} | seed={seed} | R²={row['r2']:6.3f}")

    print(f"\nResults saved to {args.out_path}")


if __name__ == '__main__':
    main()
