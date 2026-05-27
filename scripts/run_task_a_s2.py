"""Task A multimodal benchmark: tabular vs tabular+S2 on the same S2-valid subset."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import time
import warnings
import numpy as np
import pandas as pd

from src.data.preprocessing import load_and_prepare, prepare_features
from src.data.feature_sets import TARGET, get_feature_set
from src.data.s2_multimodal import (
    load_s2_index, filter_to_s2_subset, prepare_s2_arrays, concat_tabular_and_s2,
)
from src.data.splitters import grouped_split, random_split
from src.evaluation.metrics import compute_all_metrics
from src.models.linear import RidgeBaseline
from src.models.tree import XGBoostBaseline, LightGBMBaseline, RandomForestBaseline
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


def get_model_zoo():
    return {
        'Ridge': lambda **kwargs: RidgeBaseline(alpha=1.0, **kwargs),
        'RandomForest': lambda **kwargs: RandomForestBaseline(n_estimators=200, n_jobs=8, **kwargs),
        'XGBoost': lambda **kwargs: XGBoostBaseline(n_estimators=200, max_depth=6, n_jobs=8, **kwargs),
        'LightGBM': lambda **kwargs: LightGBMBaseline(n_estimators=200, max_depth=6, n_jobs=8, **kwargs),
        'MLP': lambda **kwargs: TorchMLPBaseline(**kwargs),
        'MLPGated': lambda **kwargs: TorchMultimodalMLPBaseline(fusion='gated', **kwargs),
    }


def allowed_variant(model_name: str, input_variant: str) -> bool:
    s2_variants = {'tab_s2_raw', 'tab_s2_pca64', 'tab_s2_pca128', 'tab_s2_pls64', 'tab_s2_pls128'}
    if model_name == 'MLPGated':
        return input_variant in s2_variants
    return input_variant in ({'tab'} | s2_variants)


def s2_variant_from_input(input_variant: str) -> str:
    mapping = {
        'tab_s2_raw': 'raw',
        'tab_s2_pca64': 'pca64',
        'tab_s2_pca128': 'pca128',
        'tab_s2_pls64': 'pls64',
        'tab_s2_pls128': 'pls128',
    }
    return mapping[input_variant]


def _split(df: pd.DataFrame, split_type: str, seed: int):
    if split_type == 'grouped':
        return grouped_split(df, test_size=0.2, val_size=0.1, seed=seed)
    return random_split(df, test_size=0.2, val_size=0.1, seed=seed)


def _prepare_inputs(train_df, test_df, val_df, feature_set_name, input_variant):
    drop_all_nan = False
    X_train_tab, y_train, _, encoders, medians, dropped_cols = prepare_features(
        train_df, feature_set_name, encoders=None, drop_all_nan=drop_all_nan
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

    s2_var = s2_variant_from_input(input_variant)
    y_train_for_pls = y_train if s2_var.startswith('pls') else None
    train_s2, test_s2, val_s2, bundle = prepare_s2_arrays(
        train_df, test_df, variant=s2_var, val_df=val_df, y_train=y_train_for_pls
    )
    X_train = concat_tabular_and_s2(X_train_tab, train_s2)
    X_test = concat_tabular_and_s2(X_test_tab, test_s2)
    X_val = concat_tabular_and_s2(X_val_tab, val_s2) if X_val_tab is not None and val_s2 is not None else None
    return X_train, y_train, X_test, y_test, X_val, y_val, X_train_tab.shape[1], bundle.out_dim


def fit_eval(train_df, test_df, val_df, feature_set_name, model_name, input_variant):
    X_train, y_train, X_test, y_test, X_val, y_val, tab_dim, s2_dim = _prepare_inputs(
        train_df, test_df, val_df, feature_set_name, input_variant
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


def run_a1(df, feature_set_name, model_name, input_variant, seed):
    rows = []
    for city in sorted(df['city'].unique()):
        city_df = df[df['city'] == city].reset_index(drop=True)
        if len(city_df) < 30:
            continue
        splits = _split(city_df, split_type='grouped', seed=seed)
        metrics = fit_eval(
            splits['train'], splits['test'], splits['val'],
            feature_set_name, model_name, input_variant,
        )
        if metrics is None:
            continue
        metrics['city'] = city
        rows.append(metrics)
    return rows


def run_a2(df, feature_set_name, model_name, input_variant, seed, split_type):
    splits = _split(df, split_type=split_type, seed=seed)
    metrics = fit_eval(
        splits['train'], splits['test'], splits['val'],
        feature_set_name, model_name, input_variant,
    )
    return metrics


def flush(results, out_path):
    if results:
        pd.DataFrame(results).to_csv(out_path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feature_sets', type=str, default='core_all_cities,core_all_cities_climate_plus')
    ap.add_argument('--models', type=str, default='Ridge,XGBoost,LightGBM,MLP,MLPGated')
    ap.add_argument('--input_variants', type=str, default='tab,tab_s2_raw,tab_s2_pca64')
    ap.add_argument('--tasks', type=str, default='A1,A2',
                    help='Comma-separated tasks to run: A1,A2')
    ap.add_argument('--seeds', type=str, default='42',
                    help='Comma-separated random seeds')
    ap.add_argument('--out_path', type=str, default=os.path.join(RESULTS_DIR, 'task_a_results_s2.csv'))
    args = ap.parse_args()

    feature_sets = [x for x in args.feature_sets.split(',') if x]
    model_names = [x for x in args.models.split(',') if x]
    input_variants = [x for x in args.input_variants.split(',') if x]
    tasks = {x.strip().upper() for x in args.tasks.split(',') if x.strip()}
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]

    s2_index = load_s2_index()
    all_results = []

    for fs_name in feature_sets:
        needs_external = any(col in CLIMATE_FEATURE_COLS for col in get_feature_set(fs_name))
        df = load_and_prepare(fs_name, join_external=needs_external)
        mm_df = filter_to_s2_subset(df, s2_index)
        print(f"\nFeature set: {fs_name} | rows={len(df)} | s2_subset={len(mm_df)}")

        for model_name in model_names:
            for input_variant in input_variants:
                if not allowed_variant(model_name, input_variant):
                    continue
                for seed in seeds:
                    t0 = time.time()
                    if 'A1' in tasks:
                        a1_rows = run_a1(mm_df, fs_name, model_name, input_variant, seed=seed)
                        if a1_rows:
                            macro_r2 = float(np.nanmean([r['r2'] for r in a1_rows]))
                            all_results.append({
                                'task': 'A1_per_city',
                                'feature_set': fs_name,
                                'model': model_name,
                                'input_variant': input_variant,
                                'seed': seed,
                                'split_type': 'grouped',
                                'macro_r2': macro_r2,
                                'n_cities': len(a1_rows),
                                'subset_rows': len(mm_df),
                                'elapsed_sec': time.time() - t0,
                            })
                            for city_row in a1_rows:
                                detail_row = {
                                    'task': 'A1_per_city_detail',
                                    'feature_set': fs_name,
                                    'model': model_name,
                                    'input_variant': input_variant,
                                    'seed': seed,
                                    'split_type': 'grouped',
                                    'city': city_row['city'],
                                    'subset_rows': len(mm_df),
                                    'elapsed_sec': time.time() - t0,
                                    **city_row,
                                }
                                all_results.append(detail_row)
                            flush(all_results, args.out_path)
                            print(f"  [A1] {model_name:9s} | {input_variant:13s} | seed={seed} | macro_R²={macro_r2:6.3f}")

                    if 'A2' in tasks:
                        for split_type in ['random', 'grouped']:
                            t1 = time.time()
                            a2 = run_a2(mm_df, fs_name, model_name, input_variant, seed, split_type)
                            if a2 is None:
                                continue
                            row = {
                                'task': 'A2_pooled',
                                'feature_set': fs_name,
                                'model': model_name,
                                'input_variant': input_variant,
                                'seed': seed,
                                'split_type': split_type,
                                'subset_rows': len(mm_df),
                                'elapsed_sec': time.time() - t1,
                                **a2,
                            }
                            all_results.append(row)
                            flush(all_results, args.out_path)
                            print(f"  [A2-{split_type[:3]}] {model_name:9s} | {input_variant:13s} | seed={seed} | R²={row['r2']:6.3f}")

    print(f"\nResults saved to {args.out_path}")


if __name__ == '__main__':
    main()
