"""TabPFN baseline on the same S2-eligible subset used by run_task_a_s2.py.

Fills the gaps in tables/t2_main.tex left by the original Task A runner:
  - TabPFN v2 on random-pooled split (Core, +Climate, +S2 PCA-64)
  - TabPFN v2 + S2 PCA-64 on grouped split

Why a separate script: TabPFN needs train-set subsampling (default 10K rows)
and chunked test prediction; the multi-model run_task_a_s2.py runner does not.
We keep the same S2 subset filter and PCA fit (train-only) so deltas can be
directly compared against the existing Ridge / RF / LGBM / XGB / MLP rows.

Defaults: PCA-64 only (raw 1024-d S2 has too many features for TabPFN's
context). max_train=10000 matches the existing TabPFN protocol.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse, time, warnings
import numpy as np
import pandas as pd
import torch
from tabpfn import TabPFNRegressor

from src.data.preprocessing import load_and_prepare, prepare_features
from src.data.feature_sets import TARGET, CATEGORICAL_COLS, get_feature_set
from src.data.s2_multimodal import (
    load_s2_index, filter_to_s2_subset, prepare_s2_arrays, concat_tabular_and_s2,
)
from src.data.splitters import grouped_split, random_split
from src.evaluation.metrics import compute_all_metrics

warnings.filterwarnings("ignore")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

CLIMATE_FEATURE_COLS = {
    'hdd', 'cdd', 'annual_mean_temp_c', 'annual_rh_mean',
    'annual_ssrd_mj_m2_day', 'annual_wind_ms',
}


def _split(df, split_type, seed):
    if split_type == 'grouped':
        return grouped_split(df, test_size=0.2, val_size=0.1, seed=seed)
    return random_split(df, test_size=0.2, val_size=0.1, seed=seed)


def _categorical_indices(feature_names, n_extra_s2=0):
    """Get categorical indices in the tabular block; S2 block (if any) appended afterwards is all numeric."""
    return [i for i, name in enumerate(feature_names) if name in CATEGORICAL_COLS]


def _subsample(X, y, max_train, seed):
    if len(X) <= max_train:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max_train, replace=False)
    return X[idx], y[idx]


def _prepare_inputs(train_df, test_df, feature_set_name, input_variant):
    X_train_tab, y_train, feat_names, encoders, medians, dropped_cols = prepare_features(
        train_df, feature_set_name, encoders=None, drop_all_nan=False
    )
    X_test_tab, y_test, _, _, _, _ = prepare_features(
        test_df, feature_set_name, encoders=encoders, medians=medians, drop_cols=dropped_cols
    )
    if input_variant == 'tab':
        return X_train_tab, y_train, X_test_tab, y_test, feat_names, X_train_tab.shape[1], 0

    train_s2, test_s2, _, bundle = prepare_s2_arrays(train_df, test_df, variant='pca64', val_df=None)
    X_train = concat_tabular_and_s2(X_train_tab, train_s2)
    X_test = concat_tabular_and_s2(X_test_tab, test_s2)
    return X_train, y_train, X_test, y_test, feat_names, X_train_tab.shape[1], bundle.out_dim


def fit_eval_tabpfn(X_train, y_train, X_test, y_test, feat_names, seed, max_train, device):
    X_fit, y_fit = _subsample(X_train, y_train, max_train, seed)
    clip_max = np.max(y_fit) * 2 if len(y_fit) else 1.0
    cat_idx = _categorical_indices(feat_names)

    def _run(dev):
        model = TabPFNRegressor(
            device=dev, ignore_pretraining_limits=True, random_state=seed,
            categorical_features_indices=cat_idx,
        )
        model.fit(X_fit, y_fit)
        chunk = 4096
        if len(X_test) > chunk:
            parts = [model.predict(X_test[i:i+chunk]) for i in range(0, len(X_test), chunk)]
            y_pred = np.concatenate(parts, axis=0)
        else:
            y_pred = model.predict(X_test)
        return np.clip(y_pred, 0, clip_max), dev

    use_dev = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        y_pred, used = _run(use_dev)
    except RuntimeError as e:
        if use_dev == 'cuda' and 'CUDA' in str(e):
            print(f"    CUDA OOM, retry on CPU: {e}")
            y_pred, used = _run('cpu')
        else:
            raise
    m = compute_all_metrics(y_test, y_pred)
    m['n_train_subsample'] = len(X_fit)
    m['n_train_full'] = len(X_train)
    m['n_test'] = len(X_test)
    m['device_used'] = used
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feature_sets', type=str, default='core_all_cities,core_all_cities_climate_plus')
    ap.add_argument('--input_variants', type=str, default='tab,tab_s2_pca64',
                    help='tab and/or tab_s2_pca64. raw skipped (1024-d too many for TabPFN context).')
    ap.add_argument('--split_types', type=str, default='random,grouped')
    ap.add_argument('--seeds', type=str, default='42,123,456')
    ap.add_argument('--max_train', type=int, default=10000)
    ap.add_argument('--device', type=str, default='auto')
    ap.add_argument('--out_path', type=str,
                    default=os.path.join(RESULTS_DIR, 'clean_building/task_a_s2_tabpfn_3seeds.csv'))
    args = ap.parse_args()

    feature_sets = [x for x in args.feature_sets.split(',') if x]
    input_variants = [x for x in args.input_variants.split(',') if x]
    split_types = [x for x in args.split_types.split(',') if x]
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]

    s2_index = load_s2_index()
    rows = []

    for fs_name in feature_sets:
        needs_ext = any(col in CLIMATE_FEATURE_COLS for col in get_feature_set(fs_name))
        df = load_and_prepare(fs_name, join_external=needs_ext)
        df = df[df[TARGET].notna()].copy()
        mm_df = filter_to_s2_subset(df, s2_index)
        print(f"\n[{fs_name}] full={len(df)} s2_subset={len(mm_df)}")

        for split_type in split_types:
            for input_variant in input_variants:
                for seed in seeds:
                    t0 = time.time()
                    splits = _split(mm_df, split_type=split_type, seed=seed)
                    train, test = splits['train'], splits['test']
                    X_tr, y_tr, X_te, y_te, feat, tab_dim, s2_dim = _prepare_inputs(
                        train, test, fs_name, input_variant
                    )
                    m = fit_eval_tabpfn(X_tr, y_tr, X_te, y_te, feat, seed, args.max_train, args.device)
                    m.update({
                        'task': 'A2_pooled', 'feature_set': fs_name, 'model': 'TabPFN_v2',
                        'input_variant': input_variant, 'seed': seed, 'split_type': split_type,
                        'subset_rows': len(mm_df), 'elapsed_sec': time.time() - t0,
                        'tab_dim': tab_dim, 's2_dim': s2_dim,
                    })
                    rows.append(m)
                    pd.DataFrame(rows).to_csv(args.out_path, index=False)
                    print(f"  {fs_name:32s} {split_type:8s} {input_variant:14s} seed={seed} "
                          f"r2={m['r2']:.4f} elapsed={m['elapsed_sec']:.1f}s")

    print(f"\n{len(rows)} runs → {args.out_path}")


if __name__ == '__main__':
    main()
