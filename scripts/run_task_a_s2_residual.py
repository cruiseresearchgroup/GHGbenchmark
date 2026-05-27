"""Residual-prediction MLP for S2 multimodal Task A.

Architecture (mathematical no-harm guarantee):
  1. Fit MLP_tab on (X_tab, y) → ŷ_tab
  2. Compute residuals: r = y − ŷ_tab on train set
  3. Fit MLP_s2 on (X_s2_reduced, r) → r̂_s2
  4. Final prediction: ŷ = ŷ_tab + r̂_s2

If S2 has no predictive signal beyond tab, MLP_s2 learns r̂≈0 → ŷ ≈ ŷ_tab.
This way S2 cannot hurt accuracy below tab-only.

Reuses run_task_a_s2.py's S2 subset filter and prepare_s2_arrays so PLS / PCA
variants share the same eligibility frame as the rest of the S2 experiments.

Output: results/clean_building/task_a_s2_residual_3seeds.csv
"""
import sys, os, argparse, time, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

from src.data.preprocessing import load_and_prepare, prepare_features
from src.data.feature_sets import TARGET, get_feature_set
from src.data.s2_multimodal import (
    load_s2_index, filter_to_s2_subset, prepare_s2_arrays,
)
from src.data.splitters import grouped_split, random_split
from src.evaluation.metrics import compute_all_metrics

warnings.filterwarnings("ignore")

CLIMATE_FEATURE_COLS = {
    'hdd', 'cdd', 'annual_mean_temp_c', 'annual_rh_mean',
    'annual_ssrd_mj_m2_day', 'annual_wind_ms',
}


class MLPHead(nn.Module):
    def __init__(self, d_in, hidden=128, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x)


def fit_mlp(X_train, y_train, X_test, n_epoch=80, lr=1e-3, bs=256, wd=1e-4,
            hidden=128, drop=0.2, device='cuda', seed=42, log_target=True):
    """Fit MLP and return predictions on test (in original-scale if log_target)."""
    torch.manual_seed(seed); np.random.seed(seed)
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_train); Xte = sc.transform(X_test)
    if log_target:
        y_fit = np.log1p(np.maximum(y_train, 0))
    else:
        y_fit = y_train

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(y_fit, dtype=torch.float32, device=device)
    model = MLPHead(Xtr.shape[1], hidden=hidden, drop=drop).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    ds = TensorDataset(Xtr_t, ytr_t); dl = DataLoader(ds, batch_size=bs, shuffle=True)
    for ep in range(n_epoch):
        model.train()
        for xb, yb in dl:
            opt.zero_grad(); pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        # Train predictions (for residual computation)
        y_pred_train = model(Xtr_t).squeeze(-1).cpu().numpy()
        y_pred_test = model(Xte_t).squeeze(-1).cpu().numpy()
    if log_target:
        y_pred_train = np.expm1(y_pred_train)
        y_pred_test = np.expm1(y_pred_test)
    return y_pred_train, y_pred_test


def fit_mlp_residual(X_train, r_train, X_test, n_epoch=80, lr=1e-3, bs=256,
                      wd=1e-4, hidden=128, drop=0.2, device='cuda', seed=42):
    """Fit residual MLP (no log-transform on residuals — they may be negative)."""
    torch.manual_seed(seed); np.random.seed(seed)
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_train); Xte = sc.transform(X_test)
    # Standardize residuals to help training stability; un-standardize on output
    r_mean, r_std = float(r_train.mean()), float(r_train.std() + 1e-8)
    r_norm = (r_train - r_mean) / r_std

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
    rtr_t = torch.tensor(r_norm, dtype=torch.float32, device=device)
    model = MLPHead(Xtr.shape[1], hidden=hidden, drop=drop).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    ds = TensorDataset(Xtr_t, rtr_t); dl = DataLoader(ds, batch_size=bs, shuffle=True)
    for ep in range(n_epoch):
        model.train()
        for xb, yb in dl:
            opt.zero_grad(); pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        r_pred = model(Xte_t).squeeze(-1).cpu().numpy()
    # Un-standardize
    return r_pred * r_std + r_mean


def _split(df, split_type, seed):
    if split_type == 'grouped':
        return grouped_split(df, test_size=0.2, val_size=0.1, seed=seed)
    return random_split(df, test_size=0.2, val_size=0.1, seed=seed)


def run_one(df, fs_name, split_type, seed, s2_variant, device):
    splits = _split(df, split_type=split_type, seed=seed)
    train_df, test_df = splits['train'], splits['test']

    # Tabular features
    X_train_tab, y_train, _, encoders, medians, dropped = prepare_features(
        train_df, fs_name, encoders=None, drop_all_nan=False
    )
    X_test_tab, y_test, _, _, _, _ = prepare_features(
        test_df, fs_name, encoders=encoders, medians=medians, drop_cols=dropped
    )
    X_train_tab = X_train_tab.astype(np.float32); X_test_tab = X_test_tab.astype(np.float32)

    # Step 1: tab-only MLP
    yp_tab_train, yp_tab_test = fit_mlp(
        X_train_tab, y_train, X_test_tab, device=device, seed=seed, log_target=True
    )
    clip_max = np.max(y_train) * 2 if len(y_train) else 1.0
    yp_tab_test = np.clip(yp_tab_test, 0, clip_max)
    yp_tab_train = np.clip(yp_tab_train, 0, clip_max)
    metrics_tab = compute_all_metrics(y_test, yp_tab_test)

    # Step 2: residuals
    residuals_train = y_train - yp_tab_train

    # Step 3: S2 features (pca64 or pls64)
    y_train_for_pls = y_train if s2_variant.startswith('pls') else None
    train_s2, test_s2, _, bundle = prepare_s2_arrays(
        train_df, test_df, variant=s2_variant, val_df=None, y_train=y_train_for_pls
    )

    # Step 4: residual MLP on S2
    r_pred_test = fit_mlp_residual(
        train_s2, residuals_train, test_s2, device=device, seed=seed
    )

    # Step 5: combined prediction
    yp_combined = np.clip(yp_tab_test + r_pred_test, 0, clip_max)
    metrics_combined = compute_all_metrics(y_test, yp_combined)

    return {
        'tab_only_r2': metrics_tab['r2'],
        'tab_only_mae': metrics_tab['mae'],
        'tab_only_log_mae': metrics_tab['log_mae'],
        'residual_r2': metrics_combined['r2'],
        'residual_mae': metrics_combined['mae'],
        'residual_log_mae': metrics_combined['log_mae'],
        'delta_r2': metrics_combined['r2'] - metrics_tab['r2'],
        'n_train': len(y_train), 'n_test': len(y_test),
        'tab_dim': X_train_tab.shape[1], 's2_dim': bundle.out_dim,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feature_sets', type=str, default='core_all_cities,core_all_cities_climate_plus')
    ap.add_argument('--split_types', type=str, default='random,grouped')
    ap.add_argument('--seeds', type=str, default='42,123,456')
    ap.add_argument('--s2_variants', type=str, default='pca64,pls64')
    ap.add_argument('--out_path', type=str,
                    default='results/clean_building/task_a_s2_residual_3seeds.csv')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    feature_sets = [x for x in args.feature_sets.split(',') if x]
    split_types = [x for x in args.split_types.split(',') if x]
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    s2_variants = [v for v in args.s2_variants.split(',') if v]

    s2_index = load_s2_index()
    rows = []

    for fs_name in feature_sets:
        needs_ext = any(c in CLIMATE_FEATURE_COLS for c in get_feature_set(fs_name))
        df = load_and_prepare(fs_name, join_external=needs_ext)
        df = df[df[TARGET].notna()].copy()
        mm_df = filter_to_s2_subset(df, s2_index)
        print(f"\n[{fs_name}] full={len(df)} s2_subset={len(mm_df)}")

        for split_type in split_types:
            for s2_variant in s2_variants:
                for seed in seeds:
                    t0 = time.time()
                    try:
                        m = run_one(mm_df, fs_name, split_type, seed, s2_variant, device)
                    except Exception as e:
                        print(f"  [{split_type} {s2_variant} seed={seed}] FAILED: {e}")
                        continue
                    m.update({
                        'feature_set': fs_name, 'split_type': split_type,
                        's2_variant': s2_variant, 'seed': seed,
                        'elapsed_sec': time.time() - t0,
                    })
                    rows.append(m)
                    pd.DataFrame(rows).to_csv(args.out_path, index=False)
                    print(f"  {split_type:8s} {s2_variant:8s} seed={seed} "
                          f"tab_r2={m['tab_only_r2']:+.4f} resid_r2={m['residual_r2']:+.4f} "
                          f"Δ={m['delta_r2']:+.4f}  ({m['elapsed_sec']:.1f}s)")

    print(f"\n{len(rows)} runs → {args.out_path}")


if __name__ == '__main__':
    main()
