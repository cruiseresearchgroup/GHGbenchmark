"""T1-A extension experiments:
  --which dl          MLP + FT-Transformer baselines (T1-Strict, structured_strict)
  --which seeds       5-seed bootstrap on LightGBM + TabPFN (error bars)
  --which hpsearch    Random search HP for LightGBM / XGBoost (50 configs each)
  --which temporal    Train 2018-2021 / test 2022 for all models
  --which bycompany   Held-out companies split (50/50 by nz_id)
  --which shap        Permutation importance + SHAP global summary for LightGBM
  --which bygroup     Per-sector / per-country R² breakdown for LightGBM

All outputs append to results/t1a_ext_<which>.csv (new file).
"""
from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

sys.path.insert(0, str(Path(__file__).parent))
from t1_common import (load_scope12, make_features, metrics, align, SEED)

OUT_DIR = Path("results"); OUT_DIR.mkdir(exist_ok=True)


def build_strict_split():
    df = load_scope12()
    df = df[df["subset_t1strict"]].copy()
    train = df[df["split"] == "train"].reset_index(drop=True)
    test  = df[df["split"] == "test"].reset_index(drop=True)
    X_tr = make_features(train, "structured_strict")
    X_te = make_features(test,  "structured_strict")
    X_tr, X_te = align(X_tr, X_te)
    y_tr = train["y"].values; y_te = test["y"].values
    return df, train, test, X_tr.astype(float), X_te.astype(float), y_tr, y_te


# ────────────────────────────────────────────────────────────────────────────
def run_dl():
    """MLP + FT-Transformer baselines on T1-Strict/structured_strict."""
    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[dl] device={device}")

    _, train, test, X_tr, X_te, y_tr, y_te = build_strict_split()
    sc = StandardScaler(); Xtr = sc.fit_transform(X_tr); Xte = sc.transform(X_te)
    d = Xtr.shape[1]
    print(f"[dl] d={d}, n_train={len(y_tr)}, n_test={len(y_te)}")

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)

    def train_model(model, n_epoch=80, lr=1e-3, bs=256, wd=1e-4):
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
            return model(Xte_t).squeeze(-1).cpu().numpy()

    # --- MLP ---
    class MLP(nn.Module):
        def __init__(self, d_in, hidden=256, drop=0.2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(drop),
                nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(drop),
                nn.Linear(hidden, hidden//2), nn.GELU(), nn.Dropout(drop),
                nn.Linear(hidden//2, 1),
            )
        def forward(self, x): return self.net(x)

    torch.manual_seed(SEED); np.random.seed(SEED)
    t0 = time.time()
    y_pred = train_model(MLP(d).to(device), n_epoch=80, lr=1e-3)
    m_mlp = metrics(y_te, y_pred)
    m_mlp.update(model="mlp", n_train=len(y_tr), n_test=len(y_te), wall_s=time.time()-t0)
    print(f"[dl] MLP R²={m_mlp['r2_log']:+.3f} MAE={m_mlp['mae_log']:.3f} ({m_mlp['wall_s']:.1f}s)")

    # --- FT-Transformer (lite): embed each feature as a token and apply MHA ---
    class FTTransformer(nn.Module):
        def __init__(self, d_in, d_tok=32, n_heads=4, n_layers=3, drop=0.1):
            super().__init__()
            # per-feature linear embedding: each scalar -> d_tok vector
            self.embed = nn.Parameter(torch.randn(d_in, d_tok) * 0.02)
            self.bias  = nn.Parameter(torch.randn(d_in, d_tok) * 0.02)
            self.cls   = nn.Parameter(torch.randn(1, 1, d_tok) * 0.02)
            enc_layer = nn.TransformerEncoderLayer(d_tok, n_heads, dim_feedforward=d_tok*2,
                                                   dropout=drop, batch_first=True,
                                                   activation="gelu")
            self.tx = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
            self.head = nn.Sequential(nn.LayerNorm(d_tok), nn.Linear(d_tok, 1))
        def forward(self, x):
            # x: (B, d_in); tokens: (B, d_in, d_tok)
            tok = x.unsqueeze(-1) * self.embed + self.bias
            cls = self.cls.expand(x.size(0), -1, -1)
            tok = torch.cat([cls, tok], dim=1)
            h = self.tx(tok)
            return self.head(h[:, 0])

    torch.manual_seed(SEED); np.random.seed(SEED)
    t0 = time.time()
    y_pred = train_model(FTTransformer(d).to(device), n_epoch=60, lr=5e-4, bs=128)
    m_ft = metrics(y_te, y_pred)
    m_ft.update(model="ft_transformer", n_train=len(y_tr), n_test=len(y_te), wall_s=time.time()-t0)
    print(f"[dl] FT-Transformer R²={m_ft['r2_log']:+.3f} MAE={m_ft['mae_log']:.3f} ({m_ft['wall_s']:.1f}s)")

    out = OUT_DIR / "t1a_ext_dl.csv"
    pd.DataFrame([m_mlp, m_ft]).to_csv(out, index=False)
    print(f"[dl] → {out}")


# ────────────────────────────────────────────────────────────────────────────
def run_seeds(n_seeds: int = 5):
    """Bootstrap test-set 1,000× to report CI for top models."""
    _, train, test, X_tr, X_te, y_tr, y_te = build_strict_split()

    # Fit models once (deterministic under seed=42 for LGBM; TabPFN handles its own seeding)
    results = []
    # LightGBM main
    m = LGBMRegressor(n_estimators=400, num_leaves=63, learning_rate=0.05,
                      random_state=SEED, n_jobs=8, verbose=-1, force_col_wise=True)
    m.fit(X_tr, y_tr); yp_lgbm = m.predict(X_te)

    # Also try multi-seed LightGBM (retrain with different bagging/feature seeds)
    lgbm_metrics = []
    for seed in range(SEED, SEED + n_seeds):
        m = LGBMRegressor(n_estimators=400, num_leaves=63, learning_rate=0.05,
                          random_state=seed, n_jobs=8, verbose=-1,
                          bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.9,
                          bagging_seed=seed, feature_fraction_seed=seed,
                          force_col_wise=True)
        m.fit(X_tr, y_tr)
        lgbm_metrics.append(metrics(y_te, m.predict(X_te)))
    for k in ["mae_log", "r2_log", "pearson_r"]:
        vals = [d[k] for d in lgbm_metrics]
        results.append({"model": "lightgbm", "metric": k,
                        "mean": float(np.mean(vals)),
                        "std":  float(np.std(vals, ddof=1)),
                        "min":  float(np.min(vals)),
                        "max":  float(np.max(vals)),
                        "n_seeds": n_seeds})

    # Bootstrap test-set CI on fixed LGBM predictions (no model refit)
    rng = np.random.default_rng(SEED)
    B = 1000
    n = len(y_te); idx_all = np.arange(n)
    boot = []
    for _ in range(B):
        idx = rng.choice(idx_all, n, replace=True)
        boot.append(metrics(y_te[idx], yp_lgbm[idx]))
    for k in ["mae_log", "r2_log", "pearson_r"]:
        vals = [d[k] for d in boot]
        results.append({"model": "lightgbm_bootstrap_test", "metric": k,
                        "mean": float(np.mean(vals)),
                        "std":  float(np.std(vals, ddof=1)),
                        "ci_lo": float(np.percentile(vals, 2.5)),
                        "ci_hi": float(np.percentile(vals, 97.5)),
                        "n_boot": B})

    # TabPFN seeds
    try:
        from tabpfn import TabPFNRegressor
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tabpfn_metrics = []
        rng2 = np.random.default_rng(SEED)
        for seed in range(SEED, SEED + n_seeds):
            reg = TabPFNRegressor(device=device, ignore_pretraining_limits=True,
                                  random_state=seed)
            # subsample train to 10k with seed
            if len(X_tr) > 10000:
                idx = rng2.choice(len(X_tr), 10000, replace=False)
                Xs = X_tr.iloc[idx].values; ys = y_tr[idx]
            else:
                Xs, ys = X_tr.values, y_tr
            reg.fit(Xs, ys)
            tabpfn_metrics.append(metrics(y_te, reg.predict(X_te.values)))
        for k in ["mae_log", "r2_log", "pearson_r"]:
            vals = [d[k] for d in tabpfn_metrics]
            results.append({"model": "tabpfn", "metric": k,
                            "mean": float(np.mean(vals)),
                            "std":  float(np.std(vals, ddof=1)),
                            "min":  float(np.min(vals)),
                            "max":  float(np.max(vals)),
                            "n_seeds": n_seeds})
    except Exception as e:
        print(f"[seeds] TabPFN skipped: {e}")

    out = OUT_DIR / "t1a_ext_seeds.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"[seeds] → {out}")
    for r in results:
        print(f"  {r['model']:28s} {r['metric']:11s}  mean={r['mean']:+.4f}  "
              f"{'std='+str(round(r.get('std',0),4)) if 'std' in r else ''}  "
              f"{'['+str(round(r.get('ci_lo',0),3))+','+str(round(r.get('ci_hi',0),3))+']' if 'ci_lo' in r else ''}")


# ────────────────────────────────────────────────────────────────────────────
def run_hpsearch(n_trials: int = 40):
    """Random search HP for LightGBM and XGBoost on val split."""
    df = load_scope12()
    df = df[df["subset_t1strict"]].copy()
    train = df[df["split"] == "train"].reset_index(drop=True)
    val   = df[df["split"] == "val"].reset_index(drop=True)
    test  = df[df["split"] == "test"].reset_index(drop=True)
    X_tr = make_features(train, "structured_strict")
    X_va = make_features(val,   "structured_strict")
    X_te = make_features(test,  "structured_strict")
    X_tr, X_va = align(X_tr, X_va); X_tr, X_te = align(X_tr, X_te)
    X_va = X_va.reindex(columns=X_tr.columns, fill_value=0)
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)
    y_tr = train["y"].values; y_va = val["y"].values; y_te = test["y"].values

    rng = np.random.default_rng(SEED)
    results = []

    # LightGBM
    best_val, best_cfg = None, None
    for t in range(n_trials):
        cfg = dict(
            n_estimators = int(rng.choice([300, 400, 600, 800, 1200])),
            num_leaves   = int(rng.choice([15, 31, 63, 127, 255])),
            learning_rate = float(rng.choice([0.01, 0.02, 0.05, 0.08, 0.1])),
            min_child_samples = int(rng.choice([5, 10, 20, 50])),
            reg_alpha  = float(rng.choice([0.0, 0.01, 0.1, 1.0])),
            reg_lambda = float(rng.choice([0.0, 0.01, 0.1, 1.0])),
            feature_fraction = float(rng.choice([0.7, 0.8, 0.9, 1.0])),
            bagging_fraction = float(rng.choice([0.7, 0.8, 0.9, 1.0])),
            bagging_freq = 1,
        )
        m = LGBMRegressor(random_state=SEED, n_jobs=8, verbose=-1,
                          force_col_wise=True, **cfg)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[])
        mv = metrics(y_va, m.predict(X_va))
        mt = metrics(y_te, m.predict(X_te))
        if best_val is None or mv["r2_log"] > best_val:
            best_val, best_cfg = mv["r2_log"], cfg
        results.append({"model": "lightgbm", "trial": t, "val_r2": mv["r2_log"],
                        "test_r2": mt["r2_log"], "test_mae": mt["mae_log"], **cfg})
    print(f"[hp] lightgbm best val R²={best_val:.3f} cfg={best_cfg}")
    # Refit best on train+val
    best_m = LGBMRegressor(random_state=SEED, n_jobs=8, verbose=-1,
                           force_col_wise=True, **best_cfg)
    X_tv = pd.concat([X_tr, X_va]); y_tv = np.concatenate([y_tr, y_va])
    best_m.fit(X_tv, y_tv)
    final_lgbm = metrics(y_te, best_m.predict(X_te))
    print(f"[hp] lightgbm final (train+val) test R²={final_lgbm['r2_log']:+.3f}  "
          f"MAE={final_lgbm['mae_log']:.3f}")

    # XGBoost
    best_val_xgb, best_cfg_xgb = None, None
    for t in range(n_trials):
        cfg = dict(
            n_estimators = int(rng.choice([300, 500, 800, 1200])),
            max_depth = int(rng.choice([4, 6, 8, 10])),
            learning_rate = float(rng.choice([0.01, 0.02, 0.05, 0.08, 0.1])),
            min_child_weight = float(rng.choice([1, 3, 5, 10])),
            subsample = float(rng.choice([0.7, 0.8, 0.9, 1.0])),
            colsample_bytree = float(rng.choice([0.7, 0.8, 0.9, 1.0])),
            reg_alpha = float(rng.choice([0.0, 0.01, 0.1, 1.0])),
            reg_lambda = float(rng.choice([0.0, 0.1, 1.0, 5.0])),
        )
        m = XGBRegressor(random_state=SEED, n_jobs=8, verbosity=0, **cfg)
        m.fit(X_tr, y_tr)
        mv = metrics(y_va, m.predict(X_va))
        mt = metrics(y_te, m.predict(X_te))
        if best_val_xgb is None or mv["r2_log"] > best_val_xgb:
            best_val_xgb, best_cfg_xgb = mv["r2_log"], cfg
        results.append({"model": "xgboost", "trial": t, "val_r2": mv["r2_log"],
                        "test_r2": mt["r2_log"], "test_mae": mt["mae_log"], **cfg})
    print(f"[hp] xgboost best val R²={best_val_xgb:.3f} cfg={best_cfg_xgb}")
    best_m = XGBRegressor(random_state=SEED, n_jobs=8, verbosity=0, **best_cfg_xgb)
    best_m.fit(X_tv, y_tv)
    final_xgb = metrics(y_te, best_m.predict(X_te))
    print(f"[hp] xgboost final (train+val) test R²={final_xgb['r2_log']:+.3f}  "
          f"MAE={final_xgb['mae_log']:.3f}")

    out = OUT_DIR / "t1a_ext_hpsearch.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    summary_out = OUT_DIR / "t1a_ext_hpsearch_summary.csv"
    pd.DataFrame([
        {"model":"lightgbm_tuned", **final_lgbm, "n_train": len(y_tv), "n_test": len(y_te),
         "best_cfg": json.dumps(best_cfg)},
        {"model":"xgboost_tuned",  **final_xgb,  "n_train": len(y_tv), "n_test": len(y_te),
         "best_cfg": json.dumps(best_cfg_xgb)},
    ]).to_csv(summary_out, index=False)
    print(f"[hp] → {out} + {summary_out}")


# ────────────────────────────────────────────────────────────────────────────
def run_temporal():
    """Temporal split: train 2018-2021 / test 2022 (exclude 2023 = partial year).
    All T1-Strict rows, no val set (no HP tuning). Same models as main."""
    df = load_scope12()
    df = df[df["subset_t1strict"]].copy()
    train = df[df["reporting_year"].isin([2018,2019,2020,2021])].reset_index(drop=True)
    test  = df[df["reporting_year"] == 2022].reset_index(drop=True)
    print(f"[temporal] n_train={len(train)}  n_test={len(test)}")

    results = []
    for fs in ["structured_wide", "structured_strict"]:
        X_tr = make_features(train, fs)
        X_te = make_features(test,  fs)
        X_tr, X_te = align(X_tr, X_te)
        y_tr = train["y"].values; y_te = test["y"].values

        # sector mean baseline
        sm = train.groupby("gics_11")["y"].mean()
        gm = train["y"].mean()
        y_pred = test["gics_11"].map(sm).fillna(gm).values
        results.append({"split":"temporal","feature_set":fs,"model":"sector_mean",
                        "n_train":len(train),"n_test":len(test), **metrics(y_te, y_pred)})

        # Ridge
        sc = StandardScaler(); Xtr_s = sc.fit_transform(X_tr); Xte_s = sc.transform(X_te)
        rd = Ridge(alpha=1.0, random_state=SEED); rd.fit(Xtr_s, y_tr)
        results.append({"split":"temporal","feature_set":fs,"model":"ridge",
                        "n_train":len(train),"n_test":len(test),
                        **metrics(y_te, rd.predict(Xte_s))})
        # XGB
        xgb = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                           random_state=SEED, n_jobs=8, verbosity=0)
        xgb.fit(X_tr, y_tr)
        results.append({"split":"temporal","feature_set":fs,"model":"xgboost",
                        "n_train":len(train),"n_test":len(test),
                        **metrics(y_te, xgb.predict(X_te))})
        # LGBM
        lg = LGBMRegressor(n_estimators=400, num_leaves=63, learning_rate=0.05,
                           random_state=SEED, n_jobs=8, verbose=-1, force_col_wise=True)
        lg.fit(X_tr, y_tr)
        results.append({"split":"temporal","feature_set":fs,"model":"lightgbm",
                        "n_train":len(train),"n_test":len(test),
                        **metrics(y_te, lg.predict(X_te))})

    # TabPFN on structured_strict
    try:
        from tabpfn import TabPFNRegressor
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        X_tr = make_features(train, "structured_strict")
        X_te = make_features(test,  "structured_strict")
        X_tr, X_te = align(X_tr, X_te)
        y_tr = train["y"].values; y_te = test["y"].values
        rng = np.random.default_rng(SEED)
        MAX_N = 10000
        if len(X_tr) > MAX_N:
            idx = rng.choice(len(X_tr), MAX_N, replace=False)
            Xs = X_tr.iloc[idx].values; ys = y_tr[idx]
        else:
            Xs, ys = X_tr.values, y_tr
        reg = TabPFNRegressor(device=device, ignore_pretraining_limits=True, random_state=SEED)
        reg.fit(Xs, ys)
        results.append({"split":"temporal","feature_set":"structured_strict","model":"tabpfn",
                        "n_train":len(Xs),"n_test":len(test),
                        **metrics(y_te, reg.predict(X_te.values))})
    except Exception as e:
        print(f"[temporal] TabPFN skipped: {e}")

    out = OUT_DIR / "t1a_ext_temporal.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"[temporal] → {out}")
    for r in results:
        print(f"  {r['feature_set']:18s} {r['model']:15s}  R²={r['r2_log']:+.3f}  MAE={r['mae_log']:.3f}")


# ────────────────────────────────────────────────────────────────────────────
def run_bycompany():
    """Held-out companies: split T1-Strict companies 50/50 by nz_id."""
    df = load_scope12()
    df = df[df["subset_t1strict"]].copy()
    companies = np.array(sorted(df["nz_id"].unique()))
    rng = np.random.default_rng(SEED)
    rng.shuffle(companies)
    cut = len(companies) // 2
    train_ids, test_ids = set(companies[:cut]), set(companies[cut:])
    train = df[df["nz_id"].isin(train_ids)].reset_index(drop=True)
    test  = df[df["nz_id"].isin(test_ids)].reset_index(drop=True)
    print(f"[bycompany] n_companies=(train={len(train_ids)}, test={len(test_ids)})  "
          f"n_rows=(train={len(train)}, test={len(test)})")

    results = []
    for fs in ["structured_strict"]:
        X_tr = make_features(train, fs); X_te = make_features(test, fs)
        X_tr, X_te = align(X_tr, X_te)
        y_tr = train["y"].values; y_te = test["y"].values

        sc = StandardScaler(); Xtr_s = sc.fit_transform(X_tr); Xte_s = sc.transform(X_te)
        rd = Ridge(alpha=1.0, random_state=SEED); rd.fit(Xtr_s, y_tr)
        results.append({"split":"bycompany","feature_set":fs,"model":"ridge",
                        "n_train":len(train),"n_test":len(test),
                        **metrics(y_te, rd.predict(Xte_s))})
        xgb = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                           random_state=SEED, n_jobs=8, verbosity=0)
        xgb.fit(X_tr, y_tr)
        results.append({"split":"bycompany","feature_set":fs,"model":"xgboost",
                        "n_train":len(train),"n_test":len(test),
                        **metrics(y_te, xgb.predict(X_te))})
        lg = LGBMRegressor(n_estimators=400, num_leaves=63, learning_rate=0.05,
                           random_state=SEED, n_jobs=8, verbose=-1, force_col_wise=True)
        lg.fit(X_tr, y_tr)
        results.append({"split":"bycompany","feature_set":fs,"model":"lightgbm",
                        "n_train":len(train),"n_test":len(test),
                        **metrics(y_te, lg.predict(X_te))})

    out = OUT_DIR / "t1a_ext_bycompany.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"[bycompany] → {out}")
    for r in results:
        print(f"  {r['model']:15s}  R²={r['r2_log']:+.3f}  MAE={r['mae_log']:.3f}")


# ────────────────────────────────────────────────────────────────────────────
def run_shap():
    """Permutation importance + SHAP summary for LightGBM on T1-Strict/structured_strict."""
    import shap as shaplib
    from sklearn.inspection import permutation_importance

    _, train, test, X_tr, X_te, y_tr, y_te = build_strict_split()
    lg = LGBMRegressor(n_estimators=400, num_leaves=63, learning_rate=0.05,
                       random_state=SEED, n_jobs=8, verbose=-1, force_col_wise=True)
    lg.fit(X_tr, y_tr)

    # Permutation importance (test set)
    t0 = time.time()
    pi = permutation_importance(lg, X_te, y_te, n_repeats=10, random_state=SEED,
                                 scoring="r2", n_jobs=8)
    print(f"[shap] perm importance done in {time.time()-t0:.1f}s")
    pi_df = pd.DataFrame({"feature": X_tr.columns,
                           "perm_imp_mean": pi.importances_mean,
                           "perm_imp_std":  pi.importances_std})
    pi_df = pi_df.sort_values("perm_imp_mean", ascending=False)
    (OUT_DIR / "t1a_ext_perm_importance.csv").write_text(pi_df.to_csv(index=False))
    print(f"[shap] top 10 permutation importance:")
    print(pi_df.head(10).to_string(index=False))

    # SHAP on a subsample of test (full is too slow)
    t0 = time.time()
    sample_idx = np.random.default_rng(SEED).choice(len(X_te), min(500, len(X_te)),
                                                     replace=False)
    expl = shaplib.TreeExplainer(lg)
    sv = expl.shap_values(X_te.iloc[sample_idx])
    print(f"[shap] SHAP compute {sv.shape} in {time.time()-t0:.1f}s")
    shap_df = pd.DataFrame({"feature": X_tr.columns,
                             "mean_abs_shap": np.abs(sv).mean(axis=0)})
    shap_df = shap_df.sort_values("mean_abs_shap", ascending=False)
    (OUT_DIR / "t1a_ext_shap_importance.csv").write_text(shap_df.to_csv(index=False))
    print(f"[shap] top 10 mean|SHAP|:")
    print(shap_df.head(10).to_string(index=False))


# ────────────────────────────────────────────────────────────────────────────
def run_bygroup():
    """Per-sector and per-country R² breakdown for LightGBM on T1-Strict/structured_strict."""
    _, train, test, X_tr, X_te, y_tr, y_te = build_strict_split()
    lg = LGBMRegressor(n_estimators=400, num_leaves=63, learning_rate=0.05,
                       random_state=SEED, n_jobs=8, verbose=-1, force_col_wise=True)
    lg.fit(X_tr, y_tr)
    yp = lg.predict(X_te)

    per_sector = []
    for g, idx in test.groupby("gics_11").groups.items():
        if len(idx) < 20: continue
        i = np.array([test.index.get_loc(ix) for ix in idx])
        m = metrics(y_te[i], yp[i])
        per_sector.append({"group":"sector","value":g,"n":len(i), **m})
    per_country = []
    for g, idx in test.groupby("country_iso2").groups.items():
        if len(idx) < 20: continue
        i = np.array([test.index.get_loc(ix) for ix in idx])
        m = metrics(y_te[i], yp[i])
        per_country.append({"group":"country","value":g,"n":len(i), **m})
    df_out = pd.DataFrame(per_sector + per_country).sort_values(["group","n"], ascending=[True, False])
    out = OUT_DIR / "t1a_ext_bygroup.csv"
    df_out.to_csv(out, index=False)
    print(f"[bygroup] → {out}")
    print(df_out[["group","value","n","r2_log","mae_log","median_ape"]].to_string(index=False))


# ────────────────────────────────────────────────────────────────────────────
def run_scope_split():
    """Scope 1 / Scope 2 separate targets (re-build target from NZDPU raw)."""
    from t1_common import load_scope_split
    results = []
    for which in ["s1", "s2"]:
        print(f"\n[scope:{which}]")
        df = load_scope_split(which)
        df = df[df["subset_t1strict"]].copy()
        # Use same random split by (nz_id, reporting_year) — inherited via split col
        train = df[df["split"] == "train"].reset_index(drop=True)
        test  = df[df["split"] == "test"].reset_index(drop=True)
        print(f"  n_train={len(train)}  n_test={len(test)}")
        if len(test) < 50:
            print(f"  too small, skipping"); continue
        X_tr = make_features(train, "structured_strict")
        X_te = make_features(test,  "structured_strict")
        X_tr, X_te = align(X_tr, X_te)
        y_tr = train["y"].values; y_te = test["y"].values

        sc = StandardScaler(); Xtr_s = sc.fit_transform(X_tr); Xte_s = sc.transform(X_te)
        rd = Ridge(alpha=1.0, random_state=SEED); rd.fit(Xtr_s, y_tr)
        results.append({"target":which,"model":"ridge",
                        "n_train":len(train),"n_test":len(test),
                        **metrics(y_te, rd.predict(Xte_s))})
        xgb = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                           random_state=SEED, n_jobs=8, verbosity=0)
        xgb.fit(X_tr, y_tr)
        results.append({"target":which,"model":"xgboost",
                        "n_train":len(train),"n_test":len(test),
                        **metrics(y_te, xgb.predict(X_te))})
        lg = LGBMRegressor(n_estimators=400, num_leaves=63, learning_rate=0.05,
                           random_state=SEED, n_jobs=8, verbose=-1, force_col_wise=True)
        lg.fit(X_tr, y_tr)
        results.append({"target":which,"model":"lightgbm",
                        "n_train":len(train),"n_test":len(test),
                        **metrics(y_te, lg.predict(X_te))})

        # TabPFN
        try:
            from tabpfn import TabPFNRegressor
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            rng = np.random.default_rng(SEED); MAX_N = 10000
            if len(X_tr) > MAX_N:
                idx = rng.choice(len(X_tr), MAX_N, replace=False)
                Xs = X_tr.iloc[idx].values; ys = y_tr[idx]
            else:
                Xs, ys = X_tr.values, y_tr
            reg = TabPFNRegressor(device=device, ignore_pretraining_limits=True, random_state=SEED)
            reg.fit(Xs, ys)
            results.append({"target":which,"model":"tabpfn",
                            "n_train":len(Xs),"n_test":len(test),
                            **metrics(y_te, reg.predict(X_te.values))})
        except Exception as e:
            print(f"  TabPFN skipped: {e}")

    out = OUT_DIR / "t1a_ext_scope_split.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"[scope_split] → {out}")
    for r in results:
        print(f"  {r['target']}  {r['model']:10s}  R²={r['r2_log']:+.3f}  MAE={r['mae_log']:.3f}")


# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True,
                    choices=["dl","seeds","hpsearch","temporal","bycompany","shap","bygroup","scope_split","all"])
    ap.add_argument("--n_trials", type=int, default=40)
    ap.add_argument("--n_seeds", type=int, default=5)
    args = ap.parse_args()
    w = args.which
    if w in ("dl","all"):      run_dl()
    if w in ("seeds","all"):   run_seeds(args.n_seeds)
    if w in ("hpsearch","all"):run_hpsearch(args.n_trials)
    if w in ("temporal","all"):run_temporal()
    if w in ("bycompany","all"):run_bycompany()
    if w in ("shap","all"):    run_shap()
    if w in ("bygroup","all"): run_bygroup()
    if w in ("scope_split","all"): run_scope_split()
