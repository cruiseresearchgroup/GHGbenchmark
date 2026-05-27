"""T1 fusion baseline: combine LightGBM structured prediction with Claude LLM
prediction on the overlapping 200-row LLM test subset.

Strategies:
  - lgbm_only: structured LightGBM (re-scored on the LLM 200 rows only)
  - claude_only: Claude zero-shot (from existing CSV)
  - avg: (lgbm + claude) / 2 in log space
  - ridge_stack: ridge regression with features [lgbm_pred, claude_pred] on val,
                 evaluated on test (here: 80/20 within the 200-row subset)

Output: results/t1a_fusion.csv
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor

sys.path.insert(0, str(Path(__file__).parent))
from t1_common import load_scope12, make_features, metrics, align, SEED

LLM_PRED = Path("results/t1a_llm_zero_shot_claude-haiku-4-5_predictions.csv")
OUT = Path("results/t1a_fusion.csv")


def main():
    df = load_scope12()
    df = df[df["subset_t1strict"]].copy()
    llm = pd.read_csv(LLM_PRED)
    llm = llm.rename(columns={"pred_scope12": "llm_pred_lin"})
    # Keep only LLM rows that parsed to positive values
    llm = llm[(llm["llm_pred_lin"].notna()) & (llm["llm_pred_lin"] > 0)]
    llm["llm_pred_log"] = np.log10(llm["llm_pred_lin"])

    # Merge to get features on the LLM 200-row subset
    sub = df.merge(llm[["nz_id","reporting_year","llm_pred_log"]],
                   on=["nz_id","reporting_year"], how="inner")
    print(f"LLM subset with features: {len(sub)}")

    # Train LightGBM on ALL T1-Strict train split (same as main), score on LLM subset
    train = df[df["split"] == "train"].reset_index(drop=True)
    X_tr = make_features(train, "structured_strict")
    X_sub = make_features(sub,   "structured_strict")
    X_tr, X_sub = align(X_tr, X_sub)
    y_tr = train["y"].values
    lg = LGBMRegressor(n_estimators=400, num_leaves=63, learning_rate=0.05,
                       random_state=SEED, n_jobs=8, verbose=-1, force_col_wise=True)
    lg.fit(X_tr, y_tr)
    sub["lgbm_pred_log"] = lg.predict(X_sub)
    y_true = sub["y"].values

    results = []
    results.append({"strategy":"lgbm_only","n":len(sub),
                    **metrics(y_true, sub["lgbm_pred_log"].values)})
    results.append({"strategy":"claude_only","n":len(sub),
                    **metrics(y_true, sub["llm_pred_log"].values)})
    results.append({"strategy":"avg","n":len(sub),
                    **metrics(y_true, 0.5*sub["lgbm_pred_log"] + 0.5*sub["llm_pred_log"])})

    # Ridge stacking: 50/50 split within the 200 rows (seed=SEED)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(sub))
    cut = len(sub) // 2
    val_idx, test_idx = idx[:cut], idx[cut:]
    X_stack = sub[["lgbm_pred_log","llm_pred_log"]].values
    rd = Ridge(alpha=0.1, random_state=SEED)
    rd.fit(X_stack[val_idx], y_true[val_idx])
    yp = rd.predict(X_stack[test_idx])
    results.append({"strategy":"ridge_stack_50_50","n":len(test_idx),
                    **metrics(y_true[test_idx], yp),
                    "coef_lgbm": float(rd.coef_[0]),
                    "coef_llm":  float(rd.coef_[1]),
                    "intercept": float(rd.intercept_)})

    # Grid search over α (weight on Claude) on the full 200 rows for illustration
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        yp = (1-alpha)*sub["lgbm_pred_log"] + alpha*sub["llm_pred_log"]
        results.append({"strategy":f"weighted_avg_alpha={alpha}","n":len(sub),
                        **metrics(y_true, yp.values)})

    OUT.parent.mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(OUT, index=False)
    print(f"→ {OUT}")
    for r in results:
        print(f"  {r['strategy']:28s}  n={r['n']}  R²={r['r2_log']:+.3f}  "
              f"MAE={r['mae_log']:.3f}  r={r['pearson_r']:.3f}")


if __name__ == "__main__":
    main()
