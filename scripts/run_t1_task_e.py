"""
Task T1-E: temporal forecasting of Scope 1+2 emissions.

Setup:
  - Keep companies with an unbroken (2018, 2019, 2020, 2021, 2022) panel of
    cleaned Scope 1+2 values  →  2,621 companies.
  - Context window:  2018-2021  (4 years, log10 tCO2e)
  - Forecast target: 2022        (1-step ahead, univariate)

Models:
  A. Classical / ML baselines
     persistence      y_{T+1} = y_T
     mean_3yr         y_{T+1} = mean(y_{T-2}, y_{T-1}, y_T)
     linear_trend     OLS on (t, y) over context window, extrapolate
     sector_growth    y_{T+1} = y_T × median growth-rate of sector in context
     xgboost_features same features as T1-A + lagged y's (1, 2, 3 years)
  B. Zero-shot time-series foundation model
     chronos-bolt-small  (Amazon, HuggingFace)  — univariate, no training

Metrics: same as T1-A (MAE_log, R²_log, Pearson r, MAPE, MedAPE).

Output: results/t1e_results.csv  +  results/t1e_run.log
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

BASELINE = Path("data/company-level/nzdpu_enriched/factor_baseline_v2.csv")
OUT      = Path("results/t1e_results.csv")
SEED     = 42

CONTEXT_YEARS = [2018, 2019, 2020, 2021]
TARGET_YEAR   = 2022

# ── Build the balanced panel ────────────────────────────────────────────
print("Loading panel...")
df = pd.read_csv(BASELINE)
df = df[df["scope12_actual_tco2e"].notna() &
        (df["scope12_actual_tco2e"] >= 10) &
        (df["scope12_actual_tco2e"] <= 5e8)]
df = df.sort_values(["nz_id", "reporting_year"])

all_years = CONTEXT_YEARS + [TARGET_YEAR]
wide = df.pivot_table(index=["nz_id","gics_11","country_iso2"],
                      columns="reporting_year",
                      values="scope12_actual_tco2e",
                      aggfunc="first")
# Keep only rows with every year non-null
have_all = wide[all_years].notna().all(axis=1)
panel = wide.loc[have_all, all_years].reset_index()
panel["gics_11"] = panel["gics_11"].fillna("Unknown")
print(f"  companies with full {min(all_years)}-{max(all_years)} panel: {len(panel):,}")

y_log = np.log10(panel[all_years].values)           # (N, 5)
y_ctx = y_log[:, :4]                                # 2018-2021
y_tgt = y_log[:, 4]                                 # 2022

sectors = panel["gics_11"].values
N = len(panel)


def metrics(y_true_log, y_pred_log):
    y_true = 10 ** y_true_log; y_pred = 10 ** y_pred_log
    ape = np.abs(y_true - y_pred) / np.abs(y_true)
    return dict(
        n          = int(len(y_true_log)),
        mae_log    = float(mean_absolute_error(y_true_log, y_pred_log)),
        rmse_log   = float(mean_squared_error(y_true_log, y_pred_log) ** 0.5),
        r2_log     = float(r2_score(y_true_log, y_pred_log)),
        pearson_r  = float(np.corrcoef(y_true_log, y_pred_log)[0, 1]),
        mape       = float(ape.mean()),
        median_ape = float(np.median(ape)),
    )


results = []

# ── A1. Persistence ─────────────────────────────────────────────────────
pred = y_ctx[:, -1]  # 2021
m = metrics(y_tgt, pred)
results.append({"model": "persistence", **m})
print(f"  persistence        R²={m['r2_log']:+.3f}  MAE={m['mae_log']:.3f}  r={m['pearson_r']:.3f}")

# ── A2. 3-year moving average ───────────────────────────────────────────
pred = y_ctx[:, -3:].mean(axis=1)
m = metrics(y_tgt, pred)
results.append({"model": "mean_3yr", **m})
print(f"  mean_3yr           R²={m['r2_log']:+.3f}  MAE={m['mae_log']:.3f}  r={m['pearson_r']:.3f}")

# ── A3. Linear trend (OLS on log space) ─────────────────────────────────
t = np.array(CONTEXT_YEARS, dtype=float)
t_c = t - t.mean()
den = (t_c ** 2).sum()
slope = (y_ctx * t_c).sum(axis=1) / den
intercept = y_ctx.mean(axis=1) - slope * t.mean()
pred = slope * TARGET_YEAR + intercept
m = metrics(y_tgt, pred)
results.append({"model": "linear_trend", **m})
print(f"  linear_trend       R²={m['r2_log']:+.3f}  MAE={m['mae_log']:.3f}  r={m['pearson_r']:.3f}")

# ── A4. Sector median growth (log-space) ───────────────────────────────
annual_growth = np.diff(y_ctx, axis=1).mean(axis=1)  # avg Δlog / yr per company
growth_by_sector = pd.Series(annual_growth).groupby(sectors).median()
company_growth = pd.Series(sectors).map(growth_by_sector).fillna(0.0).values
pred = y_ctx[:, -1] + company_growth  # 2021 + sector avg annual Δlog
m = metrics(y_tgt, pred)
results.append({"model": "sector_growth", **m})
print(f"  sector_growth      R²={m['r2_log']:+.3f}  MAE={m['mae_log']:.3f}  r={m['pearson_r']:.3f}")

# ── A5. XGBoost on (lagged y + features) ───────────────────────────────
# Build a design matrix: lag1, lag2, lag3, lag4, one-hot sector + country
lag = pd.DataFrame(y_ctx, columns=["y_2018","y_2019","y_2020","y_2021"])
cat = pd.get_dummies(panel[["gics_11","country_iso2"]], drop_first=False, dtype=float)
X = pd.concat([lag, cat], axis=1)
y = y_tgt

# 80/20 split
rng = np.random.default_rng(SEED)
perm = rng.permutation(N)
n_tr = int(0.8 * N)
tr, te = perm[:n_tr], perm[n_tr:]

model = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                     random_state=SEED, n_jobs=8, verbosity=0)
model.fit(X.iloc[tr], y[tr])
y_pred_te = model.predict(X.iloc[te])
m = metrics(y[te], y_pred_te)
m["n_train"] = len(tr); m["n_test"] = len(te)
results.append({"model": "xgboost_lag+feat", **m})
print(f"  xgboost_lag+feat   R²={m['r2_log']:+.3f}  MAE={m['mae_log']:.3f}  r={m['pearson_r']:.3f}  (test={len(te):,})")

# Save intermediate checkpoint
pd.DataFrame(results).to_csv(OUT, index=False)
print(f"\n  (intermediate save → {OUT})")


# ── B. Chronos zero-shot ────────────────────────────────────────────────
print("\nLoading Chronos-Bolt-Small ...")
from chronos import BaseChronosPipeline
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-small",
    device_map=device,
    torch_dtype=torch.float32,
)
print(f"  device: {device}")

# Chronos expects a list of 1D tensors (univariate series in original scale).
# Feed linear-scale emissions; we'll log10 the predictions for metric comparison.
ctx_linear = panel[CONTEXT_YEARS].values  # (N, 4)
BATCH = 256
preds_linear = []
import time
t0 = time.time()
for i in range(0, N, BATCH):
    batch = [torch.tensor(row, dtype=torch.float32) for row in ctx_linear[i:i+BATCH]]
    q, mean_pred = pipe.predict_quantiles(
        inputs=batch,
        prediction_length=1,
        quantile_levels=[0.5],
    )
    # mean_pred shape: (batch, horizon)
    preds_linear.append(mean_pred[:, 0].cpu().numpy())  # median forecast
    if (i // BATCH) % 5 == 0:
        rate = (i + BATCH) / (time.time() - t0 + 1e-6)
        print(f"    [{i+BATCH}/{N}]  {rate:.1f} series/s", flush=True)

preds_linear = np.concatenate(preds_linear)[:N]
pred_chronos = np.log10(np.clip(preds_linear, 1.0, None))
m = metrics(y_tgt, pred_chronos)
results.append({"model": "chronos_bolt_small", **m})
print(f"\n  chronos_bolt_small R²={m['r2_log']:+.3f}  MAE={m['mae_log']:.3f}  r={m['pearson_r']:.3f}")


# ── Wrap up ─────────────────────────────────────────────────────────────
OUT.parent.mkdir(parents=True, exist_ok=True)
res = pd.DataFrame(results)
res.to_csv(OUT, index=False)
print(f"\n{len(res)} results → {OUT}")
print()
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
cols = ["model","n","mae_log","rmse_log","r2_log","pearson_r","mape","median_ape"]
print(res[[c for c in cols if c in res.columns]].to_string(
    index=False, float_format=lambda x: f"{x:.3f}"))
