"""
T1-E extensions: harder variants where persistence is no longer a ceiling.

Three variants, all run on the 1,930 companies with unbroken 2018-2022 panel
of cleaned Scope 1+2 values.

  B. Growth-rate forecast
     target = Δlog = log y_2022 - log y_2021
     persistence-equivalent baseline = "predict 0 change"  →  R²=0 by design.
     Informed predictors must beat this by reading the earlier growth pattern.

  C. Turning-point subset
     Same task as main T1-E, restricted to companies whose 2020→2021
     absolute log-change exceeds 0.08 ( ~ ±20% YoY).  This isolates the
     ~15-20% of firms actually undergoing transitions (decarbonization,
     M&A, restructuring) — where "next year ≈ this year" is weakest.

  A. 2-step horizon
     Context 2018-2020 (3 years), predict 2022. Skips 2021. Tests whether
     models remain useful over a horizon persistence starts to degrade on.

Output: results/t1e_ext_results.csv  +  results/t1e_ext_run.log
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import torch
from chronos import BaseChronosPipeline

BASELINE = Path("data/company-level/nzdpu_enriched/factor_baseline_v2.csv")
OUT      = Path("results/t1e_ext_results.csv")
SEED     = 42
CONTEXT_YEARS = [2018, 2019, 2020, 2021]
TARGET_YEAR   = 2022

# ── Shared: build the balanced panel ────────────────────────────────────
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
have_all = wide[all_years].notna().all(axis=1)
panel = wide.loc[have_all, all_years].reset_index()
panel["gics_11"] = panel["gics_11"].fillna("Unknown")
print(f"  full panel: {len(panel):,}")

y_log = np.log10(panel[all_years].values)   # (N, 5): 2018 2019 2020 2021 2022


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    out = dict(
        n         = int(len(y_true)),
        mae       = float(mean_absolute_error(y_true, y_pred)),
        rmse      = float(mean_squared_error(y_true, y_pred) ** 0.5),
        r2        = float(r2_score(y_true, y_pred)),
    )
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        out["pearson_r"] = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        out["pearson_r"] = 0.0
    return out


print("\nLoading Chronos-Bolt-Small (shared across variants)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-bolt-small", device_map=device, torch_dtype=torch.float32)


def chronos_forecast(ctx_linear, horizon=1):
    """Forecast 1 or more steps ahead. Returns log10 of median forecast at
    the final horizon step."""
    BATCH = 256
    N = len(ctx_linear)
    preds = []
    for i in range(0, N, BATCH):
        batch = [torch.tensor(row, dtype=torch.float32)
                 for row in ctx_linear[i:i+BATCH]]
        _, mean_pred = pipe.predict_quantiles(
            inputs=batch, prediction_length=horizon, quantile_levels=[0.5])
        preds.append(mean_pred[:, -1].cpu().numpy())
    return np.concatenate(preds)[:N]


# ═══════════════════════════════════════════════════════════════════════
# Variant B: growth-rate forecast  (target = Δlog 2021→2022)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Variant B: growth-rate forecast (target = Δlog 2021→2022)")
print("="*60)
results = []

y_true_delta = y_log[:, 4] - y_log[:, 3]  # log y_2022 - log y_2021
print(f"  target stats: mean={y_true_delta.mean():.3f}  "
      f"std={y_true_delta.std():.3f}  |Δ|>0.1: {(np.abs(y_true_delta)>0.1).mean():.1%}")

# B0. zero-change baseline
m = metrics(y_true_delta, np.zeros_like(y_true_delta))
results.append({"variant":"B_growth_rate","model":"zero_change",**m})
print(f"  zero_change          R²={m['r2']:+.3f}  MAE={m['mae']:.4f}  r={m['pearson_r']:.3f}")

# B1. average-past-growth baseline: Δ̂_2022 = mean(Δlog 2018→2019, 2019→2020, 2020→2021)
past_growth = np.diff(y_log[:, :4], axis=1).mean(axis=1)
m = metrics(y_true_delta, past_growth)
results.append({"variant":"B_growth_rate","model":"avg_past_growth",**m})
print(f"  avg_past_growth      R²={m['r2']:+.3f}  MAE={m['mae']:.4f}  r={m['pearson_r']:.3f}")

# B2. last-year growth: Δ̂_2022 = Δlog 2020→2021
last_growth = y_log[:, 3] - y_log[:, 2]
m = metrics(y_true_delta, last_growth)
results.append({"variant":"B_growth_rate","model":"last_growth",**m})
print(f"  last_growth          R²={m['r2']:+.3f}  MAE={m['mae']:.4f}  r={m['pearson_r']:.3f}")

# B3. sector median growth: Δ̂ = sector_median(Δlog)
sectors = panel["gics_11"].values
sect_growth = pd.Series(past_growth).groupby(sectors).median()
pred = pd.Series(sectors).map(sect_growth).fillna(0.0).values
m = metrics(y_true_delta, pred)
results.append({"variant":"B_growth_rate","model":"sector_median_growth",**m})
print(f"  sector_median_growth R²={m['r2']:+.3f}  MAE={m['mae']:.4f}  r={m['pearson_r']:.3f}")

# B4. XGBoost on lagged deltas + one-hot sector/country
lag_delta = np.diff(y_log[:, :4], axis=1)  # (N, 3) deltas 2018→19, 19→20, 20→21
lag_df = pd.DataFrame(lag_delta, columns=["d18_19","d19_20","d20_21"])
lag_df["y_2021"] = y_log[:, 3]
cat = pd.get_dummies(panel[["gics_11","country_iso2"]], drop_first=False, dtype=float)
X = pd.concat([lag_df, cat], axis=1)
rng = np.random.default_rng(SEED); perm = rng.permutation(len(panel))
n_tr = int(0.8*len(panel)); tr, te = perm[:n_tr], perm[n_tr:]
model = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                     random_state=SEED, n_jobs=8, verbosity=0)
model.fit(X.iloc[tr], y_true_delta[tr])
pred_te = model.predict(X.iloc[te])
m = metrics(y_true_delta[te], pred_te)
m["n_train"] = len(tr)
results.append({"variant":"B_growth_rate","model":"xgboost_lag_delta",**m})
print(f"  xgboost_lag_delta    R²={m['r2']:+.3f}  MAE={m['mae']:.4f}  r={m['pearson_r']:.3f}  (test={len(te):,})")

# B5. Chronos: forecast 2022 level, derive Δ̂ = log(ŷ_2022) - log(y_2021)
ctx = panel[CONTEXT_YEARS].values
pred_level = np.log10(np.clip(chronos_forecast(ctx, horizon=1), 1.0, None))
pred_delta = pred_level - y_log[:, 3]
m = metrics(y_true_delta, pred_delta)
results.append({"variant":"B_growth_rate","model":"chronos_derived",**m})
print(f"  chronos_derived      R²={m['r2']:+.3f}  MAE={m['mae']:.4f}  r={m['pearson_r']:.3f}")


# ═══════════════════════════════════════════════════════════════════════
# Variant C: turning-point subset  (|Δlog 2020→2021| > 0.08)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Variant C: turning-point subset (|Δ_2020→21| > 0.08 ≈ ±20% YoY)")
print("="*60)
mask_tp = np.abs(y_log[:, 3] - y_log[:, 2]) > 0.08
print(f"  turning-point rows: {mask_tp.sum():,} / {len(panel):,} ({mask_tp.mean():.1%})")

y_ctx = y_log[mask_tp, :4]
y_tgt = y_log[mask_tp, 4]
panel_tp = panel.loc[mask_tp]
sectors_tp = sectors[mask_tp]
ctx_linear_tp = ctx[mask_tp]

# C1. persistence
pred = y_ctx[:, -1]
m = metrics(y_tgt, pred)
results.append({"variant":"C_turning_point","model":"persistence",**m})
print(f"  persistence          R²={m['r2']:+.3f}  MAE={m['mae']:.4f}  r={m['pearson_r']:.3f}")

# C2. mean_3yr
pred = y_ctx[:, -3:].mean(axis=1)
m = metrics(y_tgt, pred)
results.append({"variant":"C_turning_point","model":"mean_3yr",**m})
print(f"  mean_3yr             R²={m['r2']:+.3f}  MAE={m['mae']:.4f}  r={m['pearson_r']:.3f}")

# C3. linear_trend
t = np.array(CONTEXT_YEARS, dtype=float); t_c = t - t.mean(); den = (t_c**2).sum()
slope = (y_ctx * t_c).sum(axis=1) / den
intercept = y_ctx.mean(axis=1) - slope * t.mean()
pred = slope * TARGET_YEAR + intercept
m = metrics(y_tgt, pred)
results.append({"variant":"C_turning_point","model":"linear_trend",**m})
print(f"  linear_trend         R²={m['r2']:+.3f}  MAE={m['mae']:.4f}  r={m['pearson_r']:.3f}")

# C4. sector growth
annual = np.diff(y_ctx, axis=1).mean(axis=1)
sg = pd.Series(annual).groupby(sectors_tp).median()
pred = y_ctx[:, -1] + pd.Series(sectors_tp).map(sg).fillna(0.0).values
m = metrics(y_tgt, pred)
results.append({"variant":"C_turning_point","model":"sector_growth",**m})
print(f"  sector_growth        R²={m['r2']:+.3f}  MAE={m['mae']:.4f}  r={m['pearson_r']:.3f}")

# C5. XGBoost lag+feat
lag_tp = pd.DataFrame(y_ctx, columns=["y_2018","y_2019","y_2020","y_2021"])
cat_tp = pd.get_dummies(panel_tp[["gics_11","country_iso2"]], drop_first=False, dtype=float).reset_index(drop=True)
X_tp = pd.concat([lag_tp, cat_tp], axis=1)
N_tp = len(panel_tp); perm_tp = np.random.default_rng(SEED).permutation(N_tp)
n_tr_tp = int(0.8*N_tp); tr_tp, te_tp = perm_tp[:n_tr_tp], perm_tp[n_tr_tp:]
model = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                     random_state=SEED, n_jobs=8, verbosity=0)
model.fit(X_tp.iloc[tr_tp], y_tgt[tr_tp])
pred_te = model.predict(X_tp.iloc[te_tp])
m = metrics(y_tgt[te_tp], pred_te)
m["n_train"] = len(tr_tp)
results.append({"variant":"C_turning_point","model":"xgboost_lag+feat",**m})
print(f"  xgboost_lag+feat     R²={m['r2']:+.3f}  MAE={m['mae']:.4f}  r={m['pearson_r']:.3f}  (test={len(te_tp):,})")

# C6. Chronos
pred_tp = np.log10(np.clip(chronos_forecast(ctx_linear_tp, horizon=1), 1.0, None))
m = metrics(y_tgt, pred_tp)
results.append({"variant":"C_turning_point","model":"chronos_bolt_small",**m})
print(f"  chronos_bolt_small   R²={m['r2']:+.3f}  MAE={m['mae']:.4f}  r={m['pearson_r']:.3f}")


# ═══════════════════════════════════════════════════════════════════════
# Variant A: 2-step horizon  (context 2018-2020, target 2022, skip 2021)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Variant A: 2-step horizon (ctx 2018-2020 → target 2022, skip 2021)")
print("="*60)
ctx2 = y_log[:, :3]                  # 2018, 2019, 2020
tgt2 = y_log[:, 4]                   # 2022 (skip 2021)
ctx2_linear = panel[[2018,2019,2020]].values

# A1. persistence (last context value = 2020)
pred = ctx2[:, -1]
m = metrics(tgt2, pred)
results.append({"variant":"A_h2","model":"persistence_2020",**m})
print(f"  persistence_2020     R²={m['r2']:+.3f}  MAE={m['mae']:.4f}  r={m['pearson_r']:.3f}")

# A2. linear trend extrapolated 2 years
t2 = np.array([2018,2019,2020], dtype=float); t2c = t2 - t2.mean(); den2 = (t2c**2).sum()
slope2 = (ctx2 * t2c).sum(axis=1) / den2
int2 = ctx2.mean(axis=1) - slope2 * t2.mean()
pred = slope2 * TARGET_YEAR + int2
m = metrics(tgt2, pred)
results.append({"variant":"A_h2","model":"linear_trend_h2",**m})
print(f"  linear_trend_h2      R²={m['r2']:+.3f}  MAE={m['mae']:.4f}  r={m['pearson_r']:.3f}")

# A3. sector growth (2 years forward)
annual2 = np.diff(ctx2, axis=1).mean(axis=1)
sg2 = pd.Series(annual2).groupby(sectors).median()
pred = ctx2[:, -1] + 2 * pd.Series(sectors).map(sg2).fillna(0.0).values
m = metrics(tgt2, pred)
results.append({"variant":"A_h2","model":"sector_growth_h2",**m})
print(f"  sector_growth_h2     R²={m['r2']:+.3f}  MAE={m['mae']:.4f}  r={m['pearson_r']:.3f}")

# A4. Chronos h=2
pred_h2 = np.log10(np.clip(chronos_forecast(ctx2_linear, horizon=2), 1.0, None))
m = metrics(tgt2, pred_h2)
results.append({"variant":"A_h2","model":"chronos_bolt_small_h2",**m})
print(f"  chronos_bolt_small_h2 R²={m['r2']:+.3f}  MAE={m['mae']:.4f}  r={m['pearson_r']:.3f}")


# ═══════════════════════════════════════════════════════════════════════
OUT.parent.mkdir(parents=True, exist_ok=True)
res = pd.DataFrame(results)
res.to_csv(OUT, index=False)
print(f"\n{len(res)} results → {OUT}")
print()
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
cols_out = ["variant","model","n","mae","rmse","r2","pearson_r"]
print(res[[c for c in cols_out if c in res.columns]]
      .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
