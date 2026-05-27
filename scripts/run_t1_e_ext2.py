"""T1-E extended experiments (part 2):
  1. ARIMA baseline (forecast 2022 from 2018-2021 per company)
  2. T1-E on Scope 3 (repeat level forecast on S3 target)
  3. Exponential smoothing (Holt) as an additional classical baseline

Outputs → results/t1e_ext2_results.csv
"""
from __future__ import annotations
import warnings, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

BASELINE = Path("data/company-level/nzdpu_enriched/factor_baseline_v2.csv")
NZDPU    = Path("data/company-level/nzdpu/nzdpu_emissions.csv")
OUT      = Path("results/t1e_ext2_results.csv")
CONTEXT_YEARS = [2018, 2019, 2020, 2021]
TARGET_YEAR   = 2022


def metrics(y_true_log, y_pred_log):
    y_true = 10 ** y_true_log; y_pred = 10 ** y_pred_log
    ape = np.abs(y_true - y_pred) / np.abs(y_true)
    return dict(
        n=int(len(y_true_log)),
        mae_log=float(mean_absolute_error(y_true_log, y_pred_log)),
        rmse_log=float(mean_squared_error(y_true_log, y_pred_log) ** 0.5),
        r2_log=float(r2_score(y_true_log, y_pred_log)),
        pearson_r=float(np.corrcoef(y_true_log, y_pred_log)[0, 1]),
        mape=float(ape.mean()),
        median_ape=float(np.median(ape)),
    )


def build_panel(target: str = "s12") -> pd.DataFrame:
    if target == "s12":
        bl = pd.read_csv(BASELINE)
        bl = bl[bl["scope12_actual_tco2e"].notna()
                & (bl["scope12_actual_tco2e"] >= 10)
                & (bl["scope12_actual_tco2e"] <= 5e8)]
        val_col = "scope12_actual_tco2e"
        df = bl[["nz_id","reporting_year","gics_11","country_iso2",val_col]].copy()
    elif target == "s3":
        nz = pd.read_csv(NZDPU, low_memory=False)
        DASH = "\u2014"
        nz["s3"] = pd.to_numeric(nz["total_s3_emissions_ghg"].replace(DASH, np.nan),
                                  errors="coerce")
        nz = nz[["nz_id","reporting_year","total_s3_emissions_ghg","s3"]].rename(
            columns={"s3":"scope3_actual_tco2e"})
        nz = nz.dropna(subset=["scope3_actual_tco2e"])
        nz = nz[(nz["scope3_actual_tco2e"] >= 100)
                 & (nz["scope3_actual_tco2e"] <= 2e9)]
        bl_full = pd.read_csv(BASELINE)[["nz_id","reporting_year","gics_11","country_iso2"]]
        # Use per-company metadata (drop duplicate rows collapsing to single record per nz_id)
        meta = bl_full.drop_duplicates("nz_id")[["nz_id","gics_11","country_iso2"]]
        df = nz.merge(meta, on="nz_id", how="left")
        df = df.drop(columns=["total_s3_emissions_ghg"])
        df = df.rename(columns={"scope3_actual_tco2e":"y_raw"})
        val_col = "y_raw"
    else:
        raise ValueError(target)

    all_years = CONTEXT_YEARS + [TARGET_YEAR]
    wide = df.pivot_table(index=["nz_id","gics_11","country_iso2"],
                          columns="reporting_year", values=val_col, aggfunc="first")
    have_all = wide[all_years].notna().all(axis=1)
    panel = wide.loc[have_all, all_years].reset_index()
    panel["gics_11"] = panel["gics_11"].fillna("Unknown")
    return panel


def run_panel(panel: pd.DataFrame, tag: str) -> list[dict]:
    """Run persistence / linear / sector / ARIMA / Holt on a panel."""
    all_years = CONTEXT_YEARS + [TARGET_YEAR]
    y_log = np.log10(panel[all_years].values)
    y_ctx = y_log[:, :4]; y_tgt = y_log[:, 4]
    sectors = panel["gics_11"].values
    N = len(panel)
    print(f"[{tag}] N={N}")
    results = []

    # persistence
    results.append({"target":tag,"model":"persistence", **metrics(y_tgt, y_ctx[:, -1])})
    # mean_3yr
    results.append({"target":tag,"model":"mean_3yr",    **metrics(y_tgt, y_ctx[:, -3:].mean(axis=1))})
    # linear trend
    t = np.array(CONTEXT_YEARS, dtype=float); t_c = t - t.mean(); den = (t_c ** 2).sum()
    slope = (y_ctx * t_c).sum(axis=1) / den
    intercept = y_ctx.mean(axis=1) - slope * t.mean()
    pred = slope * TARGET_YEAR + intercept
    results.append({"target":tag,"model":"linear_trend", **metrics(y_tgt, pred)})
    # sector growth
    g = np.diff(y_ctx, axis=1).mean(axis=1)
    gbs = pd.Series(g).groupby(sectors).median()
    cg = pd.Series(sectors).map(gbs).fillna(0.0).values
    results.append({"target":tag,"model":"sector_growth", **metrics(y_tgt, y_ctx[:, -1] + cg)})

    # ARIMA (1,1,0) per company — simple specification, 4 obs is minimal
    from statsmodels.tsa.arima.model import ARIMA
    preds = np.empty(N)
    fail = 0
    t0 = time.time()
    first_err = None
    for i in range(N):
        try:
            model = ARIMA(np.asarray(y_ctx[i], dtype=float), order=(1, 1, 0),
                          enforce_stationarity=False, enforce_invertibility=False)
            fit = model.fit()
            fc = fit.forecast(steps=1)
            preds[i] = float(np.asarray(fc)[0])
        except Exception as e:
            if first_err is None: first_err = f"{type(e).__name__}: {str(e)[:120]}"
            preds[i] = y_ctx[i, -1]  # fallback = persistence
            fail += 1
    print(f"[{tag}] ARIMA fail={fail}/{N} in {time.time()-t0:.1f}s"
          + (f"  first_err={first_err}" if first_err else ""))
    results.append({"target":tag,"model":"arima_1_1_0", **metrics(y_tgt, preds),
                    "fail_count": fail})

    # Holt (exponential smoothing with trend)
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    preds = np.empty(N); fail = 0; t0 = time.time()
    first_err = None
    for i in range(N):
        try:
            fit = ExponentialSmoothing(np.asarray(y_ctx[i], dtype=float), trend="add",
                                        initialization_method="estimated").fit()
            preds[i] = float(np.asarray(fit.forecast(1))[0])
        except Exception as e:
            if first_err is None: first_err = f"{type(e).__name__}: {str(e)[:120]}"
            preds[i] = y_ctx[i, -1]; fail += 1
    print(f"[{tag}] Holt fail={fail}/{N} in {time.time()-t0:.1f}s"
          + (f"  first_err={first_err}" if first_err else ""))
    results.append({"target":tag,"model":"holt_linear", **metrics(y_tgt, preds),
                    "fail_count": fail})

    return results


if __name__ == "__main__":
    all_results = []
    # Scope 1+2
    panel12 = build_panel("s12")
    all_results += run_panel(panel12, "s12")
    # Scope 3
    try:
        panel3 = build_panel("s3")
        if len(panel3) >= 50:
            all_results += run_panel(panel3, "s3")
        else:
            print(f"[s3] too few rows ({len(panel3)}) for forecasting")
    except Exception as e:
        print(f"[s3] skipped: {e}")

    df = pd.DataFrame(all_results)
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\n→ {OUT}")
    cols = ["target","model","n","mae_log","r2_log","pearson_r","median_ape"]
    print(df[[c for c in cols if c in df.columns]].to_string(index=False,
          float_format=lambda x: f"{x:.3f}"))
