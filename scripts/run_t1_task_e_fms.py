"""
Add two more time-series foundation models to T1-E level forecast:
TimesFM (Google) and Moirai (Salesforce). Mirrors the T2 setup which
benchmarks chronos + timesfm + moirai.

Appends rows to results/t1e_results.csv alongside the existing
persistence / mean_3yr / linear_trend / sector_growth / xgboost / chronos
entries from run_t1_task_e.py.

Run in ghg env. GPU used if available.
"""

import numpy as np
import pandas as pd
import time
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch

BASELINE = Path("data/company-level/nzdpu_enriched/factor_baseline_v2.csv")
OUT      = Path("results/t1e_results.csv")

CONTEXT_YEARS = [2018, 2019, 2020, 2021]
TARGET_YEAR   = 2022
BATCH = 256

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── Rebuild panel (same as run_t1_task_e.py) ────────────────────────────
print("Loading panel...")
df = pd.read_csv(BASELINE)
df = df[df["scope12_actual_tco2e"].notna() &
        (df["scope12_actual_tco2e"] >= 10) &
        (df["scope12_actual_tco2e"] <= 5e8)]
all_years = CONTEXT_YEARS + [TARGET_YEAR]
wide = df.pivot_table(index=["nz_id","gics_11","country_iso2"],
                      columns="reporting_year",
                      values="scope12_actual_tco2e",
                      aggfunc="first")
have_all = wide[all_years].notna().all(axis=1)
panel = wide.loc[have_all, all_years].reset_index()
panel["gics_11"] = panel["gics_11"].fillna("Unknown")
print(f"  companies with full {min(all_years)}-{max(all_years)} panel: {len(panel):,}")

y_log = np.log10(panel[all_years].values)
y_tgt = y_log[:, 4]
ctx_linear = panel[CONTEXT_YEARS].values  # (N, 4)
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

new_rows = []

# ── TimesFM (Google) ────────────────────────────────────────────────────
print("\n── TimesFM 2.0 ───────────────────────────────────────────────")
import timesfm
t0 = time.time()
tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend="gpu" if device == "cuda" else "cpu",
        per_core_batch_size=32,
        horizon_len=1,
        context_len=32,  # must be multiple of input_patch_len=32
        num_layers=50,
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
)
print(f"  loaded in {time.time()-t0:.1f}s")

preds_linear = []
t0 = time.time()
for i in range(0, N, BATCH):
    batch = [ctx_linear[j].astype(np.float32) for j in range(i, min(i+BATCH, N))]
    fcst, _ = tfm.forecast(batch, freq=[0]*len(batch))  # 0 = yearly
    preds_linear.append(fcst[:, 0])
    if (i // BATCH) % 2 == 0:
        rate = (i + BATCH) / (time.time() - t0 + 1e-6)
        print(f"    [{min(i+BATCH,N)}/{N}]  {rate:.1f} series/s", flush=True)
preds_linear = np.concatenate(preds_linear)[:N]
pred_log = np.log10(np.clip(preds_linear, 1.0, None))
m = metrics(y_tgt, pred_log)
print(f"  timesfm_2p0_500m   R²={m['r2_log']:+.3f}  MAE={m['mae_log']:.3f}  r={m['pearson_r']:.3f}")
new_rows.append({"model": "timesfm_2p0_500m", **m})

# free GPU
del tfm
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ── Moirai (Salesforce, uni2ts) ────────────────────────────────────────
print("\n── Moirai-1.1-R small ────────────────────────────────────────")
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

t0 = time.time()
module = MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-small")
moirai = MoiraiForecast(
    module=module,
    prediction_length=1,
    context_length=4,
    patch_size=1,   # 1 per yearly observation
    num_samples=100,
    target_dim=1,
    feat_dynamic_real_dim=0,
    past_feat_dynamic_real_dim=0,
).to(device).eval()
print(f"  loaded in {time.time()-t0:.1f}s")

preds_linear = []
t0 = time.time()
with torch.no_grad():
    for i in range(0, N, BATCH):
        bsz = min(BATCH, N - i)
        past_target = torch.tensor(ctx_linear[i:i+bsz], dtype=torch.float32,
                                   device=device).unsqueeze(-1)    # (B, 4, 1)
        past_observed = torch.ones_like(past_target, dtype=torch.bool)
        past_is_pad   = torch.zeros((bsz, 4), dtype=torch.bool, device=device)
        out = moirai(
            past_target=past_target,
            past_observed_target=past_observed,
            past_is_pad=past_is_pad,
        )  # samples shape may be (B, num_samples, pred_len[, target_dim])
        if out.dim() == 4:
            samples = out[..., 0]
        else:
            samples = out
        median = samples.median(dim=1).values[:, 0].cpu().numpy()
        preds_linear.append(median)
        if (i // BATCH) % 2 == 0:
            rate = (i + BATCH) / (time.time() - t0 + 1e-6)
            print(f"    [{min(i+BATCH,N)}/{N}]  {rate:.1f} series/s", flush=True)
preds_linear = np.concatenate(preds_linear)[:N]
pred_log = np.log10(np.clip(preds_linear, 1.0, None))
m = metrics(y_tgt, pred_log)
print(f"  moirai_1p1_small   R²={m['r2_log']:+.3f}  MAE={m['mae_log']:.3f}  r={m['pearson_r']:.3f}")
new_rows.append({"model": "moirai_1p1_small", **m})


# ── Append to existing results ─────────────────────────────────────────
res = pd.read_csv(OUT)
res = pd.concat([res, pd.DataFrame(new_rows)], ignore_index=True)
res.to_csv(OUT, index=False)
print(f"\n{len(res)} rows → {OUT}\n")
cols = ["model","n","mae_log","rmse_log","r2_log","pearson_r","mape","median_ape"]
pd.set_option("display.width", 200)
print(res[[c for c in cols if c in res.columns]].to_string(
    index=False, float_format=lambda x: f"{x:.3f}"))
