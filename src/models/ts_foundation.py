"""
Zero-shot time-series foundation model baselines for Task E forecasting.

Each model takes per-building emission history (extracted from lag features in X)
and predicts the next year's emissions. No training — inference only.

All three classes follow the same fit/predict API as other Task E models:
  - fit(X, y): stores lag column indices, loads the pretrained model (once)
  - predict(X): extracts [lag_n, ..., lag1] per row, runs batch inference

Supported models:
  - ChronosBaseline   (Amazon, T5-based)
  - TimesFMBaseline   (Google, decoder-only 200M)
  - MoiraiBaseline    (Salesforce, any-variate)
"""

import os
import warnings
from typing import Optional

import numpy as np
import torch

# Lazy imports — only loaded when the specific baseline is instantiated.


def _extract_histories(X, lag_indices):
    """Extract per-row histories from lag columns in X.

    lag_indices should be ordered [lag_n_idx, ..., lag2_idx, lag1_idx]
    (oldest → newest).  Returns list of 1-D numpy arrays, skipping leading NaNs.
    """
    histories = []
    for i in range(X.shape[0]):
        row = X[i, lag_indices]
        # Drop leading NaNs (short history) but keep trailing
        first_valid = 0
        for j in range(len(row)):
            if np.isnan(row[j]):
                first_valid = j + 1
            else:
                break
        h = row[first_valid:]
        if len(h) == 0 or np.all(np.isnan(h)):
            h = np.array([0.0])
        histories.append(h.astype(np.float32))
    return histories


class ChronosBaseline:
    """Amazon Chronos (T5-small by default). Zero-shot probabilistic forecaster."""

    def __init__(self, model_id="amazon/chronos-t5-small", device="cuda",
                 batch_size=512, num_samples=20):
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.pipeline = None
        self.lag_indices = None  # oldest → newest

    def fit(self, X, y, **kwargs):
        if self.pipeline is None:
            from chronos import ChronosPipeline
            self.pipeline = ChronosPipeline.from_pretrained(
                self.model_id,
                device_map=self.device,
                torch_dtype=torch.float32,
            )
        return self

    def predict(self, X):
        histories = _extract_histories(X, self.lag_indices)
        all_preds = np.zeros(len(histories))
        for start in range(0, len(histories), self.batch_size):
            batch = histories[start:start + self.batch_size]
            contexts = [torch.tensor(h) for h in batch]
            with torch.no_grad():
                forecasts = self.pipeline.predict(
                    contexts, prediction_length=1,
                    num_samples=self.num_samples,
                )
            medians = np.median(forecasts.numpy(), axis=1).squeeze(-1)
            all_preds[start:start + len(batch)] = medians
        return np.clip(all_preds, 0, None)


class TimesFMBaseline:
    """Google TimesFM (200M params). Zero-shot point forecaster."""

    def __init__(self, repo_id="google/timesfm-1.0-200m-pytorch", device="cuda",
                 batch_size=1024):
        self.repo_id = repo_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.model = None
        self.lag_indices = None

    def fit(self, X, y, **kwargs):
        if self.model is None:
            import timesfm
            self.model = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    backend="gpu" if self.device == "cuda" else "cpu",
                    per_core_batch_size=self.batch_size,
                    horizon_len=1,
                    input_patch_len=32,
                    output_patch_len=128,
                    num_layers=20,
                    model_dims=1280,
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id=self.repo_id,
                ),
            )
        return self

    def predict(self, X):
        histories = _extract_histories(X, self.lag_indices)
        all_preds = np.zeros(len(histories))
        for start in range(0, len(histories), self.batch_size):
            batch = histories[start:start + self.batch_size]
            point_forecasts, _ = self.model.forecast(
                batch,
                freq=[0] * len(batch),  # 0 = unknown / low-frequency
            )
            preds = np.array([f[0] for f in point_forecasts])
            all_preds[start:start + len(batch)] = preds
        return np.clip(all_preds, 0, None)


class MoiraiBaseline:
    """Salesforce Moirai (Small). Zero-shot any-variate forecaster."""

    def __init__(self, model_id="Salesforce/moirai-1.1-R-small", device="cuda"):
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.module = None
        self.lag_indices = None

    def fit(self, X, y, **kwargs):
        if self.module is None:
            from uni2ts.model.moirai import MoiraiModule
            self.module = MoiraiModule.from_pretrained(self.model_id)
        return self

    def predict(self, X):
        from einops import rearrange
        from uni2ts.model.moirai import MoiraiForecast
        import torch

        histories = _extract_histories(X, self.lag_indices)

        # Moirai expects a GluonTS-style dataset or direct tensor input.
        # For simplicity, process in mini-batches of same-length series.
        preds = np.zeros(len(histories))

        # Group by history length for batched inference
        len_to_idxs = {}
        for i, h in enumerate(histories):
            l = len(h)
            len_to_idxs.setdefault(l, []).append(i)

        for hist_len, idxs in len_to_idxs.items():
            batch = np.stack([histories[i] for i in idxs])  # (B, hist_len)
            # Moirai uses gluonts ListDataset internally; use the forecast API
            from gluonts.dataset.pandas import PandasDataset
            import pandas as pd

            entries = []
            for row in batch:
                idx = pd.period_range(start="2000", periods=len(row), freq="Y")
                entries.append(pd.Series(row, index=idx))
            ds = PandasDataset(entries, target="target", freq="Y")

            predictor = MoiraiForecast(
                module=self.module,
                prediction_length=1,
                context_length=hist_len,
                patch_size="auto",
                num_samples=20,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
            predictor = predictor.create_predictor(batch_size=min(512, len(idxs)))

            for j, forecast in enumerate(predictor.predict(ds)):
                preds[idxs[j]] = np.median(forecast.samples)

        return np.clip(preds, 0, None)
