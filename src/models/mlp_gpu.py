"""
GPU-accelerated MLP baseline for T2 building-level regression.

Drop-in replacement for the sklearn-based MLPBaseline: 3-layer MLP with ReLU +
dropout, log1p target transform, in-class feature standardization, and early
stopping on a validation fold. Runs on CUDA when available, otherwise CPU.

Matches the fit/predict API used by the tree baselines (including the optional
`X_val`, `y_val` kwargs for external early stopping).
"""

from typing import Any, Optional

import numpy as np

import torch
import torch.nn as nn


class TorchMLPBaseline:
    DEFAULT_PARAMS = {
        "hidden_layers": (256, 128, 64),
        "dropout": 0.2,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 4096,
        "max_epochs": 200,
        "patience": 20,
        "val_fraction": 0.1,
        "seed": 42,
    }

    def __init__(self, log_target: bool = True, device: Optional[str] = None, **kwargs: Any):
        self.log_target = log_target
        self.params = {**self.DEFAULT_PARAMS, **kwargs}
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self._model: Optional[nn.Module] = None
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._is_fitted = False

    def _build_model(self, in_dim: int) -> nn.Module:
        layers = []
        prev = in_dim
        for h in self.params["hidden_layers"]:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(self.params["dropout"])]
            prev = h
        layers += [nn.Linear(prev, 1)]
        return nn.Sequential(*layers)

    def _clean(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        if y is None:
            mask = ~np.isnan(X).any(axis=1)
            return X[mask], None, mask
        y = np.asarray(y, dtype=np.float64)
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        return X[mask], y[mask], mask

    def fit(self, X, y, X_val=None, y_val=None) -> "TorchMLPBaseline":
        torch.manual_seed(self.params["seed"])
        np.random.seed(self.params["seed"])

        X, y, _ = self._clean(X, y)
        if len(X) < 2:
            raise ValueError("TorchMLPBaseline.fit received <2 valid rows")

        # Carve a validation fold if caller didn't pass one
        if X_val is None or y_val is None or len(X_val) == 0:
            n = len(X)
            n_val = max(1, int(round(n * self.params["val_fraction"])))
            rng = np.random.default_rng(self.params["seed"])
            idx = rng.permutation(n)
            val_idx, tr_idx = idx[:n_val], idx[n_val:]
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_vl, y_vl = X[val_idx], y[val_idx]
        else:
            X_vl, y_vl, _ = self._clean(X_val, y_val)
            X_tr, y_tr = X, y

        # Standardize using training fold only
        self._mean = X_tr.mean(axis=0)
        self._std = X_tr.std(axis=0)
        self._std[self._std == 0] = 1.0
        X_tr = (X_tr - self._mean) / self._std
        X_vl = (X_vl - self._mean) / self._std

        if self.log_target:
            y_tr_t = np.log1p(np.maximum(y_tr, 0))
            y_vl_t = np.log1p(np.maximum(y_vl, 0))
        else:
            y_tr_t, y_vl_t = y_tr, y_vl

        # To tensors
        dev = self.device
        X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=dev)
        y_tr_t = torch.tensor(y_tr_t, dtype=torch.float32, device=dev).unsqueeze(1)
        X_vl_t = torch.tensor(X_vl, dtype=torch.float32, device=dev)
        y_vl_t = torch.tensor(y_vl_t, dtype=torch.float32, device=dev).unsqueeze(1)

        self._model = self._build_model(X_tr.shape[1]).to(dev)
        opt = torch.optim.Adam(
            self._model.parameters(),
            lr=self.params["lr"],
            weight_decay=self.params["weight_decay"],
        )
        loss_fn = nn.MSELoss()

        best_val = float("inf")
        best_state = None
        bad_epochs = 0
        bs = self.params["batch_size"]

        for epoch in range(self.params["max_epochs"]):
            self._model.train()
            perm = torch.randperm(X_tr_t.size(0), device=dev)
            for i in range(0, perm.size(0), bs):
                sel = perm[i:i + bs]
                xb = X_tr_t[sel]
                yb = y_tr_t[sel]
                opt.zero_grad()
                pred = self._model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

            # Validation
            self._model.eval()
            with torch.no_grad():
                val_pred = self._model(X_vl_t)
                val_loss = loss_fn(val_pred, y_vl_t).item()

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in self._model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.params["patience"]:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)

        self._is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        if not self._is_fitted or self._model is None:
            raise RuntimeError("TorchMLPBaseline has not been fitted.")
        X = np.asarray(X, dtype=np.float64)
        # Fill any residual NaN rows with zeros after standardization so
        # predict() never drops rows silently (alignment with y_test).
        nan_mask = np.isnan(X).any(axis=1)
        if nan_mask.any():
            X = X.copy()
            col_means = self._mean
            for j in range(X.shape[1]):
                col_nan = np.isnan(X[:, j])
                if col_nan.any():
                    X[col_nan, j] = col_means[j]

        Xs = (X - self._mean) / self._std
        t = torch.tensor(Xs, dtype=torch.float32, device=self.device)
        self._model.eval()
        with torch.no_grad():
            out = self._model(t).cpu().numpy().ravel()
        return np.expm1(out) if self.log_target else out

    def __repr__(self):
        return (f"TorchMLPBaseline(layers={self.params['hidden_layers']}, "
                f"device={self.device})")
