"""Multimodal MLP baselines for tabular + S2 building-level regression."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn


class _TwoTowerNet(nn.Module):
    def __init__(
        self,
        tab_dim: int,
        s2_dim: int,
        tab_hidden: tuple[int, ...] = (128, 64),
        s2_hidden: tuple[int, ...] = (256, 64),
        fusion_hidden: tuple[int, ...] = (64,),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.tab_out_dim = tab_hidden[-1]
        self.s2_out_dim = s2_hidden[-1]

        self.tab_tower = self._make_tower(tab_dim, tab_hidden, dropout)
        self.s2_tower = self._make_tower(s2_dim, s2_hidden, dropout)

        fusion_in = self.tab_out_dim + self.s2_out_dim
        layers: list[nn.Module] = []
        prev = fusion_in
        for h in fusion_hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.head = nn.Sequential(*layers)

    @staticmethod
    def _make_tower(in_dim: int, hidden: tuple[int, ...], dropout: float) -> nn.Module:
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        return nn.Sequential(*layers)

    def forward(self, x_tab: torch.Tensor, x_s2: torch.Tensor) -> torch.Tensor:
        h_tab = self.tab_tower(x_tab)
        h_s2 = self.s2_tower(x_s2)
        h = torch.cat([h_tab, h_s2], dim=1)
        return self.head(h)


class _GatedFusionNet(nn.Module):
    def __init__(
        self,
        tab_dim: int,
        s2_dim: int,
        tower_hidden: tuple[int, ...] = (128, 64),
        head_hidden: tuple[int, ...] = (64,),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.out_dim = tower_hidden[-1]
        self.tab_tower = _TwoTowerNet._make_tower(tab_dim, tower_hidden, dropout)
        self.s2_tower = _TwoTowerNet._make_tower(s2_dim, tower_hidden, dropout)
        self.gate = nn.Sequential(
            nn.Linear(self.out_dim * 2, self.out_dim),
            nn.Sigmoid(),
        )

        layers: list[nn.Module] = []
        prev = self.out_dim
        for h in head_hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.head = nn.Sequential(*layers)

    def forward(self, x_tab: torch.Tensor, x_s2: torch.Tensor) -> torch.Tensor:
        h_tab = self.tab_tower(x_tab)
        h_s2 = self.s2_tower(x_s2)
        g = self.gate(torch.cat([h_tab, h_s2], dim=1))
        fused = g * h_s2 + (1.0 - g) * h_tab
        return self.head(fused)


class TorchMultimodalMLPBaseline:
    DEFAULT_PARAMS = {
        "tab_hidden": (128, 64),
        "s2_hidden": (256, 64),
        "fusion_hidden": (64,),
        "dropout": 0.2,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "batch_size": 4096,
        "max_epochs": 200,
        "patience": 20,
        "val_fraction": 0.1,
        "seed": 42,
        "fusion": "two_tower",  # or gated
    }

    def __init__(
        self,
        tab_dim: int,
        s2_dim: int,
        log_target: bool = True,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.tab_dim = tab_dim
        self.s2_dim = s2_dim
        self.log_target = log_target
        self.params = {**self.DEFAULT_PARAMS, **kwargs}

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self._model: Optional[nn.Module] = None
        self._tab_mean: Optional[np.ndarray] = None
        self._tab_std: Optional[np.ndarray] = None
        self._s2_mean: Optional[np.ndarray] = None
        self._s2_std: Optional[np.ndarray] = None
        self._is_fitted = False

    def _split(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if X.shape[1] != self.tab_dim + self.s2_dim:
            raise ValueError(
                f"Expected {self.tab_dim + self.s2_dim} total dims, got {X.shape[1]}"
            )
        return X[:, : self.tab_dim], X[:, self.tab_dim :]

    def _build_model(self) -> nn.Module:
        if self.params["fusion"] == "gated":
            return _GatedFusionNet(
                self.tab_dim,
                self.s2_dim,
                tower_hidden=self.params["tab_hidden"],
                head_hidden=self.params["fusion_hidden"],
                dropout=self.params["dropout"],
            )
        return _TwoTowerNet(
            self.tab_dim,
            self.s2_dim,
            tab_hidden=self.params["tab_hidden"],
            s2_hidden=self.params["s2_hidden"],
            fusion_hidden=self.params["fusion_hidden"],
            dropout=self.params["dropout"],
        )

    def _clean(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        if y is None:
            mask = ~np.isnan(X).any(axis=1)
            return X[mask], None, mask
        y = np.asarray(y, dtype=np.float64)
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        return X[mask], y[mask], mask

    def _standardize_fit(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X_tab, X_s2 = self._split(X)
        self._tab_mean = X_tab.mean(axis=0)
        self._tab_std = X_tab.std(axis=0)
        self._tab_std[self._tab_std == 0] = 1.0
        self._s2_mean = X_s2.mean(axis=0)
        self._s2_std = X_s2.std(axis=0)
        self._s2_std[self._s2_std == 0] = 1.0
        return (X_tab - self._tab_mean) / self._tab_std, (X_s2 - self._s2_mean) / self._s2_std

    def _standardize_apply(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X_tab, X_s2 = self._split(X)
        return (X_tab - self._tab_mean) / self._tab_std, (X_s2 - self._s2_mean) / self._s2_std

    def fit(self, X, y, X_val=None, y_val=None) -> "TorchMultimodalMLPBaseline":
        torch.manual_seed(self.params["seed"])
        np.random.seed(self.params["seed"])

        X, y, _ = self._clean(X, y)
        if len(X) < 2:
            raise ValueError("TorchMultimodalMLPBaseline.fit received <2 valid rows")

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

        X_tr_tab, X_tr_s2 = self._standardize_fit(X_tr)
        X_vl_tab, X_vl_s2 = self._standardize_apply(X_vl)

        if self.log_target:
            y_tr_t = np.log1p(np.maximum(y_tr, 0))
            y_vl_t = np.log1p(np.maximum(y_vl, 0))
        else:
            y_tr_t, y_vl_t = y_tr, y_vl

        dev = self.device
        X_tr_tab_t = torch.tensor(X_tr_tab, dtype=torch.float32, device=dev)
        X_tr_s2_t = torch.tensor(X_tr_s2, dtype=torch.float32, device=dev)
        y_tr_t = torch.tensor(y_tr_t, dtype=torch.float32, device=dev).unsqueeze(1)
        X_vl_tab_t = torch.tensor(X_vl_tab, dtype=torch.float32, device=dev)
        X_vl_s2_t = torch.tensor(X_vl_s2, dtype=torch.float32, device=dev)
        y_vl_t = torch.tensor(y_vl_t, dtype=torch.float32, device=dev).unsqueeze(1)

        self._model = self._build_model().to(dev)
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

        for _ in range(self.params["max_epochs"]):
            self._model.train()
            perm = torch.randperm(X_tr_tab_t.size(0), device=dev)
            for i in range(0, perm.size(0), bs):
                sel = perm[i:i + bs]
                xb_tab = X_tr_tab_t[sel]
                xb_s2 = X_tr_s2_t[sel]
                yb = y_tr_t[sel]
                opt.zero_grad()
                pred = self._model(xb_tab, xb_s2)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

            self._model.eval()
            with torch.no_grad():
                val_pred = self._model(X_vl_tab_t, X_vl_s2_t)
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
            raise RuntimeError("TorchMultimodalMLPBaseline has not been fitted.")
        X = np.asarray(X, dtype=np.float64)
        nan_mask = np.isnan(X).any(axis=1)
        if nan_mask.any():
            X = X.copy()
            tab, s2 = self._split(X)
            for j in range(tab.shape[1]):
                col_nan = np.isnan(tab[:, j])
                if col_nan.any():
                    tab[col_nan, j] = self._tab_mean[j]
            for j in range(s2.shape[1]):
                col_nan = np.isnan(s2[:, j])
                if col_nan.any():
                    s2[col_nan, j] = self._s2_mean[j]
            X = np.concatenate([tab, s2], axis=1)

        X_tab, X_s2 = self._standardize_apply(X)
        X_tab_t = torch.tensor(X_tab, dtype=torch.float32, device=self.device)
        X_s2_t = torch.tensor(X_s2, dtype=torch.float32, device=self.device)
        self._model.eval()
        with torch.no_grad():
            out = self._model(X_tab_t, X_s2_t).cpu().numpy().ravel()
        return np.expm1(out) if self.log_target else out

    def __repr__(self):
        return (
            f"TorchMultimodalMLPBaseline(tab_dim={self.tab_dim}, s2_dim={self.s2_dim}, "
            f"fusion={self.params['fusion']}, device={self.device})"
        )
