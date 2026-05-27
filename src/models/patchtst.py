"""
PatchTST baseline model for CarbonBench.

Implements PatchTST (Nie et al., 2023) adapted for annual emission time series.
The annual series is patched into sub-sequences, which are then processed
by a Transformer encoder with a linear head for multi-step output.

Reference: Nie et al. (2023), "A Time Series is Worth 64 Words:
Long-term Forecasting with Transformers"
"""

import numpy as np
from typing import Optional, Tuple


class PatchTSTBaseline:
    """
    PatchTST-based emission forecasting baseline.

    Patches the annual emission time series into overlapping sub-sequences
    (patches) before feeding to a Transformer encoder. This captures
    local temporal patterns more effectively than token-per-timestep approaches.

    Configuration (CarbonBench defaults):
    - d_model: 128
    - n_heads: 8
    - n_layers: 3
    - patch_len: 2 (2-year patches for annual data)
    - stride: 1 (overlapping patches)
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        patch_len: int = 2,
        stride: int = 1,
        dropout: float = 0.1,
        horizon: int = 1,
        batch_size: int = 32,
        lr: float = 1e-4,
        epochs: int = 50,
        patience: int = 10,
        log_target: bool = True,
        device: Optional[str] = None,
    ):
        """
        Args:
            d_model: Transformer embedding dimension.
            n_heads: Number of attention heads.
            n_layers: Number of Transformer encoder layers.
            patch_len: Length of each patch (in timesteps).
            stride: Stride between patches.
            dropout: Dropout rate in Transformer.
            horizon: Prediction horizon (steps ahead).
            batch_size: Training batch size.
            lr: Learning rate.
            epochs: Maximum training epochs.
            patience: Early stopping patience.
            log_target: If True, predict in log space.
            device: PyTorch device (auto-detected if None).
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.patch_len = patch_len
        self.stride = stride
        self.dropout = dropout
        self.horizon = horizon
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.log_target = log_target
        self._device = device
        self._model = None
        self._n_patches = None
        self._is_fitted = False

    def _get_device(self):
        """Auto-detect best available device."""
        try:
            import torch
            if self._device:
                return torch.device(self._device)
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        except ImportError:
            raise ImportError("PyTorch is required for PatchTSTBaseline.")

    def _compute_n_patches(self, seq_len: int) -> int:
        """Compute number of patches for a given sequence length."""
        return max(1, (seq_len - self.patch_len) // self.stride + 1)

    def _build_model(self, seq_len: int, n_features: int):
        """Build PatchTST model."""
        try:
            import torch
            import torch.nn as nn
            import math

            patch_len = self.patch_len
            stride = self.stride
            d_model = self.d_model
            n_heads = self.n_heads
            n_layers = self.n_layers
            dropout = self.dropout
            horizon = self.horizon

            n_patches = self._compute_n_patches(seq_len)

            class PatchEmbedding(nn.Module):
                """Project patches to d_model dimension."""
                def __init__(self, patch_len, n_features, d_model, dropout_rate):
                    super().__init__()
                    self.proj = nn.Linear(patch_len * n_features, d_model)
                    self.dropout = nn.Dropout(dropout_rate)

                def forward(self, x):
                    # x: [batch, n_patches, patch_len * n_features]
                    return self.dropout(self.proj(x))

            class PositionalEncoding(nn.Module):
                """Fixed sinusoidal positional encoding."""
                def __init__(self, d_model, max_len=100):
                    super().__init__()
                    pe = torch.zeros(max_len, d_model)
                    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                    div_term = torch.exp(
                        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
                    )
                    pe[:, 0::2] = torch.sin(position * div_term)
                    pe[:, 1::2] = torch.cos(position * div_term)
                    self.register_buffer("pe", pe.unsqueeze(0))

                def forward(self, x):
                    return x + self.pe[:, :x.size(1), :]

            class _PatchTSTNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.patch_embed = PatchEmbedding(patch_len, n_features, d_model, dropout)
                    self.pos_enc = PositionalEncoding(d_model)
                    encoder_layer = nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=n_heads,
                        dim_feedforward=d_model * 4,
                        dropout=dropout,
                        batch_first=True,
                        norm_first=True,  # Pre-LN for stability
                    )
                    self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
                    # Flatten all patch outputs -> predict horizon
                    self.head = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(n_patches * d_model, d_model),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(d_model, horizon),
                    )

                def forward(self, x):
                    # x: [batch, n_patches, patch_len * n_features]
                    embedded = self.patch_embed(x)           # [batch, n_patches, d_model]
                    encoded = self.pos_enc(embedded)          # [batch, n_patches, d_model]
                    context = self.transformer(encoded)       # [batch, n_patches, d_model]
                    return self.head(context)                 # [batch, horizon]

            return _PatchTSTNet(), n_patches

        except ImportError:
            raise ImportError("PyTorch is required for PatchTSTBaseline.")

    def _prepare_patches(self, X: np.ndarray) -> np.ndarray:
        """
        Convert flat feature matrix to patch sequences.

        Args:
            X: [n_samples, n_features] where first n_lags features are lag values.

        Returns:
            [n_samples, n_patches, patch_len * n_input_features]
        """
        n_lags = min(X.shape[1], 10)
        seq = X[:, :n_lags][:, ::-1]  # [n_samples, seq_len] oldest to most recent

        if X.shape[1] > n_lags:
            extra = X[:, n_lags:]  # [n_samples, n_extra]
            extra_rep = np.repeat(extra[:, np.newaxis, :], n_lags, axis=1)  # [n, seq, n_extra]
            full_seq = np.concatenate([seq[:, :, np.newaxis], extra_rep], axis=2)
        else:
            full_seq = seq[:, :, np.newaxis]  # [n, seq, 1]

        n_samples, seq_len, n_features = full_seq.shape
        n_patches = self._compute_n_patches(seq_len)

        patches = []
        for p in range(n_patches):
            start = p * self.stride
            end = start + self.patch_len
            if end <= seq_len:
                patch = full_seq[:, start:end, :]  # [n, patch_len, n_features]
            else:
                # Pad the last patch if needed
                patch = np.zeros((n_samples, self.patch_len, n_features))
                avail = seq_len - start
                patch[:, :avail, :] = full_seq[:, start:, :]
            patches.append(patch.reshape(n_samples, -1))  # [n, patch_len * n_features]

        return np.stack(patches, axis=1)  # [n, n_patches, patch_len * n_features]

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "PatchTSTBaseline":
        """Train PatchTST model."""
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError("PyTorch is required for PatchTSTBaseline.")

        device = self._get_device()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[valid], y[valid]

        y_fit = np.log1p(np.maximum(y, 0)) if self.log_target else y

        X_patches = self._prepare_patches(X)
        seq_len = min(X.shape[1], 10)
        n_features = X_patches.shape[2] // self.patch_len

        self._model, self._n_patches = self._build_model(seq_len, n_features)
        self._model = self._model.to(device)
        self._seq_len = seq_len
        self._n_features = n_features

        optimizer = torch.optim.AdamW(self._model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )
        criterion = nn.MSELoss()

        X_tensor = torch.FloatTensor(X_patches).to(device)
        y_tensor = torch.FloatTensor(y_fit.reshape(-1, 1)).to(device)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            X_val = np.asarray(X_val, dtype=float)
            y_val = np.asarray(y_val, dtype=float)
            y_val_fit = np.log1p(np.maximum(y_val, 0)) if self.log_target else y_val
            X_val_patches = self._prepare_patches(X_val)
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val_patches).to(device),
                torch.FloatTensor(y_val_fit.reshape(-1, 1)).to(device),
            )
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            self._model.train()
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = self._model(X_batch).view_as(y_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            if val_loader is not None:
                self._model.eval()
                val_loss = 0.0
                n_batches = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        pred = self._model(X_batch).view_as(y_batch)
                        val_loss += criterion(pred, y_batch).item()
                        n_batches += 1
                val_loss /= max(n_batches, 1)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

        if best_state is not None:
            self._model.load_state_dict(best_state)

        self._device_obj = device
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict emissions in tCO2e."""
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required.")

        X = np.asarray(X, dtype=float)
        X_patches = self._prepare_patches(X)

        self._model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_patches).to(self._device_obj)
            y_pred = self._model(X_tensor).cpu().numpy()

        if self.log_target:
            y_pred = np.expm1(y_pred)

        return y_pred.ravel() if y_pred.shape[1] == 1 else y_pred

    def __repr__(self) -> str:
        return (
            f"PatchTSTBaseline(d_model={self.d_model}, n_heads={self.n_heads}, "
            f"n_layers={self.n_layers}, patch_len={self.patch_len})"
        )
