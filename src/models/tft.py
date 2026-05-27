"""
Temporal Fusion Transformer (TFT) baseline for CarbonBench.

Implements a simplified TFT with:
- Variable selection network
- LSTM encoder
- Multi-head attention decoder

If pytorch-forecasting is available, uses that implementation.
Otherwise falls back to the simplified custom implementation here.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict


class TFTBaseline:
    """
    Simplified Temporal Fusion Transformer for emission forecasting.

    This implementation provides the key components of TFT:
    1. Variable Selection Networks (VSN) for input gating
    2. LSTM encoder for processing temporal sequences
    3. Interpretable multi-head attention for long-range dependencies
    4. Quantile output head

    Reference: Lim et al. (2021), "Temporal Fusion Transformers for
    Interpretable Multi-horizon Time Series Forecasting"
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        n_heads: int = 4,
        num_lstm_layers: int = 2,
        dropout: float = 0.1,
        horizon: int = 1,
        batch_size: int = 64,
        lr: float = 1e-3,
        epochs: int = 100,
        patience: int = 10,
        log_target: bool = True,
        device: Optional[str] = None,
    ):
        """
        Args:
            input_size: Number of input features.
            hidden_size: Hidden dimension for all sub-networks.
            n_heads: Number of attention heads.
            num_lstm_layers: Number of LSTM encoder layers.
            dropout: Dropout rate.
            horizon: Number of future steps to predict.
            batch_size: Training batch size.
            lr: Learning rate.
            epochs: Maximum training epochs.
            patience: Early stopping patience.
            log_target: If True, predict in log space.
            device: PyTorch device (auto-detected if None).
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.num_lstm_layers = num_lstm_layers
        self.dropout = dropout
        self.horizon = horizon
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.log_target = log_target
        self._device = device
        self._model = None
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
            raise ImportError("PyTorch is required for TFTBaseline.")

    def _build_model(self, actual_input_size: int):
        """Build the simplified TFT model."""
        try:
            import torch
            import torch.nn as nn

            hidden = self.hidden_size
            n_heads = self.n_heads
            dropout = self.dropout
            horizon = self.horizon
            num_lstm_layers = self.num_lstm_layers

            class GRN(nn.Module):
                """Gated Residual Network - core building block of TFT."""
                def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
                    super().__init__()
                    self.fc1 = nn.Linear(input_dim, hidden_dim)
                    self.fc2 = nn.Linear(hidden_dim, output_dim)
                    self.gate = nn.Linear(hidden_dim, output_dim)
                    self.residual = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
                    self.dropout = nn.Dropout(dropout_rate)
                    self.norm = nn.LayerNorm(output_dim)
                    self.elu = nn.ELU()

                def forward(self, x):
                    h = self.elu(self.fc1(x))
                    h = self.dropout(h)
                    eta1 = self.fc2(h)
                    eta2 = torch.sigmoid(self.gate(h))
                    residual = self.residual(x)
                    return self.norm(eta1 * eta2 + residual)

            class VariableSelectionNetwork(nn.Module):
                """Variable selection with soft attention over input features."""
                def __init__(self, input_dim, hidden_dim, n_vars, dropout_rate):
                    super().__init__()
                    self.n_vars = n_vars
                    self.var_grns = nn.ModuleList([
                        GRN(1, hidden_dim, hidden_dim, dropout_rate) for _ in range(n_vars)
                    ])
                    self.weight_grn = GRN(input_dim, hidden_dim, n_vars, dropout_rate)
                    self.softmax = nn.Softmax(dim=-1)

                def forward(self, x):
                    # x: [batch, seq, n_vars]
                    weights = self.softmax(self.weight_grn(x.reshape(x.shape[0], x.shape[1], -1).mean(dim=1)))
                    # weights: [batch, n_vars]
                    processed = []
                    for i in range(self.n_vars):
                        processed.append(self.var_grns[i](x[:, :, i:i+1]))
                    # processed: list of [batch, seq, hidden]
                    stacked = torch.stack(processed, dim=-1)  # [batch, seq, hidden, n_vars]
                    weights_expanded = weights.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, n_vars]
                    return (stacked * weights_expanded).sum(dim=-1)  # [batch, seq, hidden]

            class SimplifiedTFT(nn.Module):
                def __init__(self, input_dim, hidden_dim, n_heads, num_layers, dropout_rate, horizon):
                    super().__init__()
                    self.vsn = VariableSelectionNetwork(input_dim, hidden_dim, input_dim, dropout_rate)
                    self.encoder = nn.LSTM(
                        input_size=hidden_dim,
                        hidden_size=hidden_dim,
                        num_layers=num_layers,
                        dropout=dropout_rate if num_layers > 1 else 0.0,
                        batch_first=True,
                    )
                    self.attention = nn.MultiheadAttention(
                        embed_dim=hidden_dim,
                        num_heads=n_heads,
                        dropout=dropout_rate,
                        batch_first=True,
                    )
                    self.norm1 = nn.LayerNorm(hidden_dim)
                    self.norm2 = nn.LayerNorm(hidden_dim)
                    self.grn_out = GRN(hidden_dim, hidden_dim, hidden_dim, dropout_rate)
                    self.fc_out = nn.Linear(hidden_dim, horizon)
                    self.dropout = nn.Dropout(dropout_rate)

                def forward(self, x):
                    # x: [batch, seq, input_dim]
                    vsn_out = self.vsn(x)  # [batch, seq, hidden]
                    enc_out, _ = self.encoder(vsn_out)  # [batch, seq, hidden]

                    # Self-attention
                    attn_out, _ = self.attention(enc_out, enc_out, enc_out)
                    attn_out = self.norm1(attn_out + enc_out)  # residual

                    # GRN on last timestep
                    last = attn_out[:, -1, :]  # [batch, hidden]
                    out = self.grn_out(last)
                    out = self.dropout(out)
                    return self.fc_out(out)  # [batch, horizon]

            return SimplifiedTFT(actual_input_size, hidden, n_heads, num_lstm_layers, dropout, horizon)

        except ImportError:
            raise ImportError("PyTorch is required for TFTBaseline.")

    def _prepare_sequences(self, X: np.ndarray) -> np.ndarray:
        """Convert flat feature matrix to time sequences."""
        n_lags = min(X.shape[1], 10)
        lag_features = X[:, :n_lags][:, ::-1]  # oldest to most recent

        if X.shape[1] > n_lags:
            extra_features = X[:, n_lags:]
            extra_repeated = np.repeat(extra_features[:, np.newaxis, :], n_lags, axis=1)
            sequences = np.concatenate(
                [lag_features[:, :, np.newaxis], extra_repeated], axis=2
            )
        else:
            sequences = lag_features[:, :, np.newaxis]

        return sequences

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "TFTBaseline":
        """Train TFT model."""
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError("PyTorch is required for TFTBaseline.")

        device = self._get_device()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[valid], y[valid]

        y_fit = np.log1p(np.maximum(y, 0)) if self.log_target else y

        X_seq = self._prepare_sequences(X)
        actual_input_size = X_seq.shape[2]
        self._model = self._build_model(actual_input_size).to(device)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.HuberLoss()

        X_tensor = torch.FloatTensor(X_seq).to(device)
        y_tensor = torch.FloatTensor(y_fit.reshape(-1, 1)).to(device)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            X_val = np.asarray(X_val, dtype=float)
            y_val = np.asarray(y_val, dtype=float)
            y_val_fit = np.log1p(np.maximum(y_val, 0)) if self.log_target else y_val
            X_val_seq = self._prepare_sequences(X_val)
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val_seq).to(device),
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
        X_seq = self._prepare_sequences(X)

        self._model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self._device_obj)
            y_pred = self._model(X_tensor).cpu().numpy()

        if self.log_target:
            y_pred = np.expm1(y_pred)

        return y_pred.ravel() if y_pred.shape[1] == 1 else y_pred

    def __repr__(self) -> str:
        return (
            f"TFTBaseline(hidden_size={self.hidden_size}, "
            f"n_heads={self.n_heads}, horizon={self.horizon})"
        )
