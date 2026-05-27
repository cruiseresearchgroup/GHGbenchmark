"""
LSTM baseline model for CarbonBench.

PyTorch implementation of an LSTM for sequential emission forecasting.
Input: [batch, seq_len, n_features]
Output: [batch, horizon]

Default configuration: hidden_size=64, num_layers=2, dropout=0.1
"""

import numpy as np
from typing import Optional, Tuple, List


class LSTMBaseline:
    """
    LSTM-based emission forecasting baseline.

    Wraps a PyTorch LSTM with a training loop, supporting:
    - Variable-length historical sequences (up to 10 years)
    - Multi-step output (horizon >= 1)
    - Early stopping based on validation loss
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
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
            input_size: Number of input features per timestep.
            hidden_size: LSTM hidden state dimensionality.
            num_layers: Number of LSTM layers.
            dropout: Dropout rate between LSTM layers.
            horizon: Number of future steps to predict.
            batch_size: Training batch size.
            lr: Learning rate for Adam optimizer.
            epochs: Maximum training epochs.
            patience: Early stopping patience (epochs without improvement).
            log_target: If True, fit on log1p(y) and return expm1(predictions).
            device: PyTorch device ('cpu', 'cuda', 'mps'). Auto-detected if None.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
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
            raise ImportError("PyTorch is required for LSTMBaseline. Install: pip install torch")

    def _build_model(self, input_size: int):
        """Build LSTM model architecture."""
        try:
            import torch
            import torch.nn as nn

            class _LSTMNet(nn.Module):
                def __init__(self, input_size, hidden_size, num_layers, dropout, horizon):
                    super().__init__()
                    self.lstm = nn.LSTM(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout if num_layers > 1 else 0.0,
                        batch_first=True,
                    )
                    self.dropout = nn.Dropout(dropout)
                    self.fc = nn.Linear(hidden_size, horizon)

                def forward(self, x):
                    # x: [batch, seq_len, input_size]
                    lstm_out, _ = self.lstm(x)
                    # Use last timestep output
                    last_out = lstm_out[:, -1, :]
                    out = self.dropout(last_out)
                    return self.fc(out)  # [batch, horizon]

            return _LSTMNet(input_size, self.hidden_size, self.num_layers, self.dropout, self.horizon)
        except ImportError:
            raise ImportError("PyTorch is required for LSTMBaseline.")

    def _prepare_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert flat feature matrix to sequences.

        Assumes X[:, :n_lag_features] are lag features (most recent first).
        Reshapes them into sequences for LSTM input.
        """
        # Lag features are columns 0..n_lags-1 (lag1, lag2, ...)
        # We create sequences from oldest to most recent
        n_lags = min(X.shape[1], 10)  # Up to 10 years of history

        # Reverse lag order: lag_n, ..., lag_2, lag_1 (oldest to most recent)
        lag_features = X[:, :n_lags][:, ::-1]  # [n_samples, seq_len]

        # Add remaining features as context (repeated across time steps)
        if X.shape[1] > n_lags:
            extra_features = X[:, n_lags:]  # [n_samples, n_extra]
            # Repeat along time dimension
            extra_repeated = np.repeat(
                extra_features[:, np.newaxis, :], n_lags, axis=1
            )  # [n_samples, seq_len, n_extra]
            sequences = np.concatenate(
                [lag_features[:, :, np.newaxis], extra_repeated], axis=2
            )  # [n_samples, seq_len, 1+n_extra]
        else:
            sequences = lag_features[:, :, np.newaxis]  # [n_samples, seq_len, 1]

        return sequences, y

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "LSTMBaseline":
        """
        Train LSTM model.

        Args:
            X: Feature matrix, shape [n_samples, n_features].
            y: Target emissions, shape [n_samples] or [n_samples, horizon].
            X_val: Optional validation features for early stopping.
            y_val: Optional validation targets.

        Returns:
            self
        """
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError("PyTorch is required for LSTMBaseline.")

        device = self._get_device()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Remove NaN rows
        if y.ndim == 1:
            valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        else:
            valid = ~(np.isnan(X).any(axis=1) | np.isnan(y).any(axis=1))
        X, y = X[valid], y[valid]

        if self.log_target:
            if y.ndim == 1:
                y_fit = np.log1p(np.maximum(y, 0))
            else:
                y_fit = np.log1p(np.maximum(y, 0))
        else:
            y_fit = y

        # Prepare sequences
        X_seq, _ = self._prepare_sequences(X, y)
        actual_input_size = X_seq.shape[2]

        # Build model
        self._model = self._build_model(actual_input_size).to(device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        # Create data loaders
        X_tensor = torch.FloatTensor(X_seq).to(device)
        if y_fit.ndim == 1:
            y_tensor = torch.FloatTensor(y_fit.reshape(-1, 1)).to(device)
        else:
            y_tensor = torch.FloatTensor(y_fit).to(device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Validation setup
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val = np.asarray(X_val, dtype=float)
            y_val = np.asarray(y_val, dtype=float)
            y_val_fit = np.log1p(np.maximum(y_val, 0)) if self.log_target else y_val
            X_val_seq, _ = self._prepare_sequences(X_val, y_val)
            X_val_tensor = torch.FloatTensor(X_val_seq).to(device)
            if y_val_fit.ndim == 1:
                y_val_tensor = torch.FloatTensor(y_val_fit.reshape(-1, 1)).to(device)
            else:
                y_val_tensor = torch.FloatTensor(y_val_fit).to(device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.epochs):
            self._model.train()
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = self._model(X_batch)
                if pred.shape != y_batch.shape:
                    pred = pred.view_as(y_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()

            # Validation
            if val_loader is not None:
                self._model.eval()
                val_loss = 0.0
                n_batches = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        pred = self._model(X_batch)
                        if pred.shape != y_batch.shape:
                            pred = pred.view_as(y_batch)
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

        # Restore best model
        if best_state is not None:
            self._model.load_state_dict(best_state)

        self._device_obj = device
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict emissions.

        Args:
            X: Feature matrix, shape [n_samples, n_features].

        Returns:
            Predicted emissions in tCO2e, shape [n_samples] or [n_samples, horizon].
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required.")

        X = np.asarray(X, dtype=float)
        X_seq, _ = self._prepare_sequences(X, np.zeros(len(X)))

        self._model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self._device_obj)
            y_pred = self._model(X_tensor).cpu().numpy()

        if self.log_target:
            y_pred = np.expm1(y_pred)

        if y_pred.shape[1] == 1:
            return y_pred.ravel()
        return y_pred

    def __repr__(self) -> str:
        return (
            f"LSTMBaseline(hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, horizon={self.horizon})"
        )
