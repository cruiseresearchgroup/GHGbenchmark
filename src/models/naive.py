"""
Naive baseline models for CarbonBench.

Implements the simplest possible baselines:
- HistoricalMeanBaseline: predict the mean of training targets
- LastYearBaseline: predict the most recent observed emission (lag1)
"""

import numpy as np
import pandas as pd
from typing import Optional


class HistoricalMeanBaseline:
    """
    Predicts the global mean of training emissions.

    This is the simplest possible baseline. A well-performing model
    should significantly outperform this.

    Note: The mean is computed in log space (on log1p-transformed targets)
    to handle the heavy-tailed distribution of emissions.
    """

    def __init__(self, log_space: bool = True):
        """
        Args:
            log_space: If True, compute mean in log space and exponentiate
                       predictions (for targets already log-transformed, set False).
        """
        self.log_space = log_space
        self.mean_: Optional[float] = None
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HistoricalMeanBaseline":
        """
        Compute mean of training targets.

        Args:
            X: Feature matrix (not used, but kept for API compatibility).
            y: Target array of emission values.

        Returns:
            self
        """
        y = np.asarray(y, dtype=float)
        valid = y[~np.isnan(y)]

        if self.log_space:
            self.mean_ = np.mean(np.log1p(np.maximum(valid, 0)))
        else:
            self.mean_ = np.mean(valid)

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the historical mean for all instances.

        Args:
            X: Feature matrix (shape [n_samples, n_features]).

        Returns:
            Array of predictions, shape [n_samples].
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        n = len(X) if hasattr(X, "__len__") else X.shape[0]

        if self.log_space:
            return np.full(n, np.expm1(self.mean_))
        else:
            return np.full(n, self.mean_)

    def __repr__(self) -> str:
        status = f"mean={self.mean_:.3f}" if self._is_fitted else "not fitted"
        return f"HistoricalMeanBaseline({status})"


class LastYearBaseline:
    """
    Predicts last year's emission as the forecast.

    Uses the first lag feature (lag1_emission) which corresponds to the
    most recently observed emission value.

    In log space: if features are log1p-transformed, returns expm1(lag1).
    In raw space: directly returns lag1.
    """

    def __init__(self, lag1_col_index: int = 0, log_space: bool = True):
        """
        Args:
            lag1_col_index: Index of the lag-1 feature in the feature matrix.
                            By convention in CarbonBench, index 0 = lag1_emission.
            log_space: If True, the lag features are in log space (log1p-transformed),
                       and predictions will be expm1-transformed back.
        """
        self.lag1_col_index = lag1_col_index
        self.log_space = log_space
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LastYearBaseline":
        """
        No-op fit (last-year baseline has no learnable parameters).

        Args:
            X: Feature matrix (not used).
            y: Target array (not used).

        Returns:
            self
        """
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return last year's emission (lag1 feature).

        Args:
            X: Feature matrix where column `lag1_col_index` = last-year emission.

        Returns:
            Array of predictions, shape [n_samples].
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        X = np.asarray(X)
        lag1 = X[:, self.lag1_col_index]

        if self.log_space:
            return np.expm1(lag1)
        else:
            return lag1

    def __repr__(self) -> str:
        return f"LastYearBaseline(lag1_col_index={self.lag1_col_index})"


class MovingAverageBaseline:
    """
    Predicts the moving average of the last K years.

    Uses lag features to compute an unweighted average.
    """

    def __init__(self, window: int = 3, log_space: bool = True):
        """
        Args:
            window: Number of lag years to average.
            log_space: If True, features and predictions are in log space.
        """
        self.window = window
        self.log_space = log_space
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MovingAverageBaseline":
        """No-op fit."""
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Average the first `window` lag columns.

        Args:
            X: Feature matrix where columns 0..window-1 are lag features.

        Returns:
            Array of predictions, shape [n_samples].
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        X = np.asarray(X)
        n_available = min(self.window, X.shape[1])
        lag_cols = X[:, :n_available]

        avg = np.mean(lag_cols, axis=1)

        if self.log_space:
            return np.expm1(avg)
        else:
            return avg

    def __repr__(self) -> str:
        return f"MovingAverageBaseline(window={self.window})"
