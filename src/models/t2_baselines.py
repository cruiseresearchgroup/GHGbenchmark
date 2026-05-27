"""
T2 building-level baseline models for CarbonBench.

- GlobalMeanBaseline: predict training set mean
- CityTypeMeanBaseline: predict mean by (city, property_type) group
- MLPBaseline: 3-layer MLP with dropout
"""

import numpy as np
from typing import Any, Dict, Optional
from sklearn.neural_network import MLPRegressor


class GlobalMeanBaseline:
    """Predict the training set mean (in log space)."""

    def __init__(self, log_target: bool = True):
        self.log_target = log_target
        self._mean = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GlobalMeanBaseline":
        y = np.asarray(y, dtype=float)
        valid = ~np.isnan(y)
        if self.log_target:
            self._mean = np.mean(np.log1p(np.maximum(y[valid], 0)))
        else:
            self._mean = np.mean(y[valid])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0] if hasattr(X, 'shape') else len(X)
        val = np.expm1(self._mean) if self.log_target else self._mean
        return np.full(n, val)

    def __repr__(self):
        return "GlobalMeanBaseline()"


class CityTypeMeanBaseline:
    """
    Predict by (city, property_type) group mean.

    Requires that the first two columns of X are city_encoded and property_type_encoded.
    Falls back to global mean for unseen groups.
    """

    def __init__(self, log_target: bool = True):
        self.log_target = log_target
        self._group_means: Dict = {}
        self._global_mean: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CityTypeMeanBaseline":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[valid], y[valid]

        y_t = np.log1p(np.maximum(y, 0)) if self.log_target else y
        self._global_mean = np.mean(y_t)

        # Group by first two columns (city, property_type)
        for i in range(len(X)):
            key = (int(X[i, 0]), int(X[i, 1]))  # (city_code, type_code)
            if key not in self._group_means:
                self._group_means[key] = []
            self._group_means[key].append(y_t[i])

        self._group_means = {k: np.mean(v) for k, v in self._group_means.items()}
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        preds = np.zeros(len(X))
        for i in range(len(X)):
            key = (int(X[i, 0]), int(X[i, 1]))
            mean_val = self._group_means.get(key, self._global_mean)
            preds[i] = np.expm1(mean_val) if self.log_target else mean_val
        return preds

    def __repr__(self):
        return f"CityTypeMeanBaseline(n_groups={len(self._group_means)})"


class MLPBaseline:
    """
    3-layer MLP regression baseline using scikit-learn.

    Architecture: [256, 128, 64] with ReLU, dropout via early stopping.
    """

    DEFAULT_PARAMS = {
        "hidden_layer_sizes": (256, 128, 64),
        "activation": "relu",
        "solver": "adam",
        "learning_rate_init": 0.001,
        "max_iter": 500,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 20,
        "random_state": 42,
        "batch_size": 256,
    }

    def __init__(self, log_target: bool = True, **kwargs: Any):
        self.log_target = log_target
        params = {**self.DEFAULT_PARAMS, **kwargs}
        self._model = MLPRegressor(**params)
        self._is_fitted = False
        # Standardize features internally
        self._mean = None
        self._std = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPBaseline":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[valid], y[valid]

        # Standardize
        self._mean = np.nanmean(X, axis=0)
        self._std = np.nanstd(X, axis=0)
        self._std[self._std == 0] = 1.0
        X_scaled = (X - self._mean) / self._std

        y_fit = np.log1p(np.maximum(y, 0)) if self.log_target else y
        self._model.fit(X_scaled, y_fit)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted.")
        X = np.asarray(X, dtype=float)
        X_scaled = (X - self._mean) / self._std
        y_pred = self._model.predict(X_scaled)
        return np.expm1(y_pred) if self.log_target else y_pred

    def __repr__(self):
        return f"MLPBaseline(layers={self.DEFAULT_PARAMS['hidden_layer_sizes']})"
