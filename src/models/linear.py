"""
Linear baseline models for CarbonBench.

Implements linear regression variants using scikit-learn:
- LinearRegressionBaseline
- RidgeBaseline
- LassoBaseline

All models apply log1p transformation to the target and use
expm1 on predictions to convert back to tCO2e.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from typing import Optional


class LinearRegressionBaseline:
    """
    Ordinary least squares linear regression baseline.

    Fits in log space (log1p target) for better handling of
    heavy-tailed emission distributions.
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        normalize_features: bool = True,
        log_target: bool = True,
    ):
        """
        Args:
            fit_intercept: Whether to fit an intercept term.
            normalize_features: If True, standardize features before fitting.
            log_target: If True, fit on log1p(y) and return expm1(predictions).
        """
        self.fit_intercept = fit_intercept
        self.normalize_features = normalize_features
        self.log_target = log_target

        self._model = LinearRegression(fit_intercept=fit_intercept)
        self._scaler = StandardScaler() if normalize_features else None
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionBaseline":
        """
        Fit the linear regression model.

        Args:
            X: Feature matrix, shape [n_samples, n_features].
            y: Target emissions (tCO2e), shape [n_samples].

        Returns:
            self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Remove NaN rows
        valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[valid], y[valid]

        if self.log_target:
            y_fit = np.log1p(np.maximum(y, 0))
        else:
            y_fit = y

        if self._scaler is not None:
            X_fit = self._scaler.fit_transform(X)
        else:
            X_fit = X

        self._model.fit(X_fit, y_fit)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict emissions.

        Args:
            X: Feature matrix, shape [n_samples, n_features].

        Returns:
            Predicted emissions in tCO2e, shape [n_samples].
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        X = np.asarray(X, dtype=float)

        if self._scaler is not None:
            X_pred = self._scaler.transform(X)
        else:
            X_pred = X

        y_pred = self._model.predict(X_pred)

        if self.log_target:
            return np.expm1(y_pred)
        else:
            return y_pred

    @property
    def coef_(self):
        """Return model coefficients."""
        return self._model.coef_ if self._is_fitted else None

    def __repr__(self) -> str:
        return f"LinearRegressionBaseline(normalize={self.normalize_features})"


class RidgeBaseline:
    """
    Ridge (L2-regularized) regression baseline.

    Reduces overfitting through L2 regularization, which is important
    when many lag/sector features are included.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        normalize_features: bool = True,
        log_target: bool = True,
    ):
        """
        Args:
            alpha: Regularization strength. Larger = stronger regularization.
            fit_intercept: Whether to fit an intercept term.
            normalize_features: If True, standardize features before fitting.
            log_target: If True, fit on log1p(y) and return expm1(predictions).
        """
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize_features = normalize_features
        self.log_target = log_target

        self._model = Ridge(alpha=alpha, fit_intercept=fit_intercept)
        self._scaler = StandardScaler() if normalize_features else None
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeBaseline":
        """Fit Ridge regression model."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[valid], y[valid]

        if self.log_target:
            y_fit = np.log1p(np.maximum(y, 0))
        else:
            y_fit = y

        if self._scaler is not None:
            X_fit = self._scaler.fit_transform(X)
        else:
            X_fit = X

        self._model.fit(X_fit, y_fit)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict emissions in tCO2e."""
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        X = np.asarray(X, dtype=float)
        X_pred = self._scaler.transform(X) if self._scaler else X
        y_pred = self._model.predict(X_pred)
        return np.expm1(y_pred) if self.log_target else y_pred

    def __repr__(self) -> str:
        return f"RidgeBaseline(alpha={self.alpha})"


class LassoBaseline:
    """
    Lasso (L1-regularized) regression baseline.

    Performs implicit feature selection through L1 regularization,
    which can be beneficial when many features are irrelevant.
    """

    def __init__(
        self,
        alpha: float = 0.1,
        max_iter: int = 2000,
        fit_intercept: bool = True,
        normalize_features: bool = True,
        log_target: bool = True,
    ):
        """
        Args:
            alpha: Regularization strength.
            max_iter: Maximum iterations for coordinate descent.
            fit_intercept: Whether to fit an intercept term.
            normalize_features: If True, standardize features before fitting.
            log_target: If True, fit on log1p(y) and return expm1(predictions).
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.normalize_features = normalize_features
        self.log_target = log_target

        self._model = Lasso(alpha=alpha, max_iter=max_iter, fit_intercept=fit_intercept)
        self._scaler = StandardScaler() if normalize_features else None
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LassoBaseline":
        """Fit Lasso regression model."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[valid], y[valid]

        if self.log_target:
            y_fit = np.log1p(np.maximum(y, 0))
        else:
            y_fit = y

        if self._scaler is not None:
            X_fit = self._scaler.fit_transform(X)
        else:
            X_fit = X

        self._model.fit(X_fit, y_fit)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict emissions in tCO2e."""
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        X = np.asarray(X, dtype=float)
        X_pred = self._scaler.transform(X) if self._scaler else X
        y_pred = self._model.predict(X_pred)
        return np.expm1(y_pred) if self.log_target else y_pred

    @property
    def n_nonzero_features_(self) -> Optional[int]:
        """Number of non-zero coefficients (features selected)."""
        if self._is_fitted:
            return int(np.sum(self._model.coef_ != 0))
        return None

    def __repr__(self) -> str:
        return f"LassoBaseline(alpha={self.alpha})"
