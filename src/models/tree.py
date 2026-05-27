"""
Tree-based ensemble baseline models for CarbonBench.

Implements:
- RandomForestBaseline
- XGBoostBaseline
- LightGBMBaseline

All models fit in log space (log1p target) for better handling of
heavy-tailed emission distributions.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from typing import Optional, Dict, Any


class RandomForestBaseline:
    """
    Random Forest regression baseline.

    Uses scikit-learn's RandomForestRegressor with default CarbonBench
    hyperparameters as specified in the benchmark paper.
    """

    DEFAULT_PARAMS = {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_leaf": 5,
        "n_jobs": -1,
        "random_state": 42,
    }

    def __init__(
        self,
        log_target: bool = True,
        **kwargs: Any,
    ):
        """
        Args:
            log_target: If True, fit on log1p(y) and return expm1(predictions).
            **kwargs: Override default hyperparameters.
        """
        self.log_target = log_target
        params = {**self.DEFAULT_PARAMS, **kwargs}
        self._model = RandomForestRegressor(**params)
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestBaseline":
        """
        Fit Random Forest model.

        Args:
            X: Feature matrix, shape [n_samples, n_features].
            y: Target emissions (tCO2e), shape [n_samples].

        Returns:
            self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[valid], y[valid]

        y_fit = np.log1p(np.maximum(y, 0)) if self.log_target else y
        self._model.fit(X, y_fit)
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
        y_pred = self._model.predict(X)
        return np.expm1(y_pred) if self.log_target else y_pred

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        """Return feature importances from the forest."""
        return self._model.feature_importances_ if self._is_fitted else None

    def __repr__(self) -> str:
        return f"RandomForestBaseline(n_estimators={self._model.n_estimators})"


class XGBoostBaseline:
    """
    XGBoost gradient boosted tree regression baseline.

    Uses XGBoost library with CarbonBench default hyperparameters.
    """

    DEFAULT_PARAMS = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }

    def __init__(
        self,
        log_target: bool = True,
        early_stopping_rounds: Optional[int] = 20,
        **kwargs: Any,
    ):
        """
        Args:
            log_target: If True, fit on log1p(y) and return expm1(predictions).
            early_stopping_rounds: Stop if no improvement after N rounds.
            **kwargs: Override default hyperparameters.
        """
        self.log_target = log_target
        self.early_stopping_rounds = early_stopping_rounds
        self._params = {**self.DEFAULT_PARAMS, **kwargs}
        self._model = None
        self._is_fitted = False

    def _build_model(self):
        """Build the XGBoost model (lazy import)."""
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(
                **self._params,
                verbosity=0,
            )
        except ImportError:
            raise ImportError(
                "XGBoost is not installed. Install with: pip install xgboost"
            )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "XGBoostBaseline":
        """
        Fit XGBoost model.

        Args:
            X: Training feature matrix.
            y: Training targets.
            X_val: Optional validation features for early stopping.
            y_val: Optional validation targets for early stopping.

        Returns:
            self
        """
        self._model = self._build_model()

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[valid], y[valid]

        y_fit = np.log1p(np.maximum(y, 0)) if self.log_target else y

        fit_kwargs = {}
        if X_val is not None and y_val is not None and self.early_stopping_rounds:
            X_val = np.asarray(X_val, dtype=float)
            y_val = np.asarray(y_val, dtype=float)
            y_val_fit = np.log1p(np.maximum(y_val, 0)) if self.log_target else y_val

            # XGBoost >=2.0 uses early_stopping_rounds in constructor
            self._model.set_params(early_stopping_rounds=self.early_stopping_rounds)
            fit_kwargs = {
                "eval_set": [(X_val, y_val_fit)],
                "verbose": False,
            }

        self._model.fit(X, y_fit, **fit_kwargs)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict emissions in tCO2e."""
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        X = np.asarray(X, dtype=float)
        y_pred = self._model.predict(X)
        return np.expm1(y_pred) if self.log_target else y_pred

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        """Return feature importances."""
        return self._model.feature_importances_ if self._is_fitted else None

    def __repr__(self) -> str:
        return f"XGBoostBaseline(n_estimators={self._params['n_estimators']})"


class LightGBMBaseline:
    """
    LightGBM gradient boosted tree regression baseline.

    Uses LightGBM library, which is typically faster than XGBoost
    on large datasets while achieving similar accuracy.
    """

    DEFAULT_PARAMS = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    def __init__(
        self,
        log_target: bool = True,
        early_stopping_rounds: Optional[int] = 20,
        **kwargs: Any,
    ):
        """
        Args:
            log_target: If True, fit on log1p(y) and return expm1(predictions).
            early_stopping_rounds: Stop if no improvement after N rounds.
            **kwargs: Override default hyperparameters.
        """
        self.log_target = log_target
        self.early_stopping_rounds = early_stopping_rounds
        self._params = {**self.DEFAULT_PARAMS, **kwargs}
        self._model = None
        self._is_fitted = False

    def _build_model(self):
        """Build the LightGBM model (lazy import)."""
        try:
            from lightgbm import LGBMRegressor
            return LGBMRegressor(**self._params)
        except ImportError:
            raise ImportError(
                "LightGBM is not installed. Install with: pip install lightgbm"
            )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "LightGBMBaseline":
        """
        Fit LightGBM model.

        Args:
            X: Training feature matrix.
            y: Training targets.
            X_val: Optional validation features for early stopping.
            y_val: Optional validation targets for early stopping.

        Returns:
            self
        """
        self._model = self._build_model()

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[valid], y[valid]

        y_fit = np.log1p(np.maximum(y, 0)) if self.log_target else y

        fit_kwargs = {}
        if X_val is not None and y_val is not None and self.early_stopping_rounds:
            X_val = np.asarray(X_val, dtype=float)
            y_val = np.asarray(y_val, dtype=float)
            y_val_fit = np.log1p(np.maximum(y_val, 0)) if self.log_target else y_val

            fit_kwargs = {
                "eval_set": [(X_val, y_val_fit)],
                "callbacks": [],
            }
            try:
                from lightgbm import early_stopping, log_evaluation
                fit_kwargs["callbacks"] = [
                    early_stopping(self.early_stopping_rounds, verbose=False),
                    log_evaluation(period=-1),
                ]
            except ImportError:
                pass

        self._model.fit(X, y_fit, **fit_kwargs)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict emissions in tCO2e."""
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        X = np.asarray(X, dtype=float)
        y_pred = self._model.predict(X)
        return np.expm1(y_pred) if self.log_target else y_pred

    @property
    def feature_importances_(self) -> Optional[np.ndarray]:
        """Return feature importances."""
        return self._model.feature_importances_ if self._is_fitted else None

    def __repr__(self) -> str:
        return f"LightGBMBaseline(n_estimators={self._params['n_estimators']})"
