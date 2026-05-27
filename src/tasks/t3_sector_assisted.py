"""
T3: Sector-Classification-Assisted Emission Estimation Task.

This task addresses cold-start prediction: estimating emissions for entities
with NO historical emission data. The only available information is:
- Static features (size, location, sector if known)
- Sector classification (may be unknown and must be inferred)

Pipeline:
1. Sector Classifier: Predict entity sector from available features
2. Sector-Specific Regressor: Estimate emissions given (predicted) sector

Baselines:
- SectorMeanBaseline: predict mean emission of the true/predicted sector
- SectorMedianBaseline: predict median emission of the true/predicted sector
- ClassifierPlusRegressor: train sector classifier, then sector-specific regressors
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from ..evaluation.metrics import mae, rmse, mape, r2, nrmse, log_mae


def run_t3(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    model_type: str = "sector_mean",
    sector_col: str = "sector",
    target_col: str = "emissions_tco2e",
    feature_cols: Optional[List[str]] = None,
    val_data: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Run T3: Sector-classification-assisted emission estimation.

    Args:
        train_data: Training data with sector labels and emissions.
        test_data: Test data (no historical emissions; sector may be unknown).
        model_type: One of:
                    - 'sector_mean': predict mean emission of true sector
                    - 'sector_median': predict median of true sector
                    - 'classifier_regressor': sector classifier + regressor pipeline
                    - 'oracle_sector': use true sector labels (upper bound)
        sector_col: Column name for sector labels.
        target_col: Column name for emission values.
        feature_cols: Feature columns for classifier/regressor.
        val_data: Optional validation set.

    Returns:
        Dictionary with test_metrics, predictions, targets, and model info.
    """
    if model_type == "sector_mean":
        pipeline = SectorMeanPipeline(sector_col=sector_col, target_col=target_col, use_median=False)
    elif model_type == "sector_median":
        pipeline = SectorMeanPipeline(sector_col=sector_col, target_col=target_col, use_median=True)
    elif model_type in ("classifier_regressor", "oracle_sector"):
        use_oracle = (model_type == "oracle_sector")
        pipeline = ClassifierRegressorPipeline(
            sector_col=sector_col,
            target_col=target_col,
            feature_cols=feature_cols,
            use_oracle_sector=use_oracle,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    pipeline.fit(train_data)
    y_pred = pipeline.predict(test_data)
    y_true = test_data[target_col].values.astype(float)

    y_pred = np.maximum(y_pred, 0.0)

    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_eval = y_true[valid]
    y_pred_eval = y_pred[valid]

    test_metrics = _compute_t3_metrics(y_true_eval, y_pred_eval)

    # Compute sector classification accuracy for classifier_regressor pipeline
    # This is the diagnostic that reveals how sector inference quality drives
    # downstream emission estimation quality.
    sector_accuracy = np.nan
    if (
        model_type == "classifier_regressor"
        and isinstance(pipeline, ClassifierRegressorPipeline)
        and pipeline._classifier is not None
        and sector_col in test_data.columns
    ):
        try:
            X_test_feat = pipeline._get_features(test_data)
            true_sectors = test_data[sector_col].astype(str).values
            # Only evaluate on sectors seen during training
            known_sectors = set(pipeline._sector_encoder.classes_)
            valid_sector_mask = np.array([s in known_sectors for s in true_sectors])
            if valid_sector_mask.sum() > 0:
                pred_codes = pipeline._classifier.predict(X_test_feat[valid_sector_mask])
                pred_sectors = np.array([
                    pipeline._sector_encoder.classes_[c]
                    if 0 <= c < len(pipeline._sector_encoder.classes_) else "unknown"
                    for c in pred_codes
                ])
                sector_accuracy = float(
                    np.mean(pred_sectors == true_sectors[valid_sector_mask])
                )
        except Exception:
            sector_accuracy = np.nan

    test_metrics["sector_classification_accuracy"] = sector_accuracy

    result = {
        "test_metrics": test_metrics,
        "predictions": y_pred,
        "targets": y_true,
        "pipeline": pipeline,
        "model_type": model_type,
    }

    if val_data is not None:
        y_pred_val = np.maximum(pipeline.predict(val_data), 0.0)
        y_true_val = val_data[target_col].values.astype(float)
        valid_val = ~(np.isnan(y_true_val) | np.isnan(y_pred_val))
        val_metrics = _compute_t3_metrics(
            y_true_val[valid_val], y_pred_val[valid_val]
        )
        val_metrics["sector_classification_accuracy"] = np.nan  # not computed for val
        result["val_metrics"] = val_metrics

    return result


def _compute_t3_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute T3 evaluation metrics."""
    if len(y_true) == 0:
        return {k: np.nan for k in ["mae", "rmse", "mape", "r2", "nrmse", "log_mae"]}

    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "nrmse": nrmse(y_true, y_pred),
        "log_mae": log_mae(y_true, y_pred),
    }


class SectorMeanPipeline:
    """
    T3 baseline: predict the mean (or median) emission for the entity's sector.

    Requires true sector labels. Used as a lower-bound baseline showing
    what's achievable with perfect sector knowledge but no regression.
    """

    def __init__(
        self,
        sector_col: str = "sector",
        target_col: str = "emissions_tco2e",
        use_median: bool = False,
    ):
        self.sector_col = sector_col
        self.target_col = target_col
        self.use_median = use_median
        self._sector_stats: Dict[str, float] = {}
        self._global_stat: float = 0.0
        self._is_fitted = False

    def fit(self, train_data: pd.DataFrame) -> "SectorMeanPipeline":
        """Compute per-sector mean/median from training data."""
        df = train_data.copy()
        df[self.target_col] = pd.to_numeric(df[self.target_col], errors="coerce")
        df = df.dropna(subset=[self.target_col, self.sector_col])
        df = df[df[self.target_col] > 0]

        agg_fn = np.median if self.use_median else np.mean
        self._sector_stats = (
            df.groupby(self.sector_col)[self.target_col]
            .agg(agg_fn)
            .to_dict()
        )
        self._global_stat = agg_fn(df[self.target_col].values)
        self._is_fitted = True
        return self

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """Predict sector mean/median for each test entity."""
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted.")

        sectors = test_data[self.sector_col].astype(str).values
        predictions = np.array([
            self._sector_stats.get(s, self._global_stat)
            for s in sectors
        ])
        return predictions

    def __repr__(self) -> str:
        stat = "median" if self.use_median else "mean"
        return f"SectorMeanPipeline(stat={stat})"


class ClassifierRegressorPipeline:
    """
    T3 pipeline: sector classifier + sector-specific emission regressor.

    Step 1: Train a sector classifier from entity features.
    Step 2: For each sector, train a regressor to predict emissions from features.
    Step 3: At test time, predict sector (or use oracle), then predict emissions.

    This models the real-world scenario where sector may not be known
    and must be inferred from observable entity characteristics.
    """

    def __init__(
        self,
        sector_col: str = "sector",
        target_col: str = "emissions_tco2e",
        feature_cols: Optional[List[str]] = None,
        use_oracle_sector: bool = False,
        classifier_params: Optional[Dict] = None,
        regressor_params: Optional[Dict] = None,
    ):
        """
        Args:
            sector_col: Column with sector labels.
            target_col: Column with emission values.
            feature_cols: Features for classifier and regressors.
            use_oracle_sector: If True, use ground truth sector (oracle upper bound).
            classifier_params: Parameters for RandomForestClassifier.
            regressor_params: Parameters for RandomForestRegressor.
        """
        self.sector_col = sector_col
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.use_oracle_sector = use_oracle_sector

        self._classifier_params = classifier_params or {
            "n_estimators": 100, "max_depth": 8, "random_state": 42, "n_jobs": -1
        }
        self._regressor_params = regressor_params or {
            "n_estimators": 100, "max_depth": 8, "random_state": 42, "n_jobs": -1
        }

        self._sector_encoder = LabelEncoder()
        self._classifier = None
        self._sector_regressors: Dict[str, Any] = {}
        self._global_regressor = None
        self._available_feature_cols: List[str] = []
        self._global_mean: float = 0.0
        self._is_fitted = False

    def _get_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract feature matrix, filling missing values."""
        if not self._available_feature_cols:
            return np.zeros((len(df), 1))

        available = [c for c in self._available_feature_cols if c in df.columns]
        X = df[available].values.astype(float)
        # Fill NaN with column means
        col_means = np.nanmean(X, axis=0)
        col_means = np.where(np.isnan(col_means), 0, col_means)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
        return X

    def fit(self, train_data: pd.DataFrame) -> "ClassifierRegressorPipeline":
        """
        Train the sector classifier and per-sector emission regressors.

        IMPORTANT: The sector column is EXCLUDED from classifier features to
        prevent leakage — the whole point of T3 is that sector is unknown and
        must be inferred. The oracle_sector mode bypasses the classifier and
        uses the ground truth sector directly at prediction time.
        """
        df = train_data.copy()
        df[self.target_col] = pd.to_numeric(df[self.target_col], errors="coerce")
        df = df.dropna(subset=[self.target_col, self.sector_col])
        df = df[df[self.target_col] > 0]

        # Determine feature columns — ALWAYS exclude sector_col to prevent leakage.
        # The classifier must infer sector from observable features only.
        if self.feature_cols:
            # Filter out sector col even if explicitly provided
            self._available_feature_cols = [
                c for c in self.feature_cols
                if c in df.columns and c != self.sector_col
            ]
        else:
            # Auto-select numeric columns, excluding sector and target
            exclude = {self.sector_col, self.target_col}
            self._available_feature_cols = [
                c for c in df.columns
                if c not in exclude
                and df[c].dtype in [np.float64, np.int64, float, int]
            ]

        self._global_mean = np.mean(df[self.target_col].values)

        # Encode sectors
        sectors = df[self.sector_col].astype(str).values
        sector_codes = self._sector_encoder.fit_transform(sectors)

        X = self._get_features(df)

        # Train sector classifier (if not using oracle)
        if not self.use_oracle_sector and len(X[0]) > 0:
            self._classifier = RandomForestClassifier(**self._classifier_params)
            self._classifier.fit(X, sector_codes)

        # Train per-sector regressors
        y = df[self.target_col].values
        y_log = np.log1p(y)

        unique_sectors = df[self.sector_col].astype(str).unique()
        for sector in unique_sectors:
            mask = df[self.sector_col].astype(str) == sector
            X_s = X[mask]
            y_s = y_log[mask]

            if len(X_s) >= 5:
                reg = RandomForestRegressor(**self._regressor_params)
                reg.fit(X_s, y_s)
                self._sector_regressors[sector] = reg

        # Global fallback regressor
        if len(X) >= 5:
            self._global_regressor = RandomForestRegressor(**self._regressor_params)
            self._global_regressor.fit(X, y_log)

        self._is_fitted = True
        return self

    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """
        Predict emissions for test entities.

        Uses oracle sector or predicted sector to select appropriate regressor.
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline not fitted.")

        X_test = self._get_features(test_data)
        n = len(test_data)

        # Determine sectors for test entities
        if self.use_oracle_sector and self.sector_col in test_data.columns:
            predicted_sectors = test_data[self.sector_col].astype(str).values
        elif self._classifier is not None and X_test.shape[1] > 0:
            sector_code_preds = self._classifier.predict(X_test)
            # Handle unseen codes gracefully
            valid_codes = set(range(len(self._sector_encoder.classes_)))
            predicted_sectors = []
            for code in sector_code_preds:
                if code in valid_codes:
                    predicted_sectors.append(self._sector_encoder.classes_[code])
                else:
                    predicted_sectors.append(self._sector_encoder.classes_[0])
            predicted_sectors = np.array(predicted_sectors)
        else:
            # No classifier available: predict global mean
            return np.full(n, self._global_mean)

        # Predict emissions using per-sector regressors
        predictions = np.zeros(n)
        for i, (sector, x_row) in enumerate(zip(predicted_sectors, X_test)):
            x = x_row.reshape(1, -1)
            if sector in self._sector_regressors:
                log_pred = self._sector_regressors[sector].predict(x)[0]
                predictions[i] = np.expm1(log_pred)
            elif self._global_regressor is not None:
                log_pred = self._global_regressor.predict(x)[0]
                predictions[i] = np.expm1(log_pred)
            else:
                predictions[i] = self._global_mean

        return predictions

    @property
    def sector_classifier_accuracy_(self) -> Optional[float]:
        """Return classifier training accuracy (for diagnostics)."""
        return None  # Would need train data; computed externally

    def __repr__(self) -> str:
        mode = "oracle" if self.use_oracle_sector else "predicted"
        return f"ClassifierRegressorPipeline(sector={mode})"


class T3Evaluator:
    """
    T3 task evaluator for sector-classification-assisted estimation.
    """

    def __init__(
        self,
        sector_col: str = "sector",
        target_col: str = "emissions_tco2e",
        feature_cols: Optional[List[str]] = None,
    ):
        self.sector_col = sector_col
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.results_: Dict[str, Dict] = {}

    def evaluate_model(
        self,
        model_name: str,
        model_type: str,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        val_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Evaluate a T3 pipeline configuration."""
        results = run_t3(
            train_data=train_data,
            test_data=test_data,
            model_type=model_type,
            sector_col=self.sector_col,
            target_col=self.target_col,
            feature_cols=self.feature_cols,
            val_data=val_data,
        )
        self.results_[model_name] = results
        return results

    def get_results_dataframe(self) -> pd.DataFrame:
        """Compile all results into a summary DataFrame."""
        rows = []
        for model_name, results in self.results_.items():
            row = {"model": model_name}
            for metric, value in results["test_metrics"].items():
                row[f"test_{metric}"] = value
            rows.append(row)
        return pd.DataFrame(rows).set_index("model")
