"""
T1: Next-Year Emission Prediction Task.

The primary CarbonBench task: given historical emissions and static features
for an entity (facility, building, or company), predict total GHG emissions
for the next year.

Input:  historical emissions (up to 10 years) + static features
Output: scalar emission for year t+1 (in tCO2e)

Evaluation metrics: MAE, RMSE, MAPE, R², NRMSE (all in original tCO2e scale)
"""

import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..evaluation.metrics import mae, rmse, mape, r2, nrmse, log_mae


def run_t1(
    model: Any,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    metrics_fn: Optional[Callable] = None,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "target",
    entity_col: str = "entity_id",
    year_col: str = "year",
    use_val_for_early_stopping: bool = True,
) -> Dict[str, Any]:
    """
    Run T1: Next-year emission prediction.

    Args:
        model: Any model with fit(X, y) and predict(X) interface.
        train_data: Training DataFrame with features and target column.
        val_data: Validation DataFrame (used for early stopping if supported).
        test_data: Test DataFrame for final evaluation.
        metrics_fn: Optional custom metrics function. If None, uses default metrics.
        feature_cols: Columns to use as features. If None, uses all columns except
                      entity_col, year_col, and target_col.
        target_col: Name of the target column.
        entity_col: Name of the entity identifier column.
        year_col: Name of the year column.
        use_val_for_early_stopping: If True and model supports it, pass validation
                                     data to fit() for early stopping.

    Returns:
        Dictionary with keys:
        - 'test_metrics': dict of metric_name -> value
        - 'val_metrics': dict of metric_name -> value
        - 'predictions': array of test predictions
        - 'targets': array of test ground truth
        - 'model': the fitted model
    """
    # Determine feature columns
    if feature_cols is None:
        exclude_cols = {entity_col, year_col, target_col}
        feature_cols = [c for c in train_data.columns if c not in exclude_cols]

    X_train = train_data[feature_cols].values.astype(float)
    y_train = train_data[target_col].values.astype(float)

    X_val = val_data[feature_cols].values.astype(float)
    y_val = val_data[target_col].values.astype(float)

    X_test = test_data[feature_cols].values.astype(float)
    y_test = test_data[target_col].values.astype(float)

    # Train model (with optional early stopping via validation data)
    if use_val_for_early_stopping and hasattr(model, "fit"):
        import inspect
        sig = inspect.signature(model.fit)
        if "X_val" in sig.parameters:
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        else:
            model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)

    # Generate predictions
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    # Clip negative predictions (emissions cannot be negative)
    y_pred_val = np.maximum(y_pred_val, 0.0)
    y_pred_test = np.maximum(y_pred_test, 0.0)

    # Compute metrics
    compute_metrics = metrics_fn if metrics_fn else _default_metrics

    val_metrics = compute_metrics(y_val, y_pred_val)
    test_metrics = compute_metrics(y_test, y_pred_test)

    return {
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "predictions": y_pred_test,
        "targets": y_test,
        "model": model,
        "feature_cols": feature_cols,
    }


def _default_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute default T1 evaluation metrics."""
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if len(y_true) == 0:
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "r2": np.nan,
                "nrmse": np.nan, "log_mae": np.nan}

    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "nrmse": nrmse(y_true, y_pred),
        "log_mae": log_mae(y_true, y_pred),
    }


class T1Evaluator:
    """
    T1 task evaluator for running and comparing multiple models.

    Manages the full evaluation pipeline: feature extraction,
    train/val/test splitting, model fitting, and metric computation.
    """

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        target_col: str = "target",
        entity_col: str = "entity_id",
        year_col: str = "year",
    ):
        """
        Args:
            feature_cols: Feature columns to use.
            target_col: Target column name.
            entity_col: Entity identifier column.
            year_col: Year column name.
        """
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.entity_col = entity_col
        self.year_col = year_col
        self.results_: Dict[str, Dict] = {}

    def evaluate_model(
        self,
        model_name: str,
        model: Any,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Evaluate a single model on T1.

        Args:
            model_name: Name for logging/results.
            model: Model object with fit/predict interface.
            train_data: Training data.
            val_data: Validation data.
            test_data: Test data.

        Returns:
            Results dictionary.
        """
        results = run_t1(
            model=model,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            feature_cols=self.feature_cols,
            target_col=self.target_col,
            entity_col=self.entity_col,
            year_col=self.year_col,
        )
        self.results_[model_name] = results
        return results

    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Compile all results into a summary DataFrame.

        Returns:
            DataFrame with model names as index and metrics as columns.
        """
        rows = []
        for model_name, results in self.results_.items():
            row = {"model": model_name}
            for metric, value in results["test_metrics"].items():
                row[f"test_{metric}"] = value
            for metric, value in results["val_metrics"].items():
                row[f"val_{metric}"] = value
            rows.append(row)

        return pd.DataFrame(rows).set_index("model")

    def get_best_model(self, metric: str = "test_rmse") -> Tuple[str, Any]:
        """
        Return the name and fitted model with best performance on `metric`.

        Args:
            metric: Metric to rank by (lower is better).

        Returns:
            Tuple of (model_name, model_object).
        """
        best_name = None
        best_val = float("inf")

        for name, results in self.results_.items():
            split, metric_name = metric.split("_", 1)
            metrics_dict = results.get(f"{split}_metrics", {})
            val = metrics_dict.get(metric_name, float("inf"))
            if val < best_val:
                best_val = val
                best_name = name

        return best_name, self.results_[best_name]["model"] if best_name else None
