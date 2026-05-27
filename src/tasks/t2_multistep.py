"""
T2: Multi-Step Emission Forecasting Task.

Predicts emissions at multiple future horizons simultaneously:
- H=1: prediction for year t+1
- H=2: prediction for year t+2
- H=3: prediction for year t+3

Metrics are reported separately for each horizon to analyze
how performance degrades with forecasting distance.
"""

import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..evaluation.metrics import mae, rmse, mape, r2, nrmse


HORIZONS = [1, 2, 3]


def run_t2(
    model: Any,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    horizons: List[int] = HORIZONS,
    metrics_fn: Optional[Callable] = None,
    feature_cols: Optional[List[str]] = None,
    target_col_prefix: str = "target_h",
    entity_col: str = "entity_id",
    year_col: str = "year",
) -> Dict[str, Any]:
    """
    Run T2: Multi-step emission forecasting.

    The dataset for T2 should have separate target columns for each horizon:
    - target_h1: emission at year t+1
    - target_h2: emission at year t+2
    - target_h3: emission at year t+3

    Args:
        model: Model with fit(X, y) and predict(X) interface.
               For multi-output models, y shape is [n_samples, n_horizons].
               For single-output models, train one model per horizon.
        train_data: Training DataFrame.
        val_data: Validation DataFrame.
        test_data: Test DataFrame.
        horizons: List of prediction horizons to evaluate.
        metrics_fn: Optional custom metrics function.
        feature_cols: Feature columns. If None, uses all non-target columns.
        target_col_prefix: Prefix for target columns (e.g., 'target_h').
        entity_col: Entity identifier column.
        year_col: Year column.

    Returns:
        Dictionary with per-horizon metrics and combined results.
    """
    target_cols = [f"{target_col_prefix}{h}" for h in horizons]

    # Determine feature columns
    if feature_cols is None:
        exclude_cols = {entity_col, year_col} | set(target_cols)
        # Also exclude any single 'target' column if present
        exclude_cols.add("target")
        feature_cols = [c for c in train_data.columns if c not in exclude_cols]

    X_train = train_data[feature_cols].values.astype(float)
    X_val = val_data[feature_cols].values.astype(float)
    X_test = test_data[feature_cols].values.astype(float)

    compute_metrics = metrics_fn if metrics_fn else _horizon_metrics

    results_by_horizon = {}
    all_predictions = {}

    # Check if model supports multi-output (horizon > 1)
    is_multi_output = _is_multi_output_model(model)

    if is_multi_output:
        # Train once, predict all horizons simultaneously
        Y_train = _get_multi_target(train_data, target_cols)
        Y_val = _get_multi_target(val_data, target_cols)
        Y_test = _get_multi_target(test_data, target_cols)

        import inspect
        sig = inspect.signature(model.fit)
        if "X_val" in sig.parameters:
            model.fit(X_train, Y_train, X_val=X_val, y_val=Y_val)
        else:
            model.fit(X_train, Y_train)

        Y_pred_val = np.maximum(model.predict(X_val), 0)
        Y_pred_test = np.maximum(model.predict(X_test), 0)

        if Y_pred_val.ndim == 1:
            Y_pred_val = Y_pred_val.reshape(-1, 1)
        if Y_pred_test.ndim == 1:
            Y_pred_test = Y_pred_test.reshape(-1, 1)

        for i, h in enumerate(horizons):
            col = f"{target_col_prefix}{h}"
            if col not in test_data.columns:
                continue

            y_true_val = val_data[col].values.astype(float)
            y_true_test = test_data[col].values.astype(float)
            y_pred_h_val = Y_pred_val[:, i] if Y_pred_val.shape[1] > i else Y_pred_val[:, -1]
            y_pred_h_test = Y_pred_test[:, i] if Y_pred_test.shape[1] > i else Y_pred_test[:, -1]

            results_by_horizon[h] = {
                "val_metrics": compute_metrics(y_true_val, y_pred_h_val, h),
                "test_metrics": compute_metrics(y_true_test, y_pred_h_test, h),
                "predictions": y_pred_h_test,
                "targets": y_true_test,
            }
            all_predictions[f"h{h}"] = y_pred_h_test

    else:
        # Train one model per horizon
        for h in horizons:
            col = f"{target_col_prefix}{h}"
            if col not in train_data.columns:
                continue

            y_train_h = train_data[col].values.astype(float)
            y_val_h = val_data[col].values.astype(float) if col in val_data.columns else None
            y_test_h = test_data[col].values.astype(float)

            # Clone model for each horizon (create new instance with same params)
            model_h = _clone_model(model)

            import inspect
            sig = inspect.signature(model_h.fit)
            if "X_val" in sig.parameters and y_val_h is not None:
                model_h.fit(X_train, y_train_h, X_val=X_val, y_val=y_val_h)
            else:
                model_h.fit(X_train, y_train_h)

            y_pred_val_h = np.maximum(model_h.predict(X_val), 0) if y_val_h is not None else None
            y_pred_test_h = np.maximum(model_h.predict(X_test), 0)

            results_by_horizon[h] = {
                "val_metrics": compute_metrics(y_val_h, y_pred_val_h, h) if y_val_h is not None else {},
                "test_metrics": compute_metrics(y_test_h, y_pred_test_h, h),
                "predictions": y_pred_test_h,
                "targets": y_test_h,
                "model": model_h,
            }
            all_predictions[f"h{h}"] = y_pred_test_h

    # Aggregate metrics across horizons
    avg_metrics = _aggregate_horizon_metrics(results_by_horizon)

    return {
        "horizons": horizons,
        "results_by_horizon": results_by_horizon,
        "avg_metrics": avg_metrics,
        "predictions": all_predictions,
        "feature_cols": feature_cols,
    }


def _get_multi_target(df: pd.DataFrame, target_cols: List[str]) -> np.ndarray:
    """Extract multi-target matrix, keeping only available columns."""
    available = [c for c in target_cols if c in df.columns]
    return df[available].values.astype(float)


def _is_multi_output_model(model: Any) -> bool:
    """Check if model natively supports multi-output prediction."""
    # Models that support multi-output via horizon parameter
    multi_output_classes = ["LSTMBaseline", "TFTBaseline", "PatchTSTBaseline"]
    model_class = type(model).__name__
    if model_class in multi_output_classes:
        return getattr(model, "horizon", 1) > 1
    return False


def _clone_model(model: Any) -> Any:
    """Create a fresh copy of a model with the same hyperparameters."""
    import copy
    try:
        return copy.deepcopy(model)
    except Exception:
        # Fallback: create new instance with same class and init params
        new_model = type(model).__new__(type(model))
        new_model.__dict__.update(model.__dict__.copy())
        # Reset fitted state
        new_model._is_fitted = False
        if hasattr(new_model, "_model"):
            new_model._model = None
        return new_model


def _horizon_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int = 1,
) -> Dict[str, float]:
    """Compute per-horizon evaluation metrics."""
    if y_true is None or y_pred is None:
        return {}

    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if len(y_true) == 0:
        return {k: np.nan for k in ["mae", "rmse", "mape", "r2", "nrmse"]}

    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "nrmse": nrmse(y_true, y_pred),
        "horizon": horizon,
    }


def _aggregate_horizon_metrics(
    results_by_horizon: Dict[int, Dict],
) -> Dict[str, float]:
    """Compute average metrics across all horizons."""
    all_test_metrics = [
        results["test_metrics"]
        for results in results_by_horizon.values()
        if results.get("test_metrics")
    ]

    if not all_test_metrics:
        return {}

    metric_names = [k for k in all_test_metrics[0].keys() if k != "horizon"]
    avg = {}
    for metric in metric_names:
        values = [m.get(metric, np.nan) for m in all_test_metrics]
        values = [v for v in values if not np.isnan(v)]
        avg[f"avg_{metric}"] = np.mean(values) if values else np.nan

    return avg


class T2Evaluator:
    """
    T2 task evaluator for multi-step forecasting.

    Handles the evaluation pipeline for multiple models across
    multiple forecast horizons.
    """

    def __init__(
        self,
        horizons: List[int] = HORIZONS,
        feature_cols: Optional[List[str]] = None,
        target_col_prefix: str = "target_h",
        entity_col: str = "entity_id",
        year_col: str = "year",
    ):
        self.horizons = horizons
        self.feature_cols = feature_cols
        self.target_col_prefix = target_col_prefix
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
        """Evaluate a single model on T2."""
        results = run_t2(
            model=model,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            horizons=self.horizons,
            feature_cols=self.feature_cols,
            target_col_prefix=self.target_col_prefix,
            entity_col=self.entity_col,
            year_col=self.year_col,
        )
        self.results_[model_name] = results
        return results

    def get_results_dataframe(self) -> pd.DataFrame:
        """Compile per-horizon results into a summary DataFrame."""
        rows = []
        for model_name, results in self.results_.items():
            for h, h_results in results["results_by_horizon"].items():
                row = {"model": model_name, "horizon": h}
                for metric, value in h_results.get("test_metrics", {}).items():
                    if metric != "horizon":
                        row[f"test_{metric}"] = value
                rows.append(row)

        return pd.DataFrame(rows)

    def get_summary_dataframe(self) -> pd.DataFrame:
        """Return averaged metrics across horizons per model."""
        rows = []
        for model_name, results in self.results_.items():
            row = {"model": model_name}
            row.update(results.get("avg_metrics", {}))
            rows.append(row)
        return pd.DataFrame(rows).set_index("model")
