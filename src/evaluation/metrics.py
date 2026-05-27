"""
Evaluation metrics for CarbonBench.

All metrics operate on raw tCO2e values (not log-transformed),
to allow fair comparison across models regardless of their internal
transformation strategy.

Metrics implemented:
- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- MAPE: Mean Absolute Percentage Error
- R²: Coefficient of Determination
- NRMSE: Normalized RMSE (normalized by mean of y_true)
- LogMAE: MAE in log space (robust to heavy-tailed distributions)
"""

import numpy as np
from typing import Optional


def _validate_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """Validate and clean inputs, removing NaN/inf pairs."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length. "
            f"Got {len(y_true)} and {len(y_pred)}."
        )

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[valid], y_pred[valid]


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.

    MAE = (1/N) * sum(|y_true - y_pred|)

    Args:
        y_true: Ground truth emission values (tCO2e).
        y_pred: Predicted emission values (tCO2e).

    Returns:
        MAE in tCO2e.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    if len(y_true) == 0:
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.

    RMSE = sqrt((1/N) * sum((y_true - y_pred)^2))

    Args:
        y_true: Ground truth emission values (tCO2e).
        y_pred: Predicted emission values (tCO2e).

    Returns:
        RMSE in tCO2e.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    if len(y_true) == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epsilon: float = 1.0,
) -> float:
    """
    Mean Absolute Percentage Error (%).

    MAPE = (100/N) * sum(|y_true - y_pred| / max(|y_true|, epsilon))

    Uses epsilon to avoid division by zero for near-zero emissions.

    Args:
        y_true: Ground truth emission values (tCO2e).
        y_pred: Predicted emission values (tCO2e).
        epsilon: Small value to avoid division by zero (in tCO2e).

    Returns:
        MAPE as percentage.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    if len(y_true) == 0:
        return np.nan

    # Filter out near-zero true values to avoid inflated MAPE
    nonzero_mask = np.abs(y_true) > epsilon
    if nonzero_mask.sum() == 0:
        return np.nan

    y_true_nz = y_true[nonzero_mask]
    y_pred_nz = y_pred[nonzero_mask]

    return float(100.0 * np.mean(np.abs(y_true_nz - y_pred_nz) / np.abs(y_true_nz)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of Determination (R²).

    R² = 1 - SS_res / SS_tot
    where SS_res = sum((y_true - y_pred)^2)
    and   SS_tot = sum((y_true - mean(y_true))^2)

    R² = 1.0 is perfect prediction.
    R² = 0.0 means the model does no better than predicting the mean.
    R² < 0.0 means the model is worse than predicting the mean.

    Args:
        y_true: Ground truth emission values (tCO2e).
        y_pred: Predicted emission values (tCO2e).

    Returns:
        R² score.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    if len(y_true) == 0:
        return np.nan

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    return float(1.0 - ss_res / ss_tot)


def nrmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalization: str = "mean",
) -> float:
    """
    Normalized Root Mean Squared Error.

    NRMSE = RMSE / normalizer

    where normalizer can be:
    - 'mean': mean of y_true (default)
    - 'range': max(y_true) - min(y_true)
    - 'std': std of y_true

    Using 'mean' normalization makes NRMSE a relative metric
    interpretable as "RMSE as fraction of mean emission."

    Args:
        y_true: Ground truth emission values (tCO2e).
        y_pred: Predicted emission values (tCO2e).
        normalization: Normalization strategy ('mean', 'range', or 'std').

    Returns:
        NRMSE (dimensionless ratio).
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    if len(y_true) == 0:
        return np.nan

    rmse_val = np.sqrt(np.mean((y_true - y_pred) ** 2))

    if normalization == "mean":
        normalizer = np.mean(np.abs(y_true))
    elif normalization == "range":
        normalizer = np.max(y_true) - np.min(y_true)
    elif normalization == "std":
        normalizer = np.std(y_true)
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    if normalizer == 0:
        return np.nan

    return float(rmse_val / normalizer)


def log_mae(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1.0) -> float:
    """
    Mean Absolute Error in log space.

    LogMAE = (1/N) * sum(|log(y_true + 1) - log(y_pred + 1)|)

    This metric is particularly useful for T3 (cold-start estimation) where
    order-of-magnitude differences matter more than absolute values.
    It is robust to the heavy-tailed distribution of emissions.

    Args:
        y_true: Ground truth emission values (tCO2e).
        y_pred: Predicted emission values (tCO2e).
        epsilon: Added to both before log to handle zeros.

    Returns:
        LogMAE (in log-tCO2e units).
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    if len(y_true) == 0:
        return np.nan

    # Clip to avoid negative values
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)

    # log(y + epsilon) with epsilon as floor to handle near-zero values
    log_true = np.log(y_true + epsilon)
    log_pred = np.log(y_pred + epsilon)

    return float(np.mean(np.abs(log_true - log_pred)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error (%).

    sMAPE = (100/N) * sum(2 * |y_true - y_pred| / (|y_true| + |y_pred|))

    More symmetric than MAPE; less sensitive to near-zero true values.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        sMAPE as percentage.
    """
    y_true, y_pred = _validate_inputs(y_true, y_pred)
    if len(y_true) == 0:
        return np.nan

    denominator = np.abs(y_true) + np.abs(y_pred)
    nonzero = denominator > 0
    if nonzero.sum() == 0:
        return 0.0

    return float(
        100.0 * np.mean(
            2.0 * np.abs(y_true[nonzero] - y_pred[nonzero]) / denominator[nonzero]
        )
    )


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> dict:
    """
    Compute all standard CarbonBench metrics in one call.

    Args:
        y_true: Ground truth values (tCO2e).
        y_pred: Predicted values (tCO2e).
        prefix: Optional prefix for metric names (e.g., 'test_').

    Returns:
        Dictionary of metric_name -> value.
    """
    metrics = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "nrmse": nrmse(y_true, y_pred),
        "log_mae": log_mae(y_true, y_pred),
    }
    if prefix:
        return {f"{prefix}{k}": v for k, v in metrics.items()}
    return metrics
