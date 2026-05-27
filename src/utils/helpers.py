"""
General utility functions for CarbonBench.
"""

import json
import os
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional, Union


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Sets seeds for Python's random, NumPy, and (if available) PyTorch.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Dictionary with configuration values.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_results(
    results: Dict[str, Any],
    output_path: Union[str, Path],
    format: str = "json",
) -> None:
    """
    Save results dictionary to disk.

    Args:
        results: Dictionary of results to save.
        output_path: Output file path.
        format: File format ('json' or 'parquet' for DataFrames).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        # Convert numpy types to Python native for JSON serialization
        serializable = _make_serializable(results)
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)
    elif format == "parquet":
        if isinstance(results, pd.DataFrame):
            results.to_parquet(output_path, index=False)
        else:
            raise ValueError("Parquet format requires a DataFrame.")
    else:
        raise ValueError(f"Unknown format: {format}. Use 'json' or 'parquet'.")


def load_results(path: Union[str, Path]) -> Any:
    """
    Load saved results from disk.

    Args:
        path: Path to results file.

    Returns:
        Loaded results (dict for JSON, DataFrame for parquet).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        with open(path, "r") as f:
            return json.load(f)
    elif suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    elif suffix == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unknown file format: {suffix}")


def get_model_by_name(name: str, config: Optional[Dict] = None) -> Any:
    """
    Instantiate a model by name with optional configuration.

    Args:
        name: Model name. One of:
              'historical_mean', 'last_year', 'moving_average',
              'linear', 'ridge', 'lasso',
              'random_forest', 'xgboost', 'lightgbm',
              'lstm', 'tft', 'patchtst'
        config: Optional hyperparameter dictionary.

    Returns:
        Instantiated model object.

    Raises:
        ValueError: If model name is unknown.
        ImportError: If required library is not installed.
    """
    config = config or {}
    name_lower = name.lower().replace("-", "_").replace(" ", "_")

    if name_lower in ("historical_mean", "mean"):
        from ..models.naive import HistoricalMeanBaseline
        return HistoricalMeanBaseline(**config)

    elif name_lower in ("last_year", "lastyear", "persistence"):
        from ..models.naive import LastYearBaseline
        return LastYearBaseline(**config)

    elif name_lower in ("moving_average", "ma"):
        from ..models.naive import MovingAverageBaseline
        return MovingAverageBaseline(**config)

    elif name_lower in ("linear", "lr", "linear_regression"):
        from ..models.linear import LinearRegressionBaseline
        return LinearRegressionBaseline(**config)

    elif name_lower in ("ridge",):
        from ..models.linear import RidgeBaseline
        return RidgeBaseline(**config)

    elif name_lower in ("lasso",):
        from ..models.linear import LassoBaseline
        return LassoBaseline(**config)

    elif name_lower in ("random_forest", "rf", "randomforest"):
        from ..models.tree import RandomForestBaseline
        return RandomForestBaseline(**config)

    elif name_lower in ("xgboost", "xgb"):
        from ..models.tree import XGBoostBaseline
        return XGBoostBaseline(**config)

    elif name_lower in ("lightgbm", "lgbm", "lgb"):
        from ..models.tree import LightGBMBaseline
        return LightGBMBaseline(**config)

    elif name_lower in ("lstm",):
        from ..models.lstm import LSTMBaseline
        return LSTMBaseline(**config)

    elif name_lower in ("tft", "temporal_fusion_transformer"):
        from ..models.tft import TFTBaseline
        return TFTBaseline(**config)

    elif name_lower in ("patchtst", "patch_tst"):
        from ..models.patchtst import PatchTSTBaseline
        return PatchTSTBaseline(**config)

    else:
        raise ValueError(
            f"Unknown model: '{name}'. Available models: "
            "historical_mean, last_year, moving_average, linear, ridge, lasso, "
            "random_forest, xgboost, lightgbm, lstm, tft, patchtst"
        )


def format_number(value: float, unit: str = "tCO2e") -> str:
    """
    Format a number with appropriate scale suffix.

    Args:
        value: Numeric value.
        unit: Unit string to append.

    Returns:
        Formatted string (e.g., '1.23M tCO2e', '456K tCO2e').
    """
    if abs(value) >= 1e9:
        return f"{value/1e9:.2f}G {unit}"
    elif abs(value) >= 1e6:
        return f"{value/1e6:.2f}M {unit}"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.2f}K {unit}"
    else:
        return f"{value:.2f} {unit}"


def elapsed_time_str(start_time: float) -> str:
    """
    Format elapsed time since start_time.

    Args:
        start_time: Start time from time.time().

    Returns:
        Formatted string like '2m 34s' or '45s'.
    """
    elapsed = time.time() - start_time
    if elapsed >= 3600:
        h = int(elapsed // 3600)
        m = int((elapsed % 3600) // 60)
        return f"{h}h {m}m"
    elif elapsed >= 60:
        m = int(elapsed // 60)
        s = int(elapsed % 60)
        return f"{m}m {s}s"
    else:
        return f"{elapsed:.1f}s"


def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types for JSON."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj) if np.isfinite(obj) else None
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    else:
        return obj
