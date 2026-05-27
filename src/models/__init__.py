"""Baseline model implementations for CarbonBench."""
from .naive import HistoricalMeanBaseline, LastYearBaseline
from .linear import LinearRegressionBaseline, RidgeBaseline, LassoBaseline
from .tree import RandomForestBaseline, XGBoostBaseline, LightGBMBaseline

__all__ = [
    "HistoricalMeanBaseline",
    "LastYearBaseline",
    "LinearRegressionBaseline",
    "RidgeBaseline",
    "LassoBaseline",
    "RandomForestBaseline",
    "XGBoostBaseline",
    "LightGBMBaseline",
]

# Optional deep learning models (may not be available)
try:
    from .lstm import LSTMBaseline
    __all__.append("LSTMBaseline")
except ImportError:
    pass

try:
    from .tft import TFTBaseline
    __all__.append("TFTBaseline")
except ImportError:
    pass

try:
    from .patchtst import PatchTSTBaseline
    __all__.append("PatchTSTBaseline")
except ImportError:
    pass
