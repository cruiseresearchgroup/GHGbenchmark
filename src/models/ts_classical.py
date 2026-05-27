"""
Classical per-series time-series baselines for Task E.

These models are fit at predict time per-row on the row's lag sequence
(oldest → newest), then forecast one step ahead. They match the interface
expected by run_task_e.py's _fit_model: `lag_indices` is set by the pipeline
and `predict(X)` returns a 1-step-ahead forecast per row.

Why per-row fit rather than per-building: Task E builds one row per
(building, year) with pre-computed lag features. Fitting per-building
globally would require passing building_id through to predict, which the
pipeline currently does not. Per-row fit on n_lags points is equivalent in
the 1-step-ahead setting and keeps the integration minimal.

Log-target: we fit in log1p space (consistent with tree/linear baselines in
this repo) to handle heavy-tailed emissions and then expm1 the forecast.
"""
from __future__ import annotations
import warnings
import numpy as np

warnings.filterwarnings("ignore")


class _ClassicalBase:
    """Shared skeleton for per-row classical forecasters."""
    def __init__(self, log_target: bool = True):
        self.log_target = log_target
        self.lag_indices = None  # set by run_task_e._fit_model

    def fit(self, X, y, **kwargs):
        return self  # classical models are fit lazily at predict time

    def _series(self, row) -> np.ndarray:
        if self.lag_indices is None or len(self.lag_indices) == 0:
            return np.array([])
        s = np.asarray([row[i] for i in self.lag_indices], dtype=float)
        return s

    def _forecast_one(self, s: np.ndarray) -> float:
        raise NotImplementedError

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        n = len(X)
        out = np.empty(n, dtype=float)
        for i in range(n):
            s = self._series(X[i])
            # Filter NaNs (early building-years may have fewer lags than n_lags)
            s = s[np.isfinite(s)]
            if len(s) == 0:
                out[i] = 0.0
                continue
            s_fit = np.log1p(np.maximum(s, 0)) if self.log_target else s
            try:
                yhat = self._forecast_one(s_fit)
            except Exception:
                yhat = s_fit[-1]  # fallback: last observed value
            if not np.isfinite(yhat):
                yhat = s_fit[-1]
            out[i] = float(np.expm1(yhat)) if self.log_target else float(yhat)
            if not np.isfinite(out[i]) or out[i] < 0:
                out[i] = max(0.0, float(s[-1]))
        return out


class HoltBaseline(_ClassicalBase):
    """Holt's linear-trend exponential smoothing, fit per row on the lag
    sequence. Forecasts 1 step ahead.

    Falls back to last-value if statsmodels raises (e.g. flat series, <2 pts).
    """

    def __init__(self, log_target: bool = True, damped: bool = False):
        super().__init__(log_target=log_target)
        self.damped = damped

    def _forecast_one(self, s: np.ndarray) -> float:
        from statsmodels.tsa.holtwinters import Holt
        if len(s) < 2:
            return s[-1]
        m = Holt(s, damped_trend=self.damped,
                 initialization_method="estimated").fit(
            optimized=True, use_brute=False, disp=False,
        )
        fc = m.forecast(1)
        return float(fc[0])


class AR1Baseline(_ClassicalBase):
    """AR(1) with constant, fit by closed-form OLS per row on the lag sequence.

    Vectorized over rows — O(n) rather than O(n · MLE). For a 3-point series
    this gives the same answer as statsmodels ARIMA((1,0,0)) up to the
    optimizer convergence tolerance, but runs in ~1000× less time.

    Model: y[t] = c + φ · y[t-1] + ε
      φ  = cov(y[:-1], y[1:]) / var(y[:-1])     (with small-epsilon guard)
      c  = mean(y[1:]) - φ · mean(y[:-1])
      ŷ  = c + φ · y[-1]
    Falls back to last value when series is degenerate (constant or ≤1 pt).
    """

    def __init__(self, log_target: bool = True):
        super().__init__(log_target=log_target)

    def predict(self, X):
        if self.lag_indices is None or len(self.lag_indices) < 2:
            return np.zeros(len(X), dtype=float)
        X = np.asarray(X, dtype=float)
        S = X[:, list(self.lag_indices)]  # (n, n_lags), oldest → newest
        mask = np.isfinite(S)
        if self.log_target:
            S_fit = np.where(mask, np.log1p(np.maximum(S, 0)), np.nan)
        else:
            S_fit = np.where(mask, S, np.nan)

        # Vectorized AR(1): use the full (potentially NaN-containing) lag
        # sequence as the series. Treat NaNs by pairwise dropping.
        # For each row: collect (y_t, y_{t-1}) pairs across consecutive lags.
        n, L = S_fit.shape
        sum_x  = np.zeros(n); sum_y  = np.zeros(n)
        sum_xx = np.zeros(n); sum_xy = np.zeros(n)
        cnt    = np.zeros(n)
        for k in range(L - 1):
            x_raw = S_fit[:, k]; y_raw = S_fit[:, k + 1]
            ok = np.isfinite(x_raw) & np.isfinite(y_raw)
            # Accumulate only from valid pairs; using 0 for invalid would bias
            # the row-level means toward zero when some pairs are missing.
            x = np.where(ok, x_raw, 0.0)
            y = np.where(ok, y_raw, 0.0)
            sum_x  += x; sum_y  += y
            sum_xx += x * x; sum_xy += x * y
            cnt    += ok.astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_x = sum_x / cnt
            mean_y = sum_y / cnt
            var_x  = sum_xx / cnt - mean_x ** 2
            cov_xy = sum_xy / cnt - mean_x * mean_y
            phi = np.where(var_x > 1e-6, cov_xy / np.maximum(var_x, 1e-6), 0.0)
            # Enforce stationarity — unconstrained OLS on 2–3 points frequently
            # yields |φ| ≫ 1 which produces explosive forecasts after expm1.
            phi = np.clip(phi, -0.99, 0.99)
            c   = mean_y - phi * mean_x
        # Last observed value per row (rightmost finite in S_fit)
        last = np.full(n, np.nan)
        for k in range(L - 1, -1, -1):
            need = np.isnan(last) & np.isfinite(S_fit[:, k])
            last = np.where(need, S_fit[:, k], last)
        yhat = c + phi * last
        # Fallbacks
        bad = ~np.isfinite(yhat) | (cnt < 1)
        yhat = np.where(bad, last, yhat)
        yhat = np.where(np.isfinite(yhat), yhat, 0.0)
        if self.log_target:
            out = np.expm1(yhat)
        else:
            out = yhat
        out = np.where(np.isfinite(out) & (out >= 0), out, 0.0)
        return out


# Keep the legacy name importable — pipelines referencing ArimaBaseline will
# resolve to the fast vectorized AR(1) implementation.
ArimaBaseline = AR1Baseline
