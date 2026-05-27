"""
Task E: Short-horizon building-year forecasting.

This task predicts future annual building emissions using:
  - historical emissions (lag features),
  - historical climate (lagged HDD/CDD / annual climate channels), and
  - static or known-ahead building/context features from the selected feature set.

We intentionally DO NOT use contemporaneous climate of the forecast year.
If a feature set contains climate columns, Task E replaces them with lagged
climate versions so the protocol remains a genuine forecasting task rather
than a current-year regression with future weather leakage.

Subtasks
--------
E1_forecasting:
  Train on years <= 2019, validate on 2020, test on years >= 2021.

E2_covid_forecast:
  Train on years <= 2018, then evaluate separately on 2019 / 2020 / 2021.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import time
import warnings
from typing import List

import numpy as np
import pandas as pd

from src.data.preprocessing import load_and_prepare, encode_categoricals
from src.data.feature_sets import TARGET, CATEGORICAL_COLS, get_feature_set
from src.data.splitters import forecasting_split
from src.evaluation.metrics import compute_all_metrics

from src.models.t2_baselines import GlobalMeanBaseline, CityTypeMeanBaseline
from src.models.mlp_gpu import TorchMLPBaseline
from src.models.linear import RidgeBaseline
from src.models.tree import RandomForestBaseline, XGBoostBaseline, LightGBMBaseline
from src.models.ts_foundation import ChronosBaseline, TimesFMBaseline, MoiraiBaseline
from src.models.ts_classical import HoltBaseline

warnings.filterwarnings('ignore')

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

CLIMATE_FEATURE_COLS = {
    'hdd', 'cdd',
    'annual_mean_temp_c', 'annual_rh_mean',
    'annual_ssrd_mj_m2_day', 'annual_wind_ms',
}


def _flush(results, out_path):
    if not results:
        return
    pd.DataFrame(results).to_csv(out_path, index=False)


def _known_base_cols(feature_set_name: str, df: pd.DataFrame) -> List[str]:
    """
    Keep contemporaneous features that are known ahead of the forecast year.

    For Task E we intentionally drop current-year climate channels and replace
    them with lagged climate features.
    """
    feat_cols = get_feature_set(feature_set_name)
    keep = [c for c in feat_cols if c in df.columns and c not in CLIMATE_FEATURE_COLS]
    return keep


def _climate_cols_for_lags(feature_set_name: str, df: pd.DataFrame) -> List[str]:
    feat_cols = get_feature_set(feature_set_name)
    return [c for c in feat_cols if c in df.columns and c in CLIMATE_FEATURE_COLS]


def build_lag_features(df, climate_cols=None, group_col='building_id', target_col=TARGET, n_lags=3):
    """
    Add lagged target and lagged climate features per building.
    """
    climate_cols = climate_cols or []
    df = df.sort_values([group_col, 'year']).copy()

    # Emission lags
    for lag in range(1, n_lags + 1):
        df[f'ghg_lag{lag}'] = df.groupby(group_col)[target_col].shift(lag)

    df['ghg_mean_hist'] = df.groupby(group_col)[target_col].transform(
        lambda x: x.shift(1).rolling(window=n_lags, min_periods=1).mean()
    )
    df['ghg_std_hist'] = df.groupby(group_col)[target_col].transform(
        lambda x: x.shift(1).rolling(window=n_lags, min_periods=1).std()
    ).fillna(0)
    df['ghg_change_1y'] = df['ghg_lag1'] - df['ghg_lag2']

    # Climate lags
    for col in climate_cols:
        for lag in range(1, n_lags + 1):
            df[f'{col}_lag{lag}'] = df.groupby(group_col)[col].shift(lag)

    return df


def get_lag_feature_cols(climate_cols, n_lags):
    cols = [f'ghg_lag{lag}' for lag in range(1, n_lags + 1)]
    cols += ['ghg_mean_hist', 'ghg_std_hist', 'ghg_change_1y']
    for col in climate_cols:
        cols += [f'{col}_lag{lag}' for lag in range(1, n_lags + 1)]
    return cols


class LastValueBaseline:
    """Predict next year with the previous year's observed emissions."""

    def __init__(self):
        self.lag1_idx = None

    def fit(self, X, y, **kwargs):
        if self.lag1_idx is None:
            self.lag1_idx = 0
        return self

    def predict(self, X):
        return X[:, self.lag1_idx]


class LinearTrendBaseline:
    """Predict next year via 2 * lag1 - lag2."""

    def __init__(self):
        self.lag1_idx = None
        self.lag2_idx = None

    def fit(self, X, y, **kwargs):
        if self.lag1_idx is None:
            self.lag1_idx = 0
        if self.lag2_idx is None:
            self.lag2_idx = 1
        return self

    def predict(self, X):
        return 2 * X[:, self.lag1_idx] - X[:, self.lag2_idx]


def get_model_zoo():
    return {
        'LastValue': lambda: LastValueBaseline(),
        'LinearTrend': lambda: LinearTrendBaseline(),
        'GlobalMean': lambda: GlobalMeanBaseline(),
        'CityTypeMean': lambda: CityTypeMeanBaseline(),
        'Ridge': lambda: RidgeBaseline(alpha=1.0),
        'RandomForest': lambda: RandomForestBaseline(n_estimators=200, max_depth=10, n_jobs=8),
        'XGBoost': lambda: XGBoostBaseline(n_estimators=300, max_depth=6, n_jobs=8),
        'LightGBM': lambda: LightGBMBaseline(n_estimators=300, max_depth=6, n_jobs=8),
        'MLP': lambda: TorchMLPBaseline(),
        'Chronos': lambda: ChronosBaseline(),
        'TimesFM': lambda: TimesFMBaseline(),
        'Moirai': lambda: MoiraiBaseline(),
        'Holt': lambda: HoltBaseline(damped=False),
        'DampedHolt': lambda: HoltBaseline(damped=True),
    }


TS_FOUNDATION_MODELS = {'Chronos', 'TimesFM', 'Moirai'}
TS_CLASSICAL_MODELS = {'Holt', 'DampedHolt'}


def _prepare_citytype_features(train_df, test_df, val_df=None):
    """
    Forecasting variant of CityTypeMean:
    still only uses (city, property_type), intentionally ignoring lag features.
    """
    base_cols = ['city', 'property_type', TARGET]
    tr = train_df[base_cols].copy()
    te = test_df[base_cols].copy()
    va = val_df[base_cols].copy() if val_df is not None and len(val_df) > 0 else None

    tr, encoders = encode_categoricals(tr, columns=['city', 'property_type'], encoders=None)
    te, _ = encode_categoricals(te, columns=['city', 'property_type'], encoders=encoders)
    if va is not None:
        va, _ = encode_categoricals(va, columns=['city', 'property_type'], encoders=encoders)

    X_train = tr[['city', 'property_type']].values.astype(np.float64)
    y_train = tr[TARGET].values.astype(np.float64)
    X_test = te[['city', 'property_type']].values.astype(np.float64)
    y_test = te[TARGET].values.astype(np.float64)

    valid_tr = ~np.isnan(y_train)
    valid_te = ~np.isnan(y_test)
    X_train, y_train = X_train[valid_tr], y_train[valid_tr]
    X_test, y_test = X_test[valid_te], y_test[valid_te]

    X_val = y_val = None
    if va is not None:
        X_val = va[['city', 'property_type']].values.astype(np.float64)
        y_val = va[TARGET].values.astype(np.float64)
        valid_val = ~np.isnan(y_val)
        X_val, y_val = X_val[valid_val], y_val[valid_val]

    return X_train, y_train, X_test, y_test, X_val, y_val


def prepare_forecasting_features(
    df, feature_set_name, n_lags,
    encoders=None, medians=None, lag_feature_cols=None, base_feature_cols=None,
):
    """
    Prepare forecasting features:
      contemporaneous known-ahead base features
      + lagged emissions
      + lagged climate (if selected feature set includes climate columns)
    """
    base_feature_cols = base_feature_cols or _known_base_cols(feature_set_name, df)
    lag_feature_cols = lag_feature_cols or []
    feat_cols = [c for c in list(base_feature_cols) + list(lag_feature_cols) if c in df.columns]

    cat_cols = [c for c in feat_cols if c in CATEGORICAL_COLS]
    num_cols = [c for c in feat_cols if c not in CATEGORICAL_COLS]

    work = df[feat_cols + [TARGET]].copy()
    work, encoders = encode_categoricals(work, columns=cat_cols, encoders=encoders)

    # Capture lag1 nullness BEFORE median imputation so the forecasting filter
    # is real. After fillna below, X[:, lag1_idx] can never be NaN.
    lag1_valid = (
        work['ghg_lag1'].notna().values if 'ghg_lag1' in feat_cols else None
    )

    if medians is None:
        medians = {}
        for col in num_cols:
            med = work[col].median()
            medians[col] = 0.0 if pd.isna(med) else med

    for col in num_cols:
        med = medians.get(col, 0.0)
        if pd.isna(med):
            med = 0.0
        work[col] = work[col].fillna(med)

    X = work[feat_cols].values.astype(np.float64)
    y = work[TARGET].values.astype(np.float64)

    valid = ~np.isnan(y)
    if lag1_valid is not None:
        valid = valid & lag1_valid

    X = X[valid]
    y = y[valid]
    return X, y, feat_cols, encoders, medians, valid


def _clip_predictions(y_pred, y_train):
    upper = np.nanmax(y_train) * 2 if len(y_train) else None
    if upper is None or np.isnan(upper) or upper <= 0:
        return np.clip(y_pred, 0, None)
    return np.clip(y_pred, 0, upper)


def _fit_model(
    model_name, model_fn, feature_set_name,
    train_df, val_df=None,
    n_lags=3, base_feature_cols=None, lag_feature_cols=None,
):
    """Fit a model once. Returns a dict with everything needed to predict later."""
    if len(train_df) < 2:
        return None

    if model_name == 'CityTypeMean':
        X_train, y_train, _, _, X_val, y_val = _prepare_citytype_features(
            train_df, train_df.iloc[:0], val_df=val_df
        )
        feat_names = ['city', 'property_type']
        encoders = medians = None
    else:
        X_train, y_train, feat_names, encoders, medians, _ = prepare_forecasting_features(
            train_df, feature_set_name, n_lags,
            encoders=None, medians=None,
            lag_feature_cols=lag_feature_cols, base_feature_cols=base_feature_cols,
        )
        X_val = y_val = None
        if val_df is not None and len(val_df) > 0:
            X_val, y_val, _, _, _, _ = prepare_forecasting_features(
                val_df, feature_set_name, n_lags,
                encoders=encoders, medians=medians,
                lag_feature_cols=lag_feature_cols, base_feature_cols=base_feature_cols,
            )

    if len(X_train) < 2:
        return None

    model = model_fn()
    if isinstance(model, LastValueBaseline) and 'ghg_lag1' in feat_names:
        model.lag1_idx = feat_names.index('ghg_lag1')
    if isinstance(model, LinearTrendBaseline) and 'ghg_lag1' in feat_names and 'ghg_lag2' in feat_names:
        model.lag1_idx = feat_names.index('ghg_lag1')
        model.lag2_idx = feat_names.index('ghg_lag2')

    # Foundation and classical TS models need lag column indices (oldest → newest)
    if model_name in TS_FOUNDATION_MODELS or model_name in TS_CLASSICAL_MODELS:
        lag_idxs = []
        for lag in range(n_lags, 0, -1):
            col = f'ghg_lag{lag}'
            if col in feat_names:
                lag_idxs.append(feat_names.index(col))
        model.lag_indices = lag_idxs

    supports_val = model_name in ('XGBoost', 'LightGBM', 'MLP')
    if supports_val and X_val is not None and len(X_val) > 0:
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    else:
        model.fit(X_train, y_train)

    return {
        'model': model,
        'model_name': model_name,
        'feature_set_name': feature_set_name,
        'feat_names': feat_names,
        'encoders': encoders,
        'medians': medians,
        'y_train': y_train,
        'n_train': len(X_train),
        'n_lags': n_lags,
        'base_feature_cols': base_feature_cols,
        'lag_feature_cols': lag_feature_cols,
        'train_df_for_citytype': train_df if model_name == 'CityTypeMean' else None,
    }


def _predict_with_fit(fit, test_df):
    """Predict on test_df using a previously fit model. Returns (y_test, y_pred, valid_mask)."""
    if fit is None or len(test_df) == 0:
        return np.array([]), np.array([]), np.zeros(len(test_df), dtype=bool)

    model_name = fit['model_name']
    if model_name == 'CityTypeMean':
        _, _, X_test, y_test, _, _ = _prepare_citytype_features(
            fit['train_df_for_citytype'], test_df
        )
        valid = ~np.isnan(test_df[TARGET].values)
    else:
        X_test, y_test, _, _, _, valid = prepare_forecasting_features(
            test_df, fit['feature_set_name'], fit['n_lags'],
            encoders=fit['encoders'], medians=fit['medians'],
            lag_feature_cols=fit['lag_feature_cols'],
            base_feature_cols=fit['base_feature_cols'],
        )

    if len(X_test) == 0:
        return np.array([]), np.array([]), valid

    y_pred = _clip_predictions(fit['model'].predict(X_test), fit['y_train'])
    return y_test, y_pred, valid


def run_task_e1(df_with_lags, feature_set_name, model_name, model_fn, min_years=4, n_lags=3):
    """
    E1: train <= 2019, val = 2020, test >= 2021.

    Fits the model ONCE on the global train set, then reuses it to predict on
    both the full test set and per-city / per-horizon slices.
    """
    splits = forecasting_split(
        df_with_lags, min_years=min_years, train_end=2019, val_year=2020, test_start=2021
    )
    train_df = splits['train']
    val_df = splits['val']
    test_df = splits['test']

    if len(train_df) == 0 or len(test_df) == 0:
        return {'task': 'E1_forecasting', 'error': 'insufficient data'}

    climate_cols = _climate_cols_for_lags(feature_set_name, df_with_lags)
    base_feature_cols = _known_base_cols(feature_set_name, df_with_lags)
    lag_feature_cols = get_lag_feature_cols(climate_cols, n_lags)

    fit = _fit_model(
        model_name, model_fn, feature_set_name,
        train_df, val_df=val_df, n_lags=n_lags,
        base_feature_cols=base_feature_cols, lag_feature_cols=lag_feature_cols,
    )
    if fit is None:
        return {'task': 'E1_forecasting', 'error': 'no valid rows after feature prep'}

    y_test, y_pred, valid_mask = _predict_with_fit(fit, test_df)
    if len(y_test) == 0:
        return {'task': 'E1_forecasting', 'error': 'empty test after feature prep'}

    overall = compute_all_metrics(y_test, y_pred)

    test_valid = test_df.iloc[np.where(valid_mask)[0]].reset_index(drop=True).copy()
    if len(test_valid) != len(y_pred):
        return {
            'task': 'E1_forecasting',
            'error': f'valid mask mismatch: {len(test_valid)} vs {len(y_pred)}',
        }
    test_valid['__y_test'] = y_test
    test_valid['__y_pred'] = y_pred

    city_metrics = {}
    for city in sorted(test_valid['city'].unique()):
        sub = test_valid[test_valid['city'] == city]
        if len(sub) < 5:
            continue
        city_metrics[city] = compute_all_metrics(
            sub['__y_test'].values, sub['__y_pred'].values
        )

    macro = {}
    for m in ['mae', 'rmse', 'r2', 'mape', 'log_mae']:
        vals = [city_metrics[c][m] for c in city_metrics
                if not np.isnan(city_metrics[c].get(m, np.nan))]
        macro[f'macro_{m}'] = np.mean(vals) if vals else np.nan

    horizon_metrics = {}
    for yr in sorted(test_valid['year'].unique()):
        sub = test_valid[test_valid['year'] == yr]
        if len(sub) < 5:
            continue
        horizon_metrics[int(yr)] = compute_all_metrics(
            sub['__y_test'].values, sub['__y_pred'].values
        )

    result = {
        'task': 'E1_forecasting',
        'n_train': fit['n_train'],
        'n_test': len(y_test),
        'n_buildings': len(splits['building_ids']),
        'n_lags': n_lags,
        'n_climate_lag_channels': len(climate_cols),
        **{f'overall_{k}': v for k, v in overall.items()},
        **macro,
    }
    for yr, hm in horizon_metrics.items():
        result[f'r2_{yr}'] = hm.get('r2', np.nan)
        result[f'mae_{yr}'] = hm.get('mae', np.nan)
    for city, cm in city_metrics.items():
        result[f'r2_{city}'] = cm.get('r2', np.nan)
    return result


def run_task_e2(df_with_lags, feature_set_name, model_name, model_fn, min_years=4, n_lags=3):
    """E2: Train <= 2018, evaluate separately on 2019 / 2020 / 2021."""
    splits = forecasting_split(
        df_with_lags, min_years=min_years, train_end=2018, val_year=None, test_start=2019
    )
    train_df = splits['train']
    test_all = splits['test']

    if len(train_df) == 0 or len(test_all) == 0:
        return {'task': 'E2_covid_forecast', 'error': 'insufficient data'}

    climate_cols = _climate_cols_for_lags(feature_set_name, df_with_lags)
    base_feature_cols = _known_base_cols(feature_set_name, df_with_lags)
    lag_feature_cols = get_lag_feature_cols(climate_cols, n_lags)

    fit = _fit_model(
        model_name, model_fn, feature_set_name,
        train_df, val_df=None, n_lags=n_lags,
        base_feature_cols=base_feature_cols, lag_feature_cols=lag_feature_cols,
    )
    if fit is None:
        return {'task': 'E2_covid_forecast', 'error': 'no valid rows after feature prep'}

    result = {
        'task': 'E2_covid_forecast',
        'n_train': fit['n_train'],
        'n_buildings': len(splits['building_ids']),
        'n_lags': n_lags,
        'n_climate_lag_channels': len(climate_cols),
    }

    for year in [2019, 2020, 2021]:
        year_df = test_all[test_all['year'] == year]
        if len(year_df) == 0:
            continue
        y_test, y_pred, _ = _predict_with_fit(fit, year_df)
        if len(y_test) < 5:
            continue
        metrics = compute_all_metrics(y_test, y_pred)
        result[f'mae_{year}'] = metrics.get('mae', np.nan)
        result[f'r2_{year}'] = metrics.get('r2', np.nan)
        result[f'n_test_{year}'] = len(y_test)

    return result


def main():
    parser = argparse.ArgumentParser(description='Task E: Short-horizon forecasting')
    parser.add_argument(
        '--feature_sets',
        type=str,
        default='core_all_cities,core_all_cities_climate_plus',
    )
    parser.add_argument('--models', type=str, default=None)
    parser.add_argument('--subtasks', type=str, default='all',
                        help='Comma-separated among E1,E2 or all')
    parser.add_argument('--min_years', type=int, default=4,
                        help='Minimum years of data per building')
    parser.add_argument('--n_lags', type=int, default=3,
                        help='Number of historical years to expose as lag features')
    parser.add_argument('--out_path', type=str, default=os.path.join(RESULTS_DIR, 'task_e_results.csv'))
    args = parser.parse_args()

    feature_sets = [x for x in args.feature_sets.split(',') if x]
    model_zoo = get_model_zoo()
    model_names = args.models.split(',') if args.models else list(model_zoo.keys())
    subtasks = ['E1', 'E2'] if args.subtasks == 'all' else [x for x in args.subtasks.split(',') if x]

    print("=" * 70)
    print("Task E: Short-Horizon Building-Year Forecasting")
    print(f"Feature sets: {feature_sets}")
    print(f"Models: {model_names}")
    print(f"Subtasks: {subtasks}")
    print(f"Min years per building: {args.min_years}")
    print(f"Lag depth: {args.n_lags}")
    print(f"Output: {args.out_path}")
    print("=" * 70)

    all_results = []

    for fs_name in feature_sets:
        join_external = any(col in CLIMATE_FEATURE_COLS for col in get_feature_set(fs_name))
        df = load_and_prepare(fs_name, join_external=join_external)
        df = df[df[TARGET].notna()].copy()

        climate_cols = _climate_cols_for_lags(fs_name, df)
        base_feature_cols = _known_base_cols(fs_name, df)
        df = build_lag_features(df, climate_cols=climate_cols, n_lags=args.n_lags)

        eligible = df.groupby('building_id')['year'].nunique()
        n_eligible = int((eligible >= args.min_years).sum())
        lag1_cov = df['ghg_lag1'].notna().mean() * 100 if 'ghg_lag1' in df.columns else 0.0

        print("\n" + "=" * 50)
        print(f"Feature set: {fs_name}")
        print("=" * 50)
        print(f"Total rows with target: {len(df)}")
        print(f"Cities ({df['city'].nunique()}): {sorted(df['city'].unique())}")
        print(f"Buildings with >={args.min_years} years: {n_eligible}")
        print(f"Base known-ahead features: {base_feature_cols}")
        print(f"Climate lag channels: {climate_cols}")
        print(f"Lag1 coverage: {lag1_cov:.1f}%")

        for model_name in model_names:
            model_fn = model_zoo[model_name]

            if 'E1' in subtasks:
                t0 = time.time()
                e1 = run_task_e1(
                    df, fs_name, model_name, model_fn,
                    min_years=args.min_years, n_lags=args.n_lags,
                )
                e1['feature_set'] = fs_name
                e1['model'] = model_name
                all_results.append(e1)
                _flush(all_results, args.out_path)
                elapsed = time.time() - t0
                print(
                    f"  [E1] {model_name:15s} | {fs_name:28s} | "
                    f"overall_R²={e1.get('overall_r2', np.nan):6.3f} | "
                    f"macro_R²={e1.get('macro_r2', np.nan):6.3f} | "
                    f"MAE={e1.get('overall_mae', np.nan):8.1f} | "
                    f"{elapsed:.1f}s"
                )

            if 'E2' in subtasks:
                t0 = time.time()
                e2 = run_task_e2(
                    df, fs_name, model_name, model_fn,
                    min_years=args.min_years, n_lags=args.n_lags,
                )
                e2['feature_set'] = fs_name
                e2['model'] = model_name
                all_results.append(e2)
                _flush(all_results, args.out_path)
                elapsed = time.time() - t0
                print(
                    f"  [E2] {model_name:15s} | {fs_name:28s} | "
                    f"R²_2019={e2.get('r2_2019', np.nan):6.3f} | "
                    f"R²_2020={e2.get('r2_2020', np.nan):6.3f} | "
                    f"R²_2021={e2.get('r2_2021', np.nan):6.3f} | "
                    f"{elapsed:.1f}s"
                )

    print(f"\nResults saved to {args.out_path}")


if __name__ == '__main__':
    main()
