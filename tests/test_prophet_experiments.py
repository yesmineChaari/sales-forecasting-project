from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "include"))

from ml_models.prophet_daily_total import PROPHET_DAILY_TOTAL_REGRESSORS
from ml_models.prophet_experiments import (
    BASELINE_VARIANTS,
    PROPHET_VARIANTS,
    run_prophet_experiments,
    select_best_variant,
    summarize_prophet_experiment_results,
)


class FakeProphet:
    def __init__(self, **params):
        self.params = params
        self.regressors = []
        self.fit_df = None

    def add_regressor(self, name):
        self.regressors.append(name)

    def fit(self, df):
        self.fit_df = df.copy()
        return self

    def predict(self, df):
        ds = pd.to_datetime(df["ds"])
        day_index = (ds - ds.min()).dt.days.astype(float)
        yhat = np.repeat(float(self.fit_df["y"].mean()), len(df))
        yhat = yhat + day_index * 0.25

        for index, regressor_col in enumerate(self.regressors, start=1):
            yhat = yhat + pd.to_numeric(df[regressor_col], errors="coerce").fillna(0) * index

        return pd.DataFrame({"yhat": yhat})


def _daily_frame(start_date, periods):
    dates = pd.date_range(start_date, periods=periods, freq="D")
    day_index = np.arange(periods)
    weekly = (day_index % 7) * 4.0
    promo_rate = ((day_index % 10) < 3).astype(float)
    open_store_count = 8 + ((day_index % 6) != 0).astype(int)
    school_holiday_rate = ((day_index % 20) == 0).astype(float)
    state_holiday_flag = ((day_index % 31) == 0).astype(float)

    return pd.DataFrame(
        {
            "ds": dates,
            "y": (
                500.0
                + weekly
                + promo_rate * 20.0
                + open_store_count * 3.0
                - school_holiday_rate * 15.0
                - state_holiday_flag * 25.0
            ),
            "promo_store_count": promo_rate * 3.0,
            "promo_rate": promo_rate,
            "open_store_count": open_store_count,
            "school_holiday_rate": school_holiday_rate,
            "state_holiday_flag": state_holiday_flag,
        }
    )


def test_prophet_experiments_include_required_variants_and_splits():
    full_df = _daily_frame("2024-01-01", 70)
    train_df = full_df.iloc[:45]
    val_df = full_df.iloc[45:55]
    test_df = full_df.iloc[55:]

    result = run_prophet_experiments(
        train_df,
        val_df,
        test_df,
        prophet_factory=FakeProphet,
    )

    expected_variants = set(BASELINE_VARIANTS) | set(PROPHET_VARIANTS)
    assert "seasonal_naive_7" in set(result["variant"])
    assert "prophet_univariate" in set(result["variant"])
    assert "prophet_all_regressors" in set(result["variant"])
    assert set(result["variant"]) == expected_variants

    for variant in expected_variants:
        variant_splits = set(result[result["variant"] == variant]["split"])
        assert variant_splits == {"validation", "test"}

    assert result.columns.tolist() == [
        "variant",
        "split",
        "mae",
        "rmse",
        "mape",
        "r2",
        "beats_seasonal_naive_7",
    ]


def test_prophet_experiment_metrics_are_numeric_and_best_variant_is_selectable():
    full_df = _daily_frame("2024-02-01", 80)
    train_df = full_df.iloc[:50]
    val_df = full_df.iloc[50:62]
    test_df = full_df.iloc[62:]

    result = run_prophet_experiments(
        train_df,
        val_df,
        test_df,
        prophet_factory=FakeProphet,
    )

    metric_values = result[["mae", "rmse", "mape", "r2"]].to_numpy()
    assert np.issubdtype(metric_values.dtype, np.number)
    assert np.isfinite(metric_values).all()

    best_variant = select_best_variant(result, split="test")
    verdict = summarize_prophet_experiment_results(result)

    assert best_variant in set(result["variant"])
    assert verdict["best_variant"] == best_variant
    assert result.attrs["verdict"]["best_variant"] == best_variant
    assert isinstance(verdict["prophet_beats_seasonal_naive_7"], bool)
    assert isinstance(verdict["regressors_improve_over_prophet_univariate"], bool)


def test_prophet_experiments_compare_only_daily_total_variants():
    full_df = _daily_frame("2024-03-01", 75)
    result = run_prophet_experiments(
        full_df.iloc[:45],
        full_df.iloc[45:60],
        full_df.iloc[60:],
        prophet_factory=FakeProphet,
    )

    store_level_variants = {
        "xgboost",
        "xgboost_store_level",
        "lightgbm",
        "lightgbm_store_level",
        "ensemble",
        "ensemble_store_level",
    }
    expected_daily_total_variants = set(BASELINE_VARIANTS) | set(PROPHET_VARIANTS)

    assert set(result["variant"]).isdisjoint(store_level_variants)
    assert set(result["variant"]).issubset(expected_daily_total_variants)
    assert set(PROPHET_DAILY_TOTAL_REGRESSORS).issubset(full_df.columns)
