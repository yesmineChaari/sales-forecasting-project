from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "include"))

import ml_models.prophet_daily_total as prophet_daily_total
from ml_models.train_models import (
    ModelTrainer,
    PROPHET_DAILY_TOTAL_REGRESSORS,
)


def test_daily_total_frame_adds_rossmann_business_regressors():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-02",
                ]
            ),
            "store_id": [
                "store_0001",
                "store_0002",
                "store_0003",
                "store_0001",
                "store_0002",
            ],
            "sales": [100.0, 200.0, 0.0, 50.0, 70.0],
            "has_promotion": [1, 0, 1, 0, 0],
            "is_open": [1, 1, 0, 1, 1],
            "school_holiday": [0, 1, 0, 0, 0],
            "state_holiday": ["none", "a", "none", "0", "none"],
        }
    )
    trainer = object.__new__(ModelTrainer)

    daily_df = trainer._daily_total_frame(df, date_col="date", target_col="sales")

    assert daily_df.columns.tolist() == ["ds", "y"] + PROPHET_DAILY_TOTAL_REGRESSORS

    first_day = daily_df.iloc[0]
    second_day = daily_df.iloc[1]

    assert first_day["ds"] == pd.Timestamp("2024-01-01")
    assert first_day["y"] == 300.0
    assert first_day["promo_store_count"] == 2
    assert first_day["promo_rate"] == 2 / 3
    assert first_day["open_store_count"] == 2
    assert first_day["school_holiday_rate"] == 1 / 3
    assert first_day["state_holiday_flag"] == 1

    assert second_day["ds"] == pd.Timestamp("2024-01-02")
    assert second_day["y"] == 120.0
    assert second_day["promo_store_count"] == 0
    assert second_day["promo_rate"] == 0
    assert second_day["open_store_count"] == 2
    assert second_day["school_holiday_rate"] == 0
    assert second_day["state_holiday_flag"] == 0


def test_daily_total_frame_defaults_missing_optional_regressors():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "store_id": ["store_0001", "store_0002"],
            "sales": [10.0, 20.0],
        }
    )
    trainer = object.__new__(ModelTrainer)

    daily_df = trainer._daily_total_frame(df, date_col="date", target_col="sales")
    day = daily_df.iloc[0]

    assert day["y"] == 30.0
    assert day["promo_store_count"] == 0
    assert day["promo_rate"] == 0
    assert day["open_store_count"] == 2
    assert day["school_holiday_rate"] == 0
    assert day["state_holiday_flag"] == 0


def test_prophet_daily_total_fit_and_predict_use_regressors():
    class FakeProphet:
        instances = []

        def __init__(self, **params):
            self.params = params
            self.regressors = []
            self.fit_columns = None
            self.predict_columns = []
            FakeProphet.instances.append(self)

        def add_regressor(self, name):
            self.regressors.append(name)

        def fit(self, df):
            self.fit_columns = df.columns.tolist()
            return self

        def predict(self, df):
            self.predict_columns.append(df.columns.tolist())
            return pd.DataFrame({"yhat": [0.0] * len(df)})

    def _frame(date_values):
        rows = []
        for date_value in date_values:
            rows.extend(
                [
                    {
                        "date": pd.Timestamp(date_value),
                        "store_id": "store_0001",
                        "sales": 100.0,
                        "has_promotion": 1,
                        "is_open": 1,
                        "school_holiday": 0,
                        "state_holiday": "none",
                    },
                    {
                        "date": pd.Timestamp(date_value),
                        "store_id": "store_0002",
                        "sales": 200.0,
                        "has_promotion": 0,
                        "is_open": 1,
                        "school_holiday": 1,
                        "state_holiday": "a",
                    },
                ]
            )
        return pd.DataFrame(rows)

    trainer = object.__new__(ModelTrainer)
    trainer.model_config = {"prophet": {"params": {}}}
    trainer.models = {}

    original_prophet = prophet_daily_total.Prophet
    prophet_daily_total.Prophet = FakeProphet
    try:
        result = trainer.train_prophet_daily_total(
            _frame(["2024-01-01", "2024-01-02"]),
            _frame(["2024-01-03"]),
            _frame(["2024-01-04", "2024-01-05"]),
        )
    finally:
        prophet_daily_total.Prophet = original_prophet

    model = FakeProphet.instances[0]
    expected_fit_columns = ["ds", "y"] + PROPHET_DAILY_TOTAL_REGRESSORS
    expected_predict_columns = ["ds"] + PROPHET_DAILY_TOTAL_REGRESSORS

    assert model.regressors == PROPHET_DAILY_TOTAL_REGRESSORS
    assert model.fit_columns == expected_fit_columns
    assert model.predict_columns == [expected_predict_columns, expected_predict_columns]
    assert result["regressor_columns"] == PROPHET_DAILY_TOTAL_REGRESSORS
    assert result["input_example"].columns.tolist() == expected_predict_columns
