from pathlib import Path
import importlib.util
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREDICTOR_PATH = ROOT / "ui" / "utils" / "simple_predictor.py"
spec = importlib.util.spec_from_file_location("ui_simple_predictor", PREDICTOR_PATH)
ui_simple_predictor = importlib.util.module_from_spec(spec)
sys.modules["ui_simple_predictor"] = ui_simple_predictor
spec.loader.exec_module(ui_simple_predictor)

SimplePredictor = ui_simple_predictor.SimplePredictor


class FakeModelLoader:
    loaded = True
    encoders = None
    scalers = None
    feature_cols = [
        "year",
        "month",
        "day",
        "customer_traffic",
        "has_promotion",
        "is_open",
        "is_holiday",
        "school_holiday",
        "competition_distance",
        "promo2",
        "sales_lag_1",
        "sales_rolling_7_mean",
    ]

    def predict(self, X, model_type="ensemble"):
        return np.full(len(X), 100.0)


def _historical_frame():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "store_id": ["store_0001"] * len(dates),
            "sales": np.linspace(100.0, 190.0, len(dates)),
            "customer_traffic": [50] * len(dates),
            "has_promotion": [0, 1] * 5,
            "is_open": [1] * len(dates),
            "is_holiday": [0] * len(dates),
            "school_holiday": [0] * len(dates),
            "state_holiday": ["none"] * len(dates),
            "store_type": ["a"] * len(dates),
            "assortment": ["a"] * len(dates),
            "competition_distance": [500.0] * len(dates),
            "promo2": [0] * len(dates),
            "promo_interval": ["none"] * len(dates),
        }
    )


def _future_frame():
    dates = pd.date_range("2024-01-11", periods=3, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "store_id": ["store_0001"] * len(dates),
            "customer_traffic": [60] * len(dates),
            "has_promotion": [1, 0, 1],
            "is_open": [1] * len(dates),
            "is_holiday": [0] * len(dates),
            "school_holiday": [0] * len(dates),
            "state_holiday": ["none"] * len(dates),
            "store_type": ["a"] * len(dates),
            "assortment": ["a"] * len(dates),
            "competition_distance": [500.0] * len(dates),
            "promo2": [0] * len(dates),
            "promo_interval": ["none"] * len(dates),
        }
    )


def test_predictor_requires_explicit_historical_business_features():
    predictor = SimplePredictor(FakeModelLoader())
    historical_df = _historical_frame().drop(columns=["customer_traffic"])

    result = predictor.predict(
        historical_df,
        forecast_days=3,
        future_features=_future_frame(),
    )

    assert result["success"] is False
    assert "customer_traffic" in result["error"]


def test_predictor_requires_explicit_future_business_features():
    predictor = SimplePredictor(FakeModelLoader())
    future_df = _future_frame().drop(columns=["has_promotion"])

    result = predictor.predict(
        _historical_frame(),
        forecast_days=3,
        future_features=future_df,
    )

    assert result["success"] is False
    assert "has_promotion" in result["error"]


def test_predictor_uses_provided_business_features_for_forecast():
    predictor = SimplePredictor(FakeModelLoader())

    result = predictor.predict(
        _historical_frame(),
        forecast_days=3,
        future_features=_future_frame(),
    )

    assert result["success"] is True
    assert result["predictions"]["predicted_sales"].tolist() == [100.0, 100.0, 100.0]
