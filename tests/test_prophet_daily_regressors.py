from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "include"))

import ml_models.prophet_daily_total as prophet_daily_total
import ml_models.train_models as train_models_module
from ml_models.train_models import (
    ModelTrainer,
    PROPHET_DAILY_TOTAL_REGRESSORS,
    PROPHET_DAILY_TOTAL_VARIANT,
    get_prophet_variant_regressors,
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


def test_production_prophet_variant_uses_all_regressors():
    assert PROPHET_DAILY_TOTAL_VARIANT == "prophet_all_regressors"
    assert (
        get_prophet_variant_regressors(PROPHET_DAILY_TOTAL_VARIANT)
        == PROPHET_DAILY_TOTAL_REGRESSORS
    )


def test_prepare_prophet_data_preserves_zero_and_closed_rows():
    rows = []
    for day in pd.date_range("2024-01-01", periods=5, freq="D"):
        rows.extend(
            [
                {
                    "date": day,
                    "store_id": "store_0001",
                    "sales": 100.0,
                    "is_open": 1,
                },
                {
                    "date": day,
                    "store_id": "store_0002",
                    "sales": 0.0,
                    "is_open": 0,
                },
            ]
        )

    trainer = object.__new__(ModelTrainer)
    trainer.training_config = {"test_size": 0.2, "validation_size": 0.2}

    train_df, val_df, test_df = trainer.prepare_prophet_data(pd.DataFrame(rows))
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    assert len(combined_df) == len(rows)
    assert ((combined_df["is_open"] == 0) & (combined_df["sales"] == 0.0)).any()

    train_dates = set(train_df["date"].dt.normalize())
    val_dates = set(val_df["date"].dt.normalize())
    test_dates = set(test_df["date"].dt.normalize())

    assert train_dates.isdisjoint(val_dates)
    assert train_dates.isdisjoint(test_dates)
    assert val_dates.isdisjoint(test_dates)


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
    trainer.model_config = {
        "prophet": {
            "selected_variant": "prophet_all_regressors",
            "params": {},
        }
    }
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
    assert result["variant"] == "prophet_all_regressors"
    assert result["regressor_columns"] == PROPHET_DAILY_TOTAL_REGRESSORS
    assert result["input_example"].columns.tolist() == expected_predict_columns


def test_train_all_models_logs_official_prophet_all_regressor_metrics(monkeypatch):
    class FakeMLflowManager:
        def __init__(self):
            self.params = {}
            self.metrics = {}
            self.models = []
            self.ended = []

        def start_run(self, run_name=None, tags=None):
            self.run_name = run_name
            self.start_tags = tags
            return "run-1"

        def log_params(self, params):
            self.params.update(params)

        def log_metrics(self, metrics):
            self.metrics.update(metrics)

        def log_model(self, model, model_name, input_example=None):
            self.models.append(
                {
                    "model_name": model_name,
                    "input_example_columns": input_example.columns.tolist()
                    if input_example is not None
                    else None,
                }
            )

        def end_run(self, status="FINISHED"):
            self.ended.append(status)

    class FakeRegressor:
        def __init__(self, prediction):
            self.prediction = prediction
            self.feature_importances_ = np.array([1.0])

        def predict(self, X):
            return np.full(len(X), self.prediction, dtype=float)

    class FakeProphet:
        instances = []

        def __init__(self, **params):
            self.params = params
            self.regressors = []
            self.fit_columns = None
            self.fit_y = None
            self.predict_columns = []
            FakeProphet.instances.append(self)

        def add_regressor(self, name):
            self.regressors.append(name)

        def fit(self, df):
            self.fit_columns = df.columns.tolist()
            self.fit_y = df["y"].tolist()
            return self

        def predict(self, df):
            self.predict_columns.append(df.columns.tolist())
            return pd.DataFrame({"yhat": np.full(len(df), 300.0)})

    def _store_frame(date_values):
        return pd.DataFrame(
            {
                "date": pd.to_datetime(date_values),
                "store_id": ["store_0001"] * len(date_values),
                "sales": np.linspace(10.0, 10.0 + len(date_values) - 1, len(date_values)),
            }
        )

    def _prophet_frame(date_values):
        rows = []
        for date_value in date_values:
            rows.extend(
                [
                    {
                        "date": pd.Timestamp(date_value),
                        "store_id": "store_0001",
                        "sales": 300.0,
                        "has_promotion": 1,
                        "is_open": 1,
                        "school_holiday": 0,
                        "state_holiday": "none",
                    },
                    {
                        "date": pd.Timestamp(date_value),
                        "store_id": "store_0002",
                        "sales": 0.0,
                        "has_promotion": 0,
                        "is_open": 0,
                        "school_holiday": 1,
                        "state_holiday": "a",
                    },
                ]
            )
        return pd.DataFrame(rows)

    fake_manager = FakeMLflowManager()
    trainer = object.__new__(ModelTrainer)
    trainer.config = {"training": {}}
    trainer.model_config = {
        "xgboost": {"params": {}},
        "lightgbm": {"params": {}},
        "prophet": {
            "enabled": True,
            "selected_variant": "prophet_all_regressors",
            "params": {},
        },
    }
    trainer.training_config = {"test_size": 0.2, "validation_size": 0.2}
    trainer.mlflow_manager = fake_manager
    trainer.models = {}
    trainer.encoders = {}

    def fake_preprocess_features(train_df, val_df, test_df, target_col):
        trainer.feature_cols = ["feature"]
        X_train = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
        X_val = pd.DataFrame({"feature": [4.0, 5.0, 6.0]})
        X_test = pd.DataFrame({"feature": [7.0, 8.0, 9.0]})
        y_train = np.array([10.0, 11.0, 12.0])
        y_val = np.array([13.0, 14.0, 15.0])
        y_test = np.array([16.0, 17.0, 18.0])
        return X_train, X_val, X_test, y_train, y_val, y_test

    def fake_train_xgboost(X_train, y_train, X_val, y_val, use_optuna):
        model = FakeRegressor(16.0)
        trainer.models["xgboost"] = model
        return model

    def fake_train_lightgbm(X_train, y_train, X_val, y_val, use_optuna):
        model = FakeRegressor(17.0)
        trainer.models["lightgbm"] = model
        return model

    captured_tags = {}
    monkeypatch.setattr(trainer, "preprocess_features", fake_preprocess_features)
    monkeypatch.setattr(trainer, "train_xgboost", fake_train_xgboost)
    monkeypatch.setattr(trainer, "train_lightgbm", fake_train_lightgbm)
    monkeypatch.setattr(trainer, "save_artifacts", lambda: None)
    monkeypatch.setattr(prophet_daily_total, "Prophet", FakeProphet)
    monkeypatch.setattr(
        train_models_module,
        "diagnose_model_performance",
        lambda *args, **kwargs: {"recommendations": []},
    )
    monkeypatch.setattr(
        train_models_module.mlflow,
        "active_run",
        lambda: SimpleNamespace(info=SimpleNamespace(run_id="run-1")),
    )
    monkeypatch.setattr(
        train_models_module.mlflow,
        "set_tags",
        lambda tags: captured_tags.update(tags),
    )

    import utils.s3_verification as s3_verification

    monkeypatch.setattr(
        s3_verification,
        "verify_s3_artifacts",
        lambda run_id, expected_artifacts: {
            "success": True,
            "errors": [],
            "missing_artifacts": [],
        },
    )
    monkeypatch.setattr(
        s3_verification,
        "log_s3_verification_results",
        lambda results: None,
    )

    results = trainer.train_all_models(
        _store_frame(["2024-01-01", "2024-01-02", "2024-01-03"]),
        _store_frame(["2024-01-04", "2024-01-05", "2024-01-06"]),
        _store_frame(["2024-01-07", "2024-01-08", "2024-01-09"]),
        target_col="sales",
        use_optuna=False,
        prophet_train_df=_prophet_frame(["2024-01-01", "2024-01-02"]),
        prophet_val_df=_prophet_frame(["2024-01-03"]),
        prophet_test_df=_prophet_frame(["2024-01-04", "2024-01-05"]),
        prophet_uses_full_row_input=True,
    )

    prophet_model = FakeProphet.instances[0]
    expected_fit_columns = ["ds", "y"] + PROPHET_DAILY_TOTAL_REGRESSORS
    expected_predict_columns = ["ds"] + PROPHET_DAILY_TOTAL_REGRESSORS

    assert results["prophet_daily_total"]["variant"] == "prophet_all_regressors"
    assert results["prophet_daily_total"]["forecast_level"] == "daily_total"
    assert results["prophet_daily_total"]["regressor_columns"] == PROPHET_DAILY_TOTAL_REGRESSORS
    assert prophet_model.regressors == PROPHET_DAILY_TOTAL_REGRESSORS
    assert prophet_model.fit_columns == expected_fit_columns
    assert prophet_model.fit_y == [300.0, 300.0]
    assert prophet_model.predict_columns == [expected_predict_columns, expected_predict_columns]
    assert captured_tags["prophet_daily_total_variant"] == "prophet_all_regressors"
    assert captured_tags["prophet_daily_total_uses_full_row_input"] == "true"
    assert captured_tags["prophet_daily_total_comparable_with_store_level_models"] == "false"

    for metric_name in ("mae", "mape", "rmse", "r2"):
        assert f"prophet_daily_total_{metric_name}" in fake_manager.metrics

    prophet_log = [
        model_log
        for model_log in fake_manager.models
        if model_log["model_name"] == "prophet_daily_total"
    ][0]
    assert prophet_log["input_example_columns"] == expected_predict_columns
