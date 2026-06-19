from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "include"))

from pipeline.model_results import (
    build_model_registration_plan,
    canonicalize_training_results,
    evaluate_training_results,
    serializable_training_results,
)
from pipeline.model_registration import (
    register_trained_models,
    transition_registered_models,
)


def test_prophet_daily_total_is_not_compared_with_store_level_rmse():
    training_result = {
        "training_results": {
            "xgboost": {
                "forecast_level": "store_level",
                "metrics": {"rmse": 100.0},
            },
            "lightgbm": {
                "forecast_level": "store_level",
                "metrics": {"rmse": 90.0},
            },
            "ensemble_store_level": {
                "forecast_level": "store_level",
                "metrics": {"rmse": 80.0},
            },
            "prophet_daily_total": {
                "forecast_level": "daily_total",
                "metrics": {"rmse": 1.0},
            },
        },
        "mlflow_run_id": "run-1",
    }

    evaluation = evaluate_training_results(training_result)

    assert evaluation["best_store_level_model"] == "ensemble_store_level"
    assert evaluation["best_store_level_rmse"] == 80.0
    assert evaluation["best_daily_total_model"] == "prophet_daily_total"
    assert evaluation["best_daily_total_rmse"] == 1.0
    assert evaluation["best_model"] == "ensemble_store_level"
    assert "prophet_daily_total" not in evaluation["store_level_selection_candidates"]
    assert evaluation["trained_models"] == [
        "xgboost_store_level",
        "lightgbm_store_level",
        "ensemble_store_level",
        "prophet_daily_total",
    ]


def test_prophet_daily_total_is_daily_total_even_without_metadata():
    training_result = {
        "training_results": {
            "xgboost": {"metrics": {"rmse": 50.0}},
            "lightgbm": {"metrics": {"rmse": 40.0}},
            "ensemble_store_level": {"metrics": {"rmse": 45.0}},
            "prophet_daily_total": {"metrics": {"rmse": 0.5}},
        },
    }

    evaluation = evaluate_training_results(training_result)

    assert evaluation["best_store_level_model"] == "lightgbm_store_level"
    assert evaluation["best_daily_total_model"] == "prophet_daily_total"
    assert evaluation["models_by_forecast_level"]["daily_total"] == [
        "prophet_daily_total"
    ]


def test_legacy_training_result_keys_are_canonicalized_for_registration():
    canonical_results = canonicalize_training_results(
        {
            "xgboost": {"metrics": {"rmse": 100.0}},
            "lightgbm": {"metrics": {"rmse": 90.0}},
            "ensemble_store_level": {
                "metrics": {"rmse": 80.0},
                "mlflow_artifact_path": "ensemble",
            },
            "prophet_daily_total": {"metrics": {"rmse": 70.0}},
        }
    )

    assert list(canonical_results.keys()) == [
        "xgboost_store_level",
        "lightgbm_store_level",
        "ensemble_store_level",
        "prophet_daily_total",
    ]
    assert canonical_results["xgboost_store_level"]["source_model_key"] == "xgboost"
    assert canonical_results["xgboost_store_level"]["mlflow_artifact_path"] == "xgboost"
    assert canonical_results["lightgbm_store_level"]["source_model_key"] == "lightgbm"
    assert (
        canonical_results["lightgbm_store_level"]["mlflow_artifact_path"]
        == "lightgbm"
    )
    assert canonical_results["ensemble_store_level"]["mlflow_artifact_path"] == "ensemble"


def test_registration_plan_uses_registry_mapping_for_all_models():
    plan, skipped = build_model_registration_plan(
        {
            "xgboost": {"metrics": {"rmse": 100.0}},
            "lightgbm": {"metrics": {"rmse": 90.0}},
            "ensemble_store_level": {"metrics": {"rmse": 80.0}},
            "prophet_daily_total": {"metrics": {"rmse": 70.0}},
        },
        [
            "xgboost",
            "lightgbm",
            "ensemble_store_level",
            "prophet_daily_total",
        ],
        best_store_level_model="ensemble_store_level",
        best_daily_total_model="prophet_daily_total",
    )

    assert skipped == {}
    assert [
        (item["registered_name"], item["artifact_path"])
        for item in plan
    ] == [
        ("xgboost_store_level", "xgboost"),
        ("lightgbm_store_level", "lightgbm"),
        ("ensemble_store_level", "ensemble"),
        ("prophet_daily_total", "prophet_daily_total"),
    ]
    assert [item["forecast_level"] for item in plan] == [
        "store_level",
        "store_level",
        "store_level",
        "daily_total",
    ]
    assert [item["target_grain"] for item in plan] == [
        "date+store_id",
        "date+store_id",
        "date+store_id",
        "date",
    ]
    assert plan[2]["is_best_store_level"] is True
    assert plan[2]["recommended_stage"] == "Production"
    assert plan[3]["is_best_daily_total"] is True
    assert plan[3]["recommended_stage"] == "Production"


def test_training_results_are_serialized_with_model_metadata():
    results = serializable_training_results(
        {
            "prophet_daily_total": {
                "metrics": {"rmse": 1.0},
                "forecast_level": "daily_total",
                "target_grain": "date",
                "regressor_columns": ["promo_rate"],
            }
        }
    )

    assert results["prophet_daily_total"]["metrics"] == {"rmse": 1.0}
    assert results["prophet_daily_total"]["forecast_level"] == "daily_total"
    assert results["prophet_daily_total"]["target_grain"] == "date"
    assert results["prophet_daily_total"]["regressor_columns"] == ["promo_rate"]
    assert (
        results["prophet_daily_total"]["mlflow_artifact_path"]
        == "prophet_daily_total"
    )


def test_model_registration_workflow_uses_canonical_names_and_stages():
    class FakeMLflowManager:
        def __init__(self):
            self.run_tags = None
            self.registered = []
            self.version_tags = []
            self.transitions = []

        def set_run_tags(self, run_id, tags):
            self.run_tags = (run_id, tags)

        def register_model(self, run_id, registered_name, artifact_path):
            self.registered.append((run_id, registered_name, artifact_path))
            return len(self.registered)

        def set_model_version_tags(self, model_name, version, tags):
            self.version_tags.append((model_name, version, tags))

        def transition_model_stage(self, model_name, version, stage):
            self.transitions.append((model_name, version, stage))

    fake_manager = FakeMLflowManager()
    registration = register_trained_models(
        {
            "training_results": {
                "xgboost": {"metrics": {"rmse": 10.0}},
                "prophet_daily_total": {
                    "forecast_level": "daily_total",
                    "metrics": {"rmse": 1.0},
                },
            },
            "mlflow_run_id": "run-1",
        },
        {
            "best_run_id": "run-1",
            "best_store_level_model": "xgboost_store_level",
            "best_store_level_rmse": 10.0,
            "best_daily_total_model": "prophet_daily_total",
            "best_daily_total_rmse": 1.0,
            "trained_models": ["xgboost_store_level", "prophet_daily_total"],
        },
        mlflow_manager=fake_manager,
    )
    transition = transition_registered_models(
        registration,
        mlflow_manager=fake_manager,
    )

    assert fake_manager.run_tags[0] == "run-1"
    assert fake_manager.run_tags[1]["best_store_level_model"] == "xgboost_store_level"
    assert fake_manager.registered == [
        ("run-1", "xgboost_store_level", "xgboost"),
        ("run-1", "prophet_daily_total", "prophet_daily_total"),
    ]
    assert registration["production_models"] == [
        "xgboost_store_level",
        "prophet_daily_total",
    ]
    assert fake_manager.transitions == [
        ("xgboost_store_level", 1, "Production"),
        ("prophet_daily_total", 2, "Production"),
    ]
    assert transition["production_models"] == [
        "xgboost_store_level",
        "prophet_daily_total",
    ]
