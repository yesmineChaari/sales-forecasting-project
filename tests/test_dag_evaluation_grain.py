from pathlib import Path
import importlib.util
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "include"))


def _load_dag_module():
    decorators = types.ModuleType("airflow.decorators")

    def dag(*_args, **_kwargs):
        def decorator(_func):
            def wrapper(*_call_args, **_call_kwargs):
                return None

            return wrapper

        return decorator

    def task(*_args, **_kwargs):
        def decorator(func):
            return func

        return decorator

    decorators.dag = dag
    decorators.task = task

    airflow = types.ModuleType("airflow")
    operators = types.ModuleType("airflow.operators")
    bash = types.ModuleType("airflow.operators.bash")

    class BashOperator:
        def __init__(self, *_args, **_kwargs):
            pass

        def __rshift__(self, other):
            return other

    bash.BashOperator = BashOperator

    previous_modules = {
        name: sys.modules.get(name)
        for name in [
            "airflow",
            "airflow.decorators",
            "airflow.operators",
            "airflow.operators.bash",
        ]
    }

    sys.modules["airflow"] = airflow
    sys.modules["airflow.decorators"] = decorators
    sys.modules["airflow.operators"] = operators
    sys.modules["airflow.operators.bash"] = bash

    try:
        spec = importlib.util.spec_from_file_location(
            "sales_forecast_train_test_module",
            ROOT / "dags" / "sales_forecast_train.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous_module in previous_modules.items():
            if previous_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module


def test_prophet_daily_total_is_not_compared_with_store_level_rmse():
    dag_module = _load_dag_module()
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

    evaluation = dag_module._evaluate_training_results(training_result)

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
    dag_module = _load_dag_module()
    training_result = {
        "training_results": {
            "xgboost": {"metrics": {"rmse": 50.0}},
            "lightgbm": {"metrics": {"rmse": 40.0}},
            "ensemble_store_level": {"metrics": {"rmse": 45.0}},
            "prophet_daily_total": {"metrics": {"rmse": 0.5}},
        },
    }

    evaluation = dag_module._evaluate_training_results(training_result)

    assert evaluation["best_store_level_model"] == "lightgbm_store_level"
    assert evaluation["best_daily_total_model"] == "prophet_daily_total"
    assert evaluation["models_by_forecast_level"]["daily_total"] == [
        "prophet_daily_total"
    ]


def test_legacy_training_result_keys_are_canonicalized_for_registration():
    dag_module = _load_dag_module()
    canonical_results = dag_module._canonicalize_training_results(
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
    dag_module = _load_dag_module()
    plan, skipped = dag_module._build_model_registration_plan(
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
