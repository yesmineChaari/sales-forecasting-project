from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "include"))

from utils.model_registry import canonical_model_names, get_model_registry_entry


def test_legacy_training_keys_register_with_explicit_names():
    xgb_entry = get_model_registry_entry("xgboost")
    lgb_entry = get_model_registry_entry("lightgbm")
    ensemble_entry = get_model_registry_entry("ensemble_store_level")

    assert xgb_entry["artifact_path"] == "xgboost"
    assert xgb_entry["registered_name"] == "xgboost_store_level"
    assert lgb_entry["artifact_path"] == "lightgbm"
    assert lgb_entry["registered_name"] == "lightgbm_store_level"
    assert ensemble_entry["artifact_path"] == "ensemble"
    assert ensemble_entry["registered_name"] == "ensemble_store_level"


def test_model_results_can_override_artifact_path_without_renaming_registry_target():
    entry = get_model_registry_entry(
        "ensemble_store_level",
        {"mlflow_artifact_path": "ensemble_store_level"},
    )

    assert entry["artifact_path"] == "ensemble_store_level"
    assert entry["registered_name"] == "ensemble_store_level"


def test_canonical_model_names_preserve_forecast_level_names():
    assert canonical_model_names(
        ["xgboost", "lightgbm", "ensemble_store_level", "prophet_daily_total"]
    ) == [
        "xgboost_store_level",
        "lightgbm_store_level",
        "ensemble_store_level",
        "prophet_daily_total",
    ]
