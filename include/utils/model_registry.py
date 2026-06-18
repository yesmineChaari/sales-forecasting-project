"""Model registry mapping helpers for forecast-level-aware MLflow registration."""

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional


MODEL_REGISTRY_MAP: Dict[str, Dict[str, Any]] = {
    # Current training still logs the store-level tree models under legacy
    # artifact paths. Register them with explicit forecast-level names.
    "xgboost": {
        "artifact_path": "xgboost",
        "registered_name": "xgboost_store_level",
        "forecast_level": "store_level",
        "target_grain": "date+store_id",
    },
    "xgboost_store_level": {
        "artifact_path": "xgboost_store_level",
        "registered_name": "xgboost_store_level",
        "forecast_level": "store_level",
        "target_grain": "date+store_id",
    },
    "lightgbm": {
        "artifact_path": "lightgbm",
        "registered_name": "lightgbm_store_level",
        "forecast_level": "store_level",
        "target_grain": "date+store_id",
    },
    "lightgbm_store_level": {
        "artifact_path": "lightgbm_store_level",
        "registered_name": "lightgbm_store_level",
        "forecast_level": "store_level",
        "target_grain": "date+store_id",
    },
    "ensemble_store_level": {
        "artifact_path": "ensemble",
        "registered_name": "ensemble_store_level",
        "forecast_level": "store_level",
        "target_grain": "date+store_id",
        "ensemble_members": ["xgboost_store_level", "lightgbm_store_level"],
    },
    "prophet_daily_total": {
        "artifact_path": "prophet_daily_total",
        "registered_name": "prophet_daily_total",
        "forecast_level": "daily_total",
        "target_grain": "date",
    },
}


def get_model_registry_entry(
    model_key: str,
    model_results: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Return registry metadata for a trained-model result key."""
    if model_key not in MODEL_REGISTRY_MAP:
        return None

    entry = deepcopy(MODEL_REGISTRY_MAP[model_key])
    model_results = model_results or {}

    for field in ("artifact_path", "forecast_level", "target_grain"):
        result_field = "mlflow_artifact_path" if field == "artifact_path" else field
        if model_results.get(result_field):
            entry[field] = model_results[result_field]

    if model_results.get("ensemble_members"):
        entry["ensemble_members"] = model_results["ensemble_members"]

    return entry


def canonical_model_name(model_key: Optional[str]) -> Optional[str]:
    """Map a result key to its explicit registered model name."""
    if not model_key:
        return None

    entry = get_model_registry_entry(model_key)
    if not entry:
        return model_key

    return entry["registered_name"]


def canonical_model_names(model_keys: Iterable[str]) -> List[str]:
    """Map result keys to explicit model names while preserving order."""
    return [canonical_model_name(model_key) for model_key in model_keys]
