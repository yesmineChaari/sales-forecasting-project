"""Forecast-level-aware helpers for training-result metadata."""

from utils.model_registry import canonical_model_name, get_model_registry_entry


STORE_LEVEL_MODEL_SELECTION_KEYS = (
    "xgboost_store_level",
    "lightgbm_store_level",
    "ensemble_store_level",
)
DAILY_TOTAL_MODEL_SELECTION_KEYS = ("prophet_daily_total",)

STORE_LEVEL_MODEL_KEYS = {
    "xgboost",
    "xgboost_store_level",
    "lightgbm",
    "lightgbm_store_level",
    "ensemble_store_level",
}
DAILY_TOTAL_MODEL_KEYS = {"prophet_daily_total"}


def models_by_forecast_level(results):
    models_by_level = {
        "store_level": [],
        "daily_total": [],
    }

    for model_name, model_results in results.items():
        forecast_level = model_results.get("forecast_level")
        if forecast_level == "daily_total" or model_name in DAILY_TOTAL_MODEL_KEYS:
            models_by_level["daily_total"].append(model_name)
        elif forecast_level == "store_level" or model_name in STORE_LEVEL_MODEL_KEYS:
            models_by_level["store_level"].append(model_name)

    return models_by_level


def canonicalize_training_results(results):
    canonical_results = {}

    for model_key, model_results in results.items():
        model_results = dict(model_results or {})
        registry_entry = get_model_registry_entry(model_key, model_results)
        if not registry_entry:
            canonical_results[model_key] = model_results
            continue

        registered_name = registry_entry["registered_name"]
        model_results["registered_model_name"] = registered_name
        model_results["source_model_key"] = model_results.get(
            "source_model_key",
            model_key,
        )
        model_results["mlflow_artifact_path"] = registry_entry["artifact_path"]

        if not model_results.get("forecast_level"):
            model_results["forecast_level"] = registry_entry.get("forecast_level")
        if not model_results.get("target_grain"):
            model_results["target_grain"] = registry_entry.get("target_grain")
        if registry_entry.get("ensemble_members") and not model_results.get(
            "ensemble_members"
        ):
            model_results["ensemble_members"] = registry_entry["ensemble_members"]

        canonical_results[registered_name] = model_results

    return canonical_results


def serializable_training_results(results):
    serializable_results = {}

    for model_name, model_results in results.items():
        serializable_results[model_name] = {
            "metrics": model_results.get("metrics", {}),
            "forecast_level": model_results.get("forecast_level", "store_level"),
            "target_grain": model_results.get("target_grain", "date+store_id"),
            "included_in_store_ensemble": model_results.get(
                "included_in_store_ensemble",
                model_name == "ensemble_store_level",
            ),
            "ensemble_members": model_results.get("ensemble_members", []),
            "regressor_columns": model_results.get("regressor_columns", []),
            "mlflow_artifact_path": model_results.get(
                "mlflow_artifact_path",
                model_name,
            ),
        }
        if model_results.get("legacy_model_key"):
            serializable_results[model_name]["source_model_key"] = model_results[
                "legacy_model_key"
            ]

    return canonicalize_training_results(serializable_results)


def build_model_registration_plan(
    training_results,
    trained_models=None,
    best_store_level_model=None,
    best_daily_total_model=None,
):
    canonical_results = canonicalize_training_results(training_results)
    model_keys = trained_models or list(canonical_results.keys())
    best_store_level_model = canonical_model_name(best_store_level_model)
    best_daily_total_model = canonical_model_name(best_daily_total_model)

    registration_plan = []
    skipped_models = {}
    seen_registered_models = set()

    for model_key in model_keys:
        canonical_key = canonical_model_name(model_key)
        model_results = (
            canonical_results.get(model_key)
            or canonical_results.get(canonical_key)
            or {}
        )
        registry_entry = (
            get_model_registry_entry(model_key, model_results)
            or get_model_registry_entry(canonical_key, model_results)
        )
        if not registry_entry:
            skipped_models[model_key] = "no registry mapping"
            continue

        registered_name = registry_entry["registered_name"]
        if registered_name in seen_registered_models:
            skipped_models[model_key] = (
                f"duplicate registration target {registered_name}"
            )
            continue

        seen_registered_models.add(registered_name)
        is_best_store_level = registered_name == best_store_level_model
        is_best_daily_total = registered_name == best_daily_total_model
        forecast_level = registry_entry.get("forecast_level")

        recommended_stage = None
        if is_best_store_level or is_best_daily_total:
            recommended_stage = "Production"
        elif forecast_level == "store_level":
            recommended_stage = "Staging"

        registration_plan.append(
            {
                "source_result_key": model_key,
                "canonical_result_key": canonical_key,
                "registered_name": registered_name,
                "artifact_path": registry_entry["artifact_path"],
                "forecast_level": forecast_level,
                "target_grain": registry_entry.get("target_grain"),
                "is_best_store_level": is_best_store_level,
                "is_best_daily_total": is_best_daily_total,
                "recommended_stage": recommended_stage,
                "ensemble_members": registry_entry.get("ensemble_members", []),
            }
        )

    return registration_plan, skipped_models


def select_best_model_by_rmse(results, model_names):
    best_model_name = None
    best_rmse = float("inf")

    for model_name in model_names:
        model_results = results.get(model_name, {})
        metrics = model_results.get("metrics", {})
        rmse = metrics.get("rmse")
        if rmse is not None and rmse < best_rmse:
            best_rmse = rmse
            best_model_name = model_name

    if best_model_name is None:
        return None, None

    return best_model_name, best_rmse


def evaluate_training_results(training_result):
    results = canonicalize_training_results(training_result["training_results"])
    models_by_level = models_by_forecast_level(results)

    store_level_candidates = [
        model_name
        for model_name in STORE_LEVEL_MODEL_SELECTION_KEYS
        if model_name in results
    ]
    daily_total_candidates = [
        model_name
        for model_name in DAILY_TOTAL_MODEL_SELECTION_KEYS
        if model_name in results
    ]

    best_store_level_model, best_store_level_rmse = select_best_model_by_rmse(
        results,
        store_level_candidates,
    )
    best_daily_total_model, best_daily_total_rmse = select_best_model_by_rmse(
        results,
        daily_total_candidates,
    )

    return {
        "best_store_level_model": best_store_level_model,
        "best_store_level_rmse": best_store_level_rmse,
        "best_daily_total_model": best_daily_total_model,
        "best_daily_total_rmse": best_daily_total_rmse,
        "trained_models": list(results.keys()),
        "models_by_forecast_level": models_by_level,
        "store_level_selection_candidates": store_level_candidates,
        "daily_total_selection_candidates": daily_total_candidates,
        "best_run_id": training_result.get("mlflow_run_id"),
        "best_model": best_store_level_model,
    }
