"""Build serializable model-training performance reports."""

from datetime import datetime

from pipeline.model_results import (
    canonicalize_training_results,
    models_by_forecast_level,
)
from utils.model_registry import canonical_model_name, canonical_model_names


def build_performance_report(
    training_result,
    validation_summary,
    evaluation_result,
    registration_result=None,
):
    results = canonicalize_training_results(training_result["training_results"])
    validation_summary = validation_summary or {}
    evaluation_result = evaluation_result or {}
    registration_result = registration_result or {}
    models_by_level = evaluation_result.get("models_by_forecast_level")
    if not models_by_level:
        models_by_level = models_by_forecast_level(results)

    registered_models_by_level = {
        level: canonical_model_names(model_names)
        for level, model_names in models_by_level.items()
    }
    best_store_level_model = evaluation_result.get("best_store_level_model")
    best_daily_total_model = evaluation_result.get("best_daily_total_model")

    report = {
        "timestamp": datetime.now().isoformat(),
        "data_summary": {
            "total_rows": validation_summary.get("total_rows", 0),
            "total_files_available": validation_summary.get("total_files_available", 0),
            "training_files_selected": validation_summary.get(
                "total_training_files_selected",
                0,
            ),
            "files_validated": validation_summary.get("total_files_validated", 0),
            "validation_mode": validation_summary.get("validation_mode", "unknown"),
            "issues_found": validation_summary.get("issues_found", 0),
            "issues": validation_summary.get("issues", []),
        },
        "store_level_model_performance": {},
        "daily_total_model_performance": {},
        "best_store_level_model": {
            "model_name": best_store_level_model,
            "registered_model_name": canonical_model_name(best_store_level_model),
            "rmse": evaluation_result.get("best_store_level_rmse"),
        },
        "best_daily_total_model": {
            "model_name": best_daily_total_model,
            "registered_model_name": canonical_model_name(best_daily_total_model),
            "rmse": evaluation_result.get("best_daily_total_rmse"),
        },
        "models_by_forecast_level": models_by_level,
        "registered_models_by_forecast_level": registered_models_by_level,
        "model_registration": registration_result,
        "model_performance": {},
    }

    for model_name, model_results in results.items():
        metrics = model_results.get("metrics", {})
        if not metrics:
            continue

        report["model_performance"][model_name] = metrics

        if model_name in models_by_level["daily_total"]:
            report["daily_total_model_performance"][model_name] = metrics
        else:
            report["store_level_model_performance"][model_name] = metrics

    return report
