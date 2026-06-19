"""High-level MLflow model registration workflows."""

from pipeline.model_results import (
    build_model_registration_plan,
    canonicalize_training_results,
    models_by_forecast_level,
)
from utils.model_registry import canonical_model_name, canonical_model_names


def register_trained_models(
    training_result,
    evaluation_result,
    mlflow_manager=None,
):
    from utils.mlflow_utils import MLflowManager

    run_id = evaluation_result.get("best_run_id") or training_result.get(
        "mlflow_run_id"
    )
    if not run_id:
        raise ValueError("Cannot register models without an MLflow run ID")

    training_results = canonicalize_training_results(
        training_result.get("training_results", {})
    )
    trained_models = evaluation_result.get("trained_models") or list(
        training_results.keys()
    )
    models_by_level = evaluation_result.get("models_by_forecast_level")
    if not models_by_level:
        models_by_level = models_by_forecast_level(training_results)

    best_store_level_model = canonical_model_name(
        evaluation_result.get("best_store_level_model")
    )
    best_daily_total_model = canonical_model_name(
        evaluation_result.get("best_daily_total_model")
    )

    mlflow_manager = mlflow_manager or MLflowManager()
    store_level_models = canonical_model_names(models_by_level.get("store_level", []))
    daily_total_models = canonical_model_names(models_by_level.get("daily_total", []))
    mlflow_manager.set_run_tags(
        run_id,
        {
            "best_store_level_model": best_store_level_model,
            "best_store_level_rmse": evaluation_result.get("best_store_level_rmse"),
            "best_daily_total_model": best_daily_total_model,
            "best_daily_total_rmse": evaluation_result.get("best_daily_total_rmse"),
            "store_level_models": store_level_models,
            "daily_total_models": daily_total_models,
            "registered_model_names": canonical_model_names(trained_models),
        },
    )

    registered_versions = {}
    production_models = []
    staging_models = []
    registration_plan, skipped_models = build_model_registration_plan(
        training_results,
        trained_models,
        best_store_level_model,
        best_daily_total_model,
    )

    for model_key, reason in skipped_models.items():
        print(f"Skipping {model_key}; {reason}")

    for registration in registration_plan:
        registered_name = registration["registered_name"]
        artifact_path = registration["artifact_path"]
        version = mlflow_manager.register_model(
            run_id,
            registered_name,
            artifact_path,
        )

        recommended_stage = registration["recommended_stage"]
        if recommended_stage == "Production":
            production_models.append(registered_name)
        elif recommended_stage == "Staging":
            staging_models.append(registered_name)

        version_tags = {
            "forecast_level": registration["forecast_level"],
            "target_grain": registration["target_grain"],
            "is_best_store_level": registration["is_best_store_level"],
            "is_best_daily_total": registration["is_best_daily_total"],
            "source_result_key": registration["source_result_key"],
            "artifact_path": artifact_path,
        }
        if registration["ensemble_members"]:
            version_tags["ensemble_members"] = registration["ensemble_members"]

        try:
            mlflow_manager.set_model_version_tags(
                registered_name,
                version,
                version_tags,
            )
        except Exception as tag_error:
            print(
                "Warning: failed to tag registered model "
                f"{registered_name} v{version}: {tag_error}"
            )

        registered_versions[registered_name] = {
            "version": version,
            "result_key": registration["source_result_key"],
            "artifact_path": artifact_path,
            "forecast_level": registration["forecast_level"],
            "target_grain": registration["target_grain"],
            "is_best_store_level": registration["is_best_store_level"],
            "is_best_daily_total": registration["is_best_daily_total"],
            "recommended_stage": recommended_stage,
            "ensemble_members": registration["ensemble_members"],
        }
        print(
            f"Registered {registered_name} v{version} "
            f"from artifact path {artifact_path}"
        )

    return {
        "registered_versions": registered_versions,
        "best_store_level_model": best_store_level_model,
        "best_daily_total_model": best_daily_total_model,
        "registered_model_names": list(registered_versions.keys()),
        "production_models": production_models,
        "staging_models": staging_models,
        "skipped_models": skipped_models,
    }


def transition_registered_models(registration_result, mlflow_manager=None):
    from utils.mlflow_utils import MLflowManager

    mlflow_manager = mlflow_manager or MLflowManager()
    transitioned_versions = {}
    registered_only_models = []

    for model_name, version_info in registration_result.get(
        "registered_versions",
        {},
    ).items():
        version = version_info["version"]
        stage = version_info.get("recommended_stage")
        if not stage:
            registered_only_models.append(model_name)
            print(f"Leaving {model_name} v{version} registered without a stage")
            continue

        mlflow_manager.transition_model_stage(model_name, version, stage)
        try:
            mlflow_manager.set_model_version_tags(
                model_name,
                version,
                {"assigned_stage": stage},
            )
        except Exception as tag_error:
            print(
                "Warning: failed to tag assigned stage for "
                f"{model_name} v{version}: {tag_error}"
            )
        transitioned_versions[model_name] = {
            "version": version,
            "stage": stage,
        }
        print(f"Transitioned {model_name} v{version} to {stage}")

    return {
        "transitioned_versions": transitioned_versions,
        "production_models": registration_result.get("production_models", []),
        "staging_models": registration_result.get("staging_models", []),
        "registered_only_models": registered_only_models,
    }
