from datetime import datetime, timedelta
from airflow.decorators import dag, task
from airflow.operators.bash import BashOperator
import pandas as pd
import os
import sys

# Add include path
sys.path.append("/usr/local/airflow/include")

from utils.model_registry import (
    canonical_model_name,
    canonical_model_names,
    get_model_registry_entry,
)


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


def _parse_file_limit(raw_value, env_name):
    try:
        file_limit = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{env_name} must be an integer, got {raw_value!r}") from exc

    return max(file_limit, 0)


def _limit_sales_files(sales_files, raw_limit, env_name):
    file_limit = _parse_file_limit(raw_limit, env_name)
    sales_files = list(sales_files)
    selected_files = sales_files if file_limit <= 0 else sales_files[:file_limit]

    return selected_files, file_limit


def _resolve_sales_file_selection(
    sales_files,
    max_sales_files,
    validation_max_sales_files,
):
    sales_files = list(sales_files)
    training_files, training_limit = _limit_sales_files(
        sales_files,
        max_sales_files,
        "MAX_SALES_FILES",
    )
    validation_files, validation_limit = _limit_sales_files(
        training_files,
        validation_max_sales_files,
        "VALIDATION_MAX_SALES_FILES",
    )

    validation_mode = "sample"
    if validation_limit <= 0 or len(validation_files) >= len(training_files):
        validation_mode = "full"

    return {
        "total_files_available": len(sales_files),
        "training_files": training_files,
        "training_limit": training_limit,
        "validation_files": validation_files,
        "validation_limit": validation_limit,
        "validation_mode": validation_mode,
    }


def _models_by_forecast_level(results):
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


def _canonicalize_training_results(results):
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


def _build_model_registration_plan(
    training_results,
    trained_models=None,
    best_store_level_model=None,
    best_daily_total_model=None,
):
    canonical_results = _canonicalize_training_results(training_results)
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


def _select_best_model_by_rmse(results, model_names):
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


def _evaluate_training_results(training_result):
    results = _canonicalize_training_results(training_result["training_results"])
    models_by_level = _models_by_forecast_level(results)

    # Keep RMSE comparisons within the same forecast grain. Prophet forecasts
    # daily total sales, so it must never compete with store-level models.
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

    best_store_level_model, best_store_level_rmse = _select_best_model_by_rmse(
        results,
        store_level_candidates,
    )
    best_daily_total_model, best_daily_total_rmse = _select_best_model_by_rmse(
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
        # Compatibility key: this means best store-level model.
        "best_model": best_store_level_model,
    }


def _build_performance_report(
    training_result,
    validation_summary,
    evaluation_result,
    registration_result=None,
):
    results = _canonicalize_training_results(training_result["training_results"])
    validation_summary = validation_summary or {}
    evaluation_result = evaluation_result or {}
    registration_result = registration_result or {}
    models_by_level = evaluation_result.get("models_by_forecast_level")
    if not models_by_level:
        models_by_level = _models_by_forecast_level(results)

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


default_args = {
    "owner": "data_science_team",
    "depends_on_past": False,
    "start_date": datetime(2025, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

@dag(
    default_args=default_args,
    description="Train sales forecasting models",
    tags=["ml", "training", "sales_forecasting","sales"],
)
def sales_forecast_training():
    @task()
    def extract_data_task():
        from utils.rossmann_loader import RossmannDataLoader

        data_output_dir = "/tmp/rossmann_sales_data"

        loader = RossmannDataLoader(
        train_path="/usr/local/airflow/include/data/rossmann/train.csv",
        store_path="/usr/local/airflow/include/data/rossmann/store.csv",
        )

        print("Loading Rossmann Store Sales dataset...")
        file_paths = loader.prepare_data(output_dir=data_output_dir)

        total_files = sum(len(paths) for paths in file_paths.values())

        print(f"Prepared {total_files} files:")
        for data_type, paths in file_paths.items():
            print(f" - {data_type}: {len(paths)} files")

        return {
        "data_output_dir": data_output_dir,
        "file_paths": file_paths,
        "total_files": total_files,
        }
    

    @task()
    def validate_data_task(extract_result):
        file_paths = extract_result["file_paths"]
        sales_files = file_paths["sales"]
        selection = _resolve_sales_file_selection(
            sales_files,
            os.getenv("MAX_SALES_FILES", "0"),
            os.getenv("VALIDATION_MAX_SALES_FILES", "0"),
        )
        training_files = selection["training_files"]
        validation_files = selection["validation_files"]

        total_rows = 0
        issues_found = []

        total_available = selection["total_files_available"]
        total_training_selected = len(training_files)
        total_validated = len(validation_files)

        if selection["training_limit"] > 0:
            print(
                "Training file cap enabled: "
                f"using first {total_training_selected} of {total_available} "
                "sales files."
            )

        if selection["validation_mode"] == "sample":
            print(
                "Sample validation enabled: "
                f"validating first {total_validated} of "
                f"{total_training_selected} files."
            )
        else:
            print(
                "Full validation enabled: "
                f"validating all {total_validated} selected training files."
            )

        if not validation_files:
            issues_found.append("No sales files selected for validation")

        required_cols = [
        "date",
        "store_id",
        "sales",
        "customer_traffic",
        "has_promotion",
        "is_open",
        "is_holiday",
        ]

        for i, sales_file in enumerate(validation_files):
            df = pd.read_parquet(sales_file)

            if i == 0:
                print(f"Rossmann training columns: {df.columns.tolist()}")

            if df.empty:
                issues_found.append(f"Empty file: {sales_file}")
                continue

            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                issues_found.append(f"Missing columns in {sales_file}: {missing_cols}")

            total_rows += len(df)

            if "sales" in df.columns and df["sales"].min() < 0:
                issues_found.append(f"Negative sales in {sales_file}")

            if "customer_traffic" in df.columns and df["customer_traffic"].min() < 0:
                issues_found.append(f"Negative customer traffic in {sales_file}")

        validation_summary = {
            "total_files_available": total_available,
            "total_training_files_selected": total_training_selected,
            "total_files_validated": total_validated,
            "validation_mode": selection["validation_mode"],
            "max_sales_files": selection["training_limit"],
            "validation_max_sales_files": selection["validation_limit"],
            "total_rows": total_rows,
            "issues_found": len(issues_found),
            "issues": issues_found[:5],
            }

        if issues_found:
            print(f"Validation completed with {len(issues_found)} issues:")
            for issue in issues_found[:5]:
                print(f" - {issue}")
        else:
            print(f"Validation passed! Total rows checked: {total_rows}")

        return validation_summary
    
    @task()
    def train_models_task(extract_result, validation_summary):
        from ml_models.train_models import ModelTrainer

        file_paths = extract_result["file_paths"]

        print("Loading Rossmann sales data from parquet files...")

        sales_files = file_paths["sales"]

        selected_files, max_files = _limit_sales_files(
            sales_files,
            os.getenv("MAX_SALES_FILES", "0"),
            "MAX_SALES_FILES",
        )
        if max_files > 0:
            print(
                "Training file cap enabled: "
                f"loading first {len(selected_files)} of {len(sales_files)} "
                "sales files."
            )
        else:
            print(f"Training on all {len(selected_files)} sales files.")

        if not selected_files:
            raise ValueError("No sales files selected for training")

        sales_dfs = []

        for i, sales_file in enumerate(selected_files):
            df = pd.read_parquet(sales_file)
            sales_dfs.append(df)

            if (i + 1) % 50 == 0:
                print(f" Loaded {i + 1} files...")

        sales_df = pd.concat(sales_dfs, ignore_index=True)

        print(f"Combined Rossmann data shape: {sales_df.shape}")
        print(f"Columns: {sales_df.columns.tolist()}")

    # The original generated data is product-level.
    # Rossmann is already store-day level, but we still aggregate safely.
        agg_dict = {
        "sales": "sum",
        "customer_traffic": "sum",
        "has_promotion": "max",
        "is_open": "max",
        "is_holiday": "max",
        "school_holiday": "max",
        "competition_distance": "first",
        "promo2": "first",
        "store_type": "first",
        "assortment": "first",
        "state_holiday": "first",
        "promo_interval": "first",
        }

        existing_agg_dict = {
        col: agg_func
        for col, agg_func in agg_dict.items()
        if col in sales_df.columns
        }

        store_daily_sales = (
        sales_df.groupby(["date", "store_id"])
        .agg(existing_agg_dict)
        .reset_index()
        )

        store_daily_sales["date"] = pd.to_datetime(store_daily_sales["date"])

        print(f"Final Rossmann training data shape: {store_daily_sales.shape}")
        print(f"Final columns: {store_daily_sales.columns.tolist()}")

        trainer = ModelTrainer()

        categorical_cols = [
            col
            for col in [
            "store_id",
            "store_type",
            "assortment",
            "state_holiday",
            "promo_interval",
            ]
            if col in store_daily_sales.columns
        ]

        train_df, val_df, test_df = trainer.prepare_data(
            store_daily_sales,
            target_col="sales",
            date_col="date",
            group_cols=["store_id"],
            categorical_cols=categorical_cols,
        )

        print(
            f"Train shape: {train_df.shape}, "
            f"Val shape: {val_df.shape}, "
            f"Test shape: {test_df.shape}"
        )

        # Start with Optuna disabled for speed. After it works, you can change to True.
        results = trainer.train_all_models(
            train_df,
            val_df,
            test_df,
            target_col="sales",
            use_optuna=False,
        )

        for model_name, model_results in results.items():
            if "metrics" in model_results:
                print(f"\n{model_name} metrics:")
                for metric, value in model_results["metrics"].items():
                    print(f" {metric}: {value:.4f}")

        print("\nModel artifacts have been saved to MLflow/MinIO")

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
                "mlflow_artifact_path": model_results.get(
                    "mlflow_artifact_path",
                    model_name,
                ),
            }
            if model_results.get("legacy_model_key"):
                serializable_results[model_name]["source_model_key"] = (
                    model_results["legacy_model_key"]
                )

        current_run_id = getattr(trainer, "last_run_id", None)

        return {
            "training_results": _canonicalize_training_results(serializable_results),
            "mlflow_run_id": current_run_id,
        }
    
    @task()
    def evaluate_models_task(training_result):
        evaluation = _evaluate_training_results(training_result)

        if evaluation["best_store_level_model"]:
            print(
                "Best store-level model: "
                f"{evaluation['best_store_level_model']} "
                f"with RMSE: {evaluation['best_store_level_rmse']:.4f}"
            )
        else:
            print("No store-level model RMSE was available")

        if evaluation["best_daily_total_model"]:
            print(
                "Daily-total Prophet baseline: "
                f"{evaluation['best_daily_total_model']} "
                f"with RMSE: {evaluation['best_daily_total_rmse']:.4f}"
            )
        else:
            print("No daily-total model RMSE was available")

        return evaluation
    
    @task()
    def register_trained_models_task(training_result, evaluation_result):
        """Register all valid trained models and tag best-model metadata."""
        from utils.mlflow_utils import MLflowManager

        run_id = evaluation_result.get("best_run_id") or training_result.get("mlflow_run_id")
        if not run_id:
            raise ValueError("Cannot register models without an MLflow run ID")

        training_results = _canonicalize_training_results(
            training_result.get("training_results", {})
        )
        trained_models = evaluation_result.get("trained_models") or list(training_results.keys())
        models_by_level = evaluation_result.get("models_by_forecast_level")
        if not models_by_level:
            models_by_level = _models_by_forecast_level(training_results)

        best_store_level_model = canonical_model_name(
            evaluation_result.get("best_store_level_model")
        )
        best_daily_total_model = canonical_model_name(
            evaluation_result.get("best_daily_total_model")
        )

        mlflow_manager = MLflowManager()

        store_level_models = canonical_model_names(
            models_by_level.get("store_level", [])
        )
        daily_total_models = canonical_model_names(
            models_by_level.get("daily_total", [])
        )
        mlflow_manager.set_run_tags(
            run_id,
            {
                "best_store_level_model": best_store_level_model,
                "best_store_level_rmse": evaluation_result.get(
                    "best_store_level_rmse"
                ),
                "best_daily_total_model": best_daily_total_model,
                "best_daily_total_rmse": evaluation_result.get(
                    "best_daily_total_rmse"
                ),
                "store_level_models": store_level_models,
                "daily_total_models": daily_total_models,
                "registered_model_names": canonical_model_names(trained_models),
            },
        )

        registered_versions = {}
        production_models = []
        staging_models = []
        registration_plan, skipped_models = _build_model_registration_plan(
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
    
    
    @task()
    def transition_registered_models_task(registration_result):
        from utils.mlflow_utils import MLflowManager

        mlflow_manager = MLflowManager()
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


    @task()
    def generate_performance_report_task(
        training_result,
        validation_summary,
        evaluation_result,
        registration_result,
    ):
        report = _build_performance_report(
            training_result,
            validation_summary,
            evaluation_result,
            registration_result,
        )
        import json

        with open("/tmp/performance_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print("Performance report generated")
        print(
            "Store-level models: "
            f"{list(report['store_level_model_performance'].keys())}"
        )
        print(
            "Daily-total models: "
            f"{list(report['daily_total_model_performance'].keys())}"
        )
        return report
    cleanup = BashOperator(
    task_id="cleanup",
    bash_command="rm -rf /tmp/sales_data /tmp/rossmann_sales_data /tmp/performance_report.json || true",
    )
# Task dependencies using function calls
    extract_result = extract_data_task()
    validation_summary = validate_data_task(extract_result)
    training_result = train_models_task(extract_result, validation_summary)
    evaluation_result = evaluate_models_task(training_result)
    registration_result = register_trained_models_task(training_result, evaluation_result)
    transition = transition_registered_models_task(registration_result)
    report = generate_performance_report_task(
        training_result,
        validation_summary,
        evaluation_result,
        registration_result,
    )

    transition >> cleanup
    report >> cleanup

sales_forecast_training()
