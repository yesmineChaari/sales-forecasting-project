from datetime import datetime, timedelta
from airflow.decorators import dag, task
from airflow.operators.bash import BashOperator
import os
import sys

# Add include path
sys.path.append("/usr/local/airflow/include")

from data_validation.validators import (
    limit_files as _limit_sales_files,
    validate_sales_files,
)
from data_preparation.sales_training_data import (
    build_store_daily_training_data,
    read_sales_parquet_files,
    store_daily_categorical_columns,
)
from pipeline.model_results import (
    evaluate_training_results as _evaluate_training_results,
    serializable_training_results,
)
from pipeline.model_registration import (
    register_trained_models,
    transition_registered_models,
)
from pipeline.performance_report import (
    build_performance_report as _build_performance_report,
)


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
        sales_files = file_paths.get("sales", [])
        validation_summary = validate_sales_files(
            sales_files,
            max_sales_files=os.getenv("MAX_SALES_FILES", "0"),
            validation_max_sales_files=os.getenv("VALIDATION_MAX_SALES_FILES", "0"),
        )

        for message in validation_summary.get("messages", []):
            print(message)

        sample_columns = validation_summary.get("sample_columns")
        if sample_columns:
            print(f"Rossmann training columns: {sample_columns}")

        blocking_issues = validation_summary.get("blocking_issues_found", 0)
        warnings_found = validation_summary.get("warnings_found", 0)
        total_rows = validation_summary.get("total_rows", 0)
        issues_found = validation_summary.get("issues_found", 0)

        if issues_found:
            print(
                "Validation completed with "
                f"{blocking_issues} blocking issues and "
                f"{warnings_found} warnings."
            )
            for issue in validation_summary.get("issues", []):
                print(f" - {issue}")
        else:
            print(f"Validation passed! Total rows checked: {total_rows}")

        if blocking_issues:
            raise ValueError(
                "Data validation failed with "
                f"{blocking_issues} blocking issue(s). "
                "See validation task logs for details."
            )

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

        store_daily_sales = build_store_daily_training_data(selected_files)
        prophet_sales_files = file_paths.get("prophet_sales", [])
        if not prophet_sales_files:
            raise ValueError(
                "Prophet full-row sales files were not produced by extraction. "
                "Rerun extract_data_task with the current RossmannDataLoader."
            )

        prophet_selected_files, prophet_max_files = _limit_sales_files(
            prophet_sales_files,
            os.getenv("PROPHET_MAX_SALES_FILES", "0"),
            "PROPHET_MAX_SALES_FILES",
        )
        if prophet_max_files > 0:
            print(
                "Prophet file cap enabled: "
                f"loading first {len(prophet_selected_files)} of "
                f"{len(prophet_sales_files)} full-row sales files."
            )
        else:
            print(
                f"Prophet training on all {len(prophet_selected_files)} "
                "full-row sales files."
            )

        prophet_full_sales = read_sales_parquet_files(prophet_selected_files)
        prophet_uses_full_row_input = True

        print(f"Final Rossmann training data shape: {store_daily_sales.shape}")
        print(
            "Final Prophet full-row data shape: "
            f"{prophet_full_sales.shape}"
        )
        print(f"Final columns: {store_daily_sales.columns.tolist()}")

        trainer = ModelTrainer()

        categorical_cols = store_daily_categorical_columns(store_daily_sales)

        train_df, val_df, test_df = trainer.prepare_data(
            store_daily_sales,
            target_col="sales",
            date_col="date",
            group_cols=["store_id"],
            categorical_cols=categorical_cols,
        )
        prophet_train_df, prophet_val_df, prophet_test_df = trainer.prepare_prophet_data(
            prophet_full_sales,
            target_col="sales",
            date_col="date",
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
            prophet_train_df=prophet_train_df,
            prophet_val_df=prophet_val_df,
            prophet_test_df=prophet_test_df,
            prophet_uses_full_row_input=prophet_uses_full_row_input,
        )

        for model_name, model_results in results.items():
            if "metrics" in model_results:
                print(f"\n{model_name} metrics:")
                for metric, value in model_results["metrics"].items():
                    print(f" {metric}: {value:.4f}")

        print("\nModel artifacts have been saved to MLflow/MinIO")

        current_run_id = getattr(trainer, "last_run_id", None)

        return {
            "training_results": serializable_training_results(results),
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
        return register_trained_models(training_result, evaluation_result)
    
    
    @task()
    def transition_registered_models_task(registration_result):
        return transition_registered_models(registration_result)


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
