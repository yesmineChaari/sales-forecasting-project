from datetime import datetime, timedelta
from airflow.decorators import dag, task
from airflow.operators.bash import BashOperator
import pandas as pd
import os
import sys

# Add include path
sys.path.append("/usr/local/airflow/include")


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

        total_rows = 0
        issues_found = []

        print(f"Validating {len(file_paths['sales'])} Rossmann sales files...")

        required_cols = [
        "date",
        "store_id",
        "sales",
        "customer_traffic",
        "has_promotion",
        "is_open",
        "is_holiday",
        ]

        for i, sales_file in enumerate(file_paths["sales"][:10]):
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
            "total_files_validated": len(file_paths["sales"][:10]),
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

        max_files = int(os.getenv("MAX_SALES_FILES", "0"))
        selected_files = sales_files if max_files <= 0 else sales_files[:max_files]

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

        print("\nVisualization charts have been generated and saved to MLflow/MinIO")

        serializable_results = {}

        for model_name, model_results in results.items():
            serializable_results[model_name] = {
                "metrics": model_results.get("metrics", {})
            }

        current_run_id = getattr(trainer, "last_run_id", None)

        return {
            "training_results": serializable_results,
            "mlflow_run_id": current_run_id,
        }
    
    @task()
    def evaluate_models_task(training_result):
        results = training_result["training_results"]
        best_model_name = None
        best_rmse = float("inf")
        for model_name, model_results in results.items():
            if "metrics" in model_results and "rmse" in model_results["metrics"]:
                if model_results["metrics"]["rmse"] < best_rmse:
                    best_rmse = model_results["metrics"]["rmse"]
                    best_model_name = model_name
        print(f"Best model: {best_model_name} with RMSE: {best_rmse:.4f}")
        return {
            "best_model": best_model_name,
            "best_run_id": training_result.get("mlflow_run_id"),
        }
    
    @task()
    def register_best_model_task(evaluation_result):
        from utils.mlflow_utils import MLflowManager

        evaluation_result["best_model"]
        run_id = evaluation_result["best_run_id"]
        mlflow_manager = MLflowManager()
        model_versions = {}
        for model_name in ["xgboost", "lightgbm"]:
            version = mlflow_manager.register_model(run_id, model_name, model_name)
            model_versions[model_name] = version
            print(f"Registered {model_name} version: {version}")
        return model_versions
    
    
    @task()
    def transition_to_production_task(model_versions):
        from utils.mlflow_utils import MLflowManager

        mlflow_manager = MLflowManager()
        for model_name, version in model_versions.items():
            mlflow_manager.transition_model_stage(model_name, version, "Production")
            print(f"Transitioned {model_name} v{version} to Production")
        return "Models transitioned to production"


    @task()
    def generate_performance_report_task(training_result, validation_summary):
        results = training_result["training_results"]
        report = {
            "timestamp": datetime.now().isoformat(),
            "data_summary": {
                "total_rows": (
                    validation_summary.get("total_rows", 0) if validation_summary else 0
                ),
                "files_validated": (
                    validation_summary.get("total_files_validated", 0)
                    if validation_summary
                    else 0
                ),
                "issues_found": (
                    validation_summary.get("issues_found", 0)
                    if validation_summary
                    else 0
                ),
                "issues": (
                    validation_summary.get("issues", []) if validation_summary else []
                ),
            },
            "model_performance": {},
        }
        if results:
            for model_name, model_results in results.items():
                if "metrics" in model_results:
                    report["model_performance"][model_name] = model_results["metrics"]
        import json

        with open("/tmp/performance_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print("Performance report generated")
        print(f"Models trained: {list(report['model_performance'].keys())}")
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
    model_versions = register_best_model_task(evaluation_result)
    transition = transition_to_production_task(model_versions)
    report = generate_performance_report_task(training_result, validation_summary)

    report >> cleanup

sales_forecast_training()
