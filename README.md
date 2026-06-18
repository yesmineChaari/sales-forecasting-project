# Rossmann Sales Forecasting MLOps Pipeline

Airflow DAG -> Rossmann data preparation -> validation -> feature engineering -> model training -> MLflow/MinIO tracking -> model registry

## Overview

This project trains and tracks Rossmann sales forecasting models with Apache Airflow, MLflow, MinIO, and a Streamlit dashboard. The active training DAG is `sales_forecast_training` in `dags/sales_forecast_train.py`.

The main machine-learning grain is:

```text
date + store_id -> store sales
```

Prophet is handled separately as a daily-total baseline:

```text
date -> total daily sales across all stores
```

## Model Semantics

| Model | Forecast grain | Target | Compared with |
|---|---|---|---|
| XGBoost | date + store_id | store sales | LightGBM, ensemble |
| LightGBM | date + store_id | store sales | XGBoost, ensemble |
| Ensemble | date + store_id | store sales | XGBoost, LightGBM |
| Prophet | date | total daily sales | daily-total baseline only |

Prophet is not part of the store-level ensemble because it forecasts daily total sales, not individual store sales.

## Pipeline

The Airflow DAG:

1. Loads Rossmann CSV data and writes parquet files.
2. Validates selected sales files. `MAX_SALES_FILES` limits training input, and `VALIDATION_MAX_SALES_FILES` can explicitly sample validation.
3. Aggregates to `date + store_id`.
4. Creates leakage-safe lag and rolling features using past store history only.
5. Trains XGBoost, LightGBM, the XGBoost/LightGBM store-level ensemble, and optionally Prophet daily-total.
6. Logs models and artifacts to MLflow/MinIO.
7. Registers all trained model artifacts and tags/reports the best store-level model.

## Model Registry

All valid trained models are registered when available:

- `xgboost_store_level`
- `lightgbm_store_level`
- `ensemble_store_level`
- `prophet_daily_total`

The best store-level model is tagged and reported separately from the daily-total Prophet baseline. Prophet is registered as a separate daily-total model if it trains successfully.

## Serving And UI

The old FastAPI online inference service was decommissioned because correct store-level lag and rolling features require historical sales context at prediction time. A one-row request cannot safely create those features.

The Streamlit UI entry point is:

```bash
ui/inference_app.py
```

The UI reads MLflow run artifacts directly through its local loaders. It does not call a FastAPI endpoint.

## Run Locally

Start the Astronomer/Airflow stack and the extra MLflow, MinIO, and Streamlit services:

```bash
astro dev start
```

Useful local URLs:

- Airflow: `http://localhost:8080`
- MLflow: `http://localhost:5001`
- MinIO: `http://localhost:9001`
- Streamlit UI: `http://localhost:8501`

For local UI-only development after services are running:

```bash
cd ui
streamlit run inference_app.py
```

## Configuration

Common environment variables:

- `MAX_SALES_FILES`: cap training files; `0` means all files.
- `VALIDATION_MAX_SALES_FILES`: cap validation files from the selected training files; `0` means full validation.
- `ENABLE_MODEL_VISUALIZATIONS`: set to `true` to generate and log optional model visualizations.
