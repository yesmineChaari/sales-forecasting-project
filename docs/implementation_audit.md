# Implementation Audit

This document records the active sales forecasting implementation after the cleanup chunks completed so far.

## Training Grain

The active DAG is `sales_forecast_training` in `dags/sales_forecast_train.py`.

The main store-level training grain is:

```text
date + store_id -> sales
```

Rossmann data is loaded, written to parquet, concatenated, and safely aggregated by `date` and `store_id`.

## Validation Coverage

Validation uses the same sales-file selection as training by default.

- `MAX_SALES_FILES=0` means train on all sales files.
- `VALIDATION_MAX_SALES_FILES=0` means validate all selected training files.
- Positive values explicitly cap the corresponding selection and are logged as sample mode where appropriate.

## Model Outputs

`ModelTrainer.train_all_models` can return:

- `xgboost`
- `lightgbm`
- `ensemble_store_level`
- `prophet_daily_total`, if Prophet is enabled and trains successfully

XGBoost and LightGBM are store-level models. The ensemble is a store-level ensemble of XGBoost and LightGBM only.

## Prophet Behavior

Prophet is trained separately on daily total sales:

```text
date + aggregate business signals -> total sales across all stores
```

It is logged and registered as `prophet_daily_total`, evaluated only against daily aggregated targets, and excluded from store-level best-model selection and the store-level ensemble. Prophet uses the full Rossmann row stream, including closed-store and zero-sales rows. Real-data diagnostics selected the `prophet_all_regressors` variant, so production Prophet uses promotion count/rate, open-store count, school-holiday rate, and state-holiday flags.

## Registration Behavior

The DAG registers all valid trained model artifacts and records best-model metadata. Legacy artifact paths are mapped to explicit registered names where needed:

- `xgboost` artifact -> `xgboost_store_level`
- `lightgbm` artifact -> `lightgbm_store_level`
- `ensemble` artifact -> `ensemble_store_level`
- `prophet_daily_total` artifact -> `prophet_daily_total`

The best store-level model is selected only from store-level candidates. Prophet is treated as a daily-total baseline.

## Feature Engineering

Lag and rolling target-derived features are grouped by `store_id` for store-level training. Rolling features shift before rolling so current-day sales do not leak into same-day features. Initial lag/rolling nulls are filled with neutral values, not backfilled from future rows.

Holiday features use Rossmann/Germany fields such as `state_holiday`, `school_holiday`, and `is_holiday`; the previous US holiday calendar is not used.

## Serving And UI

The mismatched FastAPI online inference service was removed. Store-level online inference requires recent historical sales context to build lag and rolling features correctly.

The Streamlit app entry point is `ui/inference_app.py`. It loads MLflow run artifacts directly through local UI loaders and does not call FastAPI.

## Documentation

The root `README.md` and `ui/README.md` now describe the Rossmann pipeline, model semantics, registration behavior, UI entry point, and the decommissioned online inference API.
