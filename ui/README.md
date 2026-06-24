# Streamlit Forecast UI

This directory contains the Streamlit dashboard for exploring trained Rossmann store-level forecasting models.

## Entry Point

Run the app with:

```bash
streamlit run inference_app.py
```

The Astro Compose service starts `inference_app.py` on port `8501`.

## What The UI Loads

The active UI path uses:

- `utils/simple_model_loader.py`
- `utils/simple_predictor.py`
- MLflow run artifacts from the latest finished training run

It loads legacy pickle artifacts such as:

```text
models/xgboost/xgboost_model.pkl
models/lightgbm/lightgbm_model.pkl
models/ensemble/ensemble_model.pkl
```

Those legacy artifact folders are loaded into the same canonical store-level
model names used by registration.

This UI does not call a FastAPI endpoint.

## Models

The UI exposes store-level models only:

- `ensemble_store_level`
- `xgboost_store_level`
- `lightgbm_store_level`

`ensemble_store_level` is the saved training-time ensemble artifact, including
the calibrated validation R2 weights.

Prophet is not exposed in the UI because it is a daily-total baseline, not a store-level model.

## Input Data

CSV uploads should include the same business fields used by the UI forms:

- `date`: date column in `YYYY-MM-DD` format
- `store_id`: store identifier
- `sales`: historical sales amount
- `customer_traffic`
- `has_promotion`
- `is_open`
- `school_holiday`
- `state_holiday`
- `store_type`
- `assortment`
- `competition_distance`
- `promo2`
- `promo_interval`

`is_holiday` is computed automatically from `state_holiday` and
`school_holiday`.
The UI supports one store at a time. Uploaded rows are sorted by `date` before
lag and rolling features are built.

Manual and sample modes expose these fields through typed controls. Boolean
fields use Yes/No selectors, categorical fields use dropdowns, and numeric
fields use number inputs. Forecast-period features are entered separately before
running a prediction.

Example:

```csv
date,store_id,sales,customer_traffic,has_promotion,is_open,school_holiday,state_holiday,store_type,assortment,competition_distance,promo2,promo_interval
2024-01-01,store_0001,5234.50,520,1,1,0,none,a,a,500,0,none
2024-01-02,store_0001,4892.75,490,0,1,0,none,a,a,500,0,none
```

The dashboard is an exploration interface around MLflow artifacts and requires
recent historical sales so lag and rolling features can be built.

## Local Development

Start the project services from the repository root:

```bash
astro dev start
```

Then run the UI locally if needed:

```bash
cd ui
pip install -r requirements.txt
streamlit run inference_app.py
```

Environment variables used by the UI:

- `MLFLOW_TRACKING_URI`
- `MLFLOW_S3_ENDPOINT_URL`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

## Directory

```text
ui/
|-- inference_app.py
|-- utils/
|   |-- simple_model_loader.py
|   `-- simple_predictor.py
|-- requirements.txt
|-- Dockerfile
`-- README.md
```
