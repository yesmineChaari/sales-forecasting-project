"""Prepare Rossmann sales parquet files for store-day model training."""

import logging

import pandas as pd


logger = logging.getLogger(__name__)

STORE_DAILY_AGGREGATIONS = {
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

STORE_DAILY_CATEGORICAL_CANDIDATES = [
    "store_id",
    "store_type",
    "assortment",
    "state_holiday",
    "promo_interval",
]


def read_sales_parquet_files(sales_files, progress_interval=50):
    sales_dfs = []

    for index, sales_file in enumerate(sales_files):
        sales_dfs.append(pd.read_parquet(sales_file))

        if progress_interval and (index + 1) % progress_interval == 0:
            logger.info("Loaded %s sales files", index + 1)

    if not sales_dfs:
        raise ValueError("No sales files selected for training")

    return pd.concat(sales_dfs, ignore_index=True)


def aggregate_store_daily_sales(sales_df):
    existing_agg_dict = {
        col: agg_func
        for col, agg_func in STORE_DAILY_AGGREGATIONS.items()
        if col in sales_df.columns
    }

    store_daily_sales = (
        sales_df.groupby(["date", "store_id"])
        .agg(existing_agg_dict)
        .reset_index()
    )
    store_daily_sales["date"] = pd.to_datetime(store_daily_sales["date"])
    return store_daily_sales


def store_daily_categorical_columns(store_daily_sales):
    return [
        col
        for col in STORE_DAILY_CATEGORICAL_CANDIDATES
        if col in store_daily_sales.columns
    ]


def build_store_daily_training_data(sales_files):
    sales_df = read_sales_parquet_files(sales_files)
    return aggregate_store_daily_sales(sales_df)
