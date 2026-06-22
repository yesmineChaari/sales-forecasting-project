"""Daily-total Prophet helpers with aggregate Rossmann regressors."""

import numpy as np
import pandas as pd
from prophet import Prophet


PROPHET_DAILY_TOTAL_REGRESSORS = [
    "promo_store_count",
    "promo_rate",
    "open_store_count",
    "school_holiday_rate",
    "state_holiday_flag",
]
PROPHET_DAILY_TOTAL_VARIANT = "prophet_all_regressors"
PROPHET_DAILY_TOTAL_VARIANT_REGRESSORS = {
    "prophet_univariate": [],
    "prophet_promo_only": ["promo_store_count", "promo_rate"],
    "prophet_open_store_only": ["open_store_count"],
    "prophet_holiday_only": ["school_holiday_rate", "state_holiday_flag"],
    PROPHET_DAILY_TOTAL_VARIANT: PROPHET_DAILY_TOTAL_REGRESSORS,
}

NO_STATE_HOLIDAY_VALUES = {"", "0", "0.0", "none", "nan", "nat", "false"}


def binary_indicator(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    text = series.fillna("").astype(str).str.strip().str.lower()
    return ((numeric != 0) | text.isin({"true", "yes", "y"})).astype(int)


def state_holiday_indicator(series: pd.Series) -> pd.Series:
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return (~normalized.isin(NO_STATE_HOLIDAY_VALUES)).astype(int)


def build_daily_total_frame(
    df: pd.DataFrame,
    date_col: str = "date",
    target_col: str = "sales",
) -> pd.DataFrame:
    required_cols = {date_col, target_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns for Prophet daily-total data: {missing_cols}"
        )

    working_df = df.copy()
    working_df["ds"] = pd.to_datetime(
        working_df[date_col],
        errors="coerce",
    ).dt.normalize()
    working_df["y"] = pd.to_numeric(working_df[target_col], errors="coerce")
    working_df = working_df.dropna(subset=["ds", "y"])

    if working_df.empty:
        return pd.DataFrame(columns=["ds", "y"] + PROPHET_DAILY_TOTAL_REGRESSORS)

    if "store_id" in working_df.columns:
        working_df["_store_key"] = working_df["store_id"].astype(str)
    else:
        working_df["_store_key"] = np.arange(len(working_df))

    if "has_promotion" in working_df.columns:
        working_df["_promo_signal"] = binary_indicator(working_df["has_promotion"])
    else:
        working_df["_promo_signal"] = 0

    if "is_open" in working_df.columns:
        working_df["_open_signal"] = binary_indicator(working_df["is_open"])
    else:
        working_df["_open_signal"] = 1

    if "school_holiday" in working_df.columns:
        working_df["_school_holiday_signal"] = binary_indicator(
            working_df["school_holiday"]
        )
    else:
        working_df["_school_holiday_signal"] = 0

    if "state_holiday" in working_df.columns:
        working_df["_state_holiday_signal"] = state_holiday_indicator(
            working_df["state_holiday"]
        )
    else:
        working_df["_state_holiday_signal"] = 0

    store_day_df = (
        working_df.groupby(["ds", "_store_key"], as_index=False)
        .agg(
            y=("y", "sum"),
            promo_signal=("_promo_signal", "max"),
            open_signal=("_open_signal", "max"),
            school_holiday_signal=("_school_holiday_signal", "max"),
            state_holiday_signal=("_state_holiday_signal", "max"),
        )
    )

    daily_df = (
        store_day_df.groupby("ds", as_index=False)
        .agg(
            y=("y", "sum"),
            store_count=("_store_key", "nunique"),
            promo_store_count=("promo_signal", "sum"),
            open_store_count=("open_signal", "sum"),
            school_holiday_store_count=("school_holiday_signal", "sum"),
            state_holiday_flag=("state_holiday_signal", "max"),
        )
        .sort_values("ds")
    )

    store_count = daily_df["store_count"].replace(0, np.nan)
    daily_df["promo_rate"] = (
        daily_df["promo_store_count"] / store_count
    ).fillna(0.0)
    daily_df["school_holiday_rate"] = (
        daily_df["school_holiday_store_count"] / store_count
    ).fillna(0.0)

    daily_df = daily_df[["ds", "y"] + PROPHET_DAILY_TOTAL_REGRESSORS]
    for col in PROPHET_DAILY_TOTAL_REGRESSORS:
        daily_df[col] = pd.to_numeric(daily_df[col], errors="coerce").fillna(0.0)

    return daily_df


def get_prophet_variant_regressors(variant: str = PROPHET_DAILY_TOTAL_VARIANT):
    if variant not in PROPHET_DAILY_TOTAL_VARIANT_REGRESSORS:
        valid_variants = ", ".join(sorted(PROPHET_DAILY_TOTAL_VARIANT_REGRESSORS))
        raise ValueError(
            f"Unknown Prophet daily-total variant {variant!r}. "
            f"Valid variants: {valid_variants}"
        )
    return list(PROPHET_DAILY_TOTAL_VARIANT_REGRESSORS[variant])


def build_prophet_daily_total_model(prophet_params, regressor_cols=None):
    model = Prophet(**prophet_params)
    for regressor_col in regressor_cols or PROPHET_DAILY_TOTAL_REGRESSORS:
        model.add_regressor(regressor_col)
    return model
