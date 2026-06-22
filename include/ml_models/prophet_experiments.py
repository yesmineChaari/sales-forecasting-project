"""Lightweight daily-total Prophet diagnostics and baselines."""

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from prophet import Prophet

from ml_models.prophet_daily_total import (
    PROPHET_DAILY_TOTAL_REGRESSORS,
    PROPHET_DAILY_TOTAL_VARIANT_REGRESSORS,
    build_daily_total_frame,
)


BASELINE_VARIANTS = {
    "naive_yesterday": "yhat_naive_yesterday",
    "seasonal_naive_7": "yhat_seasonal_naive_7",
    "rolling_7": "yhat_rolling_7",
    "rolling_28": "yhat_rolling_28",
}

PROPHET_VARIANTS = {
    variant: list(regressors)
    for variant, regressors in PROPHET_DAILY_TOTAL_VARIANT_REGRESSORS.items()
}

EVALUATION_SPLITS = ("validation", "test")


def _ensure_daily_total_frame(df: pd.DataFrame) -> pd.DataFrame:
    if {"ds", "y"}.issubset(df.columns):
        daily_df = df.copy()
    else:
        daily_df = build_daily_total_frame(df)

    daily_df["ds"] = pd.to_datetime(daily_df["ds"], errors="coerce").dt.normalize()
    daily_df["y"] = pd.to_numeric(daily_df["y"], errors="coerce")
    daily_df = daily_df.dropna(subset=["ds", "y"]).sort_values("ds")

    for regressor_col in PROPHET_DAILY_TOTAL_REGRESSORS:
        if regressor_col not in daily_df.columns:
            daily_df[regressor_col] = 0.0
        daily_df[regressor_col] = pd.to_numeric(
            daily_df[regressor_col],
            errors="coerce",
        ).fillna(0.0)

    return daily_df[["ds", "y"] + PROPHET_DAILY_TOTAL_REGRESSORS].reset_index(
        drop=True
    )


def _with_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    daily_df = _ensure_daily_total_frame(df)
    daily_df["split"] = split
    return daily_df


def _metric_value(value: float) -> float:
    if pd.isna(value) or np.isinf(value):
        return float("nan")
    return float(value)


def compute_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    metric_df = pd.DataFrame({"y": y_true, "yhat": y_pred})
    metric_df = metric_df.replace([np.inf, -np.inf], np.nan).dropna()

    if metric_df.empty:
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan, "r2": np.nan}

    actual = metric_df["y"].astype(float).to_numpy()
    predicted = metric_df["yhat"].astype(float).to_numpy()
    errors = actual - predicted

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(np.square(errors)))

    nonzero_mask = actual != 0
    if nonzero_mask.any():
        mape = np.mean(
            np.abs(errors[nonzero_mask] / actual[nonzero_mask])
        ) * 100
    else:
        mape = 0.0

    ss_res = np.sum(np.square(errors))
    ss_tot = np.sum(np.square(actual - np.mean(actual)))
    if ss_tot == 0:
        r2 = 1.0 if ss_res == 0 else 0.0
    else:
        r2 = 1 - (ss_res / ss_tot)

    return {
        "mae": _metric_value(mae),
        "rmse": _metric_value(rmse),
        "mape": _metric_value(mape),
        "r2": _metric_value(r2),
    }


def _add_baseline_predictions(full_df: pd.DataFrame) -> pd.DataFrame:
    predicted_df = full_df.sort_values("ds").reset_index(drop=True).copy()

    shifted_y = predicted_df["y"].shift(1)
    predicted_df["yhat_naive_yesterday"] = shifted_y
    predicted_df["yhat_seasonal_naive_7"] = predicted_df["y"].shift(7)
    predicted_df["yhat_rolling_7"] = shifted_y.rolling(
        window=7,
        min_periods=7,
    ).mean()
    predicted_df["yhat_rolling_28"] = shifted_y.rolling(
        window=28,
        min_periods=28,
    ).mean()

    return predicted_df


def _fit_prophet_variant(
    train_df: pd.DataFrame,
    regressor_cols: List[str],
    prophet_params: Optional[Dict[str, Any]],
    prophet_factory,
):
    model = prophet_factory(**(prophet_params or {}))
    for regressor_col in regressor_cols:
        model.add_regressor(regressor_col)

    fit_cols = ["ds", "y"] + regressor_cols
    model.fit(train_df[fit_cols])
    return model


def _predict_prophet_variant(
    model,
    predict_df: pd.DataFrame,
    regressor_cols: List[str],
) -> np.ndarray:
    predict_cols = ["ds"] + regressor_cols
    forecast_df = model.predict(predict_df[predict_cols])
    if "yhat" not in forecast_df.columns:
        raise ValueError("Prophet forecast output must contain a yhat column")

    return pd.to_numeric(forecast_df["yhat"], errors="coerce").to_numpy()


def _baseline_metric_rows(full_df: pd.DataFrame) -> List[Dict[str, Any]]:
    metric_rows = []

    for variant, prediction_col in BASELINE_VARIANTS.items():
        for split in EVALUATION_SPLITS:
            split_df = full_df[full_df["split"] == split]
            metrics = compute_metrics(split_df["y"], split_df[prediction_col])
            metric_rows.append(
                {
                    "variant": variant,
                    "split": split,
                    **metrics,
                }
            )

    return metric_rows


def _prophet_metric_rows(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    prophet_params: Optional[Dict[str, Any]],
    prophet_factory,
) -> List[Dict[str, Any]]:
    metric_rows = []

    for variant, regressor_cols in PROPHET_VARIANTS.items():
        model = _fit_prophet_variant(
            train_df,
            regressor_cols,
            prophet_params,
            prophet_factory,
        )
        for split, predict_df in (("validation", val_df), ("test", test_df)):
            predictions = _predict_prophet_variant(
                model,
                predict_df,
                regressor_cols,
            )
            metrics = compute_metrics(predict_df["y"], predictions)
            metric_rows.append(
                {
                    "variant": variant,
                    "split": split,
                    **metrics,
                }
            )

    return metric_rows


def _add_seasonal_comparison(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metrics_df = metrics_df.copy()
    seasonal_rmse = (
        metrics_df[metrics_df["variant"] == "seasonal_naive_7"]
        .set_index("split")["rmse"]
        .to_dict()
    )

    def beats_seasonal(row):
        baseline_rmse = seasonal_rmse.get(row["split"])
        if row["variant"] == "seasonal_naive_7":
            return False
        if pd.isna(row["rmse"]) or baseline_rmse is None or pd.isna(baseline_rmse):
            return False
        return bool(row["rmse"] < baseline_rmse)

    metrics_df["beats_seasonal_naive_7"] = metrics_df.apply(
        beats_seasonal,
        axis=1,
    )
    return metrics_df


def run_prophet_experiments(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    prophet_params: Optional[Dict[str, Any]] = None,
    prophet_factory=Prophet,
) -> pd.DataFrame:
    """Evaluate daily-total naive baselines and Prophet variants.

    The inputs may already be daily-total frames with ``ds`` and ``y`` columns,
    or raw Rossmann-like rows that can be aggregated with
    ``build_daily_total_frame``. No store-level model outputs are accepted or
    compared here.
    """

    daily_train = _with_split(train_df, "train")
    daily_val = _with_split(val_df, "validation")
    daily_test = _with_split(test_df, "test")

    full_df = _add_baseline_predictions(
        pd.concat([daily_train, daily_val, daily_test], ignore_index=True)
    )

    metric_rows = []
    metric_rows.extend(_baseline_metric_rows(full_df))
    metric_rows.extend(
        _prophet_metric_rows(
            daily_train,
            daily_val,
            daily_test,
            prophet_params,
            prophet_factory,
        )
    )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df = _add_seasonal_comparison(metrics_df)
    metrics_df = metrics_df[
        [
            "variant",
            "split",
            "mae",
            "rmse",
            "mape",
            "r2",
            "beats_seasonal_naive_7",
        ]
    ]
    metrics_df.attrs["verdict"] = summarize_prophet_experiment_results(metrics_df)
    return metrics_df


def select_best_variant(
    metrics_df: pd.DataFrame,
    split: str = "test",
    metric: str = "rmse",
) -> Optional[str]:
    split_df = metrics_df[metrics_df["split"] == split].dropna(subset=[metric])
    if split_df.empty:
        return None

    ascending = metric != "r2"
    best_row = split_df.sort_values(metric, ascending=ascending).iloc[0]
    return str(best_row["variant"])


def summarize_prophet_experiment_results(metrics_df: pd.DataFrame) -> Dict[str, Any]:
    test_df = metrics_df[metrics_df["split"] == "test"].copy()
    best_variant = select_best_variant(metrics_df, split="test", metric="rmse")

    seasonal_rows = test_df[test_df["variant"] == "seasonal_naive_7"]
    seasonal_rmse = (
        float(seasonal_rows.iloc[0]["rmse"]) if not seasonal_rows.empty else np.nan
    )

    prophet_rows = test_df[test_df["variant"].isin(PROPHET_VARIANTS.keys())]
    best_prophet_variant = None
    best_prophet_rmse = np.nan
    if not prophet_rows.dropna(subset=["rmse"]).empty:
        best_prophet_row = prophet_rows.dropna(subset=["rmse"]).sort_values(
            "rmse"
        ).iloc[0]
        best_prophet_variant = str(best_prophet_row["variant"])
        best_prophet_rmse = float(best_prophet_row["rmse"])

    univariate_rows = test_df[test_df["variant"] == "prophet_univariate"]
    univariate_rmse = (
        float(univariate_rows.iloc[0]["rmse"]) if not univariate_rows.empty else np.nan
    )

    regressor_variants = [
        variant
        for variant in PROPHET_VARIANTS.keys()
        if variant != "prophet_univariate"
    ]
    regressor_rows = test_df[test_df["variant"].isin(regressor_variants)]
    best_regressor_variant = None
    best_regressor_rmse = np.nan
    if not regressor_rows.dropna(subset=["rmse"]).empty:
        best_regressor_row = regressor_rows.dropna(subset=["rmse"]).sort_values(
            "rmse"
        ).iloc[0]
        best_regressor_variant = str(best_regressor_row["variant"])
        best_regressor_rmse = float(best_regressor_row["rmse"])

    return {
        "prophet_beats_seasonal_naive_7": bool(
            not pd.isna(best_prophet_rmse)
            and not pd.isna(seasonal_rmse)
            and best_prophet_rmse < seasonal_rmse
        ),
        "regressors_improve_over_prophet_univariate": bool(
            not pd.isna(best_regressor_rmse)
            and not pd.isna(univariate_rmse)
            and best_regressor_rmse < univariate_rmse
        ),
        "best_variant": best_variant,
        "best_prophet_variant": best_prophet_variant,
        "best_regressor_variant": best_regressor_variant,
        "seasonal_naive_7_test_rmse": seasonal_rmse,
        "best_prophet_test_rmse": best_prophet_rmse,
        "best_regressor_test_rmse": best_regressor_rmse,
    }
