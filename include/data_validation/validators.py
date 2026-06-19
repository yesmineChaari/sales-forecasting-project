from __future__ import annotations

import logging
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

try:
    from pandera import Check, Column, DataFrameSchema
except ImportError:  # Pandera is optional for the Airflow validation path.
    Check = None
    Column = None
    DataFrameSchema = None


logger = logging.getLogger(__name__)

AIRFLOW_CONFIG_PATH = "/usr/local/airflow/include/config/ml_config.yaml"

DEFAULT_REQUIRED_COLUMNS = [
    "date",
    "store_id",
    "sales",
    "customer_traffic",
    "has_promotion",
    "is_open",
    "is_holiday",
]

DEFAULT_DATA_TYPES = {
    "date": "datetime64[ns]",
    "store_id": "object",
    "sales": "float64",
    "customer_traffic": "float64",
    "has_promotion": "int64",
    "is_open": "int64",
    "is_holiday": "int64",
    "school_holiday": "int64",
    "state_holiday": "object",
    "store_type": "object",
    "assortment": "object",
    "competition_distance": "float64",
    "promo2": "int64",
    "promo_interval": "object",
}

DEFAULT_VALUE_RANGES = {
    "sales": {"min": 0, "max": 10_000_000},
    "customer_traffic": {"min": 0},
    "competition_distance": {"min": 0},
    "has_promotion": {"min": 0, "max": 1},
    "is_open": {"min": 0, "max": 1},
    "is_holiday": {"min": 0, "max": 1},
    "school_holiday": {"min": 0, "max": 1},
    "promo2": {"min": 0, "max": 1},
}

DEFAULT_BINARY_COLUMNS = [
    "has_promotion",
    "is_open",
    "is_holiday",
    "school_holiday",
    "promo2",
]

DEFAULT_UNIQUE_KEY = ["date", "store_id"]
DEFAULT_NON_NULL_COLUMNS = DEFAULT_REQUIRED_COLUMNS
PATH_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _local_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "ml_config.yaml"


def _resolve_config_path(config_path: str) -> Path:
    path = Path(config_path)
    if path.exists():
        return path

    if str(config_path) == AIRFLOW_CONFIG_PATH:
        local_path = _local_config_path()
        if local_path.exists():
            return local_path

    raise FileNotFoundError(f"Validation config not found: {config_path}")


def _as_list(value: Any, default: Sequence[str]) -> List[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [value]
    return list(value)


def _merge_mapping(
    default: Mapping[str, Any],
    override: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    merged = dict(default)
    if override:
        merged.update(override)
    return merged


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _to_datetime(values: Any) -> pd.Series:
    try:
        return pd.to_datetime(values, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(values, errors="coerce")


def parse_file_limit(raw_value: Any, env_name: str) -> int:
    try:
        file_limit = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{env_name} must be an integer, got {raw_value!r}") from exc

    return max(file_limit, 0)


def limit_files(
    file_paths: Iterable[str],
    raw_limit: Any,
    env_name: str,
) -> Tuple[List[str], int]:
    file_limit = parse_file_limit(raw_limit, env_name)
    selected_files = list(file_paths)
    if file_limit > 0:
        selected_files = selected_files[:file_limit]

    return selected_files, file_limit


def resolve_sales_file_selection(
    sales_files: Iterable[str],
    max_sales_files: Any,
    validation_max_sales_files: Any,
) -> Dict[str, Any]:
    sales_files = list(sales_files)
    training_files, training_limit = limit_files(
        sales_files,
        max_sales_files,
        "MAX_SALES_FILES",
    )
    validation_files, validation_limit = limit_files(
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


class DataValidator:
    def __init__(
        self,
        config_path: Optional[str] = None,
        max_issue_examples: Optional[int] = None,
    ):
        config_path = config_path or os.getenv("ML_CONFIG_PATH", AIRFLOW_CONFIG_PATH)
        self.config_path = str(_resolve_config_path(config_path))

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f) or {}

        self.validation_config = self.config.get("validation") or {}
        self.required_columns = _as_list(
            self.validation_config.get("required_columns"),
            DEFAULT_REQUIRED_COLUMNS,
        )
        self.data_types = _merge_mapping(
            DEFAULT_DATA_TYPES,
            self.validation_config.get("data_types"),
        )
        self.value_ranges = _merge_mapping(
            DEFAULT_VALUE_RANGES,
            self.validation_config.get("value_ranges"),
        )
        self.non_null_columns = _as_list(
            self.validation_config.get("non_null_columns"),
            self.required_columns,
        )
        self.binary_columns = _as_list(
            self.validation_config.get("binary_columns"),
            DEFAULT_BINARY_COLUMNS,
        )
        self.unique_key = _as_list(
            self.validation_config.get("unique_key"),
            DEFAULT_UNIQUE_KEY,
        )
        self.file_date_column = self.validation_config.get("file_date_column", "date")
        self.expected_frequency = self.validation_config.get("expected_frequency", "D")
        self.warn_on_future_dates = bool(
            self.validation_config.get("warn_on_future_dates", True)
        )
        self.fail_on_future_dates = bool(
            self.validation_config.get("fail_on_future_dates", False)
        )
        self.max_outlier_percentage = float(
            self.validation_config.get("max_outlier_percentage", 10.0)
        )
        configured_max_examples = self.validation_config.get("max_issue_examples", 25)
        self.max_issue_examples = int(
            max_issue_examples
            if max_issue_examples is not None
            else configured_max_examples
        )

    @staticmethod
    def _issue(
        severity: str,
        check: str,
        message: str,
        file_path: Optional[str] = None,
        column: Optional[str] = None,
        count: Optional[int] = None,
    ) -> Dict[str, Any]:
        issue = {
            "severity": severity,
            "check": check,
            "message": message,
        }
        if file_path:
            issue["file_path"] = str(file_path)
        if column:
            issue["column"] = column
        if count is not None:
            issue["count"] = int(count)
        return issue

    @staticmethod
    def format_issue(issue: Mapping[str, Any]) -> str:
        prefix = f"{issue.get('severity', 'error').upper()} {issue.get('check')}"
        location = issue.get("file_path")
        column = issue.get("column")
        if location:
            prefix = f"{prefix} [{location}]"
        if column:
            prefix = f"{prefix} column={column}"
        return f"{prefix}: {issue.get('message')}"

    def _schema_issues(
        self,
        df: pd.DataFrame,
        file_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []

        missing_columns = sorted(set(self.required_columns) - set(df.columns))
        if missing_columns:
            issues.append(
                self._issue(
                    "error",
                    "schema",
                    f"Missing required columns: {missing_columns}",
                    file_path=file_path,
                    count=len(missing_columns),
                )
            )

        for col, expected_type in self.data_types.items():
            if col not in df.columns:
                continue
            issues.extend(
                self._coerce_column(df, col, str(expected_type), file_path=file_path)
            )

        return issues

    def _coerce_column(
        self,
        df: pd.DataFrame,
        col: str,
        expected_type: str,
        file_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        original = df[col]
        expected_type = expected_type.lower()

        if expected_type.startswith("datetime"):
            converted = _to_datetime(original)
            invalid_count = int((original.notna() & converted.isna()).sum())
            df[col] = converted
            if invalid_count:
                issues.append(
                    self._issue(
                        "error",
                        "schema",
                        f"{invalid_count} value(s) cannot be parsed as datetime",
                        file_path=file_path,
                        column=col,
                        count=invalid_count,
                    )
                )
            return issues

        if expected_type.startswith("float") or expected_type.startswith("int"):
            converted = pd.to_numeric(original, errors="coerce")
            invalid_count = int((original.notna() & converted.isna()).sum())
            if invalid_count:
                issues.append(
                    self._issue(
                        "error",
                        "schema",
                        f"{invalid_count} value(s) cannot be converted to {expected_type}",
                        file_path=file_path,
                        column=col,
                        count=invalid_count,
                    )
                )
            if expected_type.startswith("int"):
                fractional_count = int(
                    (converted.notna() & ((converted % 1) != 0)).sum()
                )
                if fractional_count:
                    issues.append(
                        self._issue(
                            "error",
                            "schema",
                            f"{fractional_count} non-integer value(s) found",
                            file_path=file_path,
                            column=col,
                            count=fractional_count,
                        )
                    )
                if not invalid_count and not fractional_count:
                    df[col] = (
                        converted.astype("Int64")
                        if converted.isna().any()
                        else converted.astype(expected_type)
                    )
                else:
                    df[col] = converted
            else:
                df[col] = converted.astype(expected_type)
            return issues

        if expected_type in {"object", "str", "string"}:
            df[col] = original.astype("string" if expected_type == "string" else "object")
            return issues

        try:
            df[col] = original.astype(expected_type)
        except Exception as exc:
            issues.append(
                self._issue(
                    "error",
                    "schema",
                    f"Cannot convert {original.dtype} to {expected_type}: {exc}",
                    file_path=file_path,
                    column=col,
                )
            )
        return issues

    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        issues = self._schema_issues(df)
        errors = [self.format_issue(issue) for issue in issues]
        is_valid = not any(issue["severity"] == "error" for issue in issues)
        return is_valid, errors

    def _quality_issues(
        self,
        df: pd.DataFrame,
        file_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        row_count = len(df)

        if row_count == 0:
            return [
                self._issue(
                    "error",
                    "data_quality",
                    "File contains no rows",
                    file_path=file_path,
                )
            ]

        duplicate_rows = int(df.duplicated().sum())
        if duplicate_rows:
            issues.append(
                self._issue(
                    "warning",
                    "duplicate_rows",
                    f"Found {duplicate_rows} exact duplicate row(s)",
                    file_path=file_path,
                    count=duplicate_rows,
                )
            )

        if all(col in df.columns for col in self.unique_key):
            duplicate_keys = int(df.duplicated(subset=self.unique_key).sum())
            if duplicate_keys:
                issues.append(
                    self._issue(
                        "error",
                        "unique_key",
                        (
                            f"Found {duplicate_keys} duplicate row(s) at "
                            f"grain {self.unique_key}"
                        ),
                        file_path=file_path,
                        count=duplicate_keys,
                    )
                )

        for col in self.non_null_columns:
            if col not in df.columns:
                continue
            null_count = int(df[col].isna().sum())
            if null_count:
                issues.append(
                    self._issue(
                        "error",
                        "nulls",
                        f"{null_count} null value(s) in required column",
                        file_path=file_path,
                        column=col,
                        count=null_count,
                    )
                )

        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            finite_values = df[col].replace([np.inf, -np.inf], np.nan)
            infinite_count = int(np.isinf(df[col]).sum())
            if infinite_count:
                issues.append(
                    self._issue(
                        "error",
                        "numeric_values",
                        f"{infinite_count} infinite value(s)",
                        file_path=file_path,
                        column=col,
                        count=infinite_count,
                    )
                )

            if col in self.value_ranges and finite_values.notna().any():
                range_config = self.value_ranges[col] or {}
                if "min" in range_config:
                    below_min = int((finite_values < range_config["min"]).sum())
                    if below_min:
                        issues.append(
                            self._issue(
                                "error",
                                "value_range",
                                (
                                    f"{below_min} value(s) below minimum "
                                    f"{range_config['min']}"
                                ),
                                file_path=file_path,
                                column=col,
                                count=below_min,
                            )
                        )
                if "max" in range_config:
                    above_max = int((finite_values > range_config["max"]).sum())
                    if above_max:
                        issues.append(
                            self._issue(
                                "error",
                                "value_range",
                                (
                                    f"{above_max} value(s) above maximum "
                                    f"{range_config['max']}"
                                ),
                                file_path=file_path,
                                column=col,
                                count=above_max,
                            )
                        )

            outliers = self._detect_outliers(finite_values.dropna())
            if row_count and (outliers / row_count) * 100 > self.max_outlier_percentage:
                issues.append(
                    self._issue(
                        "warning",
                        "outliers",
                        (
                            f"{outliers} outlier value(s), above "
                            f"{self.max_outlier_percentage:.1f}% threshold"
                        ),
                        file_path=file_path,
                        column=col,
                        count=outliers,
                    )
                )

        for col in self.binary_columns:
            if col not in df.columns:
                continue
            non_null_values = df[col].dropna()
            invalid_count = int((~non_null_values.isin([0, 1, True, False])).sum())
            if invalid_count:
                examples = non_null_values[
                    ~non_null_values.isin([0, 1, True, False])
                ].head(5)
                issues.append(
                    self._issue(
                        "error",
                        "binary_domain",
                        (
                            f"{invalid_count} non-binary value(s); "
                            f"examples={examples.tolist()}"
                        ),
                        file_path=file_path,
                        column=col,
                        count=invalid_count,
                    )
                )

        if self.file_date_column in df.columns:
            issues.extend(self._date_quality_issues(df, file_path=file_path))

        return issues

    def _date_quality_issues(
        self,
        df: pd.DataFrame,
        file_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        dates = _to_datetime(df[self.file_date_column])
        valid_dates = dates.dropna()

        if valid_dates.empty:
            return [
                self._issue(
                    "error",
                    "time_series",
                    f"No valid values in date column {self.file_date_column}",
                    file_path=file_path,
                    column=self.file_date_column,
                )
            ]

        today = pd.Timestamp.today().normalize()
        future_count = int((valid_dates.dt.normalize() > today).sum())
        if future_count and self.warn_on_future_dates:
            severity = "error" if self.fail_on_future_dates else "warning"
            issues.append(
                self._issue(
                    severity,
                    "time_series",
                    f"{future_count} future-dated row(s)",
                    file_path=file_path,
                    column=self.file_date_column,
                    count=future_count,
                )
            )

        normalized_dates = pd.DatetimeIndex(valid_dates.dt.normalize().unique())
        if file_path and len(normalized_dates) > 1:
            issues.append(
                self._issue(
                    "error",
                    "file_partition",
                    (
                        "Sales parquet files should contain one sales date; "
                        f"found {len(normalized_dates)}"
                    ),
                    file_path=file_path,
                    column=self.file_date_column,
                    count=len(normalized_dates),
                )
            )

        path_date = self._date_from_path(file_path) if file_path else None
        if path_date is not None and len(normalized_dates) == 1:
            file_date = pd.Timestamp(normalized_dates[0]).normalize()
            if file_date != path_date:
                issues.append(
                    self._issue(
                        "error",
                        "file_partition",
                        (
                            "File partition date does not match row date "
                            f"({path_date.date()} != {file_date.date()})"
                        ),
                        file_path=file_path,
                        column=self.file_date_column,
                    )
                )

        return issues

    @staticmethod
    def _date_from_path(file_path: Optional[str]) -> Optional[pd.Timestamp]:
        if not file_path:
            return None
        match = PATH_DATE_RE.search(str(file_path))
        if not match:
            return None
        return pd.Timestamp(match.group(1)).normalize()

    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        quality_report = {
            "total_rows": len(df),
            "column_stats": {},
            "quality_issues": [],
            "issue_details": [],
        }

        row_count = len(df)
        for col in df.columns:
            null_count = int(df[col].isnull().sum())
            col_stats = {
                "null_count": null_count,
                "null_percentage": 0.0
                if row_count == 0
                else (null_count / row_count) * 100,
                "unique_values": int(df[col].nunique(dropna=True)),
            }

            if pd.api.types.is_numeric_dtype(df[col]):
                finite_values = df[col].replace([np.inf, -np.inf], np.nan)
                col_stats.update(
                    {
                        "mean": _json_safe(finite_values.mean()),
                        "std": _json_safe(finite_values.std()),
                        "min": _json_safe(finite_values.min()),
                        "max": _json_safe(finite_values.max()),
                        "outliers": int(self._detect_outliers(finite_values.dropna())),
                    }
                )

            quality_report["column_stats"][col] = col_stats

        issues = self._quality_issues(df)
        quality_report["issue_details"] = issues
        quality_report["quality_issues"] = [self.format_issue(issue) for issue in issues]
        return quality_report

    def _detect_outliers(self, series: pd.Series, method: str = "iqr") -> int:
        series = series.dropna()
        if series.empty:
            return 0

        if method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                return 0
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            return int(((series < lower_bound) | (series > upper_bound)).sum())
        if method == "zscore":
            std = series.std()
            if std == 0 or pd.isna(std):
                return 0
            z_scores = np.abs((series - series.mean()) / std)
            return int((z_scores > 3).sum())
        return 0

    def create_pandera_schema(self) -> DataFrameSchema:
        if DataFrameSchema is None:
            raise ImportError(
                "pandera is required to create a Pandera schema. "
                "Install pandera or use generate_validation_report/validate_sales_files."
            )

        schema_dict = {}

        for col, dtype in self.data_types.items():
            checks = []

            if col in self.value_ranges:
                range_config = self.value_ranges[col] or {}
                if "min" in range_config:
                    checks.append(Check.greater_than_or_equal_to(range_config["min"]))
                if "max" in range_config:
                    checks.append(Check.less_than_or_equal_to(range_config["max"]))

            pandera_dtype = "datetime64" if dtype == "datetime64[ns]" else dtype
            schema_dict[col] = Column(pandera_dtype, checks=checks, nullable=True)

        return DataFrameSchema(schema_dict)

    def validate_time_series(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
        group_cols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        ts_report: Dict[str, Any] = {
            "date_range": {},
            "frequency_issues": [],
            "gaps": [],
        }

        if date_col not in df.columns:
            ts_report["frequency_issues"].append(f"Missing date column: {date_col}")
            return ts_report

        date_values = _to_datetime(df[date_col]).dropna()
        if date_values.empty:
            ts_report["frequency_issues"].append(f"No valid dates in {date_col}")
            return ts_report

        unique_dates = pd.DatetimeIndex(date_values.dt.normalize().unique()).sort_values()
        ts_report["date_range"] = {
            "start": unique_dates.min().strftime("%Y-%m-%d"),
            "end": unique_dates.max().strftime("%Y-%m-%d"),
            "days": int((unique_dates.max() - unique_dates.min()).days),
            "unique_dates": int(len(unique_dates)),
        }

        missing_dates = self._missing_dates(unique_dates)
        if len(missing_dates):
            ts_report["frequency_issues"].append(
                f"Missing {len(missing_dates)} date(s) at {self.expected_frequency} frequency"
            )
            ts_report["gaps"].append(
                {
                    "gap_count": int(len(missing_dates)),
                    "sample_missing_dates": [
                        value.strftime("%Y-%m-%d") for value in missing_dates[:10]
                    ],
                }
            )

        if group_cols:
            available_group_cols = [col for col in group_cols if col in df.columns]
            if available_group_cols:
                working_df = df[available_group_cols + [date_col]].copy()
                working_df[date_col] = _to_datetime(working_df[date_col])
                working_df = working_df.dropna(subset=[date_col])
                group_gap_reports = []
                for group, group_df in working_df.groupby(available_group_cols):
                    group_dates = pd.DatetimeIndex(
                        group_df[date_col].dt.normalize().unique()
                    ).sort_values()
                    group_missing = self._missing_dates(group_dates)
                    if len(group_missing):
                        group_gap_reports.append(
                            {
                                "group": group,
                                "gap_count": int(len(group_missing)),
                                "max_gap_days": self._max_gap_days(group_dates),
                            }
                        )
                    if len(group_gap_reports) >= self.max_issue_examples:
                        break
                if group_gap_reports:
                    ts_report["gaps"].extend(group_gap_reports)

        return ts_report

    def _missing_dates(self, unique_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        if len(unique_dates) <= 1:
            return pd.DatetimeIndex([])
        expected_dates = pd.date_range(
            unique_dates.min(),
            unique_dates.max(),
            freq=self.expected_frequency,
        )
        return expected_dates.difference(unique_dates)

    @staticmethod
    def _max_gap_days(unique_dates: pd.DatetimeIndex) -> int:
        if len(unique_dates) <= 1:
            return 0
        diffs = pd.Series(unique_dates).diff().dropna()
        if diffs.empty:
            return 0
        return int(diffs.max().days)

    def validate_prediction_data(
        self,
        df: pd.DataFrame,
        training_stats: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        errors = []

        is_valid, schema_errors = self.validate_schema(df)
        errors.extend(schema_errors)

        for col in df.select_dtypes(include=[np.number]).columns:
            if col in training_stats:
                train_mean = training_stats[col]["mean"]
                train_std = training_stats[col]["std"]
                if train_std == 0:
                    continue

                pred_mean = df[col].mean()
                if abs(pred_mean - train_mean) > 3 * train_std:
                    errors.append(
                        f"Potential distribution shift in {col}: "
                        f"mean changed from {train_mean:.2f} to {pred_mean:.2f}"
                    )

        return is_valid and len(errors) == 0, errors

    def generate_validation_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Starting data validation")

        working_df = df.copy()
        report: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "dataset_info": {
                "rows": len(working_df),
                "columns": len(working_df.columns),
                "memory_usage": working_df.memory_usage(deep=True).sum() / 1024**2,
            },
        }

        schema_issues = self._schema_issues(working_df)
        report["schema_validation"] = {
            "is_valid": not any(
                issue["severity"] == "error" for issue in schema_issues
            ),
            "errors": [self.format_issue(issue) for issue in schema_issues],
            "issue_details": schema_issues,
        }

        report["data_quality"] = self.validate_data_quality(working_df)

        all_issues = schema_issues + report["data_quality"]["issue_details"]
        if self.file_date_column in working_df.columns:
            report["time_series_validation"] = self.validate_time_series(
                working_df,
                date_col=self.file_date_column,
            )

        report["is_valid"] = not any(
            issue["severity"] == "error" for issue in all_issues
        )
        report["issues"] = [
            self.format_issue(issue) for issue in all_issues[: self.max_issue_examples]
        ]
        report["issue_details"] = all_issues[: self.max_issue_examples]
        report["issues_found"] = len(all_issues)

        logger.info("Validation complete. Valid: %s", report["is_valid"])
        return report

    def validate_sales_files(
        self,
        sales_files: Iterable[str],
        max_sales_files: Any = "0",
        validation_max_sales_files: Any = "0",
    ) -> Dict[str, Any]:
        selection = resolve_sales_file_selection(
            sales_files,
            max_sales_files,
            validation_max_sales_files,
        )
        training_files = selection["training_files"]
        validation_files = selection["validation_files"]

        messages = self._selection_messages(selection)
        issues: List[Dict[str, Any]] = []
        file_summaries: List[Dict[str, Any]] = []
        total_rows = 0
        sample_columns: List[str] = []
        all_dates: set[str] = set()
        seen_key_hashes: set[int] = set()

        if not validation_files:
            issues.append(
                self._issue(
                    "error",
                    "file_selection",
                    "No sales files selected for validation",
                )
            )

        for index, sales_file in enumerate(validation_files):
            file_issues: List[Dict[str, Any]] = []
            try:
                raw_df = pd.read_parquet(sales_file)
            except Exception as exc:
                file_issues.append(
                    self._issue(
                        "error",
                        "file_read",
                        f"Could not read parquet file: {exc}",
                        file_path=sales_file,
                    )
                )
                issues.extend(file_issues)
                file_summaries.append(
                    self._file_summary(sales_file, 0, file_issues)
                )
                continue

            if index == 0:
                sample_columns = raw_df.columns.tolist()

            working_df = raw_df.copy()
            total_rows += len(working_df)
            file_issues.extend(self._schema_issues(working_df, file_path=sales_file))
            file_issues.extend(self._quality_issues(working_df, file_path=sales_file))
            file_issues.extend(
                self._cross_file_key_issues(
                    working_df,
                    seen_key_hashes,
                    file_path=sales_file,
                )
            )
            self._collect_dates(working_df, all_dates)

            issues.extend(file_issues)
            file_summaries.append(
                self._file_summary(sales_file, len(working_df), file_issues)
            )

        issues.extend(self._batch_time_series_issues(all_dates))

        severity_counts = Counter(issue["severity"] for issue in issues)
        check_counts = Counter(issue["check"] for issue in issues)
        issue_sample = issues[: self.max_issue_examples]
        file_summary_sample = file_summaries[: self.max_issue_examples]

        return {
            "timestamp": datetime.now().isoformat(),
            "config_path": self.config_path,
            "total_files_available": selection["total_files_available"],
            "total_training_files_selected": len(training_files),
            "total_files_validated": len(validation_files),
            "validation_mode": selection["validation_mode"],
            "max_sales_files": selection["training_limit"],
            "validation_max_sales_files": selection["validation_limit"],
            "total_rows": int(total_rows),
            "issues_found": len(issues),
            "blocking_issues_found": int(severity_counts.get("error", 0)),
            "warnings_found": int(severity_counts.get("warning", 0)),
            "issues": [self.format_issue(issue) for issue in issue_sample],
            "issue_details": issue_sample,
            "issues_truncated": len(issues) > len(issue_sample),
            "issue_counts_by_severity": dict(severity_counts),
            "issue_counts_by_check": dict(check_counts),
            "total_files_with_issues": sum(
                1 for item in file_summaries if item["issues_found"]
            ),
            "file_summaries": file_summary_sample,
            "file_summaries_truncated": len(file_summaries) > len(file_summary_sample),
            "sample_columns": sample_columns,
            "date_range": self._date_range_from_strings(all_dates),
            "messages": messages,
        }

    def _selection_messages(self, selection: Mapping[str, Any]) -> List[str]:
        messages = []
        total_available = selection["total_files_available"]
        total_training_selected = len(selection["training_files"])
        total_validated = len(selection["validation_files"])

        if selection["training_limit"] > 0:
            messages.append(
                "Training file cap enabled: "
                f"using first {total_training_selected} of {total_available} "
                "sales files."
            )

        if selection["validation_mode"] == "sample":
            messages.append(
                "Sample validation enabled: "
                f"validating first {total_validated} of "
                f"{total_training_selected} files."
            )
        else:
            messages.append(
                "Full validation enabled: "
                f"validating all {total_validated} selected training files."
            )

        return messages

    def _cross_file_key_issues(
        self,
        df: pd.DataFrame,
        seen_key_hashes: set[int],
        file_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not all(col in df.columns for col in self.unique_key):
            return []

        key_df = df[self.unique_key].dropna().copy()
        if key_df.empty:
            return []

        key_hashes = pd.util.hash_pandas_object(key_df, index=False).map(int)
        duplicate_mask = key_hashes.isin(seen_key_hashes)
        duplicate_count = int(duplicate_mask.sum())
        seen_key_hashes.update(key_hashes.drop_duplicates().tolist())

        if duplicate_count:
            return [
                self._issue(
                    "error",
                    "unique_key",
                    (
                        f"Found {duplicate_count} row(s) at grain "
                        f"{self.unique_key} already present in earlier files"
                    ),
                    file_path=file_path,
                    count=duplicate_count,
                )
            ]
        return []

    def _collect_dates(self, df: pd.DataFrame, all_dates: set[str]) -> None:
        if self.file_date_column not in df.columns:
            return
        dates = _to_datetime(df[self.file_date_column]).dropna()
        for value in pd.DatetimeIndex(dates.dt.normalize().unique()):
            all_dates.add(value.strftime("%Y-%m-%d"))

    def _batch_time_series_issues(self, all_dates: set[str]) -> List[Dict[str, Any]]:
        if not all_dates:
            return []
        unique_dates = pd.DatetimeIndex(sorted(pd.Timestamp(value) for value in all_dates))
        missing_dates = self._missing_dates(unique_dates)
        if not len(missing_dates):
            return []
        return [
            self._issue(
                "warning",
                "time_series",
                (
                    f"Missing {len(missing_dates)} date(s) between selected "
                    f"validation files; sample="
                    f"{[value.strftime('%Y-%m-%d') for value in missing_dates[:10]]}"
                ),
                count=len(missing_dates),
            )
        ]

    @staticmethod
    def _date_range_from_strings(all_dates: set[str]) -> Dict[str, Any]:
        if not all_dates:
            return {}
        sorted_dates = sorted(all_dates)
        return {
            "start": sorted_dates[0],
            "end": sorted_dates[-1],
            "unique_dates": len(sorted_dates),
        }

    def _file_summary(
        self,
        file_path: str,
        rows: int,
        issues: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        severity_counts = Counter(issue["severity"] for issue in issues)
        return {
            "file_path": str(file_path),
            "rows": int(rows),
            "issues_found": len(issues),
            "blocking_issues_found": int(severity_counts.get("error", 0)),
            "warnings_found": int(severity_counts.get("warning", 0)),
            "checks_failed": sorted({str(issue["check"]) for issue in issues}),
        }


def validate_sales_files(
    sales_files: Iterable[str],
    max_sales_files: Any = "0",
    validation_max_sales_files: Any = "0",
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    validator = DataValidator(config_path=config_path)
    return validator.validate_sales_files(
        sales_files,
        max_sales_files=max_sales_files,
        validation_max_sales_files=validation_max_sales_files,
    )
