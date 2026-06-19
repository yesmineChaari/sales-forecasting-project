from pathlib import Path
import sys

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "include"))

from data_validation.validators import (
    DataValidator,
    limit_files,
    resolve_sales_file_selection,
)


def _write_validation_config(tmp_path):
    config = {
        "validation": {
            "required_columns": [
                "date",
                "store_id",
                "sales",
                "customer_traffic",
                "has_promotion",
                "is_open",
                "is_holiday",
            ],
            "data_types": {
                "date": "datetime64[ns]",
                "store_id": "object",
                "sales": "float64",
                "customer_traffic": "float64",
                "has_promotion": "int64",
                "is_open": "int64",
                "is_holiday": "int64",
            },
            "value_ranges": {
                "sales": {"min": 0},
                "customer_traffic": {"min": 0},
                "has_promotion": {"min": 0, "max": 1},
                "is_open": {"min": 0, "max": 1},
                "is_holiday": {"min": 0, "max": 1},
            },
            "non_null_columns": [
                "date",
                "store_id",
                "sales",
                "customer_traffic",
                "has_promotion",
                "is_open",
                "is_holiday",
            ],
            "unique_key": ["date", "store_id"],
            "binary_columns": ["has_promotion", "is_open", "is_holiday"],
            "file_date_column": "date",
            "expected_frequency": "D",
            "max_issue_examples": 25,
        }
    }
    config_path = tmp_path / "ml_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def _write_sales_file(tmp_path, date_label, df):
    date_value = pd.Timestamp(date_label)
    path = (
        tmp_path
        / "sales"
        / f"year={date_value.year}"
        / f"month={date_value.month:02d}"
        / f"day={date_value.day:02d}"
        / f"rossmann_sales_{date_label}.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return str(path)


def _valid_sales_frame(date_label="2024-01-01"):
    return pd.DataFrame(
        {
            "date": pd.to_datetime([date_label, date_label]),
            "store_id": ["store_0001", "store_0002"],
            "sales": [100.0, 120.0],
            "customer_traffic": [10.0, 12.0],
            "has_promotion": [0, 1],
            "is_open": [1, 1],
            "is_holiday": [0, 0],
        }
    )


def test_sales_file_validation_passes_valid_rossmann_file(tmp_path):
    config_path = _write_validation_config(tmp_path)
    sales_file = _write_sales_file(
        tmp_path,
        "2024-01-01",
        _valid_sales_frame("2024-01-01"),
    )

    summary = DataValidator(config_path=str(config_path)).validate_sales_files(
        [sales_file],
        max_sales_files="0",
        validation_max_sales_files="0",
    )

    assert summary["total_rows"] == 2
    assert summary["total_files_validated"] == 1
    assert summary["validation_mode"] == "full"
    assert summary["issues_found"] == 0
    assert summary["blocking_issues_found"] == 0
    assert summary["sample_columns"] == [
        "date",
        "store_id",
        "sales",
        "customer_traffic",
        "has_promotion",
        "is_open",
        "is_holiday",
    ]


def test_sales_file_validation_reports_schema_quality_and_grain_issues(tmp_path):
    config_path = _write_validation_config(tmp_path)
    valid_file = _write_sales_file(
        tmp_path,
        "2024-01-01",
        _valid_sales_frame("2024-01-01"),
    )
    bad_df = pd.DataFrame(
        {
            "date": ["not-a-date", "2024-01-03", "2024-01-03"],
            "store_id": ["store_0001", "store_0002", "store_0002"],
            "sales": [100.0, -1.0, 120.0],
            "customer_traffic": [10.0, -5.0, 12.0],
            "has_promotion": [0, 2, 1],
            "is_open": [1, 1, 1],
            "is_holiday": [0, 0, 0],
        }
    )
    bad_file = _write_sales_file(tmp_path, "2024-01-02", bad_df)

    summary = DataValidator(config_path=str(config_path)).validate_sales_files(
        [valid_file, bad_file],
        max_sales_files="0",
        validation_max_sales_files="0",
    )

    assert summary["total_rows"] == 5
    assert summary["blocking_issues_found"] >= 7
    assert summary["warnings_found"] >= 1
    assert summary["issue_counts_by_check"]["schema"] == 1
    assert summary["issue_counts_by_check"]["nulls"] == 1
    assert summary["issue_counts_by_check"]["value_range"] >= 3
    assert summary["issue_counts_by_check"]["binary_domain"] == 1
    assert summary["issue_counts_by_check"]["unique_key"] == 1
    assert summary["issue_counts_by_check"]["file_partition"] == 1
    assert summary["issue_counts_by_check"]["time_series"] == 1
    assert any("not be parsed as datetime" in issue for issue in summary["issues"])


def test_sales_file_selection_uses_training_cap_before_validation_cap():
    files = ["a.parquet", "b.parquet", "c.parquet"]

    selection = resolve_sales_file_selection(
        files,
        max_sales_files="2",
        validation_max_sales_files="1",
    )
    limited_files, limit = limit_files(files, raw_limit="-5", env_name="TEST_LIMIT")

    assert selection["training_files"] == ["a.parquet", "b.parquet"]
    assert selection["validation_files"] == ["a.parquet"]
    assert selection["training_limit"] == 2
    assert selection["validation_limit"] == 1
    assert selection["validation_mode"] == "sample"
    assert limited_files == files
    assert limit == 0
