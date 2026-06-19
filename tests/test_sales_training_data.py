from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "include"))

from data_preparation.sales_training_data import (
    aggregate_store_daily_sales,
    store_daily_categorical_columns,
)


def test_aggregate_store_daily_sales_preserves_training_grain():
    sales_df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-01"]
            ),
            "store_id": ["store_0001", "store_0001", "store_0002"],
            "sales": [10.0, 20.0, 30.0],
            "customer_traffic": [1.0, 2.0, 3.0],
            "has_promotion": [0, 1, 0],
            "is_open": [1, 1, 1],
            "school_holiday": [0, 1, 0],
            "store_type": ["a", "a", "b"],
        }
    )

    result = aggregate_store_daily_sales(sales_df).sort_values("store_id")

    first_store = result.iloc[0]
    assert first_store["store_id"] == "store_0001"
    assert first_store["sales"] == 30.0
    assert first_store["customer_traffic"] == 3.0
    assert first_store["has_promotion"] == 1
    assert first_store["school_holiday"] == 1
    assert first_store["store_type"] == "a"
    assert len(result) == 2


def test_store_daily_categorical_columns_uses_available_columns():
    df = pd.DataFrame(
        {
            "store_id": ["store_0001"],
            "store_type": ["a"],
            "sales": [10.0],
        }
    )

    assert store_daily_categorical_columns(df) == ["store_id", "store_type"]
