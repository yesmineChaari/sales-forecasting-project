from pathlib import Path
import sys

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "include"))

from feature_engineering.feature_pipeline import FeatureEngineer


def _write_config(tmp_path):
    config = {
        "features": {
            "date_features": [],
            "lag_features": [1],
            "rolling_features": {
                "windows": [2],
                "functions": ["mean", "std", "min", "max", "median"],
            },
        },
        "validation": {},
    }
    config_path = tmp_path / "ml_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_grouped_rolling_features_use_only_past_sales(tmp_path):
    df = pd.DataFrame(
        {
            "store_id": [1, 1, 1],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "sales": [10, 20, 30],
        }
    )
    engineer = FeatureEngineer(config_path=str(_write_config(tmp_path)))

    result = engineer.create_all_features(
        df,
        target_col="sales",
        date_col="date",
        group_cols=["store_id"],
    ).sort_values(["store_id", "date"])

    first_day = result.iloc[0]
    third_day = result.iloc[2]

    assert first_day["sales_lag_1"] == 0
    assert first_day["sales_rolling_2_mean"] == 0
    assert third_day["sales_lag_1"] == 20
    assert third_day["sales_rolling_2_mean"] == 15
    assert third_day["sales_rolling_2_min"] == 10
    assert third_day["sales_rolling_2_max"] == 20
    assert third_day["sales_rolling_2_median"] == 15
    assert third_day["sales_rolling_2_mean"] != 25
