from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "include"))

from utils.rossmann_loader import RossmannDataLoader


def test_loader_writes_full_row_prophet_sales_stream(tmp_path):
    train_path = tmp_path / "train.csv"
    store_path = tmp_path / "store.csv"

    pd.DataFrame(
        {
            "Store": [1, 1, 2],
            "DayOfWeek": [1, 2, 1],
            "Date": ["2024-01-01", "2024-01-02", "2024-01-01"],
            "Sales": [100, 0, 0],
            "Customers": [10, 0, 0],
            "Open": [1, 1, 0],
            "Promo": [0, 1, 0],
            "StateHoliday": ["0", "0", "a"],
            "SchoolHoliday": [0, 0, 1],
        }
    ).to_csv(train_path, index=False)

    pd.DataFrame(
        {
            "Store": [1, 2],
            "StoreType": ["a", "b"],
            "Assortment": ["a", "c"],
            "CompetitionDistance": [100.0, 200.0],
            "CompetitionOpenSinceMonth": [1, 1],
            "CompetitionOpenSinceYear": [2020, 2020],
            "Promo2": [0, 1],
            "Promo2SinceWeek": [0, 1],
            "Promo2SinceYear": [0, 2020],
            "PromoInterval": ["", "Jan,Apr,Jul,Oct"],
        }
    ).to_csv(store_path, index=False)

    loader = RossmannDataLoader(
        train_path=str(train_path),
        store_path=str(store_path),
    )
    file_paths = loader.prepare_data(output_dir=str(tmp_path / "prepared"))

    store_level_df = pd.concat(
        [pd.read_parquet(path) for path in file_paths["sales"]],
        ignore_index=True,
    )
    prophet_df = pd.concat(
        [pd.read_parquet(path) for path in file_paths["prophet_sales"]],
        ignore_index=True,
    )

    assert len(store_level_df) == 1
    assert store_level_df.iloc[0]["sales"] == 100.0
    assert store_level_df.iloc[0]["is_open"] == 1

    assert len(prophet_df) == 3
    assert ((prophet_df["is_open"] == 0) & (prophet_df["sales"] == 0.0)).any()
    assert ((prophet_df["is_open"] == 1) & (prophet_df["sales"] == 0.0)).any()
