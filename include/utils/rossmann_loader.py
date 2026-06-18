import os
from typing import Dict, List

import pandas as pd


class RossmannDataLoader:
    """
    Load Rossmann Store Sales data and convert it into the format expected
    by the existing Airflow training pipeline.

    Source files:
    - train.csv
    - store.csv
    """

    def __init__(
        self,
        train_path: str = "/usr/local/airflow/include/data/rossmann/train.csv",
        store_path: str = "/usr/local/airflow/include/data/rossmann/store.csv",
    ):
        self.train_path = train_path
        self.store_path = store_path

    @staticmethod
    def _normalize_state_holiday(series: pd.Series) -> pd.Series:
        no_holiday_values = {"", "0", "0.0", "none", "nan", "nat", "false"}
        normalized = series.fillna("").astype(str).str.strip().str.lower()
        return normalized.where(~normalized.isin(no_holiday_values), "none")

    def _load_and_prepare(self) -> pd.DataFrame:
        train_df = pd.read_csv(self.train_path, low_memory=False)
        store_df = pd.read_csv(self.store_path, low_memory=False)

        train_df["Date"] = pd.to_datetime(train_df["Date"])

        df = train_df.merge(store_df, on="Store", how="left")

        # Avoid zero-sales closed days because MAPE divides by y_true.
        df["Open"] = df["Open"].fillna(1).astype(int)
        df = df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()

        state_holiday = self._normalize_state_holiday(df["StateHoliday"])
        school_holiday = df["SchoolHoliday"].fillna(0).astype(int)

        prepared_df = pd.DataFrame(
            {
                "date": df["Date"],
                "store_id": "store_" + df["Store"].astype(int).astype(str).str.zfill(4),
                "sales": df["Sales"].astype(float),
                "customer_traffic": df["Customers"].fillna(0).astype(float),
                "has_promotion": df["Promo"].fillna(0).astype(int),
                "is_open": df["Open"].astype(int),
                "state_holiday": state_holiday,
                "school_holiday": school_holiday,
                "is_holiday": (
                    (state_holiday != "none")
                    | (school_holiday == 1)
                ).astype(int),
                "store_type": df["StoreType"].fillna("unknown").astype(str),
                "assortment": df["Assortment"].fillna("unknown").astype(str),
                "competition_distance": df["CompetitionDistance"].fillna(0).astype(float),
                "promo2": df["Promo2"].fillna(0).astype(int),
                "promo_interval": df["PromoInterval"].fillna("none").astype(str),
            }
        )

        prepared_df = prepared_df.sort_values(["date", "store_id"]).reset_index(drop=True)
        return prepared_df

    def prepare_data(self, output_dir: str = "/tmp/rossmann_sales_data") -> Dict[str, List[str]]:
        os.makedirs(output_dir, exist_ok=True)

        df = self._load_and_prepare()

        file_paths = {
            "sales": [],
            "promotions": [],
            "store_events": [],
            "customer_traffic": [],
            "inventory": [],
        }

        # Keep the same spirit as the original project: partition by day.
        for date_value, day_df in df.groupby(df["date"].dt.date):
            date_ts = pd.Timestamp(date_value)
            date_str = date_ts.strftime("%Y-%m-%d")

            sales_path = os.path.join(
                output_dir,
                f"sales/year={date_ts.year}/month={date_ts.month:02d}/day={date_ts.day:02d}/"
                f"rossmann_sales_{date_str}.parquet",
            )

            os.makedirs(os.path.dirname(sales_path), exist_ok=True)
            day_df.to_parquet(sales_path, index=False)
            file_paths["sales"].append(sales_path)

        metadata = {
            "source": "rossmann",
            "total_rows": len(df),
            "start_date": df["date"].min().isoformat(),
            "end_date": df["date"].max().isoformat(),
            "n_stores": df["store_id"].nunique(),
            "total_files": len(file_paths["sales"]),
        }

        metadata_path = os.path.join(output_dir, "metadata/rossmann_metadata.parquet")
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        pd.DataFrame([metadata]).to_parquet(metadata_path, index=False)

        return file_paths
