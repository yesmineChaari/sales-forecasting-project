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

    def _load_raw_data(self) -> pd.DataFrame:
        train_df = pd.read_csv(self.train_path, low_memory=False)
        store_df = pd.read_csv(self.store_path, low_memory=False)

        train_df["Date"] = pd.to_datetime(train_df["Date"])

        df = train_df.merge(store_df, on="Store", how="left")
        df["Open"] = df["Open"].fillna(1).astype(int)
        return df

    def _prepare_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

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

    def _load_and_prepare(self, include_closed_zero_sales: bool = False) -> pd.DataFrame:
        df = self._load_raw_data()

        if not include_closed_zero_sales:
            df = df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()

        return self._prepare_frame(df)

    def _write_daily_partitions(
        self,
        df: pd.DataFrame,
        output_dir: str,
        data_type: str,
        filename_prefix: str,
    ) -> List[str]:
        file_paths = []

        for date_value, day_df in df.groupby(df["date"].dt.date):
            date_ts = pd.Timestamp(date_value)
            date_str = date_ts.strftime("%Y-%m-%d")

            output_path = os.path.join(
                output_dir,
                f"{data_type}/year={date_ts.year}/month={date_ts.month:02d}/day={date_ts.day:02d}/"
                f"{filename_prefix}_{date_str}.parquet",
            )

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            day_df.to_parquet(output_path, index=False)
            file_paths.append(output_path)

        return file_paths

    def prepare_data(self, output_dir: str = "/tmp/rossmann_sales_data") -> Dict[str, List[str]]:
        os.makedirs(output_dir, exist_ok=True)

        df = self._load_and_prepare(include_closed_zero_sales=False)
        prophet_df = self._load_and_prepare(include_closed_zero_sales=True)

        file_paths = {
            "sales": [],
            "prophet_sales": [],
            "promotions": [],
            "store_events": [],
            "customer_traffic": [],
            "inventory": [],
        }

        file_paths["sales"] = self._write_daily_partitions(
            df,
            output_dir,
            "sales",
            "rossmann_sales",
        )
        file_paths["prophet_sales"] = self._write_daily_partitions(
            prophet_df,
            output_dir,
            "prophet_sales",
            "rossmann_prophet_sales",
        )

        metadata = {
            "source": "rossmann",
            "total_rows": len(df),
            "prophet_total_rows": len(prophet_df),
            "start_date": df["date"].min().isoformat(),
            "end_date": df["date"].max().isoformat(),
            "n_stores": df["store_id"].nunique(),
            "total_files": len(file_paths["sales"]),
            "prophet_total_files": len(file_paths["prophet_sales"]),
        }

        metadata_path = os.path.join(output_dir, "metadata/rossmann_metadata.parquet")
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        pd.DataFrame([metadata]).to_parquet(metadata_path, index=False)

        return file_paths
