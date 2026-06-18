import pandas as pd
import numpy as np
from typing import List, Optional
import holidays
import yaml
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    def __init__(self, config_path: str = "/usr/local/airflow/include/config/ml_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.feature_config = self.config['features']
        self.validation_config = self.config['validation']

    @staticmethod
    def _state_holiday_flags(series: pd.Series) -> pd.Series:
        no_holiday_values = {"", "0", "0.0", "none", "nan", "nat", "false"}
        normalized = series.fillna("").astype(str).str.strip().str.lower()
        return ~normalized.isin(no_holiday_values)

    @staticmethod
    def _binary_holiday_flags(series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce").fillna(0)
        text = series.fillna("").astype(str).str.strip().str.lower()
        return (numeric != 0) | text.isin({"true", "yes", "y"})

    def _create_rossmann_holiday_feature(
        self, df: pd.DataFrame, date_col: str
    ) -> pd.Series:
        holiday_flags = []

        if 'is_holiday' in df.columns:
            holiday_flags.append(self._binary_holiday_flags(df['is_holiday']))
        if 'state_holiday' in df.columns:
            holiday_flags.append(self._state_holiday_flags(df['state_holiday']))
        if 'school_holiday' in df.columns:
            holiday_flags.append(self._binary_holiday_flags(df['school_holiday']))

        if holiday_flags:
            combined = holiday_flags[0].copy()
            for flags in holiday_flags[1:]:
                combined = combined | flags
            return combined.astype(int)

        de_holidays = holidays.country_holidays("DE")
        return df[date_col].dt.date.apply(lambda value: value in de_holidays).astype(int)
        
    def create_date_features(self, df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        df = df.copy()
        
        df[date_col] = pd.to_datetime(df[date_col])
        
        date_features = self.feature_config['date_features']
        
        if 'year' in date_features:
            df['year'] = df[date_col].dt.year
        if 'month' in date_features:
            df['month'] = df[date_col].dt.month
        if 'day' in date_features:
            df['day'] = df[date_col].dt.day
        if 'dayofweek' in date_features:
            df['dayofweek'] = df[date_col].dt.dayofweek
        if 'quarter' in date_features:
            df['quarter'] = df[date_col].dt.quarter
        if 'weekofyear' in date_features:
            df['weekofyear'] = df[date_col].dt.isocalendar().week
        if 'is_weekend' in date_features:
            df['is_weekend'] = (df[date_col].dt.dayofweek >= 5).astype(int)
        if 'is_holiday' in date_features:
            # Prefer Rossmann/Germany holiday fields already present in the dataset.
            df['is_holiday'] = self._create_rossmann_holiday_feature(df, date_col)
        
        logger.info(f"Created {len(date_features)} date features")
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str, 
                           group_cols: Optional[List[str]] = None) -> pd.DataFrame:
        df = df.copy()
        lag_values = self.feature_config['lag_features']
        
        if group_cols:
            for lag in lag_values:
                df[f'{target_col}_lag_{lag}'] = df.groupby(group_cols)[target_col].shift(lag)
        else:
            for lag in lag_values:
                df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        logger.info(f"Created {len(lag_values)} lag features")
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, target_col: str,
                               group_cols: Optional[List[str]] = None) -> pd.DataFrame:
        df = df.copy()
        windows = self.feature_config['rolling_features']['windows']
        functions = self.feature_config['rolling_features']['functions']
        
        if group_cols:
            for window in windows:
                for func in functions:
                    col_name = f'{target_col}_rolling_{window}_{func}'
                    # Shift first so same-day sales never leak into features.
                    df[col_name] = df.groupby(group_cols)[target_col].transform(
                        lambda x: x.shift(1).rolling(window, min_periods=1).agg(func)
                    )
        else:
            for window in windows:
                for func in functions:
                    col_name = f'{target_col}_rolling_{window}_{func}'
                    # Shift first so same-day sales never leak into features.
                    df[col_name] = df[target_col].shift(1).rolling(window, min_periods=1).agg(func)
        
        logger.info(f"Created {len(windows) * len(functions)} rolling features")
        return df
    
    def create_interaction_features(self, df: pd.DataFrame, 
                                   categorical_cols: List[str]) -> pd.DataFrame:
        df = df.copy()
        
        for i, col1 in enumerate(categorical_cols):
            for col2 in categorical_cols[i+1:]:
                df[f'{col1}_{col2}_interaction'] = df[col1].astype(str) + "_" + df[col2].astype(str)
        
        return df
    
    def create_cyclical_features(self, df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        df = df.copy()
        
        df['month_sin'] = np.sin(2 * np.pi * df[date_col].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df[date_col].dt.month / 12)
        
        df['day_sin'] = np.sin(2 * np.pi * df[date_col].dt.day / 31)
        df['day_cos'] = np.cos(2 * np.pi * df[date_col].dt.day / 31)
        
        df['dayofweek_sin'] = np.sin(2 * np.pi * df[date_col].dt.dayofweek / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df[date_col].dt.dayofweek / 7)
        
        logger.info("Created cyclical features")
        return df
    
    def create_all_features(self, df: pd.DataFrame, target_col: str = 'sales',
                           date_col: str = 'date', 
                           group_cols: Optional[List[str]] = None,
                           categorical_cols: Optional[List[str]] = None) -> pd.DataFrame:
        
        logger.info("Starting feature engineering pipeline")
        
        # Sort by date for proper lag and rolling calculations
        if group_cols:
            df = df.sort_values(group_cols + [date_col])
        else:
            df = df.sort_values(date_col)
        
        # Create date features
        df = self.create_date_features(df, date_col)
        
        # Create lag features
        df = self.create_lag_features(df, target_col, group_cols)
        
        # Create rolling features
        df = self.create_rolling_features(df, target_col, group_cols)
        
        # Create cyclical features
        df = self.create_cyclical_features(df, date_col)
        
        # Create interaction features if categorical columns provided
        if categorical_cols:
            df = self.create_interaction_features(df, categorical_cols)
        
        # Handle missing values created by lag and rolling features
        df = self.handle_missing_values(df)
        
        logger.info(f"Feature engineering complete. Total features: {len(df.columns)}")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        # For lag and rolling features, use a neutral value for rows with no
        # prior history. Do not backfill from future target values.
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if df[col].isnull().any():
                if 'lag' in col or 'rolling' in col:
                    df[col] = df[col].fillna(0)
                else:
                    # For other features, use mean
                    df[col] = df[col].fillna(df[col].mean())
        
        return df
