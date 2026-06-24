"""
Simple predictor for sales forecasting
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

HISTORICAL_REQUIRED_COLUMNS = [
    'date',
    'store_id',
    'sales',
    'customer_traffic',
    'has_promotion',
    'is_open',
    'school_holiday',
    'state_holiday',
    'store_type',
    'assortment',
    'competition_distance',
    'promo2',
    'promo_interval',
]

FUTURE_REQUIRED_COLUMNS = [
    col for col in HISTORICAL_REQUIRED_COLUMNS if col != 'sales'
]

BINARY_COLUMNS = [
    'has_promotion',
    'is_open',
    'school_holiday',
    'promo2',
]

CATEGORICAL_INTERACTION_COLUMNS = [
    'store_id',
    'store_type',
    'assortment',
    'state_holiday',
    'promo_interval',
]


class SimplePredictor:
    """Simple predictor that works with SimpleModelLoader"""
    
    def __init__(self, model_loader):
        self.model_loader = model_loader

    def validate_required_columns(self, df: pd.DataFrame, required_columns, label: str):
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"{label} is missing required columns: {', '.join(missing_cols)}"
            )

    def validate_single_store(self, df: pd.DataFrame):
        store_count = df['store_id'].astype(str).nunique()
        if store_count != 1:
            raise ValueError(
                "The UI currently supports one store at a time. "
                "Please upload data for a single store."
            )

    def normalize_business_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for col in BINARY_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='raise').astype(int)
                invalid_values = sorted(set(df[col].unique()) - {0, 1})
                if invalid_values:
                    raise ValueError(
                        f"{col} must contain only 0/1 values. "
                        f"Invalid values: {invalid_values}"
                    )

        numeric_cols = ['sales', 'customer_traffic', 'competition_distance']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='raise')
                if (df[col] < 0).any():
                    raise ValueError(f"{col} cannot be negative")

        for col in CATEGORICAL_INTERACTION_COLUMNS:
            if col in df.columns:
                df[col] = df[col].fillna('none').astype(str)

        if 'state_holiday' in df.columns and 'school_holiday' in df.columns:
            normalized_state_holiday = (
                df['state_holiday']
                .fillna('none')
                .astype(str)
                .str.strip()
                .str.lower()
            )
            no_holiday_values = {"", "0", "0.0", "none", "nan", "nat", "false"}
            df['is_holiday'] = (
                (~normalized_state_holiday.isin(no_holiday_values))
                | (df['school_holiday'] == 1)
            ).astype(int)

        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        categorical_cols = [
            col for col in CATEGORICAL_INTERACTION_COLUMNS if col in df.columns
        ]

        for i, col1 in enumerate(categorical_cols):
            for col2 in categorical_cols[i + 1:]:
                df[f'{col1}_{col2}_interaction'] = (
                    df[col1].astype(str) + "_" + df[col2].astype(str)
                )

        return df
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction"""
        # Ensure date column is datetime
        df = df.copy()
        df = self.normalize_business_features(df)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # Extract time features
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['dayofweek'] = df['date'].dt.dayofweek
            df['quarter'] = df['date'].dt.quarter
            df['weekofyear'] = df['date'].dt.isocalendar().week
            df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
            
            # Add cyclical features
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
            df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
            df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
            df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

        df = self.create_interaction_features(df)
            
        # Add lag features if we have sales data
        if 'sales' in df.columns:
            # Multiple lag features
            for lag in [1, 2, 3, 7, 14, 21, 30]:
                df[f'sales_lag_{lag}'] = df['sales'].shift(lag)
            
            # Rolling statistics from prior sales only, matching FeatureEngineer.
            prior_sales = df['sales'].shift(1)
            for window in [3, 7, 14, 21, 30]:
                rolling_sales = prior_sales.rolling(window, min_periods=1)
                df[f'sales_rolling_{window}_mean'] = rolling_sales.mean()
                df[f'sales_rolling_{window}_std'] = rolling_sales.std()
                df[f'sales_rolling_{window}_min'] = rolling_sales.min()
                df[f'sales_rolling_{window}_max'] = rolling_sales.max()
                df[f'sales_rolling_{window}_median'] = rolling_sales.median()
            
            # Fill NaN values with appropriate defaults
            for col in df.columns:
                if 'sales_lag' in col or 'sales_rolling' in col:
                    df[col] = df[col].fillna(0)
        
        return df

    def encode_categorical_features(self, future_df: pd.DataFrame) -> pd.DataFrame:
        future_df = future_df.copy()

        if not self.model_loader.encoders:
            return future_df

        for col, encoder in self.model_loader.encoders.items():
            if col not in future_df.columns:
                continue

            try:
                if hasattr(encoder, "categories_"):
                    future_df[col] = encoder.transform(
                        future_df[[col]].astype(str)
                    ).ravel()
                elif hasattr(encoder, "classes_"):
                    class_map = {
                        value: idx for idx, value in enumerate(encoder.classes_)
                    }
                    future_df[col] = (
                        future_df[col].astype(str).map(class_map).fillna(-1).values
                    )
                else:
                    future_df[col] = encoder.transform(
                        future_df[[col]].astype(str)
                    ).ravel()
            except Exception as e:
                logger.warning(f"Error encoding {col}: {e}")
                future_df[col] = -1

        return future_df
    
    def predict(self, input_data: pd.DataFrame, model_type: str = 'ensemble_store_level',
                forecast_days: int = 30,
                future_features: pd.DataFrame = None) -> Dict[str, Any]:
        """Make predictions"""
        try:
            canonical_model_type = getattr(
                self.model_loader,
                "canonical_model_type",
                lambda value: value,
            )
            selected_model_type = canonical_model_type(model_type)

            if not self.model_loader.loaded:
                return {
                    'success': False,
                    'error': 'Models not loaded'
                }

            self.validate_required_columns(
                input_data,
                HISTORICAL_REQUIRED_COLUMNS,
                "Historical input data",
            )
            self.validate_single_store(input_data)

            if future_features is None:
                return {
                    'success': False,
                    'error': (
                        'Future feature inputs are required. Provide forecast '
                        'dates and business/store features before prediction.'
                    )
                }

            self.validate_required_columns(
                future_features,
                FUTURE_REQUIRED_COLUMNS,
                "Future feature data",
            )
            
            # Prepare historical data
            historical_df = self.prepare_features(input_data)
            
            # Create future dates
            last_date = pd.to_datetime(input_data['date']).max()
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=forecast_days,
                freq='D'
            )
            
            future_df = future_features.copy()
            future_df['date'] = pd.to_datetime(future_df['date'])
            future_df = future_df.sort_values('date').head(forecast_days)

            if len(future_df) != forecast_days:
                return {
                    'success': False,
                    'error': (
                        f'Future feature data must contain exactly '
                        f'{forecast_days} rows.'
                    )
                }

            future_dates = pd.to_datetime(future_df['date'])
            
            # Prepare features for future dates
            future_df = self.prepare_features(future_df)
            
            # Use last known values for lag features
            if len(historical_df) > 0 and 'sales' in historical_df.columns:
                # Get recent sales values for lag features
                recent_sales = historical_df['sales'].tail(30).values
                sales_mean = historical_df['sales'].mean()
                
                # Set lag features based on historical data
                for lag in [1, 2, 3, 7, 14, 21, 30]:
                    if len(recent_sales) >= lag:
                        future_df[f'sales_lag_{lag}'] = recent_sales[-lag]
                    else:
                        future_df[f'sales_lag_{lag}'] = sales_mean
                
                # Set rolling statistics based on historical data
                for window in [3, 7, 14, 21, 30]:
                    if len(recent_sales) >= window:
                        window_data = recent_sales[-window:]
                        future_df[f'sales_rolling_{window}_mean'] = np.mean(window_data)
                        future_df[f'sales_rolling_{window}_std'] = np.std(window_data)
                        future_df[f'sales_rolling_{window}_min'] = np.min(window_data)
                        future_df[f'sales_rolling_{window}_max'] = np.max(window_data)
                        future_df[f'sales_rolling_{window}_median'] = np.median(window_data)
                    else:
                        future_df[f'sales_rolling_{window}_mean'] = sales_mean
                        future_df[f'sales_rolling_{window}_std'] = 0
                        future_df[f'sales_rolling_{window}_min'] = sales_mean
                        future_df[f'sales_rolling_{window}_max'] = sales_mean
                        future_df[f'sales_rolling_{window}_median'] = sales_mean
            
            future_df = self.encode_categorical_features(future_df)
            
            # Select features based on what the model expects
            if self.model_loader.feature_cols:
                missing_features = [
                    col for col in self.model_loader.feature_cols
                    if col not in future_df.columns
                ]
                if missing_features:
                    return {
                        'success': False,
                        'error': (
                            'Could not build all model features from the '
                            'provided inputs. Missing engineered features: '
                            + ', '.join(missing_features)
                        )
                    }
                X = future_df[self.model_loader.feature_cols].values
            else:
                # Fallback to basic features (exclude string columns)
                feature_cols = ['year', 'month', 'day', 'dayofweek', 'quarter', 
                               'is_weekend', 'sales_lag_1', 'sales_lag_7',
                               'sales_rolling_mean_7', 'sales_rolling_std_7']
                # Add any store_id encoded columns
                store_cols = [col for col in future_df.columns if col.startswith('store_id_') and col != 'store_id']
                feature_cols.extend(store_cols)
                available_features = [col for col in feature_cols if col in future_df.columns]
                X = future_df[available_features].values
            
            # Make predictions
            predictions = self.model_loader.predict(X, model_type=selected_model_type)
            model_predictions = {selected_model_type: predictions}

            if selected_model_type == "ensemble_store_level" and hasattr(self.model_loader, "models"):
                for individual_model in ["xgboost_store_level", "lightgbm_store_level"]:
                    if individual_model in self.model_loader.models:
                        try:
                            model_predictions[individual_model] = self.model_loader.predict(
                                X,
                                model_type=individual_model,
                            )
                        except Exception as e:
                            logger.warning(
                                f"Could not generate {individual_model} comparison predictions: {e}"
                            )
            
            # Create results dataframe
            results_df = pd.DataFrame({
                'date': future_dates,
                'predicted_sales': predictions,
                'lower_bound': predictions * 0.9,  # Simple 10% confidence interval
                'upper_bound': predictions * 1.1
            })
            
            # Calculate summary statistics
            summary = {
                'total_predicted_sales': predictions.sum(),
                'average_daily_sales': predictions.mean(),
                'max_daily_sales': predictions.max(),
                'min_daily_sales': predictions.min(),
                'forecast_days': forecast_days
            }
            
            return {
                'success': True,
                'predictions': results_df,
                'summary': summary,
                'model_type': selected_model_type,
                'model_predictions': model_predictions
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e)
            }
