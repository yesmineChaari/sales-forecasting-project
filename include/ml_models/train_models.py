import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import yaml
import joblib
import logging
import os
from datetime import datetime

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OrdinalEncoder

import xgboost as xgb
import lightgbm as lgb
import optuna
import mlflow

from utils.mlflow_utils import MLflowManager
from feature_engineering.feature_pipeline import FeatureEngineer
from ml_models.diagnostics import diagnose_model_performance
from ml_models.ensemble_model import EnsembleModel
from ml_models.prophet_daily_total import (
    PROPHET_DAILY_TOTAL_REGRESSORS,
    build_daily_total_frame,
    build_prophet_daily_total_model,
)

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, config_path: str = "/usr/local/airflow/include/config/ml_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['models']
        self.training_config = self.config['training']
        self.mlflow_manager = MLflowManager(config_path)
        self.feature_engineer = FeatureEngineer(config_path)
        
        self.models = {}
        self.encoders = {}
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'sales',
                    date_col: str = 'date', group_cols: Optional[List[str]] = None,
                    categorical_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        
        logger.info("Preparing data for training")
        
        # Validate only the store-level training grain used by Rossmann.
        required_cols = ['date', target_col]
        if group_cols:
            required_cols.extend(group_cols)
        
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns for training: {missing_cols}")
        
        # Feature engineering
        df_features = self.feature_engineer.create_all_features(
            df, target_col, date_col, group_cols, categorical_cols
        )
        
        # Skip target encoding to avoid data leakage
        # Target encoding can cause overfitting in time series
        
        # Split data chronologically for time series
        df_sorted = df_features.sort_values(date_col)
        
        # Use more recent data for validation and testing for better performance
        # This ensures the model learns from patterns closer to what it will predict
        train_size = int(len(df_sorted) * (1 - self.training_config['test_size'] - self.training_config['validation_size']))
        val_size = int(len(df_sorted) * self.training_config['validation_size'])
        
        train_df = df_sorted[:train_size]
        val_df = df_sorted[train_size:train_size + val_size]
        test_df = df_sorted[train_size + val_size:]
        
        # Remove any rows with NaN in target column
        train_df = train_df.dropna(subset=[target_col])
        val_df = val_df.dropna(subset=[target_col])
        test_df = test_df.dropna(subset=[target_col])
        
        # Skip feature selection for now - let models handle all features
        # This allows models to learn which features are important
        
        logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def preprocess_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                    test_df: pd.DataFrame, target_col: str,
                    exclude_cols: List[str] = ['date']) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        # Separate features and target
        feature_cols = [col for col in train_df.columns if col not in exclude_cols + [target_col]]
        
        X_train = train_df[feature_cols].copy()
        X_val = val_df[feature_cols].copy()
        X_test = test_df[feature_cols].copy()
        
        y_train = train_df[target_col].values
        y_val = val_df[target_col].values
        y_test = test_df[target_col].values
        
        # Encode categorical variables. Validation/test may contain store/category
        # combinations that are absent from the chronological training split.
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col not in self.encoders:
                self.encoders[col] = OrdinalEncoder(
                    handle_unknown='use_encoded_value',
                    unknown_value=-1,
                )
                X_train[col] = self.encoders[col].fit_transform(
                    X_train[[col]].astype(str)
                ).ravel()
            else:
                X_train[col] = self.encoders[col].transform(
                    X_train[[col]].astype(str)
                ).ravel()

            known_categories = set(self.encoders[col].categories_[0])
            val_unknowns = (~X_val[col].astype(str).isin(known_categories)).sum()
            test_unknowns = (~X_test[col].astype(str).isin(known_categories)).sum()
            if val_unknowns or test_unknowns:
                logger.info(
                    "Encoding unseen categories in %s as -1 (val=%s, test=%s)",
                    col,
                    val_unknowns,
                    test_unknowns,
                )
            
            X_val[col] = self.encoders[col].transform(
                X_val[[col]].astype(str)
            ).ravel()
            X_test[col] = self.encoders[col].transform(
                X_test[[col]].astype(str)
            ).ravel()

        for split_name, frame in {
            "train": X_train,
            "validation": X_val,
            "test": X_test,
        }.items():
            non_numeric_cols = frame.select_dtypes(
                include=['object', 'category']
            ).columns.tolist()
            if non_numeric_cols:
                raise ValueError(
                    f"Non-numeric features remain after encoding in "
                    f"{split_name}: {non_numeric_cols}"
                )
        
        self.feature_cols = feature_cols
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        nonzero_mask = y_true != 0
        if nonzero_mask.any():
            mape = np.mean(
                np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])
            ) * 100
        else:
            mape = 0.0

        metrics = {
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'mape': float(mape),
            'r2': float(r2_score(y_true, y_pred))
        }
        return metrics
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    use_optuna: bool = True) -> xgb.XGBRegressor:
        
        logger.info("Training XGBoost model")
        
        if use_optuna:
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 0.5),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
                    'random_state': 42
                }
                
                params['early_stopping_rounds'] = 50
                model = xgb.XGBRegressor(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                
                y_pred = model.predict(X_val)
                return np.sqrt(mean_squared_error(y_val, y_pred))
            
            study = optuna.create_study(
                direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner()
            )
            study.optimize(objective, n_trials=self.config['training'].get('optuna_trials', 50))
            
            best_params = study.best_params
            best_params['random_state'] = 42
        else:
            best_params = self.model_config['xgboost']['params']
        
        best_params['early_stopping_rounds'] = 50
        model = xgb.XGBRegressor(**best_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
        
        self.models['xgboost'] = model
        return model
    
    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                use_optuna: bool = True) -> lgb.LGBMRegressor:
        
        logger.info("Training LightGBM model")
        
        if use_optuna:
            def objective(trial):
                params = {
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
                    'random_state': 42,
                    'verbosity': -1,
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt'
                }
                
                model = lgb.LGBMRegressor(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
                
                y_pred = model.predict(X_val)
                return np.sqrt(mean_squared_error(y_val, y_pred))
            
            study = optuna.create_study(
                direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner()
            )
            study.optimize(objective, n_trials=self.config['training'].get('optuna_trials', 50))
            
            best_params = study.best_params
            best_params['random_state'] = 42
            best_params['verbosity'] = -1
        else:
            best_params = self.model_config['lightgbm']['params']
        
        model = lgb.LGBMRegressor(**best_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
            callbacks=[lgb.early_stopping(50)])
        
        self.models['lightgbm'] = model
        return model
    
    def _daily_total_frame(self, df: pd.DataFrame, date_col: str,
                           target_col: str) -> pd.DataFrame:
        return build_daily_total_frame(df, date_col=date_col, target_col=target_col)

    def train_prophet_daily_total(self, train_df: pd.DataFrame,
                                  val_df: pd.DataFrame,
                                  test_df: pd.DataFrame,
                                  target_col: str = 'sales',
                                  date_col: str = 'date') -> Dict[str, Any]:
        logger.info("Training Prophet daily-total model")

        daily_train = self._daily_total_frame(train_df, date_col, target_col)
        daily_val = self._daily_total_frame(val_df, date_col, target_col)
        daily_test = self._daily_total_frame(test_df, date_col, target_col)

        if len(daily_train) < 2:
            raise ValueError("Prophet daily-total model requires at least two training dates")
        if daily_test.empty:
            raise ValueError("Prophet daily-total model requires test dates for evaluation")

        logger.info(
            "Prophet daily-total rows - train: %s, val: %s, test: %s",
            len(daily_train),
            len(daily_val),
            len(daily_test),
        )

        prophet_params = self.model_config['prophet']['params'].copy()
        prophet_params.update({
            'stan_backend': 'CMDSTANPY',
            'mcmc_samples': 0,
            'uncertainty_samples': 0,
        })

        try:
            model = build_prophet_daily_total_model(prophet_params)
            model.fit(daily_train)
        except Exception as e:
            logger.warning("Retrying Prophet daily-total with minimal parameters: %s", e)
            model = build_prophet_daily_total_model(
                {
                    "yearly_seasonality": True,
                    "weekly_seasonality": True,
                    "daily_seasonality": False,
                    "changepoint_prior_scale": 0.05,
                    "seasonality_prior_scale": 10.0,
                    "uncertainty_samples": 0,
                    "mcmc_samples": 0,
                }
            )
            model.fit(daily_train)

        validation_predictions = None
        if not daily_val.empty:
            validation_predictions = model.predict(
                daily_val[["ds"] + PROPHET_DAILY_TOTAL_REGRESSORS]
            )['yhat'].values

        prophet_pred = model.predict(
            daily_test[["ds"] + PROPHET_DAILY_TOTAL_REGRESSORS]
        )['yhat'].values
        prophet_metrics = self.calculate_metrics(daily_test['y'].values, prophet_pred)

        self.models['prophet_daily_total'] = model

        return {
            'model': model,
            'metrics': prophet_metrics,
            'predictions': prophet_pred,
            'validation_predictions': validation_predictions,
            'actuals': daily_test['y'].values,
            'prediction_dates': daily_test['ds'].dt.strftime('%Y-%m-%d').tolist(),
            'regressor_columns': PROPHET_DAILY_TOTAL_REGRESSORS,
            'input_example': daily_test[
                ["ds"] + PROPHET_DAILY_TOTAL_REGRESSORS
            ].head(5),
            'forecast_level': 'daily_total',
            'target_grain': 'date',
            'included_in_store_ensemble': False,
        }
    
    def train_all_models(self, train_df: pd.DataFrame, val_df: pd.DataFrame,
                        test_df: pd.DataFrame, target_col: str = 'sales',
                        use_optuna: bool = True) -> Dict[str, Dict[str, Any]]:
        
        results = {}
        
        # Start MLflow run
        run_id = self.mlflow_manager.start_run(
            run_name=f"sales_forecast_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags={
                "model_type": "store_level_ensemble",
                "store_level_ensemble": "ensemble_store_level",
                "use_optuna": str(use_optuna),
            }
        )
        self.last_run_id = run_id
        
        try:
            # Preprocess data
            X_train, X_val, X_test, y_train, y_val, y_test = self.preprocess_features(
                train_df, val_df, test_df, target_col
            )
            
            # Log data stats
            self.mlflow_manager.log_params({
                "train_size": len(train_df),
                "val_size": len(val_df),
                "test_size": len(test_df),
                "n_features": X_train.shape[1]
            })
            
            # Train XGBoost
            xgb_model = self.train_xgboost(X_train, y_train, X_val, y_val, use_optuna)
            xgb_pred = xgb_model.predict(X_test)
            xgb_metrics = self.calculate_metrics(y_test, xgb_pred)
            
            self.mlflow_manager.log_metrics({f"xgboost_{k}": v for k, v in xgb_metrics.items()})
            self.mlflow_manager.log_model(xgb_model, "xgboost", 
                                         input_example=X_train.iloc[:5])
            
            # Log feature importance
            feature_importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': xgb_model.feature_importances_
            }).sort_values('importance', ascending=False).head(20)
            
            logger.info(f"Top XGBoost features:\n{feature_importance.to_string()}")
            self.mlflow_manager.log_params({f"xgb_top_feature_{i}": f"{row['feature']} ({row['importance']:.4f})" 
                                           for i, (_, row) in enumerate(feature_importance.iterrows())})
            
            results['xgboost'] = {
                'model': xgb_model,
                'metrics': xgb_metrics,
                'predictions': xgb_pred
            }
            
            # Train LightGBM
            lgb_model = self.train_lightgbm(X_train, y_train, X_val, y_val, use_optuna)
            lgb_pred = lgb_model.predict(X_test)
            lgb_metrics = self.calculate_metrics(y_test, lgb_pred)
            
            self.mlflow_manager.log_metrics({f"lightgbm_{k}": v for k, v in lgb_metrics.items()})
            self.mlflow_manager.log_model(lgb_model, "lightgbm",
                                         input_example=X_train.iloc[:5])
            
            # Log feature importance for LightGBM
            lgb_importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': lgb_model.feature_importances_
            }).sort_values('importance', ascending=False).head(20)
            
            logger.info(f"Top LightGBM features:\n{lgb_importance.to_string()}")
            
            results['lightgbm'] = {
                'model': lgb_model,
                'metrics': lgb_metrics,
                'predictions': lgb_pred
            }
            
            # Train Prophet if enabled
            prophet_enabled = self.model_config.get('prophet', {}).get('enabled', True)
            
            if prophet_enabled:
                try:
                    prophet_results = self.train_prophet_daily_total(
                        train_df,
                        val_df,
                        test_df,
                        target_col=target_col,
                    )
                    prophet_model = prophet_results['model']
                    prophet_metrics = prophet_results['metrics']

                    mlflow.set_tags({
                        "prophet_daily_total_forecast_level": "daily_total",
                        "prophet_daily_total_target_grain": "date",
                        "prophet_daily_total_model_family": "prophet",
                        "prophet_daily_total_regressors": ",".join(
                            prophet_results.get("regressor_columns", [])
                        ),
                        "prophet_daily_total_comparable_with_store_level_models": "false",
                    })
                    self.mlflow_manager.log_metrics({
                        f"prophet_daily_total_{k}": v
                        for k, v in prophet_metrics.items()
                    })
                    self.mlflow_manager.log_model(
                        prophet_model,
                        "prophet_daily_total",
                        input_example=prophet_results.get("input_example"),
                    )

                    results['prophet_daily_total'] = prophet_results
                except Exception as e:
                    logger.warning(
                        "Skipping Prophet daily-total model; training failed: %s",
                        e,
                        exc_info=True,
                    )

            # Weighted ensemble based on validation R2. Prophet is logged as a
            # separate daily-total model and is not part of this store-level
            # ensemble.
            xgb_val_pred = xgb_model.predict(X_val)
            lgb_val_pred = lgb_model.predict(X_val)

            xgb_val_r2 = r2_score(y_val, xgb_val_pred)
            lgb_val_r2 = r2_score(y_val, lgb_val_pred)

            min_weight = 0.2
            denominator = xgb_val_r2 + lgb_val_r2
            if denominator <= 0:
                xgb_weight = 0.5
                lgb_weight = 0.5
            else:
                xgb_weight = max(min_weight, xgb_val_r2 / denominator)
                lgb_weight = max(min_weight, lgb_val_r2 / denominator)

            total_weight = xgb_weight + lgb_weight
            xgb_weight = xgb_weight / total_weight
            lgb_weight = lgb_weight / total_weight

            logger.info(
                "Store-level ensemble weights - XGBoost: %.3f, LightGBM: %.3f",
                xgb_weight,
                lgb_weight,
            )

            ensemble_weights = {
                'xgboost': xgb_weight,
                'lightgbm': lgb_weight
            }

            ensemble_pred = xgb_weight * xgb_pred + lgb_weight * lgb_pred
            
            # Create the ensemble model object
            ensemble_models = {
                'xgboost': xgb_model,
                'lightgbm': lgb_model
            }
            
            ensemble_model = EnsembleModel(ensemble_models, ensemble_weights)
            
            ensemble_model.forecast_level = 'store_level'
            ensemble_model.target_grain = 'date+store_id'
            ensemble_model.ensemble_members = [
                'xgboost_store_level',
                'lightgbm_store_level',
            ]

            # Save under the legacy key so existing UI artifact loading still works.
            self.models['ensemble'] = ensemble_model
            
            ensemble_metrics = self.calculate_metrics(y_test, ensemble_pred)
            
            mlflow.set_tags({
                "ensemble_store_level_forecast_level": "store_level",
                "ensemble_store_level_target_grain": "date+store_id",
                "ensemble_store_level_members": "xgboost_store_level,lightgbm_store_level",
            })
            self.mlflow_manager.log_metrics({f"ensemble_{k}": v for k, v in ensemble_metrics.items()})
            self.mlflow_manager.log_metrics({
                f"ensemble_store_level_{k}": v
                for k, v in ensemble_metrics.items()
            })
            self.mlflow_manager.log_model(ensemble_model, "ensemble", 
                                         input_example=X_train.iloc[:5])
            
            results['ensemble_store_level'] = {
                'model': ensemble_model,
                'metrics': ensemble_metrics,
                'predictions': ensemble_pred,
                'forecast_level': 'store_level',
                'target_grain': 'date+store_id',
                'included_in_store_ensemble': True,
                'ensemble_members': [
                    'xgboost_store_level',
                    'lightgbm_store_level',
                ],
                'mlflow_artifact_path': 'ensemble',
                'legacy_model_key': 'ensemble',
            }
            
            # Run diagnostics
            logger.info("Running model diagnostics...")
            test_predictions = {
                'xgboost': xgb_pred if 'xgboost' in results else None,
                'lightgbm': lgb_pred if 'lightgbm' in results else None,
                'ensemble_store_level': ensemble_pred
            }
            
            diagnosis = diagnose_model_performance(
                train_df, val_df, test_df, test_predictions, target_col
            )
            
            logger.info("Diagnostic recommendations:")
            for rec in diagnosis['recommendations']:
                logger.warning(f"- {rec}")
            
            visualizations_logged = False
            visualizations_enabled = (
                os.getenv("ENABLE_MODEL_VISUALIZATIONS", "false")
                .strip()
                .lower()
                == "true"
            )
            if visualizations_enabled:
                logger.info("Generating model comparison visualizations...")
                try:
                    visualizations_logged = self._generate_and_log_visualizations(results, test_df, target_col)
                    if visualizations_logged:
                        logger.info("Model visualizations generated and logged.")
                    else:
                        logger.warning(
                            "Model visualizations were enabled but no "
                            "visualization artifacts were logged."
                        )
                except Exception as viz_error:
                    logger.warning(
                        "Model visualization generation failed; continuing "
                        "without visualization artifacts: %s",
                        viz_error,
                        exc_info=True,
                    )
            else:
                logger.info(
                    "Model visualizations disabled by "
                    "ENABLE_MODEL_VISUALIZATIONS=false."
                )
            
            # Save artifacts
            self.save_artifacts()

            # Get current run ID for verification before closing the run.
            current_run_id = mlflow.active_run().info.run_id

            from utils.s3_verification import verify_s3_artifacts, log_s3_verification_results

            expected_artifacts = [
                'xgboost/MLmodel',
                'lightgbm/MLmodel',
                'ensemble/MLmodel',
                'models/xgboost/',
                'models/lightgbm/',
                'models/ensemble/',
                'encoders.pkl',
                'feature_cols.pkl',
            ]
            if 'prophet_daily_total' in results:
                expected_artifacts.extend([
                    'prophet_daily_total/MLmodel',
                    'models/prophet_daily_total/',
                ])
            if visualizations_logged:
                expected_artifacts.extend(['visualizations/', 'reports/'])

            logger.info("Verifying MinIO artifact storage...")
            verification_results = verify_s3_artifacts(
                run_id=current_run_id,
                expected_artifacts=expected_artifacts,
            )
            log_s3_verification_results(verification_results)

            if not verification_results["success"]:
                raise RuntimeError(
                    "MinIO artifact verification failed: "
                    + "; ".join(
                        verification_results["errors"]
                        + verification_results["missing_artifacts"]
                    )
                )

            self.mlflow_manager.end_run()
            
        except Exception as e:
            self.mlflow_manager.end_run(status="FAILED")
            raise e
        
        return results
    
    def _generate_and_log_visualizations(self, results: Dict[str, Any], 
                                       test_df: pd.DataFrame, 
                                       target_col: str = 'sales') -> bool:
        """Generate and log model comparison visualizations to MLflow"""
        try:
            from ml_models.model_visualization import ModelVisualizer
            import tempfile
            import os
            
            logger.info("Starting visualization generation...")
            visualizer = ModelVisualizer()
            
            # Extract metrics
            metrics_dict = {}
            for model_name, model_results in results.items():
                if model_results.get('forecast_level') == 'daily_total':
                    continue
                if 'metrics' in model_results:
                    metrics_dict[model_name] = model_results['metrics']
            
            # Prepare predictions data
            predictions_dict = {}
            for model_name, model_results in results.items():
                if model_results.get('forecast_level') == 'daily_total':
                    continue
                if 'predictions' in model_results and model_results['predictions'] is not None:
                    pred_df = test_df[['date']].copy()
                    pred_df['prediction'] = model_results['predictions']
                    predictions_dict[model_name] = pred_df
            
            # Extract feature importance if available
            feature_importance_dict = {}
            for model_name, model_results in results.items():
                if model_name in ['xgboost', 'lightgbm'] and 'model' in model_results:
                    model = model_results['model']
                    if hasattr(model, 'feature_importances_'):
                        importance_df = pd.DataFrame({
                            'feature': self.feature_cols,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        feature_importance_dict[model_name] = importance_df
            
            # Create temporary directory for visualizations
            with tempfile.TemporaryDirectory() as temp_dir:
                logger.info(f"Creating visualizations in temporary directory: {temp_dir}")
                
                # Generate all visualizations
                saved_files = visualizer.create_comprehensive_report(
                    metrics_dict=metrics_dict,
                    predictions_dict=predictions_dict,
                    actual_data=test_df,
                    feature_importance_dict=feature_importance_dict if feature_importance_dict else None,
                    save_dir=temp_dir
                )
                
                logger.info(f"Generated {len(saved_files)} visualization files: {list(saved_files.keys())}")
                
                # Log each visualization to MLflow
                for viz_name, file_path in saved_files.items():
                    if os.path.exists(file_path):
                        mlflow.log_artifact(file_path, "visualizations")
                        logger.info(f"Logged visualization: {viz_name} from {file_path}")
                    else:
                        logger.warning(f"Visualization file not found: {file_path}")
                
                # Also create a combined HTML report
                self._create_combined_html_report(saved_files, temp_dir)
                
                # Log the combined report
                combined_report = os.path.join(temp_dir, 'model_comparison_report.html')
                if os.path.exists(combined_report):
                    mlflow.log_artifact(combined_report, "reports")
                    logger.info("Logged combined HTML report")
                    return True
                return False
                    
        except Exception as e:
            logger.warning(
                "Failed to generate visualizations; continuing without "
                "visualization artifacts: %s",
                e,
                exc_info=True,
            )
            return False
    
    def _create_combined_html_report(self, saved_files: Dict[str, str], save_dir: str) -> None:
        """Create a combined HTML report with all visualizations"""
        import os
        from datetime import datetime
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Comparison Report</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }
                h1, h2 {
                    color: #333;
                }
                .section {
                    background-color: white;
                    padding: 20px;
                    margin-bottom: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .timestamp {
                    color: #666;
                    font-size: 14px;
                }
                iframe {
                    width: 100%;
                    height: 800px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    margin-top: 10px;
                }
                img {
                    max-width: 100%;
                    height: auto;
                    border-radius: 4px;
                    margin-top: 10px;
                }
            </style>
        </head>
        <body>
            <h1>Sales Forecast Model Comparison Report</h1>
            <p class="timestamp">Generated on: {timestamp}</p>
        """
        
        html_content = html_content.format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # Add each visualization section
        sections = [
            ('metrics_comparison', 'Model Performance Metrics'),
            ('predictions_comparison', 'Predictions Comparison'),
            ('residuals_analysis', 'Residuals Analysis'),
            ('error_distribution', 'Error Distribution'),
            ('feature_importance', 'Feature Importance'),
            ('summary', 'Summary Statistics')
        ]
        
        for key, title in sections:
            if key in saved_files:
                html_content += f'<div class="section"><h2>{title}</h2>'
                
                # All files are now PNG - base64 encode them
                import base64
                with open(saved_files[key], 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode()
                html_content += f'<img src="data:image/png;base64,{img_data}" alt="{title}">'
                
                html_content += '</div>'
        
        html_content += """
        </body>
        </html>
        """
        
        # Save the combined report
        with open(os.path.join(save_dir, 'model_comparison_report.html'), 'w') as f:
            f.write(html_content)
    
    def save_artifacts(self):
        import tempfile

        with tempfile.TemporaryDirectory() as artifact_dir:
            joblib.dump(self.encoders, os.path.join(artifact_dir, 'encoders.pkl'))
            joblib.dump(self.feature_cols, os.path.join(artifact_dir, 'feature_cols.pkl'))

            model_dir = os.path.join(artifact_dir, 'models')
            for model_name, model in self.models.items():
                target_dir = os.path.join(model_dir, model_name)
                os.makedirs(target_dir, exist_ok=True)
                joblib.dump(model, os.path.join(target_dir, f'{model_name}_model.pkl'))

            self.mlflow_manager.log_artifacts(artifact_dir)
        
        logger.info("Artifacts saved successfully")
