"""
Simplified model loader for Streamlit UI
"""

import os
import pickle
import sys
import importlib.util
from pathlib import Path

import joblib
import mlflow
import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

INCLUDE_PATH_CANDIDATES = [
    os.getenv("AIRFLOW_INCLUDE_PATH"),
    "/usr/local/airflow/include",
    str(Path(__file__).resolve().parents[2] / "include"),
]

for include_path in INCLUDE_PATH_CANDIDATES:
    if include_path and os.path.isdir(include_path) and include_path not in sys.path:
        sys.path.append(include_path)

STORE_LEVEL_MODEL_TYPES = (
    "ensemble_store_level",
    "xgboost_store_level",
    "lightgbm_store_level",
)

DEFAULT_MODEL_ARTIFACTS = {
    "ensemble_store_level": ("ensemble", "ensemble_model.pkl"),
    "xgboost_store_level": ("xgboost", "xgboost_model.pkl"),
    "lightgbm_store_level": ("lightgbm", "lightgbm_model.pkl"),
}

MODEL_ALIASES = {
    "xgboost": "xgboost_store_level",
    "lightgbm": "lightgbm_store_level",
    "ensemble": "ensemble_store_level",
}

MODEL_DISPLAY_NAMES = {
    "xgboost_store_level": "XGBoost",
    "lightgbm_store_level": "LightGBM",
    "ensemble_store_level": "Calibrated Ensemble",
}


def _load_model_registry_module():
    for include_path in INCLUDE_PATH_CANDIDATES:
        if not include_path:
            continue

        registry_path = Path(include_path) / "utils" / "model_registry.py"
        if not registry_path.exists():
            continue

        spec = importlib.util.spec_from_file_location(
            "sales_forecast_model_registry",
            registry_path,
        )
        registry_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(registry_module)
        return registry_module

    return None


def _model_artifacts_from_registry():
    registry_module = _load_model_registry_module()
    if registry_module is None:
        logger.warning(
            "Model registry module not found; using default UI artifact mapping"
        )
        return DEFAULT_MODEL_ARTIFACTS

    model_artifacts = {}
    for model_type in STORE_LEVEL_MODEL_TYPES:
        entry = registry_module.get_model_registry_entry(model_type)
        if not entry:
            logger.warning(
                "Model registry entry missing for %s; using default artifact mapping",
                model_type,
            )
            return DEFAULT_MODEL_ARTIFACTS

        artifact_path = entry["artifact_path"].strip("/")
        artifact_name = artifact_path.split("/")[-1]
        model_artifacts[model_type] = (
            artifact_path,
            f"{artifact_name}_model.pkl",
        )

    return model_artifacts


MODEL_ARTIFACTS = _model_artifacts_from_registry()


def canonical_model_type(model_type: str) -> str:
    return MODEL_ALIASES.get(model_type, model_type)


def display_model_type(model_type: str) -> str:
    return MODEL_DISPLAY_NAMES.get(canonical_model_type(model_type), model_type)


class SimpleModelLoader:
    """Simplified model loader that works with pickled models"""
    
    def __init__(self):
        self.mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5001')
        mlflow.set_tracking_uri(self.mlflow_uri)
        
        self.models = {}
        self.encoders = None
        self.feature_cols = None
        self.loaded = False

    @staticmethod
    def canonical_model_type(model_type: str) -> str:
        return canonical_model_type(model_type)

    @staticmethod
    def display_model_type(model_type: str) -> str:
        return display_model_type(model_type)

    def available_model_types(self):
        return [
            model_type
            for model_type in MODEL_ARTIFACTS
            if model_type in self.models
        ]

    def _load_pickle_artifact(self, artifact_path: str, label: str):
        try:
            model = joblib.load(artifact_path)
            logger.info("Loaded %s model", label)
            return model
        except Exception as e:
            logger.warning(
                "Could not load %s with joblib, trying pickle: %s",
                label,
                e,
            )
            with open(artifact_path, 'rb') as f:
                model = pickle.load(f)
            logger.info("Loaded %s model with pickle", label)
            return model
        
    def load_models_from_run(self, run_id: str) -> bool:
        """Load models from a specific MLflow run"""
        try:
            logger.info(f"Loading models from run: {run_id}")
            self.models = {}
            self.encoders = None
            self.feature_cols = None
            self.loaded = False
            
            # Download artifacts
            client = mlflow.tracking.MlflowClient()
            local_dir = f"/tmp/mlflow_models/{run_id}"
            os.makedirs(local_dir, exist_ok=True)
            
            # Download all artifacts
            artifacts_path = client.download_artifacts(run_id, "", dst_path=local_dir)
            logger.info(f"Downloaded artifacts to: {artifacts_path}")
            
            # Load encoders
            encoders_path = os.path.join(artifacts_path, "encoders.pkl")
            if os.path.exists(encoders_path):
                try:
                    self.encoders = joblib.load(encoders_path)
                    logger.info("Loaded encoders")
                except Exception as e:
                    logger.warning(f"Could not load encoders with joblib, trying pickle: {e}")
                    with open(encoders_path, 'rb') as f:
                        self.encoders = pickle.load(f)
                    logger.info("Loaded encoders with pickle")
            
            # Load feature columns
            feature_cols_path = os.path.join(artifacts_path, "feature_cols.pkl")
            if os.path.exists(feature_cols_path):
                try:
                    self.feature_cols = joblib.load(feature_cols_path)
                    logger.info(f"Loaded {len(self.feature_cols)} feature columns")
                except Exception as e:
                    logger.warning(f"Could not load feature_cols with joblib, trying pickle: {e}")
                    with open(feature_cols_path, 'rb') as f:
                        self.feature_cols = pickle.load(f)
                    logger.info(f"Loaded {len(self.feature_cols)} feature columns with pickle")
            
            # Load models
            models_dir = os.path.join(artifacts_path, "models")
            if os.path.exists(models_dir):
                for model_type, (artifact_dir, filename) in MODEL_ARTIFACTS.items():
                    model_path = os.path.join(models_dir, artifact_dir, filename)
                    if not os.path.exists(model_path):
                        logger.warning(
                            "Model artifact missing for %s: %s",
                            model_type,
                            model_path,
                        )
                        continue

                    try:
                        self.models[model_type] = self._load_pickle_artifact(
                            model_path,
                            MODEL_DISPLAY_NAMES.get(model_type, model_type),
                        )
                    except Exception as e:
                        logger.error(
                            "Skipping %s because its saved artifact could not be loaded: %s",
                            model_type,
                            e,
                        )
            
            self.loaded = len(self.models) > 0
            return self.loaded
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def get_latest_run(self) -> Optional[str]:
        """Get the latest successful run ID"""
        try:
            exp = mlflow.get_experiment_by_name("sales_forecasting")
            if not exp:
                return None
            
            runs = mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string="status = 'FINISHED'",
                order_by=["start_time DESC"],
                max_results=1
            )
            
            if len(runs) > 0:
                return runs.iloc[0]['run_id']
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest run: {e}")
            return None
    
    def predict(self, X: np.ndarray, model_type: str = 'ensemble_store_level') -> np.ndarray:
        """Make predictions with specified model"""
        model_type = canonical_model_type(model_type)
        if model_type in self.models:
            return self.models[model_type].predict(X)

        raise ValueError(f"Model type '{model_type}' not available")
