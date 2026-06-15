import os
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from typing import Dict, Any, Optional, List
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import joblib
from .service_discovery import get_mlflow_endpoint, get_minio_endpoint

logger = logging.getLogger(__name__)


class JoblibPyfuncWrapper(mlflow.pyfunc.PythonModel):
    """Minimal pyfunc wrapper around any estimator with a predict method."""

    def load_context(self, context):
        self.model = joblib.load(context.artifacts["model"])

    def predict(self, context, model_input):
        return self.model.predict(model_input)


class MLflowManager:
    def __init__(self, config_path: str = "/usr/local/airflow/include/config/ml_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        mlflow_config = self.config['mlflow']
        # Use service discovery to get tracking URI
        self.tracking_uri = get_mlflow_endpoint()
        
        self.experiment_name = mlflow_config['experiment_name']
        self.registry_name = mlflow_config['registry_name']
        
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Try to create experiment, with fallback
        try:
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            logger.warning(f"Failed to set experiment {self.experiment_name}: {e}")
            # Try with localhost if initial connection failed
            if 'mlflow' in self.tracking_uri:
                self.tracking_uri = "http://localhost:5001"
                mlflow.set_tracking_uri(self.tracking_uri)
                os.environ['MLFLOW_TRACKING_URI'] = self.tracking_uri
                logger.info(f"Retrying with localhost: {self.tracking_uri}")
                try:
                    mlflow.set_experiment(self.experiment_name)
                except Exception as e2:
                    logger.error(f"Failed to connect to MLflow: {e2}")
        
        # Configure S3 endpoint for MinIO using service discovery
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = get_minio_endpoint()
        os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID', 'minioadmin')
        os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY', 'minioadmin')
        
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> str:
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        run = mlflow.start_run(run_name=run_name, tags=tags)
        logger.info(f"Started MLflow run: {run.info.run_id}")
        return run.info.run_id
    
    def log_params(self, params: Dict[str, Any]):
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_model(self, model, model_name: str, input_example: Optional[pd.DataFrame] = None,
                  signature: Optional[Any] = None, registered_model_name: Optional[str] = None):
        """
        Log a loadable MLflow model plus the legacy pickle artifact layout.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, f"{model_name}_model.pkl")
            joblib.dump(model, model_path)

            mlflow.pyfunc.log_model(
                artifact_path=model_name,
                python_model=JoblibPyfuncWrapper(),
                artifacts={"model": model_path},
                input_example=input_example,
                signature=signature,
                registered_model_name=registered_model_name,
            )
            logger.info("Successfully logged %s as MLflow pyfunc model", model_name)

            # Preserve the previous artifact layout used by reports and manual debugging.
            mlflow.log_artifact(model_path, artifact_path=f"models/{model_name}")

            metadata = {
                "model_type": model_name,
                "framework": type(model).__module__,
                "class": type(model).__name__,
                "timestamp": datetime.now().isoformat(),
                "mlflow_model_uri": f"runs:/{mlflow.active_run().info.run_id}/{model_name}",
            }
            metadata_path = os.path.join(tmpdir, f"{model_name}_metadata.yaml")
            with open(metadata_path, 'w') as f:
                yaml.dump(metadata, f)
            mlflow.log_artifact(metadata_path, artifact_path=f"models/{model_name}")

    def _model_uri_has_mlmodel(self, run_id: str, artifact_path: str) -> bool:
        try:
            local_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path=artifact_path,
            )
        except Exception:
            return False

        return os.path.exists(os.path.join(local_path, "MLmodel"))

    def _resolve_model_uri(self, run_id: str, artifact_path: str) -> str:
        candidates = []
        if artifact_path:
            candidates.append(artifact_path.strip("/"))

        if artifact_path.startswith("models/"):
            candidates.append(artifact_path.split("/", 1)[1].strip("/"))
        else:
            candidates.append(f"models/{artifact_path.strip('/')}")

        for candidate in dict.fromkeys(candidates):
            if self._model_uri_has_mlmodel(run_id, candidate):
                return f"runs:/{run_id}/{candidate}"

        raise ValueError(
            f"No loadable MLflow model found for run {run_id} at any of: "
            f"{', '.join(dict.fromkeys(candidates))}"
        )

    def _legacy_pickle_artifact_path(self, artifact_path: str) -> str:
        artifact_path = artifact_path.strip("/")
        model_name = artifact_path.split("/")[-1]
        if artifact_path.startswith("models/"):
            return f"{artifact_path}/{model_name}_model.pkl"
        return f"models/{artifact_path}/{model_name}_model.pkl"

    def _load_legacy_pickle_model(self, run_id: str, artifact_path: str):
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=self._legacy_pickle_artifact_path(artifact_path),
        )
        return joblib.load(local_path)

    def _registered_model_name(self, model_name: str) -> str:
        return f"{self.registry_name}_{model_name}"

    def _is_run_id_version_fallback(self, version: str) -> bool:
        try:
            int(str(version))
            return False
        except ValueError:
            return True

    def _create_registered_model_if_needed(self, registered_name: str):
        try:
            self.client.create_registered_model(registered_name)
        except Exception as e:
            if "already exists" not in str(e).lower():
                logger.debug("Registered model creation skipped for %s: %s", registered_name, e)
    
    def log_artifacts(self, artifact_path: str):
        mlflow.log_artifacts(artifact_path)
    
    def log_figure(self, figure, artifact_file: str):
        mlflow.log_figure(figure, artifact_file)
    
    def end_run(self, status: str = "FINISHED"):
        # Get run ID before ending
        run = mlflow.active_run()
        run_id = run.info.run_id if run else None
        
        mlflow.end_run(status=status)
        logger.info("Ended MLflow run")
        
        # Sync artifacts to S3 after run ends
        if run_id and status == "FINISHED":
            try:
                from utils.mlflow_s3_utils import MLflowS3Manager
                s3_manager = MLflowS3Manager()
                s3_manager.sync_mlflow_artifacts_to_s3(run_id)
                logger.info(f"Synced artifacts to S3 for run {run_id}")
            except Exception as e:
                logger.warning(f"Failed to sync artifacts to S3: {e}")
    
    def get_best_model(self, metric: str = "rmse", ascending: bool = True) -> Dict[str, Any]:
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1
        )
        
        if len(runs) == 0:
            raise ValueError("No runs found in the experiment")
        
        best_run = runs.iloc[0]
        return {
            "run_id": best_run["run_id"],
            "metrics": {col.replace("metrics.", ""): val 
                       for col, val in best_run.items() 
                       if col.startswith("metrics.")},
            "params": {col.replace("params.", ""): val 
                      for col, val in best_run.items() 
                      if col.startswith("params.")}
        }
    
    def load_model(self, model_uri: str):
        """Load model from MLflow or from artifacts"""
        try:
            return mlflow.pyfunc.load_model(model_uri)
        except Exception:
            # Try loading from artifacts
            if "runs:/" in model_uri:
                run_id = model_uri.split("/")[1]
                artifact_path = "/".join(model_uri.split("/")[2:])
                return self._load_legacy_pickle_model(run_id, artifact_path)

            raise ValueError(f"Cannot load model from {model_uri}")
    
    def register_model(self, run_id: str, model_name: str, artifact_path: str) -> str:
        """Register the loadable MLflow model artifact and return its real version."""
        try:
            model_uri = self._resolve_model_uri(run_id, artifact_path)
            registered_name = self._registered_model_name(model_name)
            self._create_registered_model_if_needed(registered_name)
            model_version = mlflow.register_model(model_uri, registered_name)
            logger.info(
                "Registered model %s from %s as version %s",
                registered_name,
                model_uri,
                model_version.version,
            )
            return model_version.version
        except Exception as e:
            raise RuntimeError(
                f"Failed to register {model_name} from run {run_id}: {e}"
            ) from e
    
    def transition_model_stage(self, model_name: str, version: str, stage: str):
        if self._is_run_id_version_fallback(version):
            raise ValueError(
                f"Refusing to transition {model_name}: {version} is not an MLflow model version"
            )

        registered_name = self._registered_model_name(model_name)
        stage_error = None
        stage_set = False

        try:
            self.client.transition_model_version_stage(
                name=registered_name,
                version=version,
                stage=stage,
                archive_existing_versions=True,
            )
        except Exception as e:
            stage_error = e
            logger.warning(
                "Model stage transition failed for %s version %s: %s",
                registered_name,
                version,
                e,
            )

        try:
            model_version = self.client.get_model_version(registered_name, str(version))
            stage_set = getattr(model_version, "current_stage", None) == stage
        except Exception as e:
            logger.debug("Could not verify model stage for %s version %s: %s", registered_name, version, e)

        alias_set = False
        alias_error = None
        if hasattr(self.client, "set_registered_model_alias"):
            try:
                self.client.set_registered_model_alias(registered_name, stage, str(version))
                alias_set = True
            except Exception as e:
                alias_error = e
                logger.warning(
                    "Model alias assignment failed for %s version %s alias %s: %s",
                    registered_name,
                    version,
                    stage,
                    e,
                )

        if stage_set or alias_set:
            logger.info(
                "Marked %s version %s as %s%s",
                registered_name,
                version,
                stage,
                " using alias fallback" if alias_set and not stage_set else "",
            )
            return

        if stage_error:
            raise RuntimeError(
                f"Failed to transition {model_name} version {version} to {stage}: {stage_error}"
            ) from stage_error

        raise RuntimeError(
            f"Failed to mark {model_name} version {version} as {stage}"
            + (f": {alias_error}" if alias_error else "")
        )
    
    def get_latest_model_version(self, model_name: str, stage: Optional[str] = None) -> Dict[str, Any]:
        try:
            registered_name = self._registered_model_name(model_name)
            filter_string = f"name='{registered_name}'"
            versions = list(self.client.search_model_versions(filter_string))

            if stage:
                staged_versions = [
                    version
                    for version in versions
                    if getattr(version, "current_stage", None) == stage
                ]
                if staged_versions:
                    versions = staged_versions
                elif hasattr(self.client, "get_model_version_by_alias"):
                    alias_version = self.client.get_model_version_by_alias(
                        registered_name,
                        stage,
                    )
                    return {
                        "version": alias_version.version,
                        "stage": stage,
                        "run_id": alias_version.run_id,
                        "source": alias_version.source,
                    }
                else:
                    versions = []

            if not versions:
                raise ValueError(f"No model versions found for {model_name}")
            
            latest_version = max(versions, key=lambda x: int(x.version))
            return {
                "version": latest_version.version,
                "stage": latest_version.current_stage,
                "run_id": latest_version.run_id,
                "source": latest_version.source
            }
        except:
            # Fallback to finding the best run
            best_model = self.get_best_model()
            return {
                "version": best_model["run_id"],
                "stage": "None",
                "run_id": best_model["run_id"],
                "source": f"runs:/{best_model['run_id']}/models"
            }
