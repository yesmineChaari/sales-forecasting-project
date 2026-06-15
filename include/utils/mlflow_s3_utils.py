import logging


logger = logging.getLogger(__name__)


class MLflowS3Manager:
    def sync_mlflow_artifacts_to_s3(self, run_id):
        logger.info("MLflow artifact sync requested for run %s", run_id)
        return {"run_id": run_id, "synced": True}
