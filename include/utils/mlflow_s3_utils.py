import logging


logger = logging.getLogger(__name__)


class MLflowS3Manager:
    def sync_mlflow_artifacts_to_s3(self, run_id):
        from utils.s3_verification import verify_s3_artifacts

        logger.info("Verifying MLflow artifacts in MinIO for run %s", run_id)
        results = verify_s3_artifacts(run_id)
        if not results["success"]:
            raise RuntimeError(
                "MLflow artifacts are not available in MinIO: "
                + "; ".join(results["errors"] + results["missing_artifacts"])
            )
        return results
