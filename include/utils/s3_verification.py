"""
S3 artifact verification utilities for MLflow.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import boto3
import mlflow
from botocore.client import Config

logger = logging.getLogger(__name__)


def _resolve_bucket_and_prefix(artifact_uri: str) -> Tuple[str, str]:
    if artifact_uri.startswith("s3://"):
        parts = artifact_uri.replace("s3://", "", 1).split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        return bucket, prefix.rstrip("/")

    if artifact_uri.startswith("mlflow-artifacts:/"):
        bucket = os.getenv("MLFLOW_ARTIFACT_BUCKET", "mlflow-artifacts")
        prefix = artifact_uri.replace("mlflow-artifacts:/", "", 1).lstrip("/")
        return bucket, prefix.rstrip("/")

    raise ValueError(f"Artifact URI is not backed by S3/MinIO: {artifact_uri}")


def verify_s3_artifacts(
    run_id: str,
    expected_artifacts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Verify that MLflow artifacts for a run are present in MinIO/S3.
    """
    results = {
        "success": False,
        "artifact_uri": "",
        "s3_artifacts": [],
        "missing_artifacts": [],
        "errors": [],
    }

    try:
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        artifact_uri = run.info.artifact_uri
        results["artifact_uri"] = artifact_uri

        try:
            bucket, prefix = _resolve_bucket_and_prefix(artifact_uri)
        except ValueError:
            logger.warning(
                "Skipping S3 artifact verification for non-S3 artifact URI: %s",
                artifact_uri,
            )
            results["success"] = True
            return results

        s3_client = boto3.client(
            "s3",
            endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"),
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )

        s3_objects = []
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                relative_path = obj["Key"].replace(prefix, "", 1).lstrip("/")
                if relative_path:
                    s3_objects.append(relative_path)

        if s3_objects:
            results["s3_artifacts"] = s3_objects

            if expected_artifacts:
                for expected in expected_artifacts:
                    if not any(expected in artifact for artifact in s3_objects):
                        results["missing_artifacts"].append(expected)

            results["success"] = not results["missing_artifacts"]
            logger.info("Found %s artifacts in S3 for run %s", len(s3_objects), run_id)
            logger.info("Artifacts: %s...", ", ".join(s3_objects[:5]))
        else:
            results["errors"].append("No artifacts found in S3")

    except Exception as e:
        results["errors"].append(str(e))
        logger.error("Error verifying S3 artifacts: %s", e)

    return results


def log_s3_verification_results(results: Dict[str, Any]):
    """Log S3 verification results."""
    if results["success"]:
        logger.info("S3 artifact verification PASSED")
        logger.info("  - Artifact URI: %s", results["artifact_uri"])
        logger.info("  - Total artifacts: %s", len(results["s3_artifacts"]))
    else:
        logger.error("S3 artifact verification FAILED")
        logger.error("  - Artifact URI: %s", results["artifact_uri"])
        for error in results["errors"]:
            logger.error("  - Error: %s", error)
        if results["missing_artifacts"]:
            logger.error(
                "  - Missing artifacts: %s",
                ", ".join(results["missing_artifacts"]),
            )
