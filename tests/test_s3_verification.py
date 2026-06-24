from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "include"))

from utils import s3_verification


def test_verify_s3_artifacts_skips_non_s3_artifact_uri(monkeypatch, caplog):
    artifact_uri = "file:///tmp/mlruns/1/run-1/artifacts"

    class FakeMlflowClient:
        def get_run(self, run_id):
            return SimpleNamespace(
                info=SimpleNamespace(artifact_uri=artifact_uri)
            )

    def fail_if_s3_client_is_created(*args, **kwargs):
        raise AssertionError("S3 client should not be created for local artifacts")

    monkeypatch.setattr(
        s3_verification.mlflow.tracking,
        "MlflowClient",
        FakeMlflowClient,
    )
    monkeypatch.setattr(s3_verification.boto3, "client", fail_if_s3_client_is_created)

    with caplog.at_level("WARNING"):
        results = s3_verification.verify_s3_artifacts("run-1")

    assert results["success"] is True
    assert results["artifact_uri"] == artifact_uri
    assert results["errors"] == []
    assert results["s3_artifacts"] == []
    assert "Skipping S3 artifact verification for non-S3 artifact URI" in caplog.text
