from pathlib import Path
import importlib.util
import sys

import joblib
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
INCLUDE_PATH = ROOT / "include"
if str(INCLUDE_PATH) not in sys.path:
    sys.path.insert(0, str(INCLUDE_PATH))

LOADER_PATH = ROOT / "ui" / "utils" / "simple_model_loader.py"
spec = importlib.util.spec_from_file_location("ui_simple_model_loader", LOADER_PATH)
ui_simple_model_loader = importlib.util.module_from_spec(spec)
sys.modules["ui_simple_model_loader"] = ui_simple_model_loader
spec.loader.exec_module(ui_simple_model_loader)

from ml_models.ensemble_model import EnsembleModel

SimpleModelLoader = ui_simple_model_loader.SimpleModelLoader


class ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


def _write_model_artifact(artifacts_dir, artifact_dir, filename, model):
    model_dir = artifacts_dir / "models" / artifact_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / filename)


def test_ui_loader_uses_registry_artifact_mapping():
    assert list(ui_simple_model_loader.MODEL_ARTIFACTS) == [
        "ensemble_store_level",
        "xgboost_store_level",
        "lightgbm_store_level",
    ]
    assert ui_simple_model_loader.MODEL_ARTIFACTS["ensemble_store_level"] == (
        "ensemble",
        "ensemble_model.pkl",
    )


def _patch_mlflow_download(monkeypatch, artifacts_dir):
    class FakeMlflowClient:
        def download_artifacts(self, run_id, artifact_path, dst_path):
            return str(artifacts_dir)

    monkeypatch.setattr(
        ui_simple_model_loader.mlflow.tracking,
        "MlflowClient",
        FakeMlflowClient,
    )


def test_loader_uses_saved_calibrated_ensemble_under_registry_key(tmp_path, monkeypatch):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    _patch_mlflow_download(monkeypatch, artifacts_dir)

    xgboost = ConstantModel(10.0)
    lightgbm = ConstantModel(30.0)
    ensemble = EnsembleModel(
        {"xgboost": xgboost, "lightgbm": lightgbm},
        {"xgboost": 0.8, "lightgbm": 0.2},
    )

    _write_model_artifact(artifacts_dir, "xgboost", "xgboost_model.pkl", xgboost)
    _write_model_artifact(artifacts_dir, "lightgbm", "lightgbm_model.pkl", lightgbm)
    _write_model_artifact(artifacts_dir, "ensemble", "ensemble_model.pkl", ensemble)

    loader = SimpleModelLoader()

    assert loader.load_models_from_run("run-1") is True
    assert set(loader.models) == {
        "xgboost_store_level",
        "lightgbm_store_level",
        "ensemble_store_level",
    }
    assert "ensemble" not in loader.models
    assert loader.models["ensemble_store_level"].weights == {
        "xgboost": 0.8,
        "lightgbm": 0.2,
    }
    assert loader.predict(np.zeros((2, 1)), "ensemble").tolist() == [14.0, 14.0]


def test_loader_does_not_rebuild_equal_weight_ensemble_when_artifact_is_missing(
    tmp_path,
    monkeypatch,
):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    _patch_mlflow_download(monkeypatch, artifacts_dir)

    _write_model_artifact(
        artifacts_dir,
        "xgboost",
        "xgboost_model.pkl",
        ConstantModel(10.0),
    )
    _write_model_artifact(
        artifacts_dir,
        "lightgbm",
        "lightgbm_model.pkl",
        ConstantModel(30.0),
    )

    loader = SimpleModelLoader()

    assert loader.load_models_from_run("run-1") is True
    assert set(loader.models) == {"xgboost_store_level", "lightgbm_store_level"}
    with pytest.raises(ValueError, match="ensemble_store_level"):
        loader.predict(np.zeros((2, 1)), "ensemble")
