"""Smoke-test the pinned dependency stack without external services."""

import importlib
import json
import os
import sqlite3
import subprocess
import sys
from collections.abc import Generator, Iterator
from contextlib import closing, contextmanager
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from evidently import Report
from evidently.presets import DataDriftPreset
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mlflow.sklearn import load_model, log_model
from mlflow.utils.thread_utils import ThreadLocalVariable
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline


@pytest.mark.parametrize(
    "module",
    [
        "sqlite3",
        "pydantic_settings",
        "scipy",
        "sqlalchemy",
        "streamlit",
        "uvicorn",
    ],
)
def test_runtime_imports(module: str) -> None:
    """Import native extensions and service clients with the shared pins."""
    assert importlib.import_module(module) is not None


@contextmanager
def _isolated_mlflow_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[str]:
    """Restore MLflow configuration even if a test fails or leaves a run open."""
    previous_tracking_uri = mlflow.get_tracking_uri()
    previous_registry_uri = mlflow.get_registry_uri()
    assert mlflow.active_run() is None
    tmp_path.mkdir(parents=True, exist_ok=True)
    uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    with monkeypatch.context() as isolated:
        isolated.chdir(tmp_path)
        # Snapshot the pinned MLflow version's overrides, including an unset value.
        # Restoring only public getters would turn URI fallback into an explicit URI.
        isolated.setattr("mlflow.tracking._tracking_service.utils._tracking_uri", uri)
        isolated.setattr("mlflow.tracking._model_registry.utils._registry_uri", uri)
        isolated.setattr("mlflow.tracking.fluent._active_experiment_id", None)
        # end_run() writes this thread-local state; restore the original object
        # rather than leaving a temporary run ID attached to the old backend.
        isolated.setattr("mlflow.tracking.fluent._last_active_run_id", ThreadLocalVariable(lambda: None))
        isolated.setenv("MLFLOW_TRACKING_URI", uri)
        isolated.setenv("MLFLOW_REGISTRY_URI", uri)
        isolated.delenv("MLFLOW_EXPERIMENT_ID", raising=False)
        isolated.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
        isolated.delenv("MLFLOW_RUN_ID", raising=False)
        experiment_id = mlflow.create_experiment("dependency-compatibility")
        try:
            yield experiment_id
        finally:
            mlflow.end_run()
    assert mlflow.get_tracking_uri() == previous_tracking_uri
    assert mlflow.get_registry_uri() == previous_registry_uri
    assert mlflow.active_run() is None


@pytest.fixture
def isolated_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Provide an experiment with fully scoped MLflow run/configuration state."""
    with _isolated_mlflow_context(tmp_path, monkeypatch) as experiment_id:
        yield experiment_id


@pytest.mark.parametrize("fail", [False, True], ids=["success", "exception"])
def test_mlflow_context_restores_run_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool) -> None:
    """Preserve an inherited run ID and prior last run through either exit path."""
    with _isolated_mlflow_context(tmp_path / "original", monkeypatch) as experiment_id:
        with mlflow.start_run(experiment_id=experiment_id) as previous_run:
            previous_id = previous_run.info.run_id
        previous_uri = mlflow.get_tracking_uri()

        with monkeypatch.context() as inherited:
            inherited.setenv("MLFLOW_RUN_ID", previous_id)

            def exercise_context() -> None:
                with _isolated_mlflow_context(tmp_path / "isolated", inherited) as temporary_experiment:
                    assert "MLFLOW_RUN_ID" not in os.environ
                    assert mlflow.last_active_run() is None
                    temporary_run = mlflow.start_run(experiment_id=temporary_experiment)
                    assert temporary_run.info.run_id != previous_id
                    if fail:
                        # Leave this run open to exercise exception-path cleanup.
                        raise RuntimeError("simulated test failure")
                    mlflow.end_run()

            if fail:
                with pytest.raises(RuntimeError, match="simulated test failure"):
                    exercise_context()
            else:
                exercise_context()

            assert os.environ["MLFLOW_RUN_ID"] == previous_id
            assert mlflow.get_tracking_uri() == previous_uri
            assert mlflow.get_registry_uri() == previous_uri
            assert mlflow.active_run() is None
            restored_run = mlflow.last_active_run()
            assert restored_run is not None
            assert restored_run.info.run_id == previous_id


def test_mlflow_model_round_trip(isolated_mlflow: str) -> None:
    """Validate a registered version before promoting and reloading its alias."""
    features = pd.DataFrame(
        {"age": [60.0, 62.0, np.nan, 66.0], "sex": [0.0, 1.0, 0.0, 1.0], "test_time": [0.0, 7.0, 14.0, 21.0]}
    )
    target = np.array([10.0, 12.0, 14.0, 16.0])
    pipeline = make_pipeline(SimpleImputer(strategy="median"), LinearRegression())
    pipeline.fit(features, target)
    expected = pipeline.predict(features)
    assert np.isfinite(expected).all()

    with mlflow.start_run(experiment_id=isolated_mlflow) as run:
        mlflow.log_param("estimator", "LinearRegression")
        mlflow.log_param("feature_set", "demographics_time")
        mlflow.log_metric("mae", float(np.mean(np.abs(expected - target))))
        mlflow.log_dict({"status": "ok"}, "compatibility.json")
        info = log_model(
            pipeline,
            name="pipeline",
            input_example=features,
            # The locally fitted imputer stores a NumPy dtype; no tree types are needed.
            skops_trusted_types=["numpy.dtype"],
        )

    version = mlflow.register_model(info.model_uri, "compatibility-model")
    client = mlflow.MlflowClient()
    candidate = load_model(f"models:/compatibility-model/{version.version}")
    assert isinstance(candidate, Pipeline)
    assert isinstance(candidate.named_steps["linearregression"], LinearRegression)
    np.testing.assert_allclose(candidate.predict(features), expected)
    client.set_registered_model_alias("compatibility-model", "champion", version.version)
    loaded = load_model("models:/compatibility-model@champion")
    assert isinstance(loaded, Pipeline)
    np.testing.assert_allclose(loaded.predict(features), expected)
    parameters = client.get_run(run.info.run_id).data.params
    assert parameters["estimator"] == "LinearRegression"
    assert parameters["feature_set"] == "demographics_time"
    assert client.list_artifacts(run.info.run_id)[0].path == "compatibility.json"


def test_evidently_report(tmp_path: Path) -> None:
    """Exercise Evidently with pandas and serialize a dashboard-ready report."""
    reference = pd.DataFrame({"voice": np.linspace(0.0, 1.0, 100)})
    current = reference + 5.0
    report = Report([DataDriftPreset()]).run(reference_data=reference, current_data=current)
    assert report.dict()
    output = tmp_path / "drift.html"
    report.save_html(str(output))
    assert output.stat().st_size > 0


def test_sqlite_audit_round_trip(tmp_path: Path) -> None:
    """Persist and retrieve structured requests and predictions locally."""
    path = tmp_path / "audit.db"
    request = {"voice": 0.1, "age": 60.0}
    predictions = [12.0]
    with closing(sqlite3.connect(path, timeout=5.0)) as connection, connection:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
        connection.execute(
            "CREATE TABLE audit (request_id TEXT PRIMARY KEY, features TEXT NOT NULL, predictions TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO audit VALUES (?, ?, ?)",
            ("compatibility-request", json.dumps(request), json.dumps(predictions)),
        )

    with closing(sqlite3.connect(path, timeout=5.0)) as connection:
        row = connection.execute(
            "SELECT features, predictions FROM audit WHERE request_id = ?", ("compatibility-request",)
        ).fetchone()
    assert row is not None
    assert json.loads(row[0]) == request
    assert json.loads(row[1]) == predictions


def test_dvc_filesystem_remote(tmp_path: Path) -> None:
    """Push data to a local DVC remote and recover it without a cloud plugin."""
    repository = tmp_path / "repository"
    repository.mkdir()
    remote = tmp_path / "remote"

    def dvc(*arguments: str) -> None:
        subprocess.run(
            [sys.executable, "-m", "dvc", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

    dvc("init", "--no-scm")
    dataset = repository / "sample.csv"
    dataset.write_text("voice,motor_UPDRS\n0.1,12.0\n", encoding="utf-8")
    expected = dataset.read_bytes()
    dvc("add", "sample.csv")
    dvc("remote", "add", "--default", "local", str(remote))
    dvc("push")
    dataset.unlink()
    # Use a fresh cache so pull must restore from the filesystem remote.
    dvc("cache", "dir", str(tmp_path / "empty-cache"))
    dvc("pull")
    assert dataset.read_bytes() == expected


def test_fastapi_prediction_contract() -> None:
    """Exercise FastAPI, Pydantic, Starlette, and the HTTPX test client."""
    app = FastAPI()

    @app.post("/predict")
    def predict(values: list[float]) -> dict[str, list[float]]:
        return {"predictions": values}

    with TestClient(app) as client:
        response = client.post("/predict", json=[1.0, 2.0])
        assert response.status_code == 200
        assert response.json() == {"predictions": [1.0, 2.0]}
        assert client.post("/predict", json=["invalid"]).status_code == 422


@pytest.mark.parametrize("module", ["dvc", "streamlit"])
def test_command_line_entry_points(module: str) -> None:
    """Verify CLI startup without contacting external infrastructure."""
    result = subprocess.run(
        [sys.executable, "-m", module, "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.stdout.strip()
