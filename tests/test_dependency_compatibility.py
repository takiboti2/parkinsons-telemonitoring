"""Smoke-test the pinned dependency stack without external services."""

import importlib
import json
import sqlite3
import subprocess
import sys
from contextlib import closing
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
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


def test_mlflow_model_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Train, track, register, reload, and predict with the pinned ML stack."""
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'tracking.db'}")
    mlflow.set_experiment("dependency-compatibility")
    features = pd.DataFrame({"voice": [0.1, np.nan, 0.3, 0.4], "age": [60.0, 62.0, 64.0, 66.0]})
    target = np.array([10.0, 12.0, 14.0, 16.0])
    pipeline = make_pipeline(SimpleImputer(), RandomForestRegressor(n_estimators=2, random_state=42))
    pipeline.fit(features, target)
    expected = pipeline.predict(features)
    assert np.isfinite(expected).all()

    with mlflow.start_run() as run:
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("mae", float(np.mean(np.abs(expected - target))))
        mlflow.log_dict({"status": "ok"}, "compatibility.json")
        info = log_model(
            pipeline,
            name="pipeline",
            input_example=features,
            # Only trust these known types in this locally constructed model.
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
        )

    version = mlflow.register_model(info.model_uri, "compatibility-model")
    client = mlflow.MlflowClient()
    client.set_registered_model_alias("compatibility-model", "champion", version.version)
    loaded = load_model("models:/compatibility-model@champion")
    assert isinstance(loaded, Pipeline)
    np.testing.assert_allclose(loaded.predict(features), expected)
    assert client.get_run(run.info.run_id).data.params["random_state"] == "42"
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
