"""Exercise the project's real pipelines against persistent HTTP-served MLflow."""

import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import httpx
import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException
from mlflow.models import get_model_info
from mlflow.sklearn import log_model
from mlflow.tracking.fluent import ActiveModelContext
from mlflow.utils.thread_utils import ThreadLocalVariable

from parkinsons_ml.schema import CANDIDATE_FEATURES
from parkinsons_ml.training import build_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def http_tracking_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Restore fluent state and environment even if a real-service assertion fails."""
    assert mlflow.active_run() is None
    monkeypatch.chdir(tmp_path)
    # Preserve unset URI overrides as unset, rather than restoring getter defaults.
    monkeypatch.setattr("mlflow.tracking._tracking_service.utils._tracking_uri", None)
    monkeypatch.setattr("mlflow.tracking._model_registry.utils._registry_uri", None)
    monkeypatch.setattr("mlflow.tracking.fluent._active_experiment_id", None)
    monkeypatch.setattr("mlflow.tracking.fluent._last_active_run_id", ThreadLocalVariable(lambda: None))
    monkeypatch.setattr("mlflow.tracking.fluent._last_logged_model_id", ThreadLocalVariable(lambda: None))
    monkeypatch.setattr("mlflow.tracking.fluent._ACTIVE_MODEL_CONTEXT", ThreadLocalVariable(ActiveModelContext))
    # Developer credentials, experiment IDs, and storage overrides must not leak
    # into the temporary server/client, or make tests alter a development store.
    for name in list(os.environ):
        if name.startswith(("MLFLOW_", "_MLFLOW_")):
            monkeypatch.delenv(name)
    # URI/model setters may introduce previously absent environment variables.
    # Register their original absence too, so teardown removes those additions.
    for name in ("MLFLOW_TRACKING_URI", "MLFLOW_REGISTRY_URI", "MLFLOW_ACTIVE_MODEL_ID"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "3")
    try:
        yield
    finally:
        mlflow.end_run()


@contextmanager
def running_server(store: Path, port: int) -> Generator[str]:
    """Start the real launcher from elsewhere; reuse its files on restart."""
    store.mkdir(parents=True, exist_ok=True)
    uri = f"http://127.0.0.1:{port}"
    log = store / "server.log"
    environment = {name: value for name, value in os.environ.items() if not name.startswith("MLFLOW_")}
    environment["MLFLOW_DISABLE_TELEMETRY"] = "true"
    environment["PARKINSONS_MLFLOW_STORAGE_DIR"] = str(store)
    environment["PARKINSONS_MLFLOW_PORT"] = str(port)
    # The launcher must override a conflicting inherited registry store. Without
    # its explicit --registry-store-uri, registering a model creates this database.
    environment["MLFLOW_REGISTRY_STORE_URI"] = f"sqlite:///{store / 'unexpected-registry.db'}"
    with log.open("a", encoding="utf-8") as output:
        process = subprocess.Popen(
            [str(PROJECT_ROOT / "scripts/start_mlflow.sh")],
            cwd=store,
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            # Bounded readiness, not a fixed startup sleep. Server logs remain in
            # pytest's temporary directory for diagnosis when startup fails.
            deadline = time.monotonic() + 60
            with httpx.Client(timeout=1, trust_env=False) as probe:
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(f"MLflow exited during startup:\n{log.read_text()}")
                    try:
                        if probe.get(f"{uri}/health").status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.1)
                else:
                    raise TimeoutError(f"MLflow readiness timed out:\n{log.read_text()}")
            yield uri
        finally:
            # Kill the entire process group, including the CLI's server child.
            # This also runs when logging/loading an artifact raises unexpectedly.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)


def free_port() -> int:
    """Let the OS choose a localhost port instead of sharing a fixed test port."""
    with socket.socket() as connection:
        connection.bind(("127.0.0.1", 0))
        return int(connection.getsockname()[1])


@pytest.mark.parametrize("feature_set", ["baseline", "candidate"])
def test_http_artifact_model_and_registry_survive_restart(
    tmp_path: Path,
    http_tracking_state: None,
    feature_set: Literal["baseline", "candidate"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = tmp_path / "server"
    client_directory = tmp_path / "client"
    client_directory.mkdir()
    monkeypatch.chdir(client_directory)
    features = pd.DataFrame(
        {
            "age": [50.0, 60.0, np.nan, 80.0],
            "sex": [0.0, 1.0, np.nan, 0.0],
            "test_time": [-5.0, 1.0, 2.0, np.nan],
        }
    )
    if feature_set == "candidate":
        for name in CANDIDATE_FEATURES[3:]:
            features[name] = [1.0, np.nan, 3.0, 4.0]
    pipeline = build_pipeline(feature_set)
    pipeline.fit(features, [10.0, 12.0, 14.0, 16.0])
    expected = pipeline.predict(features)
    report_bytes = b'{"schema_version":"1","purpose":"http-integration"}\n'
    report = client_directory / "validation_summary.json"
    report.write_bytes(report_bytes)
    port = free_port()

    with running_server(store, port) as uri:
        mlflow.set_tracking_uri(uri)
        mlflow.set_registry_uri(uri)
        experiment_id = mlflow.create_experiment("http-models", tags={"purpose": "integration-test"})
        with mlflow.start_run(experiment_id=experiment_id, tags={"purpose": "integration-test"}) as run:
            mlflow.log_param("feature_set", feature_set)
            mlflow.log_metric("mae", float(np.mean(np.abs(expected - [10.0, 12.0, 14.0, 16.0]))))
            mlflow.log_artifact(str(report))
            info = log_model(
                pipeline,
                name="pipeline",
                input_example=features,
                code_paths=[str(PROJECT_ROOT / "src/parkinsons_ml")],
                skops_trusted_types=["numpy.dtype", "parkinsons_ml.training.FeaturePreparation"],
            )
        version = mlflow.register_model(info.model_uri, f"http-{feature_set}")
        model_uri = f"models:/http-{feature_set}/{version.version}"
        client = mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)
        stored_run = client.get_run(run.info.run_id)
        artifact_uri = stored_run.info.artifact_uri
        assert artifact_uri is not None
        assert artifact_uri.startswith("mlflow-artifacts:/")
        experiment = client.get_experiment(experiment_id)
        assert experiment.artifact_location.startswith("mlflow-artifacts:/")
        # An actual HTTP GET proves this report is served by MLflow, not read
        # directly from its filesystem by the client artifact SDK.
        artifact_path = artifact_uri.removeprefix("mlflow-artifacts:")
        with httpx.Client(trust_env=False, timeout=10) as http:
            response = http.get(f"{uri}/api/2.0/mlflow-artifacts/artifacts{artifact_path}/validation_summary.json")
            response.raise_for_status()
            assert response.content == report_bytes

    with running_server(store, port) as uri:
        client = mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)
        persisted = client.get_run(run.info.run_id)
        assert persisted.info.status == "FINISHED"
        assert persisted.data.params["feature_set"] == feature_set
        assert np.isfinite(persisted.data.metrics["mae"])
        assert persisted.data.tags["purpose"] == "integration-test"
        downloaded = client.download_artifacts(run.info.run_id, report.name, str(client_directory))
        assert Path(downloaded).read_bytes() == report_bytes
        assert info.model_id is not None
        assert client.get_logged_model(info.model_id).artifact_location.startswith("mlflow-artifacts:/")
        assert get_model_info(model_uri).signature is not None
        # Use a fresh interpreter and a different working directory to catch
        # dependence on fluent state or the server's current working directory.
        environment = dict(os.environ, MLFLOW_TRACKING_URI=uri, MLFLOW_REGISTRY_URI=uri)
        environment.pop("PYTHONPATH", None)
        payload = features.to_json(orient="split")
        assert isinstance(payload, str)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import json
import sys
import mlflow.sklearn
import pandas as pd
payload = json.loads(sys.argv[2])
frame = pd.DataFrame(payload["data"], columns=payload["columns"])
model = mlflow.sklearn.load_model(sys.argv[1])
print(json.dumps(model.predict(frame).tolist()))
""",
                model_uri,
                payload,
            ],
            cwd=client_directory,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        np.testing.assert_allclose(json.loads(result.stdout), expected)
    assert (store / "mlflow.db").is_file()
    assert (store / "artifacts").is_dir()
    assert not (store / "unexpected-registry.db").exists()
    assert not (client_directory / "mlruns").exists()
    assert not (client_directory / "mlflow.db").exists()


def test_http_artifact_failure_marks_run_failed_and_outage_does_not_fallback(
    tmp_path: Path,
    http_tracking_state: None,
) -> None:
    with running_server(tmp_path / "server", free_port()) as uri:
        mlflow.set_tracking_uri(uri)
        mlflow.set_registry_uri(uri)
        experiment_id = mlflow.create_experiment("http-failures")
        client = mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)
        failed_run_id: str | None = None
        with pytest.raises(MlflowException):
            with mlflow.start_run(experiment_id=experiment_id) as run:
                failed_run_id = run.info.run_id
                client.download_artifacts(run.info.run_id, "missing-report.json", str(tmp_path))
        assert failed_run_id is not None
        assert client.get_run(failed_run_id).info.status == "FAILED"
    # With the server stopped, the same explicitly HTTP-configured client must
    # raise rather than silently creating a separate local tracking database.
    with pytest.raises(MlflowException):
        client.create_run(experiment_id)
    assert not (tmp_path / "mlflow.db").exists()
    assert not (tmp_path / "mlruns").exists()


@pytest.mark.parametrize(
    "setting",
    [
        "MLFLOW_READ_REPLICA_BACKEND_STORE_URI",
        "MLFLOW_ARTIFACTS_ONLY",
        "MLFLOW_ARTIFACTS_PRESIGNED_ONLY",
    ],
)
def test_launcher_rejects_conflicting_server_settings_before_writing_storage(
    tmp_path: Path,
    http_tracking_state: None,
    setting: str,
) -> None:
    store = tmp_path / "rejected-store"
    # Treat the value as sensitive; failure output must identify only the setting.
    value = "sqlite:///sensitive-replica.db" if setting.endswith("STORE_URI") else "true"
    environment = dict(os.environ, PARKINSONS_MLFLOW_STORAGE_DIR=str(store))
    environment[setting] = value
    result = subprocess.run(
        [str(PROJECT_ROOT / "scripts/start_mlflow.sh")],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert f"Unset {setting}:" in result.stderr
    assert value not in result.stderr
    assert not store.exists()
