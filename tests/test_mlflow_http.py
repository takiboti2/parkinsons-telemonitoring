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
from dataclasses import replace
from pathlib import Path

import httpx
import mlflow
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from mlflow.exceptions import MlflowException
from mlflow.models import get_model_info
from mlflow.sklearn import load_model, log_model
from mlflow.tracking.fluent import ActiveModelContext
from mlflow.utils.thread_utils import ThreadLocalVariable
from sklearn.linear_model import LinearRegression

from parkinsons_ml import api, experimentation, inference, promotion, tracking, training
from parkinsons_ml.schema import BASELINE_FEATURES, CANDIDATE_FEATURES
from parkinsons_ml.training import FeaturePreparation, build_pipeline

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


def test_training_comparison_http(
    tmp_path: Path,
    prepared_directory: Path,
    http_tracking_state: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = training.training_config(
        {"experiment_name": "comparison", "extra_feature_sets": {"voice": ["ppe", "sex"]}}
    )
    store, port = tmp_path / "server", free_port()
    original_read = Path.read_bytes

    def read(path: Path) -> bytes:
        assert path.name != "test.csv"
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    with running_server(store, port) as uri:
        previous_uris = mlflow.get_tracking_uri(), mlflow.get_registry_uri()
        previous_environment = {key: os.environ.get(key) for key in ("MLFLOW_TRACKING_URI", "MLFLOW_REGISTRY_URI")}
        result = training.run_experiment(config, prepared_directory, uri)
        assert (mlflow.get_tracking_uri(), mlflow.get_registry_uri()) == previous_uris
        assert {key: os.environ.get(key) for key in previous_environment} == previous_environment
        client = mlflow.MlflowClient(tracking_uri=uri)
        # Exercise the preserved module command in a fresh process. This catches
        # import cycles and accidental serialization of __main__.FeaturePreparation
        # that direct calls within an already-imported test process could miss.
        parameters_path = tmp_path / "cli-params.yaml"
        parameters_path.write_text("train:\n  experiment_name: cli-comparison\n")
        command = subprocess.run(
            [
                sys.executable,
                "-m",
                "parkinsons_ml.training",
                "--params",
                str(parameters_path),
                "--data",
                str(prepared_directory),
            ],
            env=dict(os.environ, MLFLOW_TRACKING_URI=uri, MLFLOW_REGISTRY_URI=uri),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert command.returncode == 0, command.stderr
        cli_result = json.loads(command.stdout)
        assert client.get_run(cli_result["run_id"]).info.status == "FINISHED"
        assert {item["name"] for item in cli_result["configurations"]} == {"baseline", "candidate"}
        client.create_experiment("unsafe-artifacts", artifact_location=str(tmp_path / "unsafe"))
        with pytest.raises(ValueError, match="HTTP-served"):
            tracking.run_experiment(
                experimentation.training_config({"experiment_name": "unsafe-artifacts"}), prepared_directory, uri
            )
        assert client.search_registered_models() == []
        parent = client.get_run(result.run_id)
        assert parent.info.status == "FINISHED"
        assert parent.data.tags["winner"] == result.winner.name
        assert parent.data.tags["dataset_id"] == result.dataset_id
        assert len(result.configurations) == 3
        for child in result.configurations:
            run = client.get_run(child.run_id)
            assert run.info.status == "FINISHED"
            assert run.data.tags["mlflow.parentRunId"] == result.run_id
            assert json.loads(run.data.params["features"]) == list(child.features)
            assert run.data.metrics["oof_mae"] == child.metrics.mae
            assert len(client.get_metric_history(child.run_id, "validation_mae")) == 3
            parameters = json.loads(
                Path(client.download_artifacts(child.run_id, "pipeline_parameters.json", str(tmp_path))).read_text()
            )
            assert parameters["estimator"]["class"] == "LinearRegression"
            assert parameters["estimator"]["parameters"] == LinearRegression().get_params(deep=False)
            assert parameters["preprocessing"] == {
                "features": list(child.features),
                "continuous_imputation": "median",
                "sex_imputation": "most_frequent",
                "test_time": "zero_clip",
            }
            reference = client.download_artifacts(child.run_id, "monitoring/reference.csv", str(tmp_path))
            frame = pd.read_csv(reference)
            assert tuple(frame.columns) == child.features
            identity = json.loads(
                Path(client.download_artifacts(child.run_id, "monitoring/identity.json", str(tmp_path))).read_text()
            )
            assert experimentation.digest(Path(reference).read_bytes()) == identity["sha256"]
            predictions = pd.read_csv(client.download_artifacts(child.run_id, "predictions/oof.csv", str(tmp_path)))
            assert set(predictions.columns) == {"measurement", "fold", "actual", "prediction"}
            assert len(predictions) == 18
        assert mlflow.active_run() is None
    with running_server(store, port) as uri:
        mlflow.set_tracking_uri(uri)
        mlflow.set_registry_uri(uri)
        model = load_model(result.winner.model_uri)
        assert model is not None
        frame = experimentation.load_training_snapshot(prepared_directory).train
        assert type(model.named_steps["features"]) is FeaturePreparation
        assert np.isfinite(model.predict(frame)).all()
        assert mlflow.MlflowClient().get_run(result.run_id).info.status == "FINISHED"


def test_training_failures_do_not_select_partial_winner(
    tmp_path: Path,
    prepared_directory: Path,
    http_tracking_state: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = experimentation.training_config({"experiment_name": "failures"})
    with running_server(tmp_path / "server", free_port()) as uri:
        client = mlflow.MlflowClient(tracking_uri=uri)

        def fail(*args: object, **kwargs: object) -> None:
            raise RuntimeError("required operation failed")

        def invalid_reload(uri: str) -> None:
            return None

        for operation in ("evaluate", "log_model", "load_model", "log_metrics", "log_text", "fit", "invalid_reload"):
            with monkeypatch.context() as patch:
                if operation == "invalid_reload":
                    patch.setattr(tracking, "load_model", invalid_reload)
                elif operation == "fit":
                    patch.setattr(LinearRegression, "fit", fail)
                else:
                    target = mlflow if operation in {"log_metrics", "log_text"} else tracking
                    patch.setattr(target, operation, fail)
                with pytest.raises((RuntimeError, ValueError)):
                    tracking.run_experiment(config, prepared_directory, uri)
            assert mlflow.active_run() is None
        experiment = client.get_experiment_by_name("failures")
        assert experiment is not None
        runs = client.search_runs([experiment.experiment_id])
        assert all(run.info.status == "FAILED" and "winner" not in run.data.tags for run in runs)
        evaluate = experimentation.evaluate

        def fail_second(
            snapshot: experimentation.TrainingSnapshot, features: tuple[str, ...]
        ) -> experimentation.Evaluation:
            if len(features) > 3:
                raise RuntimeError("second configuration failed")
            return evaluate(snapshot, features)

        with monkeypatch.context() as patch:
            patch.setattr(tracking, "evaluate", fail_second)
            with pytest.raises(RuntimeError, match="second configuration"):
                tracking.run_experiment(config, prepared_directory, uri)
        runs = client.search_runs([experiment.experiment_id])
        assert any(run.info.status == "FINISHED" for run in runs)
        assert all("winner" not in run.data.tags for run in runs)
        # A validation error is recorded in a failed parent, before children exist.
        (prepared_directory / "train.csv").write_text("broken")
        with pytest.raises(ValueError, match="checksum"):
            tracking.run_experiment(config, prepared_directory, uri)
        # Caller-owned runs must remain active and unmodified.
        mlflow.set_tracking_uri(uri)
        with mlflow.start_run(experiment_id=experiment.experiment_id) as caller:
            with pytest.raises(ValueError, match="unrelated"):
                tracking.run_experiment(config, prepared_directory, uri)
            assert mlflow.active_run() is caller


def test_champion_policy_rollback_and_recovery_http(
    tmp_path: Path,
    prepared_directory: Path,
    http_tracking_state: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, port = tmp_path / "server", free_port()
    name = "promotion-test"
    lock = tmp_path / "promotion.lock"
    with running_server(store, port) as uri:
        mlflow.set_tracking_uri(uri)
        mlflow.set_registry_uri(uri)
        client = mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)
        config = experimentation.training_config({"experiment_name": "promotion-comparisons"})
        first = tracking.run_experiment(config, prepared_directory, uri)
        second = tracking.run_experiment(config, prepared_directory, uri)
        assert first.promotion is not None and first.promotion.decision == "disabled"
        # Set controlled authoritative validation outcomes on real completed runs;
        # the returned ComparisonResult still holds its original scores. Promotion
        # must read MLflow rather than trusting those stale caller values.
        client.log_metric(first.winner.run_id, "oof_mae", 2.0)
        client.log_metric(second.winner.run_id, "oof_mae", 1.0)

        def promote(result: experimentation.ComparisonResult, model: str = name) -> experimentation.PromotionResult:
            return promotion.promote_winner(client, result, experimentation.PromotionConfig(True, model), lock)

        bootstrap = promote(first)
        assert bootstrap.decision == "promoted" and bootstrap.reason == "bootstrap"
        old = bootstrap.candidate_version
        assert old is not None and bootstrap.candidate_mae == 2.0
        assert promotion.champion_version(client, name) == old
        assert promote(first).decision == "retained"  # exact tie
        assert len(client.search_model_versions(f"name = '{name}'")) == 1
        client.log_metric(second.winner.run_id, "oof_mae", 3.0)
        retained = promote(second)
        assert retained.decision == "retained", retained
        assert len(client.search_model_versions(f"name = '{name}'")) == 1
        client.log_metric(second.winner.run_id, "oof_mae", 1.0)
        better = promote(second)
        assert better.decision == "promoted" and better.previous_mae == 2.0 and better.candidate_mae == 1.0
        assert promotion.champion_version(client, name) == better.candidate_version
        # Return to the first immutable version to exercise failure paths against
        # an independently completed, eligible candidate without retraining it.
        client.set_registered_model_alias(name, "champion", old)

        previous_identity = client.get_run(second.winner.run_id).data.tags["dataset_id"]
        client.set_tag(second.winner.run_id, "dataset_id", "different")
        client.set_tag(second.run_id, "dataset_id", "different")
        incomparable = promote(second)
        assert incomparable.decision == "rejected" and "incomparable" in incomparable.reason
        client.set_tag(second.winner.run_id, "dataset_id", previous_identity)
        client.set_tag(second.run_id, "dataset_id", previous_identity)
        assert promotion.champion_version(client, name) == old

        client.delete_model_version_tag(name, old, "promotion_policy")
        assert promote(second).decision == "rejected"
        client.set_model_version_tag(name, old, "promotion_policy", promotion.POLICY)
        client.set_model_version_tag(name, old, "oof_mae", "999.0")
        assert promote(second).decision == "rejected"
        client.set_model_version_tag(name, old, "oof_mae", "2.0")
        assert promote(replace(second, run_id=first.run_id)).decision == "rejected"
        assert promote(replace(second, configurations=(*second.configurations, first.winner))).decision == "rejected"
        for model_uri in ("runs:/unapproved", first.winner.model_uri):
            wrong = replace(second, winner=replace(second.winner, model_uri=model_uri))
            assert promote(wrong).decision == "rejected"

        # A finished child cannot deploy from an unfinished/failed comparison.
        client.set_terminated(second.run_id, "FAILED")
        assert promote(second).decision == "rejected"
        client.set_terminated(second.run_id, "FINISHED")
        with monkeypatch.context() as patch:

            def invalid_model(uri: str) -> None:
                return None

            patch.setattr(promotion, "load_model", invalid_model)
            assert promote(second).decision == "rejected"
        assert promotion.champion_version(client, name) == old

        original_set = client.set_registered_model_alias

        def applied_then_timed_out() -> object:
            # Return a callable with per-operation state: fail only the update,
            # allowing the subsequent automatic rollback request to succeed.
            applied = False

            def update(model: str, alias: str, version: str) -> None:
                nonlocal applied
                original_set(model, alias, version)
                if not applied:
                    applied = True
                    raise TimeoutError("response lost after applying alias")

            return update

        with monkeypatch.context() as patch:
            patch.setattr(client, "set_registered_model_alias", applied_then_timed_out())
            failed = promote(second)
        assert failed.decision == "failed" and failed.confirmed
        assert promotion.champion_version(client, name) == old
        assert promotion.PENDING not in client.get_registered_model(name).tags

        with monkeypatch.context() as patch:
            patch.setattr(client, "set_registered_model_alias", applied_then_timed_out())
            failed_bootstrap = promote(second, "bootstrap-failure")
        assert failed_bootstrap.decision == "failed" and failed_bootstrap.confirmed
        assert promotion.champion_version(client, "bootstrap-failure") is None

        # Required decision logging failures also roll back rather than reporting
        # an unaudited promotion as successful.
        original_log = client.log_dict
        logged_failure = False

        def fail_decision_once(run_id: str, dictionary: dict[str, object], artifact_file: str) -> None:
            nonlocal logged_failure
            if artifact_file == "decision.json" and not logged_failure:
                logged_failure = True
                raise ConnectionError("audit response lost")
            original_log(run_id, dictionary, artifact_file)

        with monkeypatch.context() as patch:
            patch.setattr(client, "log_dict", fail_decision_once)
            assert promote(second).decision == "failed"
        assert promotion.champion_version(client, name) == old

        # Failed rollback leaves a durable marker. A later invocation first
        # restores the old alias, then makes its own fresh eligibility decision.
        def unreachable(*args: object, **kwargs: object) -> None:
            raise ConnectionError("unreachable")

        with monkeypatch.context() as patch:
            patch.setattr(client, "set_registered_model_alias", applied_then_timed_out())
            patch.setattr(promotion, "restore_alias", unreachable)
            uncertain = promote(second)
        assert uncertain.decision == "failed" and not uncertain.confirmed
        assert client.get_registered_model(name).tags[promotion.PENDING] == uncertain.run_id
        assert promotion.champion_version(client, name) == uncertain.candidate_version

        # A separate API reader must not serve the transient new alias or repair
        # it itself. The durable marker protects serving after the writer dies too.
        api_params = tmp_path / "pending-api.yaml"
        api_params.write_text(
            f"train:\n  experiment_name: promotion-comparisons\n  promotion:\n" f"    registered_model_name: {name}\n"
        )
        with monkeypatch.context() as patch:
            patch.setattr(api, "PARAMS_PATH", api_params)
            patch.setattr(tracking, "PROMOTION_LOCK", lock)
            key = "pending-promotion-api-test-key-only"
            application = api.create_app(api.APISettings(tracking_uri=uri, api_key=SecretStr(key)))
            with TestClient(application) as http:
                for _ in range(2):
                    response = http.post(
                        "/predict",
                        json={"records": [{"age": 60, "sex": 0, "test_time": 0}]},
                        headers={"X-API-Key": key},
                    )
                    assert response.status_code == 503
                assert http.get("/health/ready").status_code == 503
                assert client.get_registered_model(name).tags[promotion.PENDING] == uncertain.run_id

        client.log_metric(second.winner.run_id, "oof_mae", 3.0)
        assert promote(second).decision == "retained"
        assert promotion.champion_version(client, name) == old
        assert promotion.PENDING not in client.get_registered_model(name).tags
        assert uncertain.run_id is not None
        assert client.get_run(uncertain.run_id).data.tags["state"] == "complete"

        # A committed success with failed marker cleanup is retained on recovery,
        # not incorrectly rolled back as if the process had died before commit.
        client.log_metric(second.winner.run_id, "oof_mae", 1.0)
        with monkeypatch.context() as patch:
            patch.setattr(client, "delete_registered_model_tag", unreachable)
            committed = promote(second)
        assert committed.decision == "promoted"
        assert promotion.PENDING in client.get_registered_model(name).tags
        assert promote(second).decision == "retained"
        assert promotion.champion_version(client, name) == committed.candidate_version
        assert promotion.PENDING not in client.get_registered_model(name).tags
    with running_server(store, port) as uri:
        client = mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)
        assert promotion.champion_version(client, name) == committed.candidate_version
        assert committed.run_id is not None
        assert client.get_run(committed.run_id).info.status == "FINISHED"


def test_automatic_promotion_only_after_finished_training_http(
    tmp_path: Path,
    prepared_directory: Path,
    http_tracking_state: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tracking, "PROMOTION_LOCK", tmp_path / "promotion.lock")
    config = experimentation.training_config(
        {"experiment_name": "automatic", "promotion": {"enabled": True, "registered_model_name": "automatic-model"}}
    )
    with running_server(tmp_path / "server", free_port()) as uri:
        result = tracking.run_experiment(config, prepared_directory, uri)
        assert result.promotion is not None and result.promotion.decision == "promoted"
        client = mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)
        assert client.get_run(result.run_id).info.status == "FINISHED"
        assert promotion.champion_version(client, "automatic-model") == result.promotion.candidate_version

        # Promotion failure remains separate from successful training.
        def busy(*args: object, **kwargs: object) -> experimentation.PromotionResult:
            return experimentation.PromotionResult("failed", "promotion_busy", "automatic-model")

        with monkeypatch.context() as patch:
            patch.setattr(tracking, "promote_winner", busy)
            completed = tracking.run_experiment(config, prepared_directory, uri)
        assert completed.promotion is not None and completed.promotion.decision == "failed"
        assert client.get_run(completed.run_id).info.status == "FINISHED"


def test_api_training_and_logged_immutable_inference_http(
    tmp_path: Path,
    prepared_directory: Path,
    http_tracking_state: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters = tmp_path / "api-params.yaml"
    parameters.write_text(
        "train:\n  experiment_name: api-integration\n  promotion:\n"
        "    enabled: true\n    registered_model_name: api-model\n"
    )
    monkeypatch.setattr(api, "PARAMS_PATH", parameters)
    monkeypatch.setattr(api, "DATA_DIRECTORY", prepared_directory)
    monkeypatch.setattr(tracking, "PROMOTION_LOCK", tmp_path / "api-promotion.lock")
    original_read = Path.read_bytes

    def read(path: Path) -> bytes:
        assert path.name != "test.csv"
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    key = "isolated-http-api-key-not-a-real-secret"
    headers = {"X-API-Key": key}
    frame = pd.read_csv(prepared_directory / "train.csv").loc[:2, list(CANDIDATE_FEATURES)]
    records = [{name: float(row[name]) for name in CANDIDATE_FEATURES} for _, row in frame.iterrows()]
    records[0]["test_time"] = -3.0
    downloads: list[Path] = []
    original_download = inference.download_artifacts

    def download(*, artifact_uri: str, dst_path: str) -> str:
        downloads.append(Path(dst_path))
        return original_download(artifact_uri=artifact_uri, dst_path=dst_path)

    monkeypatch.setattr(inference, "download_artifacts", download)
    with running_server(tmp_path / "server", free_port()) as uri:
        application = api.create_app(api.APISettings(tracking_uri=uri, api_key=SecretStr(key)))
        client = mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)
        with TestClient(application) as http:
            assert http.get("/health/live").status_code == 200
            assert http.get("/health/ready").status_code == 503
            assert http.post("/predict", json={"records": records}, headers=headers).status_code == 503
            trained = http.post("/train", json={}, headers=headers)
            assert trained.status_code == 200, trained.text
            comparison = trained.json()
            assert comparison["promotion"]["decision"] == "promoted"
            assert http.get("/health/ready").status_code == 200
            previous_path = list(sys.path)
            response = http.post("/predict", json={"records": records}, headers=headers)
            assert response.status_code == 200, response.text
            assert sys.path == previous_path and len(downloads) == 1
            assert not downloads[-1].exists()
            result = response.json()
            first_version = result["model_version"]
            run = client.get_run(result["run_id"])
            assert run.info.status == "FINISHED" and run.data.tags["purpose"] == "inference"
            assert run.data.tags["request_id"] == result["request_id"]
            assert run.data.tags["model_version"] == first_version and run.data.metrics["batch_size"] == 3
            with tracking.tracking_context(uri):
                model = load_model(f"models:/api-model/{first_version}")
                assert model is not None
                expected = model.predict(pd.DataFrame(records))
            np.testing.assert_allclose(result["predictions"], expected)
            inputs = pd.read_csv(client.download_artifacts(result["run_id"], "inputs.csv", str(tmp_path)))
            assert tuple(inputs.columns) == model.named_steps["features"].features
            assert inputs.loc[0, "test_time"] == 0.0
            predictions = pd.read_csv(client.download_artifacts(result["run_id"], "predictions.csv", str(tmp_path)))
            assert predictions["row"].tolist() == [0, 1, 2]
            np.testing.assert_allclose(predictions["prediction"], result["predictions"])

            # Switch to the already-evaluated other configuration. No application
            # restart or cache invalidation should be needed for the next request.
            other = next(
                item for item in comparison["configurations"] if item["run_id"] != comparison["winner"]["run_id"]
            )
            logged = client.get_logged_model(other["model_uri"].removeprefix("models:/"))
            origin = client.get_run(other["run_id"])
            second_version = client.create_model_version(
                "api-model",
                logged.artifact_location,
                run_id=other["run_id"],
                model_id=logged.model_id,
                tags={name: origin.data.tags[name] for name in promotion.IDENTITIES},
            )
            client.set_registered_model_alias("api-model", "champion", second_version.version)
            with promotion.promotion_lock(tracking.PROMOTION_LOCK):
                blocked = http.post("/predict", json={"records": records}, headers=headers)
                assert blocked.status_code == 503
            previous_path = list(sys.path)
            response = http.post("/predict", json={"records": records}, headers=headers)
            assert response.status_code == 200 and response.json()["model_version"] == second_version.version
            assert response.json()["model_version"] != first_version
            assert sys.path == previous_path and not downloads[-1].exists()
            assert client.get_run(result["run_id"]).data.tags["model_version"] == first_version

            # A failed artifact upload cannot produce an optimistic success or a
            # FINISHED inference record. Failure also releases the API guard.
            original_log = mlflow.MlflowClient.log_text

            def fail_upload(self: mlflow.MlflowClient, run_id: str, text: str, artifact_file: str) -> None:
                if artifact_file == "predictions.csv":
                    raise ConnectionError("private dependency details")
                original_log(self, run_id, text, artifact_file)

            with monkeypatch.context() as patch:
                patch.setattr(mlflow.MlflowClient, "log_text", fail_upload)
                failed = http.post("/predict", json={"records": records}, headers=headers)
            assert failed.status_code == 503 and "private" not in failed.text
            runs = client.search_runs([run.info.experiment_id], "tags.purpose = 'inference'")
            assert any(item.info.status == "FAILED" for item in runs)
            assert http.post("/predict", json={"records": records}, headers=headers).status_code == 200

            def bad_artifact(model_uri: str) -> None:
                raise ValueError("private artifact error")

            with monkeypatch.context() as patch:
                patch.setattr(inference, "load_model", bad_artifact)
                failed = http.post("/predict", json={"records": records}, headers=headers)
                assert failed.status_code == 500 and "private" not in failed.text
                assert not downloads[-1].exists()
                # Readiness intentionally checks registry metadata, not downloads.
                assert http.get("/health/ready").status_code == 200


@pytest.mark.parametrize(
    "feature_set, selected",
    [
        ("baseline", BASELINE_FEATURES),
        ("candidate", CANDIDATE_FEATURES),
        ("voice_subset", ("ppe", "jitter")),
    ],
)
def test_http_artifact_model_and_registry_survive_restart(
    tmp_path: Path,
    http_tracking_state: None,
    feature_set: str,
    selected: tuple[str, ...],
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
    for name in CANDIDATE_FEATURES[3:]:
        features[name] = [1.0, np.nan, 3.0, 4.0]
    pipeline = build_pipeline(selected)
    selection_step = pipeline.named_steps["features"]
    assert isinstance(selection_step, FeaturePreparation)
    canonical = selection_step.features
    assert canonical == tuple(sorted(selected))
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
            mlflow.log_param("features", json.dumps(list(canonical)))
            mlflow.log_metric("mae", float(np.mean(np.abs(expected - [10.0, 12.0, 14.0, 16.0]))))
            mlflow.log_artifact(str(report))
            info = log_model(
                pipeline,
                name="pipeline",
                input_example=features.loc[:, list(canonical)],
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
        assert json.loads(persisted.data.params["features"]) == list(canonical)
        assert np.isfinite(persisted.data.metrics["mae"])
        assert persisted.data.tags["purpose"] == "integration-test"
        downloaded = client.download_artifacts(run.info.run_id, report.name, str(client_directory))
        assert Path(downloaded).read_bytes() == report_bytes
        assert info.model_id is not None
        assert client.get_logged_model(info.model_id).artifact_location.startswith("mlflow-artifacts:/")
        signature = get_model_info(model_uri).signature
        assert signature is not None
        assert signature.inputs is not None
        assert signature.inputs.input_names() == list(canonical)
        # Use a fresh interpreter and a different working directory to catch
        # dependence on fluent state or the server's current working directory.
        environment = dict(os.environ, MLFLOW_TRACKING_URI=uri, MLFLOW_REGISTRY_URI=uri)
        environment.pop("PYTHONPATH", None)
        payload = features.loc[:, list(canonical)].to_json(orient="split")
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
