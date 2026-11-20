"""Immutable-model inference, input representation, and mandatory logging failures."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
import sys
from typing import cast
from unittest.mock import MagicMock

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, RESOURCE_DOES_NOT_EXIST
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

from parkinsons_ml import inference, promotion, tracking
from parkinsons_ml.schema import POLICY_VERSION, SCHEMA_VERSION
from parkinsons_ml.training import build_pipeline


@pytest.fixture
def version() -> ModelVersion:
    result = MagicMock(spec=ModelVersion)
    result.name, result.version, result.run_id, result.status = "model", "7", "training", "READY"
    result.tags = {
        "dataset_id": "dataset",
        "split_id": "split",
        "schema_version": SCHEMA_VERSION,
        "policy_version": POLICY_VERSION,
        "evaluation_id": "evaluation",
    }
    return cast(ModelVersion, result)


@pytest.fixture
def pipeline() -> Pipeline:
    model = build_pipeline(("age", "sex", "test_time"))
    model.fit(
        pd.DataFrame({"age": [50.0, 60.0, 70.0], "sex": [0.0, 1.0, 0.0], "test_time": [0.0, 1.0, 2.0]}),
        [10.0, 12.0, 14.0],
    )
    return model


def test_resolves_only_one_alias_version(
    version: ModelVersion, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = MagicMock(spec=MlflowClient)
    client.get_registered_model.return_value.aliases = {"champion": "7"}
    client.get_registered_model.return_value.tags = {}
    client.get_model_version.return_value = version
    monkeypatch.setattr(tracking, "PROMOTION_LOCK", tmp_path / "promotion.lock")
    with promotion.promotion_lock(tracking.PROMOTION_LOCK, shared=True):
        assert inference.resolve_champion(client, "model") is version
    client.get_registered_model.assert_called_once_with("model")
    client.get_model_version.assert_called_once_with("model", "7")
    client.get_registered_model.return_value.aliases = {}
    with pytest.raises(inference.UnavailableError, match="No champion"):
        inference.resolve_champion(client, "model")
    client.get_registered_model.side_effect = MlflowException("missing", RESOURCE_DOES_NOT_EXIST)
    with pytest.raises(inference.UnavailableError, match="No champion"):
        inference.resolve_champion(client, "model")
    client.get_registered_model.side_effect = ConnectionError("private server details")
    with pytest.raises(inference.UnavailableError, match="registry is unavailable"):
        inference.resolve_champion(client, "model")


def test_not_ready_version(version: ModelVersion, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock(spec=MlflowClient)
    cast(MagicMock, version).status = "PENDING_REGISTRATION"
    client.get_model_version.return_value = version
    client.get_registered_model.return_value.aliases = {"champion": "7"}
    client.get_registered_model.return_value.tags = {}
    monkeypatch.setattr(tracking, "PROMOTION_LOCK", tmp_path / "promotion.lock")
    with pytest.raises(inference.UnavailableError, match="not ready"):
        inference.resolve_champion(client, "model")


def test_pending_and_active_promotions_are_not_served(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock(spec=MlflowClient)
    lock = tmp_path / "promotion.lock"
    monkeypatch.setattr(tracking, "PROMOTION_LOCK", lock)
    client.get_registered_model.return_value.aliases = {"champion": "2"}
    client.get_registered_model.return_value.tags = {promotion.PENDING: "unconfirmed-operation"}
    with pytest.raises(inference.UnavailableError, match="not yet confirmed"):
        inference.resolve_champion(client, "model")
    client.get_model_version.assert_not_called()
    with promotion.promotion_lock(lock):
        with pytest.raises(inference.UnavailableError, match="in progress"):
            inference.resolve_champion(client, "model")


@pytest.mark.parametrize(
    "problem",
    ["none", "provenance", "policy", "not_pipeline", "selector", "estimator", "features", "signature", "unsorted"],
)
def test_loads_only_approved_immutable_pipeline(
    version: ModelVersion, pipeline: Pipeline, problem: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = MagicMock()
    info.signature.inputs.input_names.return_value = ["age", "sex", "test_time"]
    returned: object = pipeline
    if problem == "provenance":
        version.tags.clear()
    elif problem == "policy":
        version.tags["policy_version"] = "unsupported"
    elif problem == "not_pipeline":
        returned = None
    elif problem == "selector":
        pipeline.steps[0] = ("features", object())
    elif problem == "estimator":
        pipeline.steps[-1] = ("linearregression", object())
    elif problem == "features":
        pipeline.named_steps["features"].features = ("subject",)
    elif problem == "signature":
        info.signature.inputs.input_names.return_value = ["age"]
    elif problem == "unsorted":
        pipeline.named_steps["features"].features = ("sex", "age", "test_time")

    def download(*, artifact_uri: str, dst_path: str) -> str:
        assert artifact_uri == "models:/model/7"
        return dst_path

    def load(path: str) -> object:
        assert Path(path).is_dir()
        return returned

    def signature(path: str) -> object:
        assert Path(path).name == "MLmodel"
        return info

    monkeypatch.setattr(inference, "download_artifacts", download)
    monkeypatch.setattr(inference, "load_model", load)
    monkeypatch.setattr(inference.Model, "load", signature)
    if problem == "none":
        with inference.load_pipeline(version) as (loaded, features):
            assert loaded is pipeline and features == ("age", "sex", "test_time")
    else:
        with pytest.raises(inference.ModelError):
            with inference.load_pipeline(version):
                pytest.fail("An invalid model must not be exposed to prediction")


@pytest.mark.parametrize(
    "stage, error, expected",
    [
        ("download", MlflowException("missing", RESOURCE_DOES_NOT_EXIST), inference.ModelError),
        ("download", MlflowException("offline", INTERNAL_ERROR), inference.UnavailableError),
        ("download", ConnectionError("offline"), inference.UnavailableError),
        ("load", ValueError("bad artifact"), inference.ModelError),
    ],
)
def test_model_loading_error_classification(
    version: ModelVersion, stage: str, error: Exception, expected: type[Exception], monkeypatch: pytest.MonkeyPatch
) -> None:
    def download(*, artifact_uri: str, dst_path: str) -> str:
        if stage == "download":
            raise error
        return dst_path

    def fail(path: str) -> None:
        raise error

    monkeypatch.setattr(inference, "download_artifacts", download)
    monkeypatch.setattr(inference, "load_model", fail)
    with pytest.raises(expected):
        with inference.load_pipeline(version):
            pytest.fail("A failed load must not expose a model")


@pytest.mark.parametrize("failure", ["none", "load", "predict"])
def test_download_and_import_paths_have_request_lifetimes(
    version: ModelVersion, pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    downloads: list[Path] = []
    before = list(sys.path)
    info = MagicMock()
    info.signature.inputs.input_names.return_value = ["age", "sex", "test_time"]

    def download(*, artifact_uri: str, dst_path: str) -> str:
        local = Path(dst_path) / "model"
        local.mkdir()
        downloads.append(local)
        return str(local)

    def load(path: str) -> Pipeline:
        sys.path.insert(0, str(Path(path) / "code"))
        if failure == "load":
            raise ValueError("bad local artifact")
        return pipeline

    def metadata(path: str) -> object:
        return info

    monkeypatch.setattr(inference, "download_artifacts", download)
    monkeypatch.setattr(inference, "load_model", load)
    monkeypatch.setattr(inference.Model, "load", metadata)
    for _ in range(3):
        if failure == "load":
            with pytest.raises(inference.ModelError):
                with inference.load_pipeline(version):
                    pytest.fail("Load should fail")
        elif failure == "predict":
            with pytest.raises(RuntimeError):
                with inference.load_pipeline(version):
                    assert downloads[-1].is_dir()
                    raise RuntimeError("Prediction aborted")
        else:
            with inference.load_pipeline(version) as (loaded, _):
                assert loaded is pipeline and downloads[-1].is_dir()
                assert sys.path != before
        assert sys.path == before
        assert not downloads[-1].parent.exists()


@pytest.mark.parametrize(
    "record",
    [
        {"age": True},
        {"age": "60"},
        {"age": np.inf},
        {"age": float("nan")},
        {"age": 10**400},
        {"age": 2.5},
        {"age": 60, "sex": 2},
        {"age": 60, "motor_updrs": 20},
        {"age": 60, "subject": 1},
        {},
    ],
)
def test_invalid_predictors(record: dict[str, object]) -> None:
    with pytest.raises(inference.InputError):
        inference.prepare_inputs(cast(list[dict[str, float | int | None]], [record]), ("age",))


def test_pre_imputation_inputs_preserve_nulls_clip_time_and_drop_unused() -> None:
    records: list[dict[str, float | int | None]] = [
        {"age": None, "test_time": -4, "sex": 0},
        {"age": 70, "test_time": 3, "sex": 1},
    ]
    frame = inference.prepare_inputs(records, ("age", "test_time"))
    assert tuple(frame.columns) == ("age", "test_time")
    assert pd.isna(frame.iloc[0, 0]) and frame["test_time"].tolist() == [0.0, 3.0]
    with pytest.raises(inference.InputError, match="empty"):
        inference.prepare_inputs([], ("age",))
    with pytest.raises(inference.InputError, match="Every|Each"):
        inference.prepare_inputs([{"age": 60}, {"age": 70, "test_time": 3}], ("age", "test_time"))


@pytest.fixture
def inference_client(version: ModelVersion, pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    client = MagicMock(spec=MlflowClient)
    client.get_experiment.return_value.artifact_location = "mlflow-artifacts:/test"
    client.create_run.return_value.info.run_id = "inference-run"

    @contextmanager
    def connection(uri: str) -> Generator[MlflowClient]:
        yield client

    monkeypatch.setattr(inference, "tracking_context", connection)
    monkeypatch.setattr(inference, "resolve_champion", MagicMock(return_value=version))

    @contextmanager
    def loaded(registered: ModelVersion) -> Generator[tuple[Pipeline, tuple[str, ...]]]:
        yield pipeline, ("age", "sex", "test_time")

    monkeypatch.setattr(inference, "load_pipeline", loaded)
    return client


def test_successful_inference_finishes_required_logging(inference_client: MagicMock) -> None:
    result = inference.predict_batch([{"age": 60, "sex": 1, "test_time": -2}], "http://localhost", "model")
    assert result.run_id == "inference-run" and result.model_version == "7"
    assert len(result.predictions) == 1
    inference_client.set_terminated.assert_called_with("inference-run", "FINISHED")
    assert inference_client.log_text.call_count == 2


@pytest.mark.parametrize(
    "operation", ["get_run", "create_run", "log_text", "log_param", "log_metric", "set_terminated"]
)
def test_inference_dependency_failures_are_unavailable(inference_client: MagicMock, operation: str) -> None:
    getattr(inference_client, operation).side_effect = MlflowException("private dependency detail", INTERNAL_ERROR)
    with pytest.raises(inference.UnavailableError):
        inference.predict_batch([{"age": 60, "sex": 1, "test_time": 0}], "http://localhost", "model")
    if operation not in {"get_run", "create_run"}:
        inference_client.set_terminated.assert_any_call("inference-run", "FAILED")


def test_invalid_logging_location_is_a_model_error(inference_client: MagicMock) -> None:
    inference_client.get_experiment.return_value.artifact_location = "file:///unapproved"
    with pytest.raises(inference.ModelError, match="HTTP-served"):
        inference.predict_batch([{"age": 60, "sex": 1, "test_time": 0}], "http://localhost", "model")
    inference_client.create_run.assert_not_called()


@pytest.mark.parametrize("output", [np.array([np.inf]), np.array([1.0, 2.0]), RuntimeError("prediction failure")])
def test_invalid_model_predictions_are_model_errors(
    inference_client: MagicMock, pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch, output: object
) -> None:
    predicted = MagicMock(side_effect=output) if isinstance(output, Exception) else MagicMock(return_value=output)
    monkeypatch.setattr(pipeline, "predict", predicted)
    with pytest.raises(inference.ModelError):
        inference.predict_batch([{"age": 60, "sex": 1, "test_time": 0}], "http://localhost", "model")
    inference_client.set_terminated.assert_called_once_with("inference-run", "FAILED")


def test_unexpected_bug_is_not_disguised_as_dependency_outage(inference_client: MagicMock) -> None:
    inference_client.log_param.side_effect = RuntimeError("programming error")
    with pytest.raises(RuntimeError, match="programming error"):
        inference.predict_batch([{"age": 60, "sex": 1, "test_time": 0}], "http://localhost", "model")
    inference_client.set_terminated.assert_called_once_with("inference-run", "FAILED")


def test_failure_update_outage_never_returns_predictions(inference_client: MagicMock) -> None:
    inference_client.log_text.side_effect = MlflowException("offline", INTERNAL_ERROR)
    inference_client.set_terminated.side_effect = MlflowException("offline", INTERNAL_ERROR)
    with pytest.raises(inference.UnavailableError):
        inference.predict_batch([{"age": 60, "sex": 1, "test_time": 0}], "http://localhost", "model")
    inference_client.set_terminated.assert_called_once_with("inference-run", "FAILED")


def test_empty_large_and_active_run_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    batches: tuple[list[dict[str, float | int | None]], ...] = ([], [{"age": 60}] * 3)
    for records in batches:
        with pytest.raises(inference.InputError, match="Batch size"):
            inference.predict_batch(records, "http://localhost", "model", max_batch_size=2)
    monkeypatch.setattr(mlflow, "active_run", MagicMock(return_value=object()))
    with pytest.raises(inference.ModelError, match="another active"):
        inference.predict_batch([{"age": 60}], "http://localhost", "model")
