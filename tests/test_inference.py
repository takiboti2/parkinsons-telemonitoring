"""Immutable-model inference, input representation, and mandatory logging failures."""

from collections.abc import Generator
from contextlib import contextmanager
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

from parkinsons_ml import inference
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


def test_resolves_only_one_alias_version(version: ModelVersion, monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock(spec=MlflowClient)
    alias = MagicMock(return_value="7")
    client.get_model_version.return_value = version
    monkeypatch.setattr(inference, "champion_version", alias)
    assert inference.resolve_champion(client, "model") is version
    alias.assert_called_once_with(client, "model")
    client.get_model_version.assert_called_once_with("model", "7")
    alias.return_value = None
    with pytest.raises(inference.UnavailableError, match="No champion"):
        inference.resolve_champion(client, "model")
    alias.side_effect = ConnectionError("private server details")
    with pytest.raises(inference.UnavailableError, match="registry is unavailable"):
        inference.resolve_champion(client, "model")


def test_not_ready_version(version: ModelVersion, monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock(spec=MlflowClient)
    cast(MagicMock, version).status = "PENDING_REGISTRATION"
    client.get_model_version.return_value = version
    monkeypatch.setattr(inference, "champion_version", MagicMock(return_value="7"))
    with pytest.raises(inference.UnavailableError, match="not ready"):
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

    def load(uri: str) -> object:
        assert uri == "models:/model/7"
        return returned

    def signature(uri: str) -> object:
        return info

    monkeypatch.setattr(inference, "load_model", load)
    monkeypatch.setattr(inference, "get_model_info", signature)
    if problem == "none":
        loaded, features = inference.load_pipeline(version)
        assert loaded is pipeline and features == ("age", "sex", "test_time")
    else:
        with pytest.raises(inference.ModelError):
            inference.load_pipeline(version)


@pytest.mark.parametrize(
    "error, expected",
    [
        (MlflowException("missing", RESOURCE_DOES_NOT_EXIST), inference.ModelError),
        (MlflowException("offline", INTERNAL_ERROR), inference.UnavailableError),
        (ValueError("bad artifact"), inference.ModelError),
    ],
)
def test_model_loading_error_classification(
    version: ModelVersion, error: Exception, expected: type[Exception], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(uri: str) -> None:
        raise error

    monkeypatch.setattr(inference, "load_model", fail)
    with pytest.raises(expected):
        inference.load_pipeline(version)


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
    assert np.isnan(frame.iloc[0, 0]) and frame["test_time"].tolist() == [0.0, 3.0]
    with pytest.raises(inference.InputError, match="empty"):
        inference.prepare_inputs([], ("age",))
    with pytest.raises(inference.InputError, match="Every|Each"):
        inference.prepare_inputs([{"age": 60}, {"age": 70, "test_time": 3}], ("age", "test_time"))


@pytest.mark.parametrize(
    "failure",
    [
        "none",
        "create_run",
        "get_run",
        "local_artifacts",
        "log_text",
        "log_param",
        "log_metric",
        "set_terminated",
        "predict",
        "nonfinite",
        "failed_failure_update",
    ],
)
def test_required_logging_and_prediction_fail_closed(
    version: ModelVersion, pipeline: Pipeline, failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = MagicMock(spec=MlflowClient)
    client.get_experiment.return_value.artifact_location = "mlflow-artifacts:/test"
    client.create_run.return_value.info.run_id = "inference-run"

    @contextmanager
    def connection(uri: str) -> Generator[MlflowClient]:
        yield client

    monkeypatch.setattr(inference, "tracking_context", connection)
    monkeypatch.setattr(inference, "resolve_champion", MagicMock(return_value=version))
    monkeypatch.setattr(inference, "load_pipeline", MagicMock(return_value=(pipeline, ("age", "sex", "test_time"))))

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("private dependency detail")

    if failure in {"create_run", "get_run", "log_text", "log_param", "log_metric", "set_terminated"}:
        getattr(client, failure).side_effect = fail
    elif failure == "local_artifacts":
        client.get_experiment.return_value.artifact_location = "file:///unapproved"
    elif failure == "predict":
        monkeypatch.setattr(pipeline, "predict", fail)
    elif failure == "nonfinite":
        monkeypatch.setattr(pipeline, "predict", MagicMock(return_value=np.array([np.inf])))
    elif failure == "failed_failure_update":
        client.log_text.side_effect = fail
        client.set_terminated.side_effect = fail
    records: list[dict[str, float | int | None]] = [{"age": 60, "sex": 1, "test_time": -2}]
    if failure == "none":
        result = inference.predict_batch(records, "http://localhost", "model")
        assert result.run_id == "inference-run" and result.model_version == "7"
        assert len(result.predictions) == 1
        client.set_terminated.assert_called_with("inference-run", "FINISHED")
        assert client.log_text.call_count == 2
    else:
        with pytest.raises((inference.UnavailableError, inference.ModelError)):
            inference.predict_batch(records, "http://localhost", "model")
        if failure not in {"create_run", "get_run", "local_artifacts"}:
            client.set_terminated.assert_any_call("inference-run", "FAILED")


def test_empty_large_and_active_run_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    batches: tuple[list[dict[str, float | int | None]], ...] = ([], [{"age": 60}] * 3)
    for records in batches:
        with pytest.raises(inference.InputError, match="Batch size"):
            inference.predict_batch(records, "http://localhost", "model", max_batch_size=2)
    monkeypatch.setattr(mlflow, "active_run", MagicMock(return_value=object()))
    with pytest.raises(inference.ModelError, match="another active"):
        inference.predict_batch([{"age": 60}], "http://localhost", "model")
