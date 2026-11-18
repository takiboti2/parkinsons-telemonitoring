"""Promotion contracts and failure boundaries independent of server availability."""

import json
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, RESOURCE_DOES_NOT_EXIST
from mlflow.tracking import MlflowClient

from parkinsons_ml import promotion
from parkinsons_ml.experimentation import (
    ComparisonResult,
    ConfigurationResult,
    Metrics,
    PromotionConfig,
    PromotionResult,
    TrainConfig,
    training_config,
)
from parkinsons_ml.training import build_pipeline


@pytest.fixture
def comparison() -> ComparisonResult:
    child = ConfigurationResult(
        "baseline", ("age", "sex", "test_time"), "child", "models:/m-example", Metrics(1.0, 1.0, 0.0, None)
    )
    return ComparisonResult("parent", (child,), child, "dataset", "split")


@pytest.mark.parametrize(
    "settings",
    [
        None,
        [],
        {"enabled": "true"},
        {"enabled": 1},
        {"unknown": True},
        {"registered_model_name": ""},
        {"registered_model_name": "a/b"},
    ],
)
def test_invalid_promotion_config(settings: object) -> None:
    with pytest.raises(ValueError):
        training_config({"experiment_name": "test", "promotion": settings})


def test_promotion_config_defaults() -> None:
    assert not training_config({"experiment_name": "test"}).promotion.enabled
    config = training_config({"experiment_name": "test", "promotion": {"enabled": True}})
    assert config.promotion == PromotionConfig(True, "motor-updrs")
    with pytest.raises(ValueError, match="validated PromotionConfig"):
        TrainConfig("test", config.feature_sets, cast(PromotionConfig, {}))


def test_disabled_and_busy_do_not_touch_registry(comparison: ComparisonResult, tmp_path: Path) -> None:
    client = cast(MlflowClient, MagicMock(spec=MlflowClient))
    assert promotion.promote_winner(client, comparison, PromotionConfig(), tmp_path / "lock").decision == "disabled"
    assert not cast(MagicMock, client).mock_calls
    with promotion.promotion_lock(tmp_path / "lock"):
        result = promotion.promote_winner(client, comparison, PromotionConfig(True), tmp_path / "lock")
        assert result.reason == "promotion_busy"
    assert not cast(MagicMock, client).mock_calls
    # Exception teardown must release the guard for a subsequent process/operation.
    with pytest.raises(RuntimeError):
        with promotion.promotion_lock(tmp_path / "lock"):
            raise RuntimeError("aborted")
    with promotion.promotion_lock(tmp_path / "lock"):
        pass


def test_missing_champion_is_not_an_outage() -> None:
    client = MagicMock(spec=MlflowClient)
    client.get_registered_model.side_effect = MlflowException("missing", error_code=RESOURCE_DOES_NOT_EXIST)
    assert promotion.champion_version(client, "model") is None
    promotion.reconcile(client, "model")
    client.get_registered_model.side_effect = MlflowException("outage", error_code=INTERNAL_ERROR)
    with pytest.raises(MlflowException):
        promotion.champion_version(client, "model")
    with pytest.raises(MlflowException):
        promotion.reconcile(client, "model")


@pytest.mark.parametrize(
    "problem",
    [
        "unfinished_child",
        "no_parent",
        "unfinished_parent",
        "not_winner",
        "missing_identity",
        "mismatched_identity",
        "missing_score",
        "nan",
        "negative",
    ],
)
def test_invalid_authoritative_evaluation(problem: str) -> None:
    client = MagicMock(spec=MlflowClient)
    tags = {key: key for key in promotion.IDENTITIES}
    child = MagicMock()
    child.info.status = "FINISHED"
    child.data.tags = {**tags, "mlflow.parentRunId": "parent"}
    child.data.params = {"features": '["age"]'}
    child.data.metrics = {"oof_mae": 1.0}
    parent = MagicMock()
    parent.info.status = "FINISHED"
    parent.data.tags = {**tags, "winner_run_id": "child"}
    if problem == "unfinished_child":
        child.info.status = "FAILED"
    elif problem == "no_parent":
        del child.data.tags["mlflow.parentRunId"]
    elif problem == "unfinished_parent":
        parent.info.status = "RUNNING"
    elif problem == "not_winner":
        parent.data.tags["winner_run_id"] = "other"
    elif problem == "missing_identity":
        del child.data.tags["dataset_id"]
    elif problem == "mismatched_identity":
        parent.data.tags["split_id"] = "different"
    elif problem == "missing_score":
        child.data.metrics = {}
    else:
        child.data.metrics["oof_mae"] = np.nan if problem == "nan" else -1.0
    client.get_run.side_effect = {"child": child, "parent": parent}.__getitem__
    with pytest.raises(ValueError):
        promotion.evaluation_record(client, "child")


def test_unresolved_intent_blocks_new_work(
    comparison: ComparisonResult, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = cast(MlflowClient, MagicMock(spec=MlflowClient))

    def broken(client: MlflowClient, name: str) -> None:
        raise ConnectionError("unreachable")

    monkeypatch.setattr(promotion, "reconcile", broken)
    result = promotion.promote_winner(client, comparison, PromotionConfig(True), tmp_path / "lock")
    assert not result.confirmed and result.decision == "failed"
    assert not cast(MagicMock, client).mock_calls


def test_unavailable_audit_does_not_claim_promotion(comparison: ComparisonResult, tmp_path: Path) -> None:
    client = MagicMock(spec=MlflowClient)
    client.get_registered_model.return_value.tags = {}
    client.get_run.side_effect = ConnectionError("unreachable")
    result = promotion.promote_winner(client, comparison, PromotionConfig(True), tmp_path / "lock")
    assert result.decision == "failed" and result.run_id is None
    client.set_registered_model_alias.assert_not_called()


def test_lock_filesystem_failure_is_reported(comparison: ComparisonResult, tmp_path: Path) -> None:
    client = cast(MlflowClient, MagicMock(spec=MlflowClient))
    occupied = tmp_path / "file"
    occupied.write_text("not a directory")
    result = promotion.promote_winner(client, comparison, PromotionConfig(True), occupied / "lock")
    assert result.decision == "failed" and "FileExistsError" in result.reason


@pytest.mark.parametrize("old,new", [(None, "bad"), ("bad", "2"), ("0", "2"), (None, "0")])
def test_invalid_recovery_record_stays_blocked(old: str | None, new: str, tmp_path: Path) -> None:

    client = MagicMock(spec=MlflowClient)
    client.get_registered_model.return_value.tags = {promotion.PENDING: "audit"}
    intent = tmp_path / "intent.json"
    intent.write_text(json.dumps({"registered_model_name": "model", "previous_version": old, "candidate_version": new}))
    client.download_artifacts.return_value = str(intent)
    with pytest.raises(ValueError, match="Invalid outstanding"):
        promotion.reconcile(client, "model")
    client.delete_registered_model_tag.assert_not_called()


def test_external_alias_is_never_overwritten() -> None:
    client = MagicMock(spec=MlflowClient)
    client.get_registered_model.return_value.aliases = {"champion": "external"}
    with pytest.raises(RuntimeError, match="outside"):
        promotion.restore_alias(client, "model", "1", "2")
    client.set_registered_model_alias.assert_not_called()
    client.delete_registered_model_alias.assert_not_called()


def test_uncertain_audit_remains_pending() -> None:
    client = MagicMock(spec=MlflowClient)
    result = PromotionResult("failed", "rollback_unconfirmed", "model", run_id="audit", confirmed=False)
    promotion.finish(client, result)
    client.set_tag.assert_any_call("audit", "state", "pending")


@pytest.mark.parametrize(
    "problem",
    [
        "none",
        "not_pipeline",
        "selection",
        "signature",
        "checksum",
        "columns",
        "refit_columns",
        "missing_predictions",
        "nonfinite",
        "wrong_predictions",
    ],
)
def test_immutable_version_validation(problem: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock(spec=MlflowClient)
    inputs = pd.DataFrame({"age": [50.0, 60.0, 70.0]})
    pipeline = build_pipeline(("age",))
    pipeline.fit(inputs, [1.0, 2.0, 3.0])
    expected = pd.DataFrame({"prediction": pipeline.predict(inputs)})
    if problem == "columns":
        inputs = inputs.rename(columns={"age": "motor_updrs"})
    elif problem == "refit_columns":
        expected = expected.rename(columns={"prediction": "other"})
    elif problem == "missing_predictions":
        expected = expected.iloc[:1]
    elif problem == "nonfinite":
        expected["prediction"] = np.nan
    elif problem == "wrong_predictions":
        expected["prediction"] += 1
    reference = tmp_path / "reference.csv"
    predictions = tmp_path / "refit.csv"
    identity = tmp_path / "identity.json"
    inputs.to_csv(reference, index=False)
    expected.to_csv(predictions, index=False)
    identity.write_text(
        json.dumps({"sha256": "bad" if problem == "checksum" else promotion.digest(reference.read_bytes())})
    )
    paths = {
        "monitoring/reference.csv": str(reference),
        "monitoring/identity.json": str(identity),
        "predictions/refit.csv": str(predictions),
    }

    def download(run_id: str, path: str) -> str:
        return paths[path]

    client.download_artifacts.side_effect = download

    def load(uri: str) -> object:
        return None if problem == "not_pipeline" else pipeline

    info = MagicMock()
    info.signature.inputs.input_names.return_value = ["sex"] if problem == "signature" else ["age"]

    def model_info(uri: str) -> object:
        return info

    monkeypatch.setattr(promotion, "load_model", load)
    monkeypatch.setattr(promotion, "get_model_info", model_info)
    record = promotion.EvaluationRecord(1.0, {}, ("sex",) if problem == "selection" else ("age",))
    if problem == "none":
        promotion.validate_version(client, "model", "1", "run", record)
    else:
        with pytest.raises((ValueError, AssertionError)):
            promotion.validate_version(client, "model", "1", "run", record)


def test_rollback_must_verify_restoration() -> None:
    client = MagicMock(spec=MlflowClient)
    client.get_registered_model.return_value.aliases = {"champion": "2"}
    with pytest.raises(RuntimeError, match="Rollback could not be confirmed"):
        promotion.restore_alias(client, "model", "1", "2")


def test_completed_intent_disagreement_is_not_cleared(tmp_path: Path) -> None:
    client = MagicMock(spec=MlflowClient)
    client.get_registered_model.return_value.tags = {promotion.PENDING: "audit"}
    client.get_registered_model.return_value.aliases = {"champion": "3"}
    client.get_run.return_value.data.tags = {"state": "complete", "decision": "promoted"}
    path = tmp_path / "intent.json"
    path.write_text(json.dumps({"registered_model_name": "model", "previous_version": "1", "candidate_version": "2"}))
    client.download_artifacts.return_value = str(path)
    with pytest.raises(RuntimeError, match="disagrees"):
        promotion.reconcile(client, "model")
    client.delete_registered_model_tag.assert_not_called()
