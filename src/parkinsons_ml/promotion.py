"""Single-host champion selection with strict MAE improvement and alias rollback."""

import fcntl
import json
import math
from collections.abc import Generator
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from mlflow.entities import LoggedModelStatus
from mlflow.exceptions import MlflowException
from mlflow.artifacts import download_artifacts
from mlflow.models import Model
from mlflow.sklearn import load_model
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from parkinsons_ml.artifacts import artifact_workspace
from parkinsons_ml.experimentation import (
    ComparisonResult,
    PromotionConfig,
    PromotionResult,
    digest,
    json_mapping,
    validate_predictions,
)
from parkinsons_ml.training import FeaturePreparation, validate_features

POLICY = "strict-comparable-oof-mae-v1"
IDENTITIES = ("dataset_id", "split_id", "schema_version", "policy_version", "evaluation_id")
PENDING = "pending_promotion_run_id"


@contextmanager
def promotion_lock(path: Path, *, shared: bool = False) -> Generator[None]:
    """Guard registry updates/read snapshots, not training; never unlink the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        mode = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        fcntl.flock(lock, mode | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def champion_version(client: MlflowClient, name: str) -> str | None:
    # A missing alias is bootstrap, but an outage is not. Inspect aliases on the
    # model so unrelated INVALID_PARAMETER_VALUE errors cannot masquerade as absence.
    try:
        return client.get_registered_model(name).aliases.get("champion")
    except MlflowException as error:
        if error.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        return None


@dataclass(frozen=True)
class EvaluationRecord:
    mae: float
    identities: dict[str, str]
    features: tuple[str, ...]


def evaluation_record(client: MlflowClient, run_id: str) -> EvaluationRecord:
    """Read authoritative run metadata, never a caller-supplied ranking score."""
    run = client.get_run(run_id)
    parent_id = run.data.tags.get("mlflow.parentRunId")
    if run.info.status != "FINISHED" or not parent_id:
        raise ValueError("Candidate/incumbent must originate from a completed comparison")
    parent = client.get_run(parent_id)
    if parent.info.status != "FINISHED" or parent.data.tags.get("winner_run_id") != run_id:
        raise ValueError("Originating comparison must have finished and selected this run")
    identities = {key: run.data.tags.get(key, "") for key in IDENTITIES}
    if any(not value or parent.data.tags.get(key) != value for key, value in identities.items()):
        raise ValueError("Missing or inconsistent evaluation identities")
    mae = run.data.metrics.get("oof_mae")
    if mae is None or not math.isfinite(mae) or mae < 0:
        raise ValueError("OOF MAE must be present, finite, and nonnegative")
    features = validate_features(json.loads(run.data.params.get("features", "null")))
    return EvaluationRecord(mae, identities, features)


def finish(client: MlflowClient, result: PromotionResult) -> None:
    """Complete audit logging before declaring the alias operation committed."""
    assert result.run_id is not None
    client.log_dict(result.run_id, {**asdict(result), "promotion_policy": POLICY}, "decision.json")
    client.set_tag(result.run_id, "decision", result.decision)
    client.set_terminated(result.run_id, "FAILED" if result.decision == "failed" else "FINISHED")
    client.set_tag(result.run_id, "state", "complete" if result.confirmed else "pending")


def restore_alias(client: MlflowClient, name: str, old: str | None, new: str) -> None:
    """Roll back only our version; never overwrite an unrelated external update."""
    current = champion_version(client, name)
    if current not in {old, new}:
        raise RuntimeError("Champion changed outside the promotion guard")
    if current == new:
        if old is None:
            client.delete_registered_model_alias(name, "champion")
        else:
            client.set_registered_model_alias(name, "champion", old)
    if champion_version(client, name) != old:
        raise RuntimeError("Rollback could not be confirmed")


def reconcile(client: MlflowClient, name: str) -> None:
    """Resolve at most one outstanding alias intent before another promotion."""
    try:
        pending = client.get_registered_model(name).tags.get(PENDING)
    except MlflowException as error:
        if error.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
        return
    if not pending:
        return
    with artifact_workspace() as directory:
        intent = json_mapping(Path(client.download_artifacts(pending, "intent.json", directory)).read_bytes())
    old, new = intent.get("previous_version"), intent.get("candidate_version")
    if (
        intent.get("registered_model_name") != name
        or not isinstance(new, str)
        or not new.isdecimal()
        or int(new) < 1
        or (old is not None and (not isinstance(old, str) or not old.isdecimal() or int(old) < 1))
    ):
        raise ValueError("Invalid outstanding promotion intent")
    run = client.get_run(pending)
    # An acknowledged success may leave only marker cleanup outstanding. Do not
    # undo it. An interrupted, uncommitted update is conservatively rolled back.
    if run.data.tags.get("state") == "complete":
        expected = new if run.data.tags.get("decision") == "promoted" else old
        if champion_version(client, name) != expected:
            raise RuntimeError("Completed promotion audit disagrees with champion")
    else:
        restore_alias(client, name, old, new)
        finish(
            client,
            PromotionResult(
                "failed",
                "interrupted_update_rolled_back",
                name,
                run_id=pending,
                previous_version=old,
                candidate_version=new,
            ),
        )
    client.delete_registered_model_tag(name, PENDING)


def validate_version(client: MlflowClient, name: str, version: str, run_id: str, record: EvaluationRecord) -> None:
    """Check the immutable registry version against already-saved refit artifacts."""
    uri = f"models:/{name}/{version}"
    with artifact_workspace() as directory:
        local = download_artifacts(artifact_uri=uri, dst_path=directory)
        pipeline = load_model(local)
        if not isinstance(pipeline, Pipeline):
            raise ValueError("Registered model is not a complete pipeline")
        selector = pipeline.named_steps.get("features")
        if (
            tuple(pipeline.named_steps) != ("features", "imputation", "linearregression")
            or not isinstance(selector, FeaturePreparation)
            or selector.features != record.features
            or type(pipeline.named_steps.get("linearregression")) is not LinearRegression
        ):
            raise ValueError("Registered pipeline does not match the approved model/selection")
        signature = Model.load(str(Path(local) / "MLmodel")).signature
        if signature is None or signature.inputs is None or signature.inputs.input_names() != list(record.features):
            raise ValueError("Registered model signature differs from selected features")
        reference_path = Path(client.download_artifacts(run_id, "monitoring/reference.csv", directory))
        identity = json_mapping(
            Path(client.download_artifacts(run_id, "monitoring/identity.json", directory)).read_bytes()
        )
        if identity.get("sha256") != digest(reference_path.read_bytes()):
            raise ValueError("Monitoring reference checksum mismatch")
        inputs = pd.read_csv(reference_path)
        expected = pd.read_csv(client.download_artifacts(run_id, "predictions/refit.csv", directory))
        if tuple(inputs.columns) != record.features or tuple(expected.columns) != ("prediction",):
            raise ValueError("Unexpected refit verification columns")
        predicted = np.asarray(pipeline.predict(inputs), dtype=np.float64)
        validate_predictions(predicted, (len(inputs),))
        values = expected["prediction"].to_numpy(dtype=np.float64)
        validate_predictions(values, predicted.shape)
        np.testing.assert_allclose(predicted, values, rtol=1e-10, atol=1e-10)


def promote_winner(
    client: MlflowClient, comparison: ComparisonResult, config: PromotionConfig, lock_path: Path
) -> PromotionResult:
    """A failed promotion never turns successful training into a failed comparison."""
    result = PromotionResult("failed", "promotion_not_started", config.registered_model_name)
    if not config.enabled:
        return replace(result, decision="disabled", reason="promotion_disabled")
    try:
        with promotion_lock(lock_path):
            try:
                reconcile(client, config.registered_model_name)
            except Exception:
                # Keep the registry marker intact. Another automatic update must
                # not proceed while a prior alias operation remains uncertain.
                return replace(result, reason="outstanding_update_requires_reconciliation", confirmed=False)
            return promote_locked(client, comparison, config, result)
    except BlockingIOError:
        return replace(result, reason="promotion_busy")
    except Exception as error:
        return replace(result, reason=f"promotion_failed:{type(error).__name__}")


def promote_locked(
    client: MlflowClient, comparison: ComparisonResult, config: PromotionConfig, result: PromotionResult
) -> PromotionResult:
    """Caller owns the single-host guard; registry/audit operations are sequential."""
    name = config.registered_model_name
    updating = False
    try:
        parent = client.get_run(comparison.run_id)
        audit = client.create_run(
            parent.info.experiment_id,
            tags={
                "purpose": "champion-promotion",
                "mlflow.runName": f"promotion:{name}",
                "registered_model_name": name,
                "comparison_run_id": comparison.run_id,
                "promotion_policy": POLICY,
                "state": "assessing",
            },
        )
        result = replace(result, run_id=audit.info.run_id)
        candidate = comparison.winner
        record = evaluation_record(client, candidate.run_id)
        if client.get_run(candidate.run_id).data.tags.get("mlflow.parentRunId") != comparison.run_id:
            raise ValueError("Winner does not belong to this comparison")
        for child in comparison.configurations:
            run = client.get_run(child.run_id)
            if run.info.status != "FINISHED" or run.data.tags.get("mlflow.parentRunId") != comparison.run_id:
                raise ValueError("All comparison configurations must have completed successfully")
        old = champion_version(client, name)
        result = replace(result, previous_version=old, candidate_mae=record.mae)
        if old is not None:
            incumbent = client.get_model_version(name, old)
            if not incumbent.run_id or incumbent.tags.get("promotion_policy") != POLICY:
                raise ValueError("Champion lacks approved evaluation provenance")
            previous = evaluation_record(client, incumbent.run_id)
            result = replace(result, previous_mae=previous.mae)
            if any(
                incumbent.tags.get(key) != value for key, value in previous.identities.items()
            ) or incumbent.tags.get("oof_mae") != str(previous.mae):
                raise ValueError("Champion registry provenance differs from its originating evaluation")
            if record.identities != previous.identities:
                raise ValueError("Candidate and champion evaluation identities are incomparable")
            if record.mae >= previous.mae:
                result = replace(result, decision="retained", reason="no_strict_oof_mae_improvement")
                finish(client, result)
                return result
        # Bind registration to the selected run's logged model, not an arbitrary
        # URI supplied by a caller. Eligibility is decided before creating versions.
        if not candidate.model_uri.startswith("models:/") or "/" in candidate.model_uri.removeprefix("models:/"):
            raise ValueError("Expected an immutable logged model URI")
        logged = client.get_logged_model(candidate.model_uri.removeprefix("models:/"))
        if logged.source_run_id != candidate.run_id or logged.status != LoggedModelStatus.READY:
            raise ValueError("Logged model does not belong to the completed winner")
        try:
            client.get_registered_model(name)
        except MlflowException as error:
            if error.error_code != "RESOURCE_DOES_NOT_EXIST":
                raise
            client.create_registered_model(name)
        version = client.create_model_version(
            name,
            logged.artifact_location,
            run_id=candidate.run_id,
            model_id=logged.model_id,
            tags={**record.identities, "promotion_policy": POLICY, "oof_mae": str(record.mae)},
        )
        result = replace(result, candidate_version=version.version)
        validate_version(client, name, version.version, candidate.run_id, record)
        # Persist the small intent before any alias update. It is not an alternate
        # deployment pointer; champion remains the only source of active-model truth.
        client.log_dict(audit.info.run_id, asdict(result), "intent.json")
        client.set_tag(audit.info.run_id, "state", "pending")
        updating = True
        client.set_registered_model_tag(name, PENDING, audit.info.run_id)
        if champion_version(client, name) != old:
            raise RuntimeError("Champion changed outside the promotion guard")
        client.set_registered_model_alias(name, "champion", version.version)
        if champion_version(client, name) != version.version:
            raise RuntimeError("Champion activation could not be confirmed")
        result = replace(result, decision="promoted", reason="bootstrap" if old is None else "lower_oof_mae")
        finish(client, result)
    except Exception as error:
        confirmed = True
        if updating:
            try:
                assert result.candidate_version is not None
                restore_alias(client, name, result.previous_version, result.candidate_version)
            except Exception:
                confirmed = False
        result = replace(
            result,
            decision="rejected" if isinstance(error, ValueError) and not updating else "failed",
            reason=(str(error) if isinstance(error, ValueError) else type(error).__name__)
            + ":"
            + (
                "rolled_back"
                if updating and confirmed
                else "rollback_unconfirmed" if updating else "candidate_not_promoted"
            ),
            confirmed=confirmed,
        )
        if result.run_id is not None:
            # Audit outages must not conceal the actual alias outcome. Leaving the
            # marker blocks another promotion until reconciliation can complete.
            try:
                finish(client, result)
            except Exception:
                return result
        if not confirmed:
            return result
    # Marker cleanup is retryable after a fully audited success/rollback. Recovery
    # checks that terminal audit against the alias before clearing a leftover marker.
    if updating:
        with suppress(Exception):
            client.delete_registered_model_tag(name, PENDING)
    return result
