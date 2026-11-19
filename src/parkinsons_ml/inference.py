"""Predict with one immutable champion version and require complete MLflow logging."""

import math
import sys
import time
from collections.abc import Generator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import mlflow
import numpy as np
import pandas as pd
from mlflow.entities.model_registry import ModelVersion
from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.sklearn import load_model
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from parkinsons_ml.experimentation import validate_predictions
from parkinsons_ml import tracking
from parkinsons_ml.promotion import IDENTITIES, PENDING, promotion_lock
from parkinsons_ml.schema import CANDIDATE_FEATURES, POLICY_VERSION, SCHEMA_VERSION
from parkinsons_ml.tracking import tracking_context
from parkinsons_ml.training import FeaturePreparation, validate_features


class InputError(ValueError):
    """The caller supplied invalid prediction records, not an invalid model."""


class UnavailableError(RuntimeError):
    """The registry/logging dependency or champion selection is unavailable."""


class ModelError(RuntimeError):
    """The selected immutable artifact cannot safely serve predictions."""


@dataclass(frozen=True)
class PredictionResult:
    request_id: str
    run_id: str
    registered_model_name: str
    model_version: str
    predictions: tuple[float, ...]


def resolve_champion(client: MlflowClient, name: str) -> ModelVersion:
    """Capture a committed version under the same brief guard as alias writers."""
    try:
        with promotion_lock(tracking.PROMOTION_LOCK, shared=True):
            try:
                model = client.get_registered_model(name)
            except MlflowException as error:
                # An unregistered model is a bootstrap state, not an outage.
                if error.error_code == "RESOURCE_DOES_NOT_EXIST":
                    raise UnavailableError("No champion is available") from error
                raise
            # A crashed promoter releases its file lock but leaves this marker.
            # Serving must fail closed until promotion reconciles the intent.
            if model.tags.get(PENDING):
                raise UnavailableError("Champion promotion is not yet confirmed")
            version = model.aliases.get("champion")
            if version is None:
                raise UnavailableError("No champion is available")
            resolved = client.get_model_version(name, version)
            if resolved.status != "READY":
                raise UnavailableError("Champion version is not ready")
            return resolved
    except BlockingIOError as error:
        raise UnavailableError("Champion promotion is in progress") from error
    except (MlflowException, OSError) as error:
        raise UnavailableError("Champion registry is unavailable") from error


@contextmanager
def load_pipeline(version: ModelVersion) -> Generator[tuple[Pipeline, tuple[str, ...]]]:
    """Own model downloads/import-path changes for the entire prediction operation."""
    if not version.run_id or any(not version.tags.get(key) for key in IDENTITIES):
        raise ModelError("Champion provenance is incomplete")
    if (version.tags["schema_version"], version.tags["policy_version"]) != (SCHEMA_VERSION, POLICY_VERSION):
        raise ModelError("Champion schema/policy is unsupported")
    uri = f"models:/{version.name}/{version.version}"
    previous_path = list(sys.path)
    with TemporaryDirectory(prefix="parkinsons-inference-") as temporary:
        try:
            try:
                local_path = download_artifacts(artifact_uri=uri, dst_path=temporary)
            except MlflowException as error:
                if error.error_code in {"RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"}:
                    raise ModelError("Champion artifact is unavailable or invalid") from error
                raise UnavailableError("Model artifact service is unavailable") from error
            except OSError as error:
                raise UnavailableError("Model artifact download is unavailable") from error
            try:
                # Read the signature from the same download, not a second remote
                # lookup. Post-load checks are compatibility checks, not a sandbox:
                # model artifacts and their writers must still be trusted.
                pipeline = load_model(local_path)
                signature = Model.load(str(Path(local_path) / "MLmodel")).signature
            except (MlflowException, OSError, ValueError, TypeError, KeyError, AttributeError) as error:
                raise ModelError("Champion artifact could not be loaded") from error
            if not isinstance(pipeline, Pipeline) or tuple(pipeline.named_steps) != (
                "features",
                "imputation",
                "linearregression",
            ):
                raise ModelError("Champion is not a complete approved pipeline")
            selector = pipeline.named_steps["features"]
            if (
                not isinstance(selector, FeaturePreparation)
                or type(pipeline.named_steps["linearregression"]) is not LinearRegression
            ):
                raise ModelError("Champion contains an unsupported estimator or selector")
            try:
                features = validate_features(selector.features)
            except ValueError as error:
                raise ModelError("Champion selection is invalid") from error
            if (
                selector.features != features
                or signature is None
                or signature.inputs is None
                or signature.inputs.input_names() != list(features)
            ):
                raise ModelError("Champion signature/selection is inconsistent")
            yield pipeline, features
        finally:
            # The API serializes model operations. Remove SDK-added import paths
            # before deleting their directories, on failures as well as success.
            # Keep the files alive until prediction/logging finishes for lazy imports.
            sys.path[:] = previous_path


def prepare_inputs(records: list[dict[str, float | int | None]], features: tuple[str, ...]) -> pd.DataFrame:
    """Validate every submitted predictor, then retain only the model's selection."""
    if not records:
        raise InputError("records must not be empty")
    for row in records:
        if set(row) - set(CANDIDATE_FEATURES):
            raise InputError("Records contain unknown or forbidden predictor names")
        # Check per-record keys before pandas can turn an omitted key into NaN.
        # Explicit null is allowed for imputation; an absent selected field is not.
        if set(features) - set(row):
            raise InputError("Each record must include every selected predictor")
        for value in row.values():
            try:
                if value is not None and (type(value) not in (float, int) or not math.isfinite(value)):
                    raise InputError("Predictors must be finite numbers or null, not strings or booleans")
            except OverflowError as error:
                raise InputError("Predictors must be representable finite numbers") from error
    frame = pd.DataFrame(records, dtype=float)
    submitted = validate_features(list(frame.columns))
    try:
        values = FeaturePreparation(submitted).transform(frame)
    except ValueError as error:
        raise InputError("Predictor encodings are invalid") from error
    # This is the same selected, clipped, pre-imputation representation used by
    # the training monitoring reference. Extra approved predictors are not logged.
    prepared = pd.DataFrame(values, columns=pd.Index(submitted))
    return prepared.loc[:, list(features)]


def predict_batch(
    records: list[dict[str, float | int | None]], tracking_uri: str, model_name: str, max_batch_size: int = 1000
) -> PredictionResult:
    """No final-test reads, refitting, alias changes, or unlogged successful output."""
    if not 1 <= len(records) <= max_batch_size:
        raise InputError("Batch size is outside the allowed range")
    if mlflow.active_run() is not None:
        raise ModelError("Prediction cannot take ownership of another active MLflow run")
    request_id = uuid4().hex
    started = time.perf_counter()
    with tracking_context(tracking_uri) as client:
        version = resolve_champion(client, model_name)
        with load_pipeline(version) as (pipeline, features):
            inputs = prepare_inputs(records, features)
            assert version.run_id is not None  # checked by load_pipeline before any logging
            # Direct client APIs avoid fluent active runs and never re-resolve champion.
            try:
                source = client.get_run(version.run_id)
                experiment = client.get_experiment(source.info.experiment_id)
                if not experiment.artifact_location.startswith("mlflow-artifacts:/"):
                    raise ModelError("Inference requires HTTP-served artifact storage")
                run = client.create_run(
                    source.info.experiment_id,
                    tags={
                        "purpose": "inference",
                        "mlflow.runName": "prediction-batch",
                        "request_id": request_id,
                        "registered_model_name": model_name,
                        "model_version": version.version,
                        "training_run_id": version.run_id,
                        "reference_run_id": version.run_id,
                        "reference_artifact": "monitoring/reference.csv",
                        "representation": "selected-pre-imputation-v1",
                        **{key: version.tags[key] for key in IDENTITIES},
                    },
                )
            except (MlflowException, OSError) as error:
                raise UnavailableError("Inference logging could not be started") from error
            try:
                client.log_text(run.info.run_id, inputs.to_csv(index=False), "inputs.csv")
                try:
                    predicted = np.asarray(pipeline.predict(inputs), dtype=np.float64)
                    validate_predictions(predicted, (len(inputs),))
                except (ValueError, TypeError, RuntimeError, ArithmeticError) as error:
                    raise ModelError("Champion prediction failed or returned invalid values") from error
                output = pd.DataFrame({"row": np.arange(len(inputs)), "prediction": predicted})
                client.log_text(run.info.run_id, output.to_csv(index=False), "predictions.csv")
                client.log_param(run.info.run_id, "features", ",".join(features))
                client.log_metric(run.info.run_id, "batch_size", len(inputs))
                client.log_metric(run.info.run_id, "processing_seconds", time.perf_counter() - started)
                client.set_terminated(run.info.run_id, "FINISHED")
            except Exception as error:
                # Always attempt to close an opened run, including unexpected bugs,
                # but only classify actual dependency failures as unavailability.
                with suppress(Exception):
                    client.set_terminated(run.info.run_id, "FAILED")
                if isinstance(error, (MlflowException, OSError)):
                    raise UnavailableError("Required inference logging failed") from error
                raise
            return PredictionResult(request_id, run.info.run_id, model_name, version.version, tuple(predicted.tolist()))
