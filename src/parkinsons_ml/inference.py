"""Predict with one immutable champion version and require complete MLflow logging."""

import math
import time
from contextlib import suppress
from dataclasses import dataclass
from uuid import uuid4

import mlflow
import numpy as np
import pandas as pd
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.models import get_model_info
from mlflow.sklearn import load_model
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from parkinsons_ml.experimentation import validate_predictions
from parkinsons_ml.promotion import IDENTITIES, champion_version
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
    """Resolve the alias once; all later work uses this exact version number."""
    try:
        version = champion_version(client, name)
        if version is None:
            raise UnavailableError("No champion is available")
        resolved = client.get_model_version(name, version)
        if resolved.status != "READY":
            raise UnavailableError("Champion version is not ready")
        return resolved
    except UnavailableError:
        raise
    except Exception as error:
        raise UnavailableError("Champion registry is unavailable") from error


def load_pipeline(version: ModelVersion) -> tuple[Pipeline, tuple[str, ...]]:
    """Load fresh artifacts without caching or consulting the alias again."""
    if not version.run_id or any(not version.tags.get(key) for key in IDENTITIES):
        raise ModelError("Champion provenance is incomplete")
    if (version.tags["schema_version"], version.tags["policy_version"]) != (SCHEMA_VERSION, POLICY_VERSION):
        raise ModelError("Champion schema/policy is unsupported")
    uri = f"models:/{version.name}/{version.version}"
    try:
        pipeline = load_model(uri)
        signature = get_model_info(uri).signature
    except MlflowException as error:
        if error.error_code in {"RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE"}:
            raise ModelError("Champion artifact is unavailable or invalid") from error
        raise UnavailableError("Model artifact service is unavailable") from error
    except Exception as error:
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
        or (signature.inputs.input_names() != list(features))
    ):
        raise ModelError("Champion signature/selection is inconsistent")
    return pipeline, features


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
        pipeline, features = load_pipeline(version)
        inputs = prepare_inputs(records, features)
        assert version.run_id is not None  # checked by load_pipeline before any logging
        # Use client APIs rather than fluent active runs. API operation serialization
        # still protects the process-wide routing needed by model artifact loaders.
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
        except ModelError:
            raise
        except Exception as error:
            raise UnavailableError("Inference logging could not be started") from error
        try:
            client.log_text(run.info.run_id, inputs.to_csv(index=False), "inputs.csv")
            try:
                predicted = np.asarray(pipeline.predict(inputs), dtype=np.float64)
                validate_predictions(predicted, (len(inputs),))
            except Exception as error:
                raise ModelError("Champion prediction failed or returned invalid values") from error
            output = pd.DataFrame({"row": np.arange(len(inputs)), "prediction": predicted})
            client.log_text(run.info.run_id, output.to_csv(index=False), "predictions.csv")
            client.log_param(run.info.run_id, "features", ",".join(features))
            client.log_metric(run.info.run_id, "batch_size", len(inputs))
            client.log_metric(run.info.run_id, "processing_seconds", time.perf_counter() - started)
            client.set_terminated(run.info.run_id, "FINISHED")
        except Exception as error:
            # A server outage can prevent the remote failure update too. Never
            # convert that uncertainty into successful, unlogged predictions.
            with suppress(Exception):
                client.set_terminated(run.info.run_id, "FAILED")
            if isinstance(error, ModelError):
                raise
            raise UnavailableError("Required inference logging failed") from error
        return PredictionResult(request_id, run.info.run_id, model_name, version.version, tuple(predicted.tolist()))
