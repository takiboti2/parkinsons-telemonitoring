"""Compare LinearRegression pipelines using saved patient folds and HTTP MLflow."""

import argparse
import hashlib
import io
import ipaddress
import json
import os
import re
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow.models import infer_signature
from mlflow.sklearn import load_model, log_model
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline

from parkinsons_ml.data import PrepareConfig, parse_subject
from parkinsons_ml.schema import (
    BASELINE_FEATURES,
    CANDIDATE_FEATURES,
    COLUMN_MAPPING,
    EXPORT_COLUMNS,
    MISSING_TOKENS,
    POLICY_VERSION,
    SCHEMA_VERSION,
    clip_test_time,
)


def validate_features(features: object) -> tuple[str, ...]:
    """Validate a runtime selection and return its canonical alphabetical order."""
    # Accept ordered inputs so duplicates remain detectable. Strings, sets, and
    # generators are not supported selections; check types before hashing names.
    if isinstance(features, (str, bytes)) or not isinstance(features, Sequence):
        raise ValueError("features must be an ordered sequence of column names, not a string")
    selected: list[str] = []
    for name in features:
        if not isinstance(name, str):
            raise ValueError("features must contain only string column names")
        selected.append(name)
    if not selected:
        raise ValueError("features must not be empty")
    duplicates = sorted({name for name in selected if selected.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate features: {duplicates}")
    unknown = sorted(set(selected) - set(CANDIDATE_FEATURES))
    if unknown:
        raise ValueError(f"Features outside CANDIDATE_FEATURES: {unknown}")
    # Sort only after validation: converting to a set would silently hide duplicate
    # configuration entries. Equivalent lists now have identical saved selections.
    return tuple(sorted(selected))


class FeaturePreparation(TransformerMixin, BaseEstimator):
    """Select approved features, validate numeric inputs and clip times consistently."""

    def __init__(self, features: tuple[str, ...] = BASELINE_FEATURES) -> None:
        # Store constructor arguments unchanged so scikit-learn can clone an
        # unfitted instance for each validation fold.
        self.features = features

    def fit(self, X: pd.DataFrame, y: object = None) -> "FeaturePreparation":
        values = self.transform(X)
        # An all-missing fitting feature has no training-derived imputation value.
        # Reject it instead of allowing the imputer to silently drop that column.
        selected = validate_features(self.features)
        empty = [name for index, name in enumerate(selected) if np.isnan(values[:, index]).all()]
        if empty:
            raise ValueError(f"Entirely missing fitting features: {empty}")
        return self

    def transform(self, X: pd.DataFrame) -> NDArray[np.float64]:
        # Also validate direct transformer use and sklearn parameter updates;
        # constructor arguments stay unchanged so cloning remains supported.
        selected = validate_features(self.features)
        if X.columns.has_duplicates:
            raise ValueError("Duplicate prediction columns")
        missing = set(selected) - set(X.columns)
        if missing:
            raise ValueError(f"Missing prediction features: {sorted(missing)}")
        # Select and order only approved predictors. Extra target/identifier
        # columns in prepared training tables cannot leak into the estimator.
        frame = X.loc[:, list(selected)]
        # Complex columns are numeric to pandas, but casting them to float would
        # silently discard imaginary components. Reject them before conversion.
        if any(pd.api.types.is_complex_dtype(frame[name]) for name in selected):
            raise ValueError("Prediction features must be real, not complex")
        if any(
            not pd.api.types.is_numeric_dtype(frame[name]) or pd.api.types.is_bool_dtype(frame[name])
            for name in selected
        ):
            raise ValueError("Prediction features must be numeric, not strings or booleans")
        # Work on a copy so clipping cannot mutate a caller's DataFrame. Convert
        # pandas nullable numeric values into the NaN representation sklearn uses.
        values = frame.to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
        if np.isinf(values).any():
            raise ValueError("Infinite prediction values")
        # Missing values are allowed, but observed sex/age values must still obey
        # their encodings. Imputation must not disguise invalid observed values.
        if "sex" in selected:
            sex = values[:, selected.index("sex")]
            if not np.isin(sex[~np.isnan(sex)], [0.0, 1.0]).all():
                raise ValueError("Invalid sex encoding")
        if "age" in selected:
            age = values[:, selected.index("age")]
            if not np.equal(age[~np.isnan(age)], np.floor(age[~np.isnan(age)])).all():
                raise ValueError("Noninteger age")
        if "test_time" in selected:
            index = selected.index("test_time")
            # This idempotent rule also runs for standalone serialized-model calls,
            # keeping inference consistent with prepared train/test representation.
            values[:, index] = [clip_test_time(float(value)) for value in values[:, index]]
        return values


def build_pipeline(features: Sequence[str] = BASELINE_FEATURES) -> Pipeline:
    """Construct an unfitted pipeline suitable for fold fitting and serialization."""
    selected = validate_features(features)
    # Learn continuous medians and the categorical mode separately. A median of
    # binary sex values could otherwise produce the invalid category 0.5.
    imputation = ColumnTransformer(
        [
            (name, SimpleImputer(strategy="most_frequent" if name == "sex" else "median"), [index])
            for index, name in enumerate(selected)
        ]
    )
    # Return unfitted steps: fold-specific fit calls learn imputation values only
    # from that fold's training patients, and serialization retains every step.
    return Pipeline(
        [
            # One imputer per column preserves canonical alphabetical order
            # through to the estimator, including a sex-only selection.
            ("features", FeaturePreparation(selected)),
            ("imputation", imputation),
            ("linearregression", LinearRegression()),
        ]
    )


# Change this identity when scoring, weighting, or tie-breaking semantics change;
# downstream promotion must not treat different evaluation policies as comparable.
EVALUATION_VERSION = "pooled-measurement-oof-v1"
# Training is deliberately repository-only. Validate this source-layout assumption
# before contacting MLflow rather than misidentifying a wheel's site-packages root.
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class FeatureSet:
    # Store the validated alphabetical tuple, not the caller's mutable feature list.
    name: str
    features: tuple[str, ...]


@dataclass(frozen=True)
class TrainConfig:
    # This is the expanded comparison, including the two mandatory configurations.
    experiment_name: str
    feature_sets: tuple[FeatureSet, ...]

    def __post_init__(self) -> None:
        # Validate the expanded configuration once, whether it comes from YAML or
        # a Python caller. Do not reconstruct a YAML-shaped mapping in orchestration.
        if type(self.experiment_name) is not str or not self.experiment_name.strip():
            raise ValueError("experiment_name must be a nonempty string")
        names: set[str] = set()
        selections: set[tuple[str, ...]] = set()
        required = {
            "baseline": validate_features(BASELINE_FEATURES),
            "candidate": validate_features(CANDIDATE_FEATURES),
        }
        for item in self.feature_sets:
            if type(item.name) is not str or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", item.name):
                raise ValueError("Feature-set names must be lowercase identifiers of at most 64 characters")
            if item.name in names:
                raise ValueError("Duplicate feature-set name")
            selected = validate_features(item.features)
            if item.features != selected or (item.name in required and selected != required[item.name]):
                raise ValueError(
                    "Feature sets must use canonical selections and the required baseline/candidate features"
                )
            if selected in selections:
                raise ValueError("Duplicate canonical feature configuration")
            names.add(item.name)
            selections.add(selected)
        if not required.keys() <= names:
            raise ValueError("Both baseline and candidate configurations are required")
        # Normalize run order without requiring callers to pre-sort their settings.
        object.__setattr__(self, "feature_sets", tuple(sorted(self.feature_sets, key=lambda item: item.name)))


class UniqueKeyLoader(yaml.SafeLoader):
    """Keep YAML mappings from silently hiding duplicate configuration names."""


def unique_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode) -> dict[str, object]:
    result: dict[str, object] = {}
    # Check pairs before building a dict: ordinary YAML loading would overwrite an
    # earlier configuration with the same name and hide the user's mistake.
    for key, value in loader.construct_pairs(node, deep=True):
        if not isinstance(key, str) or key in result:
            raise ValueError("YAML keys must be unique strings")
        result[key] = value
    return result


# Register on our loader only; importing this module must not change PyYAML's
# default behavior for preparation or other callers in the same process.
UniqueKeyLoader.add_constructor("tag:yaml.org,2002:map", unique_mapping)


def training_config(value: object) -> TrainConfig:
    """Validate YAML settings, including mandatory and redundant selections."""
    # Treat misspelled options as errors rather than silently using defaults.
    if not isinstance(value, dict) or set(value) - {"experiment_name", "extra_feature_sets"}:
        raise ValueError("train must contain only experiment_name and extra_feature_sets")
    name = value.get("experiment_name")
    extras = value.get("extra_feature_sets", {})
    if not isinstance(name, str):
        raise ValueError("experiment_name must be a nonempty string")
    if not isinstance(extras, dict):
        raise ValueError("extra_feature_sets must be a mapping")
    # Extras extend this fixed comparison; they cannot replace its reference models.
    selections = [
        FeatureSet("baseline", validate_features(BASELINE_FEATURES)),
        FeatureSet("candidate", validate_features(CANDIDATE_FEATURES)),
    ]
    for key, features in extras.items():
        if key in {"baseline", "candidate"}:
            raise ValueError("Reserved feature-set name")
        selected = validate_features(features)
        selections.append(FeatureSet(key, selected))
    # TrainConfig enforces names, required selections, and canonical uniqueness.
    return TrainConfig(name, tuple(selections))


def json_mapping(payload: bytes) -> dict[str, object]:
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON mapping")
    return cast(dict[str, object], value)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class TrainingSnapshot:
    """Private per-comparison copies; callers must not mutate the frames."""

    train: pd.DataFrame
    folds: NDArray[np.int64]
    artifacts: dict[str, bytes]
    dataset_id: str
    split_id: str


def subject_ids(frame: pd.DataFrame) -> list[int]:
    ids: list[int] = []
    # Reuse preparation's exact decimal parser: converting identifiers through
    # binary floats could merge distinct patients and invalidate grouped validation.
    for token in frame["subject"]:
        value, reason = parse_subject(str(token))
        if value is None or reason is not None:
            raise ValueError("Invalid subject identifier")
        ids.append(value)
    return ids


def load_training_snapshot(directory: Path) -> TrainingSnapshot:
    """Check bytes before parsing; deliberately never read or hash test.csv."""
    # An explicit allowlist keeps the final test table out of experimentation.
    # Reuse these captured bytes for validation, parsing, and artifact uploads;
    # configurations must not reread potentially changing files between fits.
    names = ("manifest.json", "train.csv", "split_assignments.csv", "validation_summary.json", "cleaning_events.csv")
    artifacts = {name: (directory / name).read_bytes() for name in names}
    manifest = json_mapping(artifacts["manifest.json"])
    if (manifest.get("schema_version"), manifest.get("policy_version")) != (SCHEMA_VERSION, POLICY_VERSION):
        raise ValueError("Unsupported prepared schema/policy")
    if manifest.get("column_mapping") != COLUMN_MAPPING or manifest.get("missing_tokens") != sorted(MISSING_TOKENS):
        raise ValueError("Inconsistent prepared representation")
    # Verify only the artifacts consumed here. The manifest's test.csv checksum
    # remains lineage metadata, not permission to inspect the held-out table.
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Missing artifact checksums")
    for name in names[1:]:
        if outputs.get(name) != digest(artifacts[name]):
            raise ValueError(f"Artifact checksum mismatch: {name}")
    split_id = digest(artifacts["split_assignments.csv"])
    if manifest.get("split_sha256") != split_id:
        raise ValueError("Split identity mismatch")
    parameters = manifest.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("Missing preparation parameters")
    # Apply the original parameter constraints, then require every field explicitly:
    # defaults must not conceal an incomplete preparation record.
    config = PrepareConfig(**parameters)
    if set(parameters) != set(asdict(config)):
        raise ValueError("Incomplete preparation parameters")
    raw_hash = manifest.get("raw_sha256")
    if not isinstance(raw_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", raw_hash):
        raise ValueError("Invalid raw dataset identity")
    summary = json_mapping(artifacts["validation_summary.json"])
    if summary.get("status") != "complete":
        raise ValueError("Preparation is not complete")
    # Prepared CSVs encode missing values as empty fields. Preserve identifiers as
    # text until exact parsing, and avoid pandas' broader default missing tokens.
    train = pd.read_csv(io.BytesIO(artifacts["train.csv"]), dtype=str, keep_default_na=False, na_values=[""])
    assignments = pd.read_csv(
        io.BytesIO(artifacts["split_assignments.csv"]), dtype={"subject": str}, keep_default_na=False, na_values=[""]
    )
    # Require the exact exported schemas, including order, before interpreting rows.
    if tuple(train.columns) != EXPORT_COLUMNS or tuple(assignments.columns) != (
        "subject",
        "partition",
        "validation_fold",
    ):
        raise ValueError("Invalid prepared columns")
    train["subject"] = subject_ids(train)
    assignments["subject"] = subject_ids(assignments)
    for column in EXPORT_COLUMNS[1:]:
        train[column] = train[column].astype(float)
    if train.empty or train.duplicated().any() or assignments["subject"].duplicated().any():
        raise ValueError("Empty training data or duplicate rows/assignments")
    # Validate all candidate predictors without fitting preprocessing or changing
    # rows. Missing predictors remain imputable; invalid targets never do.
    FeaturePreparation(validate_features(CANDIDATE_FEATURES)).transform(train)
    target = train["motor_updrs"]
    if not pd.api.types.is_numeric_dtype(target) or not np.isfinite(target.to_numpy(dtype=float)).all():
        raise ValueError("Target must be finite and present")
    # The serialized transformer clips inference times, but prepared inputs must
    # already satisfy that policy; do not silently repair stale training artifacts.
    if bool(train["test_time"].dropna().lt(0).any()):
        raise ValueError("Prepared test_time must already be clipped")
    if not bool(assignments["partition"].isin(["train", "test"]).all()):
        raise ValueError("Invalid partition")
    # Patient-level metadata is enough to establish partition membership without
    # opening test.csv. Duplicate assignment IDs were rejected above.
    saved_train = assignments.loc[assignments["partition"].eq("train")]
    saved_test = assignments.loc[assignments["partition"].eq("test")]
    if saved_test.empty or saved_test["validation_fold"].notna().any():
        raise ValueError("Test assignments must have no validation fold")
    if set(train["subject"]) != set(saved_train["subject"]):
        raise ValueError("Training patients do not match saved assignments")
    fold_values = saved_train["validation_fold"].to_numpy(dtype=float)
    if not np.isfinite(fold_values).all() or set(fold_values) != set(range(config.validation_folds)):
        raise ValueError("Invalid or missing validation folds")
    # Expand the authoritative patient folds into train measurement order once.
    # Every feature configuration will reuse this same row-aligned array.
    mapping = dict(zip(saved_train["subject"], saved_train["validation_fold"], strict=True))
    folds = np.array([mapping[subject] for subject in train["subject"]], dtype=np.int64)
    # Cross-check the preparation report against observed train/assignment counts;
    # a successful status alone does not establish internally consistent artifacts.
    partitions = summary.get("partitions")
    fold_summary = summary.get("folds")
    expected_train = {"rows": len(train), "patients": int(train["subject"].nunique())}
    if not isinstance(partitions, dict) or not isinstance(partitions.get("train"), dict):
        raise ValueError("Missing partition summary")
    if any(partitions["train"].get(key) != value for key, value in expected_train.items()):
        raise ValueError("Training summary mismatch")
    expected_folds = {
        str(fold): {"rows": int((folds == fold).sum()), "patients": int(train.loc[folds == fold, "subject"].nunique())}
        for fold in range(config.validation_folds)
    }
    if fold_summary != expected_folds:
        raise ValueError("Fold summary mismatch")
    # This is a private per-comparison snapshot, not a deeply immutable container.
    # Callers must honor its no-mutation contract for both frames and fold labels.
    return TrainingSnapshot(train, folds, artifacts, digest(artifacts["manifest.json"]), split_id)


@dataclass(frozen=True)
class Metrics:
    mae: float
    rmse: float
    # Undefined R² is represented explicitly in reports, never as a forced score.
    r2: float | None
    r2_reason: str | None


def validate_predictions(predicted: NDArray[np.float64], expected_shape: tuple[int, ...]) -> None:
    """Check prediction structure without calculating unused evaluation scores."""
    if predicted.shape != expected_shape or not np.isfinite(predicted).all():
        raise ValueError("Invalid prediction shape or nonfinite predictions")


def regression_metrics(actual: NDArray[np.float64], predicted: NDArray[np.float64]) -> Metrics:
    validate_predictions(predicted, actual.shape)
    # R² needs at least two observations and nonzero target variance. MAE/RMSE
    # remain useful in those cases, so retain them alongside the undefined reason.
    reason = "fewer_than_two_measurements" if len(actual) < 2 else ("constant_target" if np.ptp(actual) == 0 else None)
    result = Metrics(
        float(mean_absolute_error(actual, predicted)),
        float(root_mean_squared_error(actual, predicted)),
        None if reason else float(r2_score(actual, predicted, force_finite=False)),
        reason,
    )
    # Finite predictions can still overflow during metric arithmetic; such results
    # cannot safely participate in selection or be emitted as portable JSON.
    if not all(np.isfinite(value) for value in (result.mae, result.rmse, result.r2) if value is not None):
        raise ValueError("Nonfinite evaluation metrics")
    return result


@dataclass(frozen=True)
class Evaluation:
    # Keep pooled scores separate from equal-weight fold summaries: unequal fold
    # sizes, and the nonlinear RMSE calculation, make these different statistics.
    pooled: Metrics
    folds: tuple[Metrics, ...]
    fold_summary: dict[str, float | None]
    predictions: NDArray[np.float64]


def evaluate(snapshot: TrainingSnapshot, features: tuple[str, ...]) -> Evaluation:
    """Fit fresh estimators; every measurement gets exactly one held-out prediction."""
    train, folds = snapshot.train, snapshot.folds
    if folds.shape != (len(train),) or not np.isfinite(folds).all():
        raise ValueError("Incomplete out-of-fold assignments")
    target = train["motor_updrs"].to_numpy(dtype=np.float64)
    # Preserve original measurement positions while folds select noncontiguous rows.
    # Track coverage separately so an unfilled slot cannot masquerade as a score.
    predictions = np.full(len(train), np.nan)
    coverage = np.zeros(len(train), dtype=int)
    results: list[Metrics] = []
    for fold in sorted(set(folds)):
        validation = folds == fold
        fitting = ~validation
        if not fitting.any() or set(train.loc[fitting, "subject"]) & set(train.loc[validation, "subject"]):
            raise ValueError("Invalid patient-disjoint evaluation")
        # Rebuild all steps for every fold. Reusing a fitted imputer would leak
        # validation-patient statistics into later fits, even with a fresh regressor.
        pipeline = build_pipeline(features)
        pipeline.fit(train.loc[fitting], target[fitting])
        predicted = np.asarray(pipeline.predict(train.loc[validation]), dtype=np.float64)
        results.append(regression_metrics(target[validation], predicted))
        predictions[validation] = predicted
        coverage[validation] += 1
    if not np.equal(coverage, 1).all():
        raise ValueError("Incomplete out-of-fold predictions")
    # Summaries weight folds equally and use population standard deviation (ddof=0).
    # If any fold has undefined R², do not silently summarize only the valid folds.
    summaries: dict[str, float | None] = {}
    for name in ("mae", "rmse", "r2"):
        values = [getattr(item, name) for item in results]
        for statistic in ("mean", "std"):
            summaries[f"fold_{name}_{statistic}"] = (
                None if None in values else float(np.mean(values) if statistic == "mean" else np.std(values))
            )
    # Calculate the selection score on concatenated held-out predictions, giving
    # each measurement equal weight rather than each patient or each fold.
    return Evaluation(regression_metrics(target, predictions), tuple(results), summaries, predictions)


@dataclass(frozen=True)
class ConfigurationResult:
    # Link comparable validation scores to this configuration's all-train refit.
    name: str
    features: tuple[str, ...]
    run_id: str
    model_uri: str
    metrics: Metrics


@dataclass(frozen=True)
class ComparisonResult:
    # This success-only contract is shared by the CLI and future API/bootstrap
    # callers; it conveys a selection, not registration or deployment approval.
    run_id: str
    configurations: tuple[ConfigurationResult, ...]
    winner: ConfigurationResult
    dataset_id: str
    split_id: str
    evaluation_id: str = EVALUATION_VERSION


def select_winner(results: Sequence[ConfigurationResult]) -> ConfigurationResult:
    # Use exact MAE ties, not a tolerance: prefer baseline, then a smaller feature
    # set, then name so input ordering cannot decide the winner.
    return min(results, key=lambda item: (item.metrics.mae, item.name != "baseline", len(item.features), item.name))


def validate_repository() -> None:
    """Fail early and explicitly when repository-only experimentation is unavailable."""
    if (
        Path(__file__).resolve().parent != ROOT / "src/parkinsons_ml"
        or not (ROOT / "uv.lock").is_file()
        or not (ROOT / "pyproject.toml").is_file()
    ):
        raise ValueError(
            "Training requires a source checkout containing src/parkinsons_ml, uv.lock, and pyproject.toml"
        )
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], cwd=ROOT, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(
            "Training requires Git and a Git source checkout; installed wheels are not supported"
        ) from error
    if Path(result.stdout.strip()).resolve() != ROOT:
        raise ValueError("Training source must belong to its own Git checkout")


def lineage() -> dict[str, object]:
    """Allowlisted hashes only; never upload working-tree contents or environment."""

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()

    # HEAD alone cannot describe uncommitted development code. Include source and
    # dependency hashes plus dirty status without uploading unrelated changed files.
    paths = [ROOT / "uv.lock", ROOT / "pyproject.toml", *sorted((ROOT / "src/parkinsons_ml").glob("*.py"))]
    return {
        "git_revision": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in paths},
    }


def log_metrics(metrics: Metrics, prefix: str, step: int = 0) -> None:
    # MLflow scalars must be numeric; preserve undefined-score reasons in JSON
    # reports instead. Prefixes distinguish fold validation from pooled OOF scores.
    mlflow.log_metrics(
        {
            f"{prefix}{name}": value
            for name, value in asdict(metrics).items()
            if name != "r2_reason" and value is not None
        },
        step=step,
    )


def log_configuration(snapshot: TrainingSnapshot, selection: FeatureSet) -> ConfigurationResult:
    """Run inside a child context so any required failure marks that child failed."""
    mlflow.log_params(
        {
            "features": json.dumps(selection.features),
            "feature_count": len(selection.features),
            "estimator": "LinearRegression",
            "continuous_imputation": "median",
            "sex_imputation": "most_frequent",
            "test_time": "zero_clip",
        }
    )
    evaluation = evaluate(snapshot, selection.features)
    # Saved folds are contiguous and zero-based, so metric steps match fold IDs.
    # Keep all folds in one child run rather than creating another run hierarchy.
    for fold, metrics in enumerate(evaluation.folds):
        log_metrics(metrics, "validation_", fold)
        mlflow.log_metrics(
            {
                "validation_rows": int((snapshot.folds == fold).sum()),
                "validation_patients": int(snapshot.train.loc[snapshot.folds == fold, "subject"].nunique()),
            },
            step=fold,
        )
    log_metrics(evaluation.pooled, "oof_")
    mlflow.log_metrics({key: value for key, value in evaluation.fold_summary.items() if value is not None})
    # Unlike scalar metrics, this report preserves null scores and their reasons.
    mlflow.log_dict(
        {
            "pooled": asdict(evaluation.pooled),
            "folds": [asdict(item) for item in evaluation.folds],
            "fold_summary": evaluation.fold_summary,
        },
        "evaluation.json",
    )
    train = snapshot.train
    # Use snapshot-local measurement positions for alignment, not patient IDs.
    # Grouping assignments stay in separate preparation/audit artifacts.
    predictions = pd.DataFrame(
        {
            "measurement": np.arange(len(train)),
            "fold": snapshot.folds,
            "actual": train["motor_updrs"],
            "prediction": evaluation.predictions,
        }
    )
    mlflow.log_text(predictions.to_csv(index=False), "predictions/oof.csv")
    # Validation is complete for this configuration. Its saved model is a separate
    # all-train refit; fitted values below must never replace held-out selection scores.
    pipeline = build_pipeline(selection.features)
    pipeline.fit(train, train["motor_updrs"])
    # Examples, signatures, and references expose only canonical selected predictors,
    # even though pipeline selection also protects fits on the wider training table.
    inputs = train.loc[:, list(selection.features)]
    expected = np.asarray(pipeline.predict(inputs), dtype=np.float64)
    validate_predictions(expected, (len(train),))
    # Serialize preprocessing together with LinearRegression and package custom code.
    # Keep the skops trust list narrow rather than trusting arbitrary loaded types.
    info = log_model(
        pipeline,
        name="pipeline",
        input_example=inputs.head(5),
        signature=infer_signature(inputs, expected),
        code_paths=[str(ROOT / "src/parkinsons_ml")],
        skops_trusted_types=["numpy.dtype", "parkinsons_ml.training.FeaturePreparation"],
    )
    # Verify the uploaded model URI, not a local pickle or the in-memory estimator.
    # A serialization/loading mismatch fails the child and therefore the comparison.
    loaded = load_model(info.model_uri)
    if not isinstance(loaded, Pipeline):
        raise ValueError("Reloaded model must be a complete pipeline")
    np.testing.assert_allclose(loaded.predict(inputs), expected, rtol=1e-12, atol=1e-12)
    # Retain natural missingness before learned imputation for later drift analysis.
    # Hash the same CSV representation that is uploaded, with no patient identifier.
    reference_csv = inputs.to_csv(index=False)
    mlflow.log_text(reference_csv, "monitoring/reference.csv")
    mlflow.log_dict(
        {
            "representation": "selected-pre-imputation-v1",
            "features": list(selection.features),
            "sha256": digest(reference_csv.encode()),
            "rows": len(inputs),
        },
        "monitoring/identity.json",
    )
    mlflow.log_text(pd.DataFrame({"prediction": expected}).to_csv(index=False), "predictions/refit.csv")
    # Describe our fixed pipeline explicitly instead of filtering a generic deep
    # parameter tree, which silently drops non-primitive settings.
    mlflow.log_dict(
        {
            "estimator": {
                "class": "LinearRegression",
                "parameters": pipeline.named_steps["linearregression"].get_params(deep=False),
            },
            "preprocessing": {
                "features": list(selection.features),
                "continuous_imputation": "median",
                "sex_imputation": "most_frequent",
                "test_time": "zero_clip",
            },
        },
        "pipeline_parameters.json",
    )
    run = mlflow.active_run()
    assert run is not None
    return ConfigurationResult(selection.name, selection.features, run.info.run_id, info.model_uri, evaluation.pooled)


def run_experiment(config: TrainConfig, directory: Path, tracking_uri: str) -> ComparisonResult:
    """Synchronous comparison; no final-test access, registry writes, or partial winner."""
    # TrainConfig validates itself at construction, including direct Python use.
    # Require an explicit server and keep embedded credentials/query values out of
    # URIs that MLflow may persist or display. No local tracking fallback is allowed.
    parsed = urlsplit(tracking_uri)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("An explicit HTTP/HTTPS tracking URI without embedded credentials is required")
    # Plain HTTP is suitable only for same-host loopback traffic. Parse IP literals
    # directly rather than resolving arbitrary DNS names that could later change.
    if parsed.scheme == "http" and parsed.hostname != "localhost":
        try:
            loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            loopback = False
        if not loopback:
            raise ValueError("Remote MLflow servers require HTTPS; HTTP is allowed only for loopback hosts")
    # Fluent MLflow can resume an environment-specified run even without an active
    # context. Refuse both cases rather than taking ownership of a caller's run.
    if mlflow.active_run() is not None or os.environ.get("MLFLOW_RUN_ID"):
        raise ValueError("Cannot start comparison inside an unrelated active run")
    # Repository/Git errors must not leave newly created experiments or failed runs.
    validate_repository()
    # URI setters affect process state and environment variables. Save their prior
    # values so successful comparisons and failures both restore caller routing.
    old_tracking, old_registry = mlflow.get_tracking_uri(), mlflow.get_registry_uri()
    environment = {key: os.environ.get(key) for key in ("MLFLOW_TRACKING_URI", "MLFLOW_REGISTRY_URI")}
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(tracking_uri)
        client = mlflow.MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
        experiment = client.get_experiment_by_name(config.experiment_name)
        experiment_id = experiment.experiment_id if experiment else client.create_experiment(config.experiment_name)
        # HTTP tracking alone is insufficient: an existing experiment could still
        # point clients at a local artifact directory instead of the server proxy.
        if not client.get_experiment(experiment_id).artifact_location.startswith("mlflow-artifacts:/"):
            raise ValueError("Experiment must use HTTP-served mlflow-artifacts storage")
        # Open the diagnostic parent before validating prepared inputs. Exceptions
        # propagate through the contexts, marking runs failed when MLflow is reachable.
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name="comparison",
            tags={"purpose": "training-comparison", "evaluation_id": EVALUATION_VERSION},
        ) as parent:
            mlflow.log_dict(asdict(config), "configuration.json")
            snapshot = load_training_snapshot(directory)
            # Attach the same comparability identities to the parent and every child.
            tags = {
                "dataset_id": snapshot.dataset_id,
                "split_id": snapshot.split_id,
                "schema_version": SCHEMA_VERSION,
                "policy_version": POLICY_VERSION,
                "evaluation_id": EVALUATION_VERSION,
            }
            mlflow.set_tags(tags)
            mlflow.log_dict(lineage(), "lineage.json")
            # Grouping/audit metadata is intentionally separate from prediction artifacts.
            for name, payload in snapshot.artifacts.items():
                if name != "train.csv":
                    mlflow.log_text(payload.decode(), f"preparation/{name}")
            results: list[ConfigurationResult] = []
            # Evaluate/refit/log sequentially against one shared snapshot. Completed
            # children may remain for diagnosis if any later configuration fails.
            for selection in config.feature_sets:
                with mlflow.start_run(experiment_id=experiment_id, nested=True, run_name=selection.name, tags=tags):
                    mlflow.log_metrics(
                        {"train_rows": len(snapshot.train), "train_patients": int(snapshot.train["subject"].nunique())}
                    )
                    results.append(log_configuration(snapshot, selection))
            # Publish a winner only after every required child operation succeeds;
            # never select opportunistically from an incomplete comparison.
            winner = select_winner(results)
            result = ComparisonResult(
                parent.info.run_id, tuple(results), winner, snapshot.dataset_id, snapshot.split_id
            )
            mlflow.log_dict(asdict(result), "comparison.json")
            mlflow.set_tags(
                {"winner": winner.name, "winner_run_id": winner.run_id, "winner_model_uri": winner.model_uri}
            )
            return result
    finally:
        # Restore routing even if server access, fitting, or artifact operations fail.
        # Preserve unset environment variables as unset, not as getter defaults.
        mlflow.set_tracking_uri(old_tracking)
        mlflow.set_registry_uri(old_registry)
        for key, value in environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    # Keep this adapter thin: configuration checks and orchestration are also used
    # directly by Python callers, without depending on argparse or printed output.
    settings = yaml.load(args.params.read_text(), Loader=UniqueKeyLoader)
    if not isinstance(settings, dict):
        raise ValueError("Parameters must contain a train mapping")
    config = training_config(settings.get("train"))
    result = run_experiment(config, args.data, os.environ.get("MLFLOW_TRACKING_URI", ""))
    print(json.dumps(asdict(result), allow_nan=False, indent=2))


if __name__ == "__main__":
    # Keep custom transformer classes importable under their canonical module,
    # rather than serializing a python -m invocation's __main__ definitions.
    from parkinsons_ml.training import main as training_main

    training_main()
