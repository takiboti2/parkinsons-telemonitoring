"""Validate prepared inputs and compare pipelines without depending on MLflow."""

import hashlib
import io
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from parkinsons_ml.data import PrepareConfig, parse_subject
from parkinsons_ml.schema import (
    BASELINE_FEATURES,
    CANDIDATE_FEATURES,
    COLUMN_MAPPING,
    EXPORT_COLUMNS,
    MISSING_TOKENS,
    POLICY_VERSION,
    SCHEMA_VERSION,
)
from parkinsons_ml.training import FeaturePreparation, build_pipeline, validate_features

# Change this identity when scoring, weighting, or tie-breaking semantics change;
# downstream promotion must not treat different evaluation policies as comparable.
EVALUATION_VERSION = "pooled-measurement-oof-v1"


@dataclass(frozen=True)
class FeatureSet:
    # Store the validated alphabetical tuple, not the caller's mutable feature list.
    name: str
    features: tuple[str, ...]


@dataclass(frozen=True)
class PromotionConfig:
    """Registry naming only; score comparison and alias policy are fixed in code."""

    enabled: bool = False
    registered_model_name: str = "motor-updrs"

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("promotion.enabled must be a boolean")
        if type(self.registered_model_name) is not str or not re.fullmatch(
            r"[a-zA-Z][a-zA-Z0-9_.-]{0,127}", self.registered_model_name
        ):
            raise ValueError("registered_model_name must be a simple identifier of at most 128 characters")


@dataclass(frozen=True)
class PromotionResult:
    """Keep successful training distinct from registry decisions and failures."""

    decision: Literal["promoted", "retained", "rejected", "disabled", "failed"]
    reason: str
    registered_model_name: str
    run_id: str | None = None
    previous_version: str | None = None
    candidate_version: str | None = None
    previous_mae: float | None = None
    candidate_mae: float | None = None
    confirmed: bool = True


@dataclass(frozen=True)
class TrainConfig:
    # This is the expanded comparison, including the two mandatory configurations.
    experiment_name: str
    feature_sets: tuple[FeatureSet, ...]
    promotion: PromotionConfig = PromotionConfig()

    def __post_init__(self) -> None:
        # Validate the expanded configuration once, whether it comes from YAML or
        # a Python caller. Do not reconstruct a YAML-shaped mapping in orchestration.
        if type(self.promotion) is not PromotionConfig:
            raise ValueError("promotion must be a validated PromotionConfig")
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
    if not isinstance(value, dict) or set(value) - {"experiment_name", "extra_feature_sets", "promotion"}:
        raise ValueError("train must contain only experiment_name, extra_feature_sets, and promotion")
    name = value.get("experiment_name")
    extras = value.get("extra_feature_sets", {})
    if not isinstance(name, str):
        raise ValueError("experiment_name must be a nonempty string")
    if not isinstance(extras, dict):
        raise ValueError("extra_feature_sets must be a mapping")
    promotion = value.get("promotion", {})
    if not isinstance(promotion, dict):
        raise ValueError("promotion must be a mapping")
    try:
        promotion_config = PromotionConfig(**promotion)
    except TypeError as error:
        raise ValueError("Unknown promotion configuration keys") from error
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
    return TrainConfig(name, tuple(selections), promotion_config)


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
    promotion: PromotionResult | None = None


def select_winner(results: Sequence[ConfigurationResult]) -> ConfigurationResult:
    # Use exact MAE ties, not a tolerance: prefer baseline, then a smaller feature
    # set, then name so input ordering cannot decide the winner.
    return min(results, key=lambda item: (item.metrics.mae, item.name != "baseline", len(item.features), item.name))
