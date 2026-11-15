"""Deterministic preparation with auditable row cleaning and patient splits."""

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal, TypeAlias, TypedDict

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold, train_test_split

from parkinsons_ml.schema import (
    CANDIDATE_FEATURES,
    COLUMN_MAPPING,
    COLUMNS,
    EXPORT_COLUMNS,
    MISSING_TOKENS,
    POLICY_VERSION,
    SCHEMA_VERSION,
    validate_headers,
)

LOGGER = logging.getLogger(__name__)
Event: TypeAlias = dict[str, str | int]


class CleaningSummary(TypedDict):
    """Counts produced by cleaning, before a dataset can necessarily be split."""

    input_rows: int
    invalid_rows_dropped: int
    duplicate_rows_dropped: int
    unique_rows_dropped: int
    retained_rows: int
    clipped_rows: int
    reason_counts: dict[str, int]
    missing_before: dict[str, int]
    missing_after: dict[str, int]
    identifiable_patients_before: int
    patients_after: int
    patients_completely_removed: int


class PartitionSummary(TypedDict):
    """Patient and measurement proportions for one exported partition."""

    rows: int
    patients: int
    row_fraction: float
    patient_fraction: float


class FoldSummary(TypedDict):
    """Validation-patient and measurement counts for a saved fold."""

    patients: int
    rows: int


class ExtremeBounds(TypedDict):
    """Usable train-derived bounds and descriptive counts."""

    status: Literal["ok"]
    lower: float
    upper: float
    train_count: int
    test_count: int


class SkippedExtreme(TypedDict):
    """An explicit reason why no statistical bounds are available."""

    status: Literal["insufficient_observations", "zero_iqr", "unrepresentable_bounds"]
    train_count: None
    test_count: None


ExtremeReport: TypeAlias = dict[str, ExtremeBounds | SkippedExtreme]


class PreparationSummary(CleaningSummary):
    """A complete successful report; failure diagnostics can contain less data."""

    partitions: dict[str, PartitionSummary]
    folds: dict[str, FoldSummary]
    extremes: ExtremeReport
    status: Literal["complete"]


@dataclass(frozen=True)
class PrepareConfig:
    """Parameters participating in the DVC preparation identity."""

    seed: int = 42
    test_fraction: float = 0.20
    validation_folds: int = 5
    extreme_iqr_multiplier: float = 1.5

    def __post_init__(self) -> None:
        # Reject unusable settings before splitting. Explicit integer checks also
        # exclude booleans, which Python otherwise treats as integer values.
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if type(self.test_fraction) not in (int, float) or not 0 < self.test_fraction < 1:
            raise ValueError("test_fraction must be between zero and one")
        if type(self.validation_folds) is not int or self.validation_folds < 2:
            raise ValueError("validation_folds must be an integer of at least two")
        if (
            type(self.extreme_iqr_multiplier) not in (int, float)
            or not math.isfinite(self.extreme_iqr_multiplier)
            or self.extreme_iqr_multiplier <= 0
        ):
            raise ValueError("extreme_iqr_multiplier must be positive and finite")


def write_json(path: Path, value: object) -> None:
    """Write portable JSON without nonstandard NaN/Infinity values."""
    # Stable ordering and no run timestamp allow identical inputs to produce
    # byte-identical artifacts; nonfinite statistics must be handled explicitly.
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def missingness(frame: pd.DataFrame) -> dict[str, int]:
    return {name: int(frame[name].isna().sum()) for name in COLUMNS}


def parse_subject(token: str) -> tuple[int | None, str | None]:
    """Validate patient identifiers without a lossy float conversion."""
    if token in MISSING_TOKENS:
        return None, "missing_required"
    try:
        # Binary64 can merge distinct large identifiers. Decimal checks the exact
        # token before conversion to an arbitrary-precision Python integer.
        value = Decimal(token)
    except InvalidOperation:
        return None, "malformed_numeric"
    if not value.is_finite():
        return None, "nonfinite_numeric"
    if value != value.to_integral_value():
        return None, "noninteger_value"
    return int(value), None


def clean_data(raw: Path) -> tuple[pd.DataFrame, list[Event], CleaningSummary]:
    """Read raw CSV into a DataFrame, then validate and clean with pandas masks."""
    events: list[Event] = []
    try:
        # Keep headers as the first row so pandas cannot silently rename duplicates.
        # Keep literal strings so missing tokens and malformed values stay distinct.
        frame = pd.read_csv(
            raw,
            header=None,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            skip_blank_lines=False,
            engine="python",
            on_bad_lines="error",
            encoding="utf-8",
        )
    except pd.errors.EmptyDataError as error:
        raise ValueError("Invalid raw schema: empty CSV") from error
    headers = frame.iloc[0].tolist()
    validate_headers(headers)
    # Retaining the existing index gives the first data record number 1. Do not
    # reset it after filtering: cleaning events refer back to these source records.
    tokens = frame.iloc[1:].copy()
    tokens.columns = pd.Index(headers)
    input_rows = len(tokens)
    # The Python parser uses None for absent fields, but preserves explicit empty
    # fields as strings. Reject truncated records without losing their row numbers.
    truncated = tokens.isna().any(axis=1)
    width_errors = int(truncated.sum())
    events.extend(
        {"row": int(index), "action": "drop", "reason": "row_width", "column": ""} for index in tokens.index[truncated]
    )
    tokens = tokens.loc[~truncated].rename(columns=COLUMN_MAPPING)
    tokens = pd.DataFrame({name: tokens[name].str.strip() for name in COLUMNS}, index=tokens.index)
    # Save genuine missingness before coercion: a malformed string may also become
    # NaN, but must be dropped rather than silently accepted for later imputation.
    missing = tokens.isin(list(MISSING_TOKENS))
    numeric = pd.DataFrame(tokens.mask(missing).apply(pd.to_numeric, errors="coerce"), dtype=float)
    invalid = pd.Series(False, index=tokens.index)

    def record(mask: pd.Series, column: str, reason: str) -> None:
        # A row can have several audit reasons but is removed only once, using
        # the union of all invalid-value masks.
        nonlocal invalid
        invalid |= mask
        events.extend(
            {"row": int(index), "action": "drop", "reason": reason, "column": column} for index in tokens.index[mask]
        )

    subjects: list[int | None] = []
    for index, token in zip(tokens.index, tokens["subject"], strict=True):
        subject, reason = parse_subject(str(token))
        subjects.append(subject)
        if reason is not None:
            invalid.loc[index] = True
            events.append({"row": int(index), "action": "drop", "reason": reason, "column": "subject"})
    # Python integers in an object column preserve identifiers beyond binary64/int64.
    numeric["subject"] = pd.Series(subjects, index=tokens.index, dtype=object)
    identifiable_patients = set(numeric["subject"].dropna())
    # Validate every supplied numeric column, including the excluded total score.
    # Missing predictors remain valid; only labels and grouping IDs are required.
    for name in COLUMNS:
        if name == "subject":
            continue
        nonfinite = (~missing[name]) & (
            tokens[name].str.fullmatch(r"[+-]?(?:nan|inf(?:inity)?)", case=False) | np.isinf(numeric[name])
        )
        # Distinguish nonfinite spellings from the explicitly accepted NA tokens.
        malformed = numeric[name].isna() & ~missing[name] & ~nonfinite
        record(nonfinite, name, "nonfinite_numeric")
        record(malformed, name, "malformed_numeric")
        if name == "motor_updrs":
            record(pd.Series(missing[name], index=tokens.index), name, "missing_required")
        elif name in ("age", "sex"):
            # Inspect original tokens: float rounding can turn a fractional age
            # or a nearly-binary sex code into a valid-looking integer.
            constrained = ~(missing[name] | nonfinite | malformed)
            invalid_discrete = pd.Series(False, index=tokens.index)
            for index, token in zip(tokens.index[constrained], tokens.loc[constrained, name], strict=True):
                value = Decimal(str(token))
                invalid_discrete.loc[index] = (
                    value != value.to_integral_value() if name == "age" else value not in (Decimal(0), Decimal(1))
                )
            record(invalid_discrete, name, "noninteger_value" if name == "age" else "invalid_sex")
    valid = numeric.loc[~invalid].copy()
    # Compare all 22 columns before dropping total_updrs or clipping times, so
    # distinct repeated measurements are not mistaken for exact duplicate copies.
    duplicates = valid.duplicated(keep="first")
    first_rows = valid.groupby(list(COLUMNS), dropna=False, sort=False)["subject"].transform("idxmin")
    # Equal rows share a subject value; idxmin therefore identifies the first
    # source record in each group. Include missing values in duplicate groups.
    for index in valid.index[duplicates]:
        events.append(
            {
                "row": int(index),
                "action": "drop",
                "reason": "duplicate",
                "column": "",
                "duplicate_of": int(first_rows.loc[index]),
            }
        )
    retained = valid.loc[~duplicates].copy()
    # Record only actions on retained rows. Negative infinity was already invalid;
    # pandas clip preserves missing times for training-fitted imputation.
    for index in retained.index[retained["test_time"].lt(0)]:
        events.append({"row": int(index), "action": "clip", "reason": "negative_test_time", "column": "test_time"})
    retained["test_time"] = retained["test_time"].clip(lower=0)
    events.sort(key=lambda event: (int(event["row"]), str(event["column"]), str(event["reason"])))
    patients_after = set(retained["subject"])
    # Reason totals can overlap, unlike the unique dropped-row totals. Reporting
    # patient loss also exposes cases where cleaning removes every visit of a patient.
    invalid_rows = int(invalid.sum()) + width_errors
    duplicate_rows = int(duplicates.sum())
    counts = dict(sorted(Counter(str(event["reason"]) for event in events).items()))
    for reason, count in counts.items():
        LOGGER.warning("Preparation %s: %d events", reason, count)
    summary: CleaningSummary = {
        "input_rows": input_rows,
        "invalid_rows_dropped": invalid_rows,
        "duplicate_rows_dropped": duplicate_rows,
        "unique_rows_dropped": invalid_rows + duplicate_rows,
        "retained_rows": len(retained),
        "clipped_rows": sum(event["action"] == "clip" for event in events),
        "reason_counts": counts,
        "missing_before": {name: int(missing[name].sum()) for name in COLUMNS},
        "missing_after": missingness(retained),
        "identifiable_patients_before": len(identifiable_patients),
        "patients_after": len(patients_after),
        "patients_completely_removed": len(identifiable_patients - patients_after),
    }
    return retained, events, summary


def assign_splits(rows: pd.DataFrame, config: PrepareConfig) -> pd.DataFrame:
    """Assign each patient once, with reproducible grouped validation folds."""
    # Sort unique IDs before seeded sampling so the patient assignment does not
    # depend on the order in which their measurements appeared in the CSV.
    patients: list[int] = sorted(int(value) for value in rows["subject"].unique())
    test_count = math.ceil(len(patients) * config.test_fraction)
    if len(patients) - test_count < config.validation_folds or test_count < 1:
        raise ValueError("Insufficient patients for nonempty train/test and requested grouped folds")
    train, test = train_test_split(patients, test_size=config.test_fraction, random_state=config.seed)
    # Only train patients participate in validation. The dummy feature array is
    # used for its length; GroupKFold assigns rows according to patient groups.
    groups = rows.loc[rows["subject"].isin(list(train)), "subject"].to_numpy()
    folds: dict[int, int] = {}
    splitter = GroupKFold(n_splits=config.validation_folds, shuffle=True, random_state=config.seed)
    for fold, (_, validation) in enumerate(splitter.split(np.zeros(len(groups)), groups=groups)):
        for index in validation:
            folds[int(groups[int(index)])] = fold
    test_patients = set(test)
    # Export one assignment per patient for both models to reuse. Test patients
    # deliberately have no validation-fold number.
    return pd.DataFrame(
        [
            {
                "subject": int(patient),
                "partition": "test" if patient in test_patients else "train",
                "validation_fold": "" if patient in test_patients else folds[int(patient)],
            }
            for patient in patients
        ]
    )


def extreme_report(train: pd.DataFrame, test: pd.DataFrame, multiplier: float) -> ExtremeReport:
    """Descriptive flags only; thresholds never learn from test values."""
    report: ExtremeReport = {}
    for name in CANDIDATE_FEATURES:
        if name == "sex":
            # Binary category codes are not continuous outlier measurements.
            continue
        values = train[name].dropna()
        # Test observations never influence reference thresholds. Explicit skip
        # statuses avoid presenting undefined or degenerate bounds as real checks.
        if len(values) < 4:
            report[name] = {"status": "insufficient_observations", "train_count": None, "test_count": None}
            continue
        q1, q3 = (float(value) for value in values.quantile([0.25, 0.75]))
        iqr = q3 - q1
        if iqr == 0:
            report[name] = {"status": "zero_iqr", "train_count": None, "test_count": None}
            continue
        lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
        if not math.isfinite(lower) or not math.isfinite(upper):
            report[name] = {"status": "unrepresentable_bounds", "train_count": None, "test_count": None}
            continue
        # The report uses a zero floor for every evaluated numeric feature. This
        # changes descriptive flags only; observed values and upper bounds stay intact.
        lower = max(0.0, lower)
        report[name] = {
            "status": "ok",
            "lower": lower,
            "upper": upper,
            "train_count": int((train[name].lt(lower) | train[name].gt(upper)).sum()),
            "test_count": int((test[name].lt(lower) | test[name].gt(upper)).sum()),
        }
    return report


def write_csv(path: Path, columns: tuple[str, ...], rows: pd.DataFrame) -> None:
    """Export float repr precision, with empty fields for missing values."""
    # Fix the column order and omit the DataFrame index. Source-record references
    # live in the audit artifact rather than becoming accidental model features.
    rows.reindex(columns=list(columns)).to_csv(path, index=False, na_rep="", lineterminator="\n")


def write_failure(
    path: Path,
    error: Exception,
    summary: CleaningSummary | PreparationSummary | None = None,
    events: list[Event] | None = None,
) -> None:
    """Record configuration and preparation failures through the same boundary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        path,
        {
            "status": "failed",
            "error": str(error),
            "summary": {**(summary or {}), "status": "failed"},
            "events": events or [],
        },
    )
    LOGGER.error("Preparation failed; diagnostics: %s", path)


def publish(stage: Path, output: Path) -> None:
    """Publish only a fully written artifact set, without persistent rollback state."""
    # All processing and staging succeeded before this point. Replacing a nonempty
    # directory requires removing it first; failure in this short final window can
    # leave outputs missing or incomplete. Rerun preparation or restore with DVC.
    if output.exists():
        shutil.rmtree(output)
    os.replace(stage, output)


def prepare(raw: Path, output: Path, config: PrepareConfig, failure_report: Path) -> PreparationSummary:
    """Prepare artifacts without depending on MLflow or mutating the raw dataset."""
    # No report exists until cleaning returns. Keep partial failure diagnostics
    # distinct from the fully populated report promised to successful callers.
    summary: CleaningSummary | PreparationSummary | None = None
    events: list[Event] = []
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        raw_hash = checksum(raw)
        rows, events, summary = clean_data(raw)
        assignments = assign_splits(rows, config)
        # Expand patient-level assignments back to measurement rows. No row-level
        # random split is allowed, as that would leak patients across partitions.
        patient_partitions = {
            int(subject): str(partition)
            for subject, partition in zip(assignments["subject"], assignments["partition"], strict=True)
        }
        partitions = pd.Series(rows["subject"], index=rows.index).map(patient_partitions)
        train = rows.loc[partitions.eq("train")].copy()
        test = rows.loc[partitions.eq("test")].copy()
        partition_summary: dict[str, PartitionSummary] = {
            name: {
                "rows": len(partition),
                "patients": int(partition["subject"].nunique()),
                "row_fraction": len(partition) / len(rows),
                "patient_fraction": int(partition["subject"].nunique()) / len(assignments),
            }
            for name, partition in (("train", train), ("test", test))
        }
        fold_summary: dict[str, FoldSummary] = {
            str(fold): {
                "patients": int(assignments["validation_fold"].eq(fold).sum()),
                "rows": int(
                    train["subject"].isin(assignments.loc[assignments["validation_fold"].eq(fold), "subject"]).sum()
                ),
            }
            for fold in range(config.validation_folds)
        }
        report: PreparationSummary = {
            **summary,
            "partitions": partition_summary,
            "folds": fold_summary,
            "extremes": extreme_report(train, test, config.extreme_iqr_multiplier),
            "status": "complete",
        }
        summary = report
        # Build the entire artifact set before publishing any of it. A failed
        # export leaves existing outputs in place and removes temporary files.
        # Temporary-directory cleanup must not turn a committed publication into
        # a reported failure either; leftover scratch files are safe to clean later.
        with tempfile.TemporaryDirectory(
            prefix=f".{output.name}-",
            dir=output.parent,
            ignore_cleanup_errors=True,
        ) as temporary:
            stage = Path(temporary) / "artifacts"
            stage.mkdir()
            write_csv(stage / "train.csv", EXPORT_COLUMNS, train)
            write_csv(stage / "test.csv", EXPORT_COLUMNS, test)
            write_csv(
                stage / "split_assignments.csv",
                ("subject", "partition", "validation_fold"),
                assignments,
            )
            write_csv(
                stage / "cleaning_events.csv",
                ("row", "action", "reason", "column", "duplicate_of"),
                pd.DataFrame(events),
            )
            write_json(stage / "validation_summary.json", report)
            split_hash = checksum(stage / "split_assignments.csv")
            # Hash the other artifacts before writing the manifest: including its
            # own checksum would create a circular dependency.
            write_json(
                stage / "manifest.json",
                {
                    "raw_sha256": raw_hash,
                    "schema_version": SCHEMA_VERSION,
                    "policy_version": POLICY_VERSION,
                    "column_mapping": COLUMN_MAPPING,
                    "missing_tokens": sorted(MISSING_TOKENS),
                    "parameters": asdict(config),
                    "split_sha256": split_hash,
                    "outputs": {path.name: checksum(path) for path in sorted(stage.iterdir())},
                },
            )
            if checksum(raw) != raw_hash:
                # Detect a source change during the run rather than attributing
                # prepared artifacts to a raw-file identity that is no longer valid.
                raise ValueError("Raw data changed during preparation")
            publish(stage, output)
    except (ValueError, OSError) as error:
        # Diagnostics are separate from successful DVC outputs. Re-raise so the
        # caller/DVC sees a failure rather than accepting incomplete preparation.
        write_failure(failure_report, error, summary, events)
        raise
    LOGGER.info("Prepared %d train and %d test rows", len(train), len(test))
    return report


def main() -> None:
    """Run the local DVC stage with approved repository-relative defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    failure_report = Path("documents/reports/prepare-failure.json")
    try:
        settings = yaml.safe_load(args.params.read_text(encoding="utf-8"))
        if not isinstance(settings, dict) or not isinstance(settings.get("prepare"), dict):
            raise ValueError("Parameters must contain a prepare mapping")
        # Validate configuration inside the diagnostic boundary, including unknown
        # keys and incorrect runtime types from YAML, before touching active data.
        config = PrepareConfig(**settings["prepare"])
    except (OSError, ValueError, TypeError, OverflowError, yaml.YAMLError) as error:
        write_failure(failure_report, error)
        raise
    prepare(
        Path("data/raw/parkinsons_updrs.csv"),
        Path("data/processed"),
        config,
        failure_report,
    )


if __name__ == "__main__":
    main()
