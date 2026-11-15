"""Deterministic preparation with auditable row cleaning and patient splits."""

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
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
    clip_test_time,
    validate_headers,
)

LOGGER = logging.getLogger(__name__)
Row: TypeAlias = dict[str, float]
Event: TypeAlias = dict[str, str | int]


@dataclass(frozen=True)
class PrepareConfig:
    """Parameters participating in the DVC preparation identity."""

    seed: int = 42
    test_fraction: float = 0.20
    validation_folds: int = 5
    extreme_iqr_multiplier: float = 1.5

    def __post_init__(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if not 0 < self.test_fraction < 1:
            raise ValueError("test_fraction must be between zero and one")
        if type(self.validation_folds) is not int or self.validation_folds < 2:
            raise ValueError("validation_folds must be an integer of at least two")
        if not math.isfinite(self.extreme_iqr_multiplier) or self.extreme_iqr_multiplier <= 0:
            raise ValueError("extreme_iqr_multiplier must be positive and finite")


def write_json(path: Path, value: object) -> None:
    """Write portable JSON without nonstandard NaN/Infinity values."""
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def missingness(rows: list[Row]) -> dict[str, int]:
    return {name: sum(math.isnan(row[name]) for row in rows) for name in COLUMNS}


def clean_data(raw: Path) -> tuple[list[Row], list[Event], dict[str, object]]:
    """Parse explicit missing tokens, reject bad rows, then deduplicate and clip."""
    missing_before = dict.fromkeys(COLUMNS, 0)
    retained: list[Row] = []
    events: list[Event] = []
    seen: dict[tuple[float | None, ...], int] = {}
    identifiable_patients: set[float] = set()
    invalid_rows = 0
    duplicate_rows = 0
    input_rows = 0
    with raw.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream, strict=True)
        headers = next(reader, [])
        validate_headers(headers)
        for number, values in enumerate(reader, start=1):
            input_rows += 1
            if len(values) != len(headers):
                events.append({"row": number, "action": "drop", "reason": "row_width", "column": ""})
                invalid_rows += 1
                continue
            row: Row = {}
            reasons: list[tuple[str, str]] = []
            for raw_name, text in zip(headers, values, strict=True):
                name = COLUMN_MAPPING[raw_name]
                token = text.strip()
                if token in MISSING_TOKENS:
                    missing_before[name] += 1
                    value = math.nan
                    if name in ("subject", "motor_updrs"):
                        reasons.append((name, "missing_required"))
                else:
                    try:
                        value = float(token)
                    except ValueError:
                        value = math.nan
                        reasons.append((name, "malformed_numeric"))
                    else:
                        if not math.isfinite(value):
                            reasons.append((name, "nonfinite_numeric"))
                        elif name in ("subject", "age") and not value.is_integer():
                            reasons.append((name, "noninteger_value"))
                        elif name == "sex" and value not in (0, 1):
                            reasons.append((name, "invalid_sex"))
                row[name] = value
            subject = row["subject"]
            if math.isfinite(subject) and subject.is_integer():
                identifiable_patients.add(subject)
            if reasons:
                invalid_rows += 1
                events.extend(
                    {"row": number, "action": "drop", "reason": reason, "column": name} for name, reason in reasons
                )
                continue
            key = tuple(None if math.isnan(row[name]) else row[name] for name in COLUMNS)
            if key in seen:
                duplicate_rows += 1
                events.append(
                    {"row": number, "action": "drop", "reason": "duplicate", "column": "", "duplicate_of": seen[key]}
                )
                continue
            seen[key] = number
            if row["test_time"] < 0:
                events.append({"row": number, "action": "clip", "reason": "negative_test_time", "column": "test_time"})
            row["test_time"] = clip_test_time(row["test_time"])
            retained.append(row)
    patients_after = {row["subject"] for row in retained}
    counts = dict(sorted(Counter(str(event["reason"]) for event in events).items()))
    for reason, count in counts.items():
        LOGGER.warning("Preparation %s: %d events", reason, count)
    summary: dict[str, object] = {
        "input_rows": input_rows,
        "invalid_rows_dropped": invalid_rows,
        "duplicate_rows_dropped": duplicate_rows,
        "unique_rows_dropped": invalid_rows + duplicate_rows,
        "retained_rows": len(retained),
        "clipped_rows": sum(event["action"] == "clip" for event in events),
        "reason_counts": counts,
        "missing_before": missing_before,
        "missing_after": missingness(retained),
        "identifiable_patients_before": len(identifiable_patients),
        "patients_after": len(patients_after),
        "patients_completely_removed": len(identifiable_patients - patients_after),
    }
    return retained, events, summary


def assign_splits(rows: list[Row], config: PrepareConfig) -> list[dict[str, int | str]]:
    """Assign each patient once, with reproducible grouped validation folds."""
    patients = sorted({row["subject"] for row in rows})
    test_count = math.ceil(len(patients) * config.test_fraction)
    if len(patients) - test_count < config.validation_folds or test_count < 1:
        raise ValueError("Insufficient patients for nonempty train/test and requested grouped folds")
    train, test = train_test_split(patients, test_size=config.test_fraction, random_state=config.seed)
    train_patients = set(train)
    groups = [int(row["subject"]) for row in rows if row["subject"] in train_patients]
    folds: dict[int, int] = {}
    splitter = GroupKFold(n_splits=config.validation_folds, shuffle=True, random_state=config.seed)
    for fold, (_, validation) in enumerate(splitter.split(np.zeros(len(groups)), groups=groups)):
        for index in validation:
            folds[groups[int(index)]] = fold
    test_patients = set(test)
    return [
        {
            "subject": int(patient),
            "partition": "test" if patient in test_patients else "train",
            "validation_fold": "" if patient in test_patients else folds[int(patient)],
        }
        for patient in patients
    ]


def extreme_report(train: list[Row], test: list[Row], multiplier: float) -> dict[str, object]:
    """Descriptive flags only; thresholds never learn from test values."""
    report: dict[str, object] = {}
    for name in CANDIDATE_FEATURES:
        if name == "sex":
            continue
        values = [row[name] for row in train if not math.isnan(row[name])]
        if len(values) < 4:
            report[name] = {"status": "insufficient_observations", "train_count": None, "test_count": None}
            continue
        q1, q3 = (float(value) for value in np.quantile(values, [0.25, 0.75]))
        iqr = q3 - q1
        if iqr == 0:
            report[name] = {"status": "zero_iqr", "train_count": None, "test_count": None}
            continue
        lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
        if not math.isfinite(lower) or not math.isfinite(upper):
            report[name] = {"status": "unrepresentable_bounds", "train_count": None, "test_count": None}
            continue
        report[name] = {
            "status": "ok",
            "lower": lower,
            "upper": upper,
            "train_count": sum(row[name] < lower or row[name] > upper for row in train),
            "test_count": sum(row[name] < lower or row[name] > upper for row in test),
        }
    return report


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    """Export float repr precision, with empty fields for missing values."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: "" if isinstance(value, float) and math.isnan(value) else value for key, value in row.items()}
            )


def publish(stage: Path, output: Path) -> None:
    """Swap complete directories; preserve previous outputs if activation fails."""
    backup = output.with_name(f".{output.name}.previous")
    if output.exists():
        os.replace(output, backup)
    try:
        os.replace(stage, output)
    except OSError:
        if backup.exists():
            os.replace(backup, output)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def prepare(raw: Path, output: Path, config: PrepareConfig, failure_report: Path) -> dict[str, object]:
    """Prepare artifacts without depending on MLflow or mutating the raw dataset."""
    summary: dict[str, object] = {}
    events: list[Event] = []
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        backup = output.with_name(f".{output.name}.previous")
        # Reconcile an interruption between either of the publication renames.
        if backup.exists():
            if output.exists():
                shutil.rmtree(backup)
            else:
                os.replace(backup, output)
        raw_hash = checksum(raw)
        rows, events, summary = clean_data(raw)
        assignments = assign_splits(rows, config)
        partitions = {int(item["subject"]): item["partition"] for item in assignments}
        train = [row for row in rows if partitions[int(row["subject"])] == "train"]
        test = [row for row in rows if partitions[int(row["subject"])] == "test"]
        summary["partitions"] = {
            name: {
                "rows": len(partition),
                "patients": len({row["subject"] for row in partition}),
                "row_fraction": len(partition) / len(rows),
                "patient_fraction": len({row["subject"] for row in partition}) / len(assignments),
            }
            for name, partition in (("train", train), ("test", test))
        }
        summary["folds"] = {
            str(fold): {
                "patients": sum(item["validation_fold"] == fold for item in assignments),
                "rows": sum(
                    row["subject"] in {item["subject"] for item in assignments if item["validation_fold"] == fold}
                    for row in train
                ),
            }
            for fold in range(config.validation_folds)
        }
        summary["extremes"] = extreme_report(train, test, config.extreme_iqr_multiplier)
        summary["status"] = "complete"
        with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
            stage = Path(temporary) / "artifacts"
            stage.mkdir()
            write_csv(stage / "train.csv", EXPORT_COLUMNS, [dict(row) for row in train])
            write_csv(stage / "test.csv", EXPORT_COLUMNS, [dict(row) for row in test])
            write_csv(
                stage / "split_assignments.csv",
                ("subject", "partition", "validation_fold"),
                [dict(item) for item in assignments],
            )
            write_csv(
                stage / "cleaning_events.csv",
                ("row", "action", "reason", "column", "duplicate_of"),
                [dict(event) for event in events],
            )
            write_json(stage / "validation_summary.json", summary)
            split_hash = checksum(stage / "split_assignments.csv")
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
                raise ValueError("Raw data changed during preparation")
            publish(stage, output)
    except (ValueError, OSError, csv.Error) as error:
        summary["status"] = "failed"
        failure_report.parent.mkdir(parents=True, exist_ok=True)
        write_json(failure_report, {"status": "failed", "error": str(error), "summary": summary, "events": events})
        LOGGER.error("Preparation failed; diagnostics: %s", failure_report)
        raise
    LOGGER.info("Prepared %d train and %d test rows", len(train), len(test))
    return summary


def main() -> None:
    """Run the local DVC stage with approved repository-relative defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = yaml.safe_load(args.params.read_text(encoding="utf-8"))
    config = PrepareConfig(**settings["prepare"])
    prepare(
        Path("data/raw/parkinsons_updrs.csv"),
        Path("data/processed"),
        config,
        Path("documents/reports/prepare-failure.json"),
    )


if __name__ == "__main__":
    main()
