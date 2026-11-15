"""Preparation behavior, failure safety and reproducibility."""

import csv
import hashlib
import json
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from parkinsons_ml import data
from parkinsons_ml.data import PrepareConfig, assign_splits, clean_data, extreme_report, prepare
from parkinsons_ml.schema import (
    CANDIDATE_FEATURES,
    COLUMN_MAPPING,
    COLUMNS,
    RAW_COLUMNS,
    clip_test_time,
    to_snake_case,
    validate_headers,
)


@pytest.fixture
def records() -> list[dict[str, object]]:
    """Distinct measurements with enough patients for five training folds."""
    return [
        dict(
            zip(
                RAW_COLUMNS,
                [patient, 50 + patient, patient % 2, visit * 7 - 1, 10 + patient + visit / 10, 20 + patient]
                + [0.01 * (patient + visit + index) for index in range(16)],
                strict=True,
            )
        )
        for patient in range(1, 13)
        for visit in range(2)
    ]


def save_raw(path: Path, records: list[dict[str, object]], headers: tuple[str, ...] = RAW_COLUMNS) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    return path


def snapshot(path: Path) -> dict[str, bytes]:
    return {item.name: item.read_bytes() for item in path.iterdir()}


def run_prepare(tmp_path: Path, raw: Path, config: PrepareConfig = PrepareConfig()) -> dict[str, object]:
    return prepare(raw, tmp_path / "processed", config, tmp_path / "reports" / "failure.json")


def test_schema_mapping_and_failures() -> None:
    assert to_snake_case(" Éxample (Value) ") == "example_value"
    assert COLUMN_MAPPING["subject#"] == "subject"
    assert COLUMN_MAPPING["Jitter(%)"] == "jitter"
    assert len(set(COLUMNS)) == 22
    validate_headers(list(reversed(RAW_COLUMNS)))
    for headers in (RAW_COLUMNS[:-1], RAW_COLUMNS + ("other",), RAW_COLUMNS + ("age",), RAW_COLUMNS + ("motor updrs",)):
        with pytest.raises(ValueError, match="Invalid raw schema: missing=.*unexpected=.*duplicates="):
            validate_headers(headers)
    assert math.isnan(clip_test_time(math.nan))
    assert clip_test_time(-1) == 0
    with pytest.raises(ValueError, match="Infinite"):
        clip_test_time(-math.inf)


def test_cleaning_reasons_missingness_and_audit(tmp_path: Path, records: list[dict[str, object]], caplog) -> None:
    records[0]["age"] = "not-a-number"
    records[0]["sex"] = 3
    records[1]["subject#"] = ""
    records[2]["motor_UPDRS"] = "NA"
    records[3]["subject#"] = 2.5
    records[4]["age"] = 52.5
    records[5]["NHR"] = "inf"
    records[6]["HNR"] = "-inf"
    records[7]["PPE"] = "NAN"  # Not an accepted missing token.
    records[8]["age"] = "NA"
    records[8]["sex"] = "null"
    records[8]["test_time"] = ""
    records[8]["Jitter(%)"] = "NaN"
    records[8]["total_UPDRS"] = ""
    records.append(dict(records[8]))
    records.append(dict(records[10], test_time=-2))  # Clipping makes these identical; keep both.
    raw = save_raw(tmp_path / "raw.csv", records)
    with raw.open("a", encoding="utf-8") as stream:
        stream.write("1,2\n")
    with caplog.at_level(logging.WARNING):
        rows, events, report = clean_data(raw)
    assert report["input_rows"] == 27
    assert report["invalid_rows_dropped"] == 9
    assert report["duplicate_rows_dropped"] == 1
    assert report["retained_rows"] == 17
    assert report["unique_rows_dropped"] == 10
    assert report["patients_completely_removed"] == 4
    assert report["missing_before"]["age"] == 2  # Malformed numeric is not missing.
    assert report["missing_after"]["age"] == 1
    assert report["reason_counts"]["invalid_sex"] == 1
    assert report["reason_counts"]["nonfinite_numeric"] == 3
    assert "malformed_numeric" in caplog.text
    duplicate = next(event for event in events if event["reason"] == "duplicate")
    assert duplicate["duplicate_of"] == 9
    assert all(row["test_time"] >= 0 or math.isnan(row["test_time"]) for row in rows)
    assert sum(row["subject"] == 6 and row["test_time"] == 0 for row in rows) == 2
    assert sum(math.isnan(row["sex"]) for row in rows) == 1


@pytest.mark.parametrize("token", ["", "NA", "N/A", "NaN", "nan", "NULL", "null", " NA "])
def test_missing_predictors_preserved(tmp_path: Path, records: list[dict[str, object]], token: str) -> None:
    records[0]["PPE"] = token
    rows, _, summary = clean_data(save_raw(tmp_path / "raw.csv", records))
    assert math.isnan(rows[0]["ppe"])
    assert summary["retained_rows"] == len(records)


def test_complete_preparation_reproducible_and_disjoint(tmp_path: Path, records: list[dict[str, object]]) -> None:
    records[0]["PPE"] = ""
    raw = save_raw(tmp_path / "raw.csv", records, tuple(reversed(RAW_COLUMNS)))
    original = raw.read_bytes()
    report = run_prepare(tmp_path, raw)
    output = tmp_path / "processed"
    first = snapshot(output)
    run_prepare(tmp_path, raw)
    assert snapshot(output) == first
    assert raw.read_bytes() == original
    train = pd.read_csv(output / "train.csv", float_precision="round_trip")
    test = pd.read_csv(output / "test.csv", float_precision="round_trip")
    assignments = pd.read_csv(output / "split_assignments.csv")
    assert set(train.subject).isdisjoint(test.subject)
    assert len(train) + len(test) == 24
    assert assignments.subject.is_unique
    assert set(assignments.query("partition == 'train'").validation_fold) == set(range(5))
    assert assignments.query("partition == 'test'").validation_fold.isna().all()
    assert len(assignments.query("partition == 'test'")) == 3
    assert not {"total_updrs", "validation_fold"} & set(train.columns)
    assert report["status"] == "complete"
    assert report["partitions"]["train"]["patients"] == 9
    assert sum(fold["rows"] for fold in report["folds"].values()) == len(train)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["raw_sha256"] == hashlib.sha256(original).hexdigest()
    assert manifest["column_mapping"] == COLUMN_MAPPING
    for name, digest in manifest["outputs"].items():
        assert hashlib.sha256(first[name]).hexdigest() == digest
    rows, _, _ = clean_data(raw)
    for frame in (train, test):
        expected = pd.DataFrame([row for row in rows if row["subject"] in set(frame.subject)])
        np.testing.assert_allclose(
            frame[list(CANDIDATE_FEATURES)], expected[list(CANDIDATE_FEATURES)], rtol=0, atol=0, equal_nan=True
        )
    assert not (tmp_path / "reports" / "failure.json").exists()


def test_extremes_only_learn_from_train(tmp_path: Path, records: list[dict[str, object]]) -> None:
    rows, _, _ = clean_data(save_raw(tmp_path / "raw.csv", records))
    report = extreme_report(rows[:20], rows[20:], 1.5)
    changed = [dict(row, ppe=1e100) for row in rows[20:]]
    shifted = extreme_report(rows[:20], changed, 1.5)
    assert report["ppe"]["lower"] == shifted["ppe"]["lower"]
    assert report["ppe"]["upper"] == shifted["ppe"]["upper"]
    assert shifted["ppe"]["test_count"] == 4
    assert extreme_report(rows[:2], [], 1.5)["age"]["status"] == "insufficient_observations"
    constant = [dict(rows[0]) for _ in range(5)]
    assert extreme_report(constant, rows, 1.5)["age"]["status"] == "zero_iqr"
    overflowing = [dict(rows[0], ppe=value) for value in (-1e308, -1e308, 1e308, 1e308)]
    with np.errstate(over="ignore", invalid="ignore"):
        assert extreme_report(overflowing, [], 1.5)["ppe"]["status"] == "unrepresentable_bounds"


@pytest.mark.parametrize(
    "settings",
    [
        {"seed": -1},
        {"seed": True},
        {"seed": 2**32},
        {"test_fraction": 0},
        {"test_fraction": 1},
        {"test_fraction": math.nan},
        {"validation_folds": 1},
        {"validation_folds": 2.5},
        {"extreme_iqr_multiplier": 0},
        {"extreme_iqr_multiplier": math.inf},
    ],
)
def test_invalid_parameters(settings: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        PrepareConfig(**settings)


def test_insufficient_data_preserves_outputs_and_diagnostics(tmp_path: Path, records: list[dict[str, object]]) -> None:
    raw = save_raw(tmp_path / "raw.csv", records)
    run_prepare(tmp_path, raw)
    before = snapshot(tmp_path / "processed")
    for record in records:
        record["motor_UPDRS"] = ""
    save_raw(raw, records)
    with pytest.raises(ValueError, match="Insufficient patients"):
        run_prepare(tmp_path, raw)
    assert snapshot(tmp_path / "processed") == before
    failure = json.loads((tmp_path / "reports" / "failure.json").read_text())
    assert failure["status"] == "failed"
    assert failure["summary"]["retained_rows"] == 0
    assert len(failure["events"]) == 24
    with pytest.raises(ValueError, match="Insufficient"):
        assign_splits([{"subject": 1.0}], PrepareConfig())


def test_bad_schema_and_csv_produce_failure_reports(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    raw.write_text("bad,column\n1,2\n")
    with pytest.raises(ValueError, match="Invalid raw schema"):
        run_prepare(tmp_path, raw)
    assert json.loads((tmp_path / "reports" / "failure.json").read_text())["summary"] == {"status": "failed"}
    raw.write_text(",".join(RAW_COLUMNS) + '\n"unterminated\n')
    with pytest.raises(csv.Error):
        run_prepare(tmp_path, raw)


def test_publication_failure_rolls_back(tmp_path: Path, records: list[dict[str, object]], monkeypatch) -> None:
    raw = save_raw(tmp_path / "raw.csv", records)
    run_prepare(tmp_path, raw)
    before = snapshot(tmp_path / "processed")
    replace = os.replace

    def fail_activation(source: Path, destination: Path) -> None:
        if source.name == "artifacts":
            raise OSError("activation failed")
        replace(source, destination)

    monkeypatch.setattr(data.os, "replace", fail_activation)
    with pytest.raises(OSError, match="activation failed"):
        run_prepare(tmp_path, raw)
    assert snapshot(tmp_path / "processed") == before
    assert not (tmp_path / ".processed.previous").exists()
    assert not list(tmp_path.glob(".processed-*"))
    monkeypatch.undo()
    output = tmp_path / "new-output"
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "marker").write_text("complete")
    data.publish(stage, output)
    assert (output / "marker").read_text() == "complete"


@pytest.mark.parametrize("activated", [True, False])
def test_interrupted_publication_reconciled(tmp_path: Path, records: list[dict[str, object]], activated: bool) -> None:
    raw = save_raw(tmp_path / "raw.csv", records)
    run_prepare(tmp_path, raw)
    output = tmp_path / "processed"
    before = snapshot(output)
    output.rename(tmp_path / ".processed.previous")
    if activated:
        output.mkdir()
        for name, content in before.items():
            (output / name).write_bytes(content)
    raw.write_text("invalid\n")
    with pytest.raises(ValueError):
        run_prepare(tmp_path, raw)
    assert snapshot(output) == before
    assert not (tmp_path / ".processed.previous").exists()


def test_raw_change_during_preparation_aborts(tmp_path: Path, records: list[dict[str, object]], monkeypatch) -> None:
    raw = save_raw(tmp_path / "raw.csv", records)
    original_publish = data.extreme_report

    def change_raw(*args):
        raw.write_text("changed")
        return original_publish(*args)

    monkeypatch.setattr(data, "extreme_report", change_raw)
    with pytest.raises(ValueError, match="Raw data changed"):
        run_prepare(tmp_path, raw)
    assert not (tmp_path / "processed").exists()


def test_prepare_entrypoint(tmp_path: Path, records: list[dict[str, object]], monkeypatch) -> None:
    save_raw(tmp_path / "data/raw/parkinsons_updrs.csv", records)
    (tmp_path / "params.yaml").write_text("prepare: {seed: 7, validation_folds: 5}\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["prepare"])
    data.main()
    assert json.loads((tmp_path / "data/processed/manifest.json").read_text())["parameters"]["seed"] == 7
