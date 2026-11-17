"""CSV ingestion must preserve schema and audit semantics before cleaning."""

from pathlib import Path

import pandas as pd
import pytest

from parkinsons_ml.data import clean_data
from parkinsons_ml.schema import RAW_COLUMNS


@pytest.mark.parametrize("headers", [RAW_COLUMNS[:-1], RAW_COLUMNS + ("extra",), RAW_COLUMNS + ("age",)])
def test_header_validation_precedes_cleaning(tmp_path: Path, headers: tuple[str, ...]) -> None:
    raw = tmp_path / "raw.csv"
    raw.write_text(",".join(headers) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid raw schema"):
        clean_data(raw)


def test_empty_file_fails_loudly(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    raw.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid raw schema: empty CSV"):
        clean_data(raw)


def test_extra_record_fields_are_not_silently_discarded(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    raw.write_text(",".join(RAW_COLUMNS) + "\n" + ",".join(["1"] * 23) + "\n", encoding="utf-8")
    with pytest.raises(pd.errors.ParserError):
        clean_data(raw)


def test_literal_missing_fields_differ_from_truncated_records(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    values = ["1", "60", "0", "-1", "12.5", "20"] + ["0.01"] * 16
    values[-1] = ""
    raw.write_text(
        ",".join(RAW_COLUMNS) + "\n" + ",".join(values) + "\n" + ",".join(values[:-1]) + "\n",
        encoding="utf-8",
    )
    frame, events, summary = clean_data(raw)
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.index) == [1]
    assert frame[["ppe"]].isna().to_numpy().all()
    assert frame[["test_time"]].eq(0).to_numpy().all()
    assert summary["input_rows"] == 2
    assert summary["invalid_rows_dropped"] == 1
    assert {"row": 2, "action": "drop", "reason": "row_width", "column": ""} in events
