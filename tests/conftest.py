"""Small prepared dataset shared by experimentation and HTTP integration tests."""

from pathlib import Path

import pandas as pd
import pytest

from parkinsons_ml.data import PrepareConfig, prepare
from parkinsons_ml.schema import RAW_COLUMNS


@pytest.fixture
def prepared_directory(tmp_path: Path) -> Path:
    rows = [
        [patient, 50 + patient, patient % 2, visit * 7, 10 + patient + visit / 10, 20 + patient]
        + [0.01 * (patient + visit + index) for index in range(16)]
        for patient in range(1, 13)
        for visit in range(2)
    ]
    raw = tmp_path / "raw.csv"
    pd.DataFrame(rows, columns=pd.Index(RAW_COLUMNS)).to_csv(raw, index=False)
    output = tmp_path / "processed"
    prepare(raw, output, PrepareConfig(validation_folds=3), tmp_path / "failure.json")
    return output
