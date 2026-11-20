"""Small prepared dataset shared by experimentation and HTTP integration tests."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest
import yaml

from parkinsons_ml import tracking
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


@pytest.fixture
def image_root(tmp_path: Path, prepared_directory: Path) -> Path:
    root = tmp_path / "image"
    (root / "src/parkinsons_ml").mkdir(parents=True)
    for source in (tracking.ROOT / "src/parkinsons_ml").glob("*.py"):
        (root / "src/parkinsons_ml" / source.name).write_bytes(source.read_bytes())
    for name in ("uv.lock", "pyproject.toml", "requirements.txt", "dvc.yaml", "dvc.lock"):
        (root / name).write_bytes((tracking.ROOT / name).read_bytes())
    (root / "params.yaml").write_text(
        yaml.safe_dump(
            {
                "prepare": asdict(PrepareConfig(validation_folds=3)),
                "train": {
                    "experiment_name": "bootstrap-test",
                    "promotion": {
                        "enabled": True,
                        "registered_model_name": "bootstrap-model",
                    },
                },
            }
        )
    )
    payload = (prepared_directory.parent / "raw.csv").read_bytes()
    metadata = {
        "format_version": 1,
        "filename": "parkinsons_updrs.csv",
        "source": "isolated test fixture",
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "dvc_md5": hashlib.md5(payload, usedforsecurity=False).hexdigest(),
    }
    (root / "documents").mkdir()
    (root / "documents/data-bundle.json").write_text(json.dumps(metadata))
    (root / "dataset.dvc").write_text(
        yaml.safe_dump(
            {
                "outs": [{"size": len(payload), "md5": metadata["dvc_md5"]}],
            }
        )
    )
    raw = root / "data/raw"
    raw.mkdir(parents=True)
    (raw / "parkinsons_updrs.csv").write_bytes(payload)
    return root
