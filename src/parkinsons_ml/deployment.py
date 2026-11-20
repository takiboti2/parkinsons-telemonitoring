"""Offline bundle delivery and one-shot, fail-closed single-host bootstrap."""

import argparse
import hashlib
import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path

import yaml
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from parkinsons_ml.data import PrepareConfig, prepare
from parkinsons_ml.experimentation import UniqueKeyLoader, load_training_snapshot, training_config
from parkinsons_ml.inference import load_pipeline, resolve_champion
from parkinsons_ml.promotion import champion_version, promotion_lock, reconcile
from parkinsons_ml.tracking import (
    ROOT,
    PROMOTION_LOCK,
    run_experiment,
    tracking_context,
    validate_repository,
    validate_tracking_uri,
)

LOGGER = logging.getLogger(__name__)


def bundle_metadata(root: Path) -> dict[str, object]:
    metadata = json.loads((root / "documents/data-bundle.json").read_bytes())
    dvc_path = root / "dataset.dvc"
    if not dvc_path.exists():
        dvc_path = root / "data/raw/parkinsons_updrs.csv.dvc"
    tracked = yaml.safe_load(dvc_path.read_text())["outs"][0]
    if (
        not isinstance(metadata, dict)
        or metadata.get("format_version") != 1
        or metadata.get("filename") != "parkinsons_updrs.csv"
        or metadata.get("size") != tracked["size"]
        or metadata.get("dvc_md5") != tracked["md5"]
    ):
        raise ValueError("Bundle pin disagrees with DVC metadata")
    return metadata


def verified_data(root: Path, path: Path) -> bytes:
    expected = bundle_metadata(root)
    if path.is_symlink() or not path.is_file() or path.stat().st_size != expected["size"]:
        raise ValueError("Raw dataset is missing, linked, or has an unexpected size")
    payload = path.read_bytes()
    if (
        hashlib.sha256(payload).hexdigest() != expected["sha256"]
        or hashlib.md5(payload, usedforsecurity=False).hexdigest() != expected["dvc_md5"]
    ):
        raise ValueError("Raw dataset checksum does not match the Git/DVC pin")
    return payload


def create_bundle(root: Path, destination: Path) -> None:
    payload = verified_data(root, root / "data/raw/parkinsons_updrs.csv")
    metadata = bundle_metadata(root)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "parkinsons_updrs.csv").write_bytes(payload)
    (destination / "manifest.json").write_text(json.dumps(metadata, sort_keys=True) + "\n")


def install_bundle(root: Path, bundle: Path) -> None:
    expected = bundle_metadata(root)
    if set(path.name for path in bundle.iterdir()) != {"manifest.json", "parkinsons_updrs.csv"}:
        raise ValueError("Bundle must contain exactly the raw CSV and manifest.json")
    if (bundle / "manifest.json").is_symlink() or json.loads((bundle / "manifest.json").read_bytes()) != expected:
        raise ValueError("Bundle manifest differs from the Git-tracked expectation")
    payload = verified_data(root, bundle / "parkinsons_updrs.csv")
    raw = root / "data/raw/parkinsons_updrs.csv"
    if raw.exists() or raw.is_symlink():
        verified_data(root, raw)
        return
    raw.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation never overwrites an existing dataset, including on races.
    with raw.open("xb") as output:
        output.write(payload)


def wait_for_mlflow(client: MlflowClient, attempts: int = 6) -> None:
    for attempt in range(attempts):
        try:
            client.search_experiments(max_results=1)
            return
        except (MlflowException, OSError):
            if attempt == attempts - 1:
                raise RuntimeError("MLflow did not become available") from None
            time.sleep(2)


def bootstrap(root: Path, bundle: Path, uri: str) -> None:
    validate_tracking_uri(uri)
    validate_repository()
    install_bundle(root, bundle)
    parameters = yaml.load((root / "params.yaml").read_text(), Loader=UniqueKeyLoader)
    if not isinstance(parameters, dict) or not isinstance(parameters.get("prepare"), dict):
        raise ValueError("Bootstrap requires preparation parameters")
    preparation = PrepareConfig(**parameters["prepare"])
    config = training_config(parameters.get("train"))
    if not config.promotion.enabled:
        raise ValueError("Bootstrap requires champion promotion to be enabled")
    processed = root / "data/processed"
    if not processed.exists() or not any(processed.iterdir()):
        prepare(root / "data/raw/parkinsons_updrs.csv", processed, preparation, root / "data/prepare-failure.json")
    snapshot = load_training_snapshot(processed)  # never opens or hashes test.csv
    manifest = json.loads(snapshot.artifacts["manifest.json"])
    if manifest["raw_sha256"] != bundle_metadata(root)["sha256"] or manifest["parameters"] != asdict(preparation):
        raise ValueError("Existing processed data differs from the pinned bootstrap configuration")
    with tracking_context(uri) as client:
        wait_for_mlflow(client)
        with promotion_lock(PROMOTION_LOCK):
            reconcile(client, config.promotion.registered_model_name)
            existing = champion_version(client, config.promotion.registered_model_name)
        if existing is not None:
            with load_pipeline(resolve_champion(client, config.promotion.registered_model_name)):
                return
    result = run_experiment(config, processed, uri)
    if result.promotion is None or result.promotion.decision not in {"promoted", "retained"}:
        raise RuntimeError("Bootstrap did not establish a confirmed champion")
    with tracking_context(uri) as client:
        with load_pipeline(resolve_champion(client, config.promotion.registered_model_name)):
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["bundle", "bootstrap"])
    parser.add_argument("--directory", type=Path, default=Path("/bundle"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        if args.action == "bundle":
            create_bundle(ROOT, args.directory)
        else:
            bootstrap(ROOT, args.directory, os.environ["MLFLOW_TRACKING_URI"])
    except Exception as error:
        LOGGER.error("Deployment operation failed (%s)", type(error).__name__)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
