"""Bundle integrity, image provenance and bootstrap failure contracts."""

import sys
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from mlflow.tracking import MlflowClient

from parkinsons_ml import deployment, provenance, tracking


def test_verified_bundle_roundtrip_preserves_existing_data(image_root: Path, tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    deployment.create_bundle(image_root, bundle)
    raw = image_root / "data/raw/parkinsons_updrs.csv"
    payload = raw.read_bytes()
    raw.unlink()
    deployment.install_bundle(image_root, bundle)
    assert raw.read_bytes() == payload
    modified = raw.stat().st_mtime_ns
    deployment.install_bundle(image_root, bundle)
    assert raw.stat().st_mtime_ns == modified
    raw.write_bytes(b"different existing data")
    with pytest.raises(ValueError):
        deployment.install_bundle(image_root, bundle)
    assert raw.read_bytes() == b"different existing data"


@pytest.mark.parametrize("problem", ["data", "manifest", "extra", "link", "size", "dvc"])
def test_bundle_rejection(image_root: Path, tmp_path: Path, problem: str) -> None:
    bundle = tmp_path / "bundle"
    deployment.create_bundle(image_root, bundle)
    (image_root / "data/raw/parkinsons_updrs.csv").unlink()
    raw = bundle / "parkinsons_updrs.csv"
    if problem == "data":
        payload = bytearray(raw.read_bytes())
        payload[-1] ^= 1
        raw.write_bytes(payload)
    elif problem == "manifest":
        (bundle / "manifest.json").write_text("{}")
    elif problem == "extra":
        (bundle / "unexpected.csv").write_text("unexpected")
    elif problem == "link":
        raw.unlink()
        raw.symlink_to(image_root / "dataset.dvc")
    elif problem == "size":
        raw.write_text("short")
    else:
        (image_root / "dataset.dvc").write_text("outs:\n- md5: wrong\n  size: 1\n")
    with pytest.raises(ValueError):
        deployment.install_bundle(image_root, bundle)


@pytest.mark.parametrize(
    "problem",
    [
        "none",
        "revision",
        "file",
        "extra_source",
        "manifest",
        "nested_source",
        "nested_changed",
        "deployment",
    ],
)
def test_image_provenance_binds_files(image_root: Path, problem: str) -> None:
    if problem == "revision":
        with pytest.raises(ValueError):
            provenance.write_provenance(image_root, "not-a-revision")
        return
    provenance.write_provenance(image_root, "a" * 40)
    if problem == "nested_changed":
        nested = image_root / "src/parkinsons_ml/package"
        nested.mkdir()
        (nested / "__init__.py").write_text("original")
        provenance.write_provenance(image_root, "a" * 40)
        (nested / "__init__.py").write_text("changed")
    elif problem == "nested_source":
        nested = image_root / "src/parkinsons_ml/package"
        nested.mkdir()
        (nested / "__init__.py").write_text("extra")
    elif problem == "deployment":
        (image_root / "compose.yaml").write_text("changed runtime")
    if problem == "file":
        (image_root / "params.yaml").write_text("changed")
    elif problem == "extra_source":
        (image_root / "src/parkinsons_ml/extra.py").write_text("changed")
    elif problem == "manifest":
        (image_root / "build-provenance.json").write_text("{}")
    if problem == "none":
        result = provenance.image_lineage(image_root)
        assert result["source"] == "image-build" and result["build_revision"] == "a" * 40
        assert "git_dirty" not in result
    else:
        with pytest.raises(ValueError):
            provenance.image_lineage(image_root)


def test_tracking_accepts_verified_image_without_git(image_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provenance.write_provenance(image_root, "a" * 40)
    monkeypatch.setattr(tracking, "ROOT", image_root)
    monkeypatch.setattr(tracking, "__file__", str(image_root / "src/parkinsons_ml/tracking.py"))
    tracking.validate_repository()
    assert tracking.lineage() == provenance.image_lineage(image_root)
    (image_root / "uv.lock").write_text("tampered")
    with pytest.raises(ValueError, match="provenance"):
        tracking.validate_repository()


@pytest.mark.parametrize("bundle_state", ["missing", "empty", "modified"])
def test_verified_persistent_raw_does_not_require_bundle(
    image_root: Path,
    tmp_path: Path,
    bundle_state: str,
) -> None:
    bundle = tmp_path / "delivery"
    if bundle_state != "missing":
        bundle.mkdir()
        if bundle_state == "modified":
            (bundle / "manifest.json").write_text("invalid")
    raw = image_root / "data/raw/parkinsons_updrs.csv"
    before = raw.read_bytes(), raw.stat().st_mtime_ns
    deployment.install_bundle(image_root, bundle)
    assert (raw.read_bytes(), raw.stat().st_mtime_ns) == before
    raw.write_text("corrupt persistent state")
    with pytest.raises(ValueError, match="size"):
        deployment.install_bundle(image_root, bundle)


def test_checkout_ignores_stale_image_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    # Real checkouts (including worktrees with a .git file) keep Git validation
    # and lineage even if someone has generated an obsolete image manifest.
    monkeypatch.setattr(tracking, "image_lineage", MagicMock(side_effect=AssertionError("Must use Git")))
    original_exists = Path.exists

    def exists(path: Path) -> bool:
        return path == tracking.ROOT / "build-provenance.json" or original_exists(path)

    monkeypatch.setattr(Path, "exists", exists)
    tracking.validate_repository()
    assert "git_revision" in tracking.lineage()


@pytest.mark.parametrize(
    "error",
    [
        ValueError("https://user:secret@example.org/private"),
        RuntimeError("private path and credentials"),
    ],
)
def test_cli_does_not_echo_external_exception_messages(
    error: Exception,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(sys, "argv", ["deployment", "bundle"])
    monkeypatch.setattr(deployment, "create_bundle", MagicMock(side_effect=error))
    with pytest.raises(SystemExit):
        deployment.main()
    assert type(error).__name__ in caplog.text
    assert str(error) not in caplog.text


@pytest.mark.parametrize(
    "uri, allowed",
    [
        ("http://mlflow:5000", True),
        ("http://MLFLOW:5000", True),
        ("http://mlflow:5001", False),
        ("http://mlflow.evil:5000", False),
        ("http://example.org:5000", False),
        ("http://10.0.0.1:5000", False),
        ("http://user:secret@mlflow:5000", False),
    ],
)
def test_http_exception_is_exact_and_explicit(uri: str, allowed: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARKINSONS_HTTP_MLFLOW_HOSTS", "mlflow:5000")
    if allowed:
        tracking.validate_tracking_uri(uri)
    else:
        with pytest.raises(ValueError):
            tracking.validate_tracking_uri(uri)
    monkeypatch.delenv("PARKINSONS_HTTP_MLFLOW_HOSTS")
    with pytest.raises(ValueError):
        tracking.validate_tracking_uri(uri)


@pytest.mark.parametrize("allowlist", ["*", "mlflow", "10.0.0.1:5000", "mlflow:99999", "mlflow:0"])
def test_invalid_http_exception_rejected(allowlist: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARKINSONS_HTTP_MLFLOW_HOSTS", allowlist)
    with pytest.raises(ValueError, match="allowlist"):
        tracking.validate_tracking_uri("http://mlflow:5000")


def test_bounded_startup_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock(spec=MlflowClient)
    client.search_experiments.side_effect = [ConnectionError(), []]
    sleep = MagicMock()
    monkeypatch.setattr(deployment.time, "sleep", sleep)
    deployment.wait_for_mlflow(client, attempts=2)
    sleep.assert_called_once_with(2)
    client.search_experiments.side_effect = ConnectionError()
    with pytest.raises(RuntimeError, match="available"):
        deployment.wait_for_mlflow(client, attempts=2)


def test_bundle_cli_and_sanitized_failure(
    image_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(deployment, "ROOT", image_root)
    destination = tmp_path / "cli-bundle"
    monkeypatch.setattr(sys, "argv", ["deployment", "bundle", "--directory", str(destination)])
    deployment.main()
    assert (destination / "manifest.json").is_file()
    with pytest.raises(SystemExit):
        deployment.main()
    assert "Bundle destination already exists" in caplog.text and str(destination) not in caplog.text


def test_provenance_cli(image_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provenance, "__file__", str(image_root / "src/parkinsons_ml/provenance.py"))
    monkeypatch.setattr(sys, "argv", ["provenance", "--revision", "a" * 40])
    provenance.main()
    assert provenance.image_lineage(image_root)["build_revision"] == "a" * 40


@pytest.mark.parametrize("problem", ["parameters", "disabled", "processed_config", "partial"])
def test_bootstrap_fails_before_registry_for_invalid_local_state(
    image_root: Path,
    prepared_directory: Path,
    tmp_path: Path,
    problem: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = tmp_path / "bundle"
    deployment.create_bundle(image_root, bundle)
    monkeypatch.setattr(deployment, "validate_repository", MagicMock())
    registry = MagicMock(side_effect=AssertionError("Invalid local state must not contact MLflow"))
    monkeypatch.setattr(deployment, "tracking_context", registry)
    parameters = yaml.safe_load((image_root / "params.yaml").read_text())
    if problem == "parameters":
        (image_root / "params.yaml").write_text("[]")
    elif problem == "disabled":
        parameters["train"]["promotion"]["enabled"] = False
        (image_root / "params.yaml").write_text(yaml.safe_dump(parameters))
    elif problem == "processed_config":
        shutil.copytree(prepared_directory, image_root / "data/processed")
        parameters["prepare"]["seed"] = 999
        (image_root / "params.yaml").write_text(yaml.safe_dump(parameters))
    else:
        (image_root / "data/processed").mkdir()
        (image_root / "data/processed/leftover.csv").write_text("partial")
    with pytest.raises((ValueError, OSError)):
        deployment.bootstrap(image_root, bundle, "http://localhost:5000")
    registry.assert_not_called()
