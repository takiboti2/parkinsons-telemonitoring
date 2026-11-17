"""Configuration, prepared-snapshot integrity and grouped evaluation contracts."""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from parkinsons_ml import training
from parkinsons_ml.schema import BASELINE_FEATURES, CANDIDATE_FEATURES
from parkinsons_ml.training import (
    ConfigurationResult,
    FeatureSet,
    Metrics,
    TrainConfig,
    digest,
    evaluate,
    load_training_snapshot,
    regression_metrics,
    run_experiment,
    select_winner,
    training_config,
)


@pytest.mark.parametrize(
    "settings",
    [
        None,
        [],
        {},
        {"experiment_name": " "},
        {"experiment_name": 4},
        {"experiment_name": "test", "unknown": True},
        {"experiment_name": "test", "extra_feature_sets": []},
        *[
            {"experiment_name": "test", "extra_feature_sets": {name: ["age"]}}
            for name in ("baseline", "candidate", "Invalid name", "", 1)
        ],
        {"experiment_name": "test", "extra_feature_sets": {"extra": list(reversed(BASELINE_FEATURES))}},
        {"experiment_name": "test", "extra_feature_sets": {"a": ["ppe"], "b": ["ppe"]}},
        {"experiment_name": "test", "extra_feature_sets": {"a": ["ppe", "ppe"]}},
    ],
)
def test_invalid_configuration(settings: object) -> None:
    with pytest.raises(ValueError):
        training_config(settings)


def test_canonical_configuration() -> None:
    result = training_config({"experiment_name": "test", "extra_feature_sets": {"voice": ["sex", "ppe"]}})
    assert [item.name for item in result.feature_sets] == ["baseline", "candidate", "voice"]
    assert result.feature_sets[1].features == tuple(sorted(CANDIDATE_FEATURES))
    assert result.feature_sets[2].features == ("ppe", "sex")


def rewrite_manifest(directory: Path, name: str, payload: bytes) -> None:
    (directory / name).write_bytes(payload)
    path = directory / "manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["outputs"][name] = digest(payload)
    if name == "split_assignments.csv":
        manifest["split_sha256"] = digest(payload)
    path.write_text(json.dumps(manifest))


def test_snapshot_never_reads_test(prepared_directory: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path.read_bytes

    def read(path: Path) -> bytes:
        assert path.name != "test.csv"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    snapshot = load_training_snapshot(prepared_directory)
    assert len(snapshot.train) == 18
    assert set(snapshot.folds) == {0, 1, 2}
    assert snapshot.split_id == digest(snapshot.artifacts["split_assignments.csv"])
    assert snapshot.dataset_id == digest(snapshot.artifacts["manifest.json"])


@pytest.mark.parametrize(
    "change",
    [
        "schema_version",
        "policy_version",
        "column_mapping",
        "missing_tokens",
        "outputs",
        "parameters",
        "split_sha256",
        "raw_sha256",
    ],
)
def test_invalid_manifest(prepared_directory: Path, change: str) -> None:
    path = prepared_directory / "manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest[change] = None
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_training_snapshot(prepared_directory)


def test_corrupt_artifact(prepared_directory: Path) -> None:
    (prepared_directory / "train.csv").write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_training_snapshot(prepared_directory)


@pytest.mark.parametrize("change", ["status", "partitions", "counts", "folds"])
def test_invalid_summary(prepared_directory: Path, change: str) -> None:
    summary = json.loads((prepared_directory / "validation_summary.json").read_bytes())
    if change == "counts":
        summary["partitions"]["train"]["rows"] += 1
    else:
        summary[change] = None
    rewrite_manifest(prepared_directory, "validation_summary.json", json.dumps(summary).encode())
    with pytest.raises(ValueError):
        load_training_snapshot(prepared_directory)


@pytest.mark.parametrize(
    "change", ["columns", "subject", "duplicate", "empty", "target", "negative_time", "sex", "infinite", "age"]
)
def test_invalid_training_data(prepared_directory: Path, change: str) -> None:
    frame = pd.read_csv(prepared_directory / "train.csv")
    if change == "columns":
        frame = frame.drop(columns="age")
    elif change == "subject":
        frame["subject"] = frame["subject"].astype(str)
        frame.loc[0, "subject"] = "1.5"
    elif change == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif change == "empty":
        frame = frame.iloc[:0]
    else:
        column, value = {
            "target": ("motor_updrs", np.nan),
            "negative_time": ("test_time", -1.0),
            "sex": ("sex", 2.0),
            "infinite": ("ppe", np.inf),
            "age": ("age", 2.5),
        }[change]
        frame[column] = frame[column].astype(float)
        frame.loc[0, column] = value
    rewrite_manifest(prepared_directory, "train.csv", frame.to_csv(index=False).encode())
    with pytest.raises(ValueError):
        load_training_snapshot(prepared_directory)


@pytest.mark.parametrize("change", ["columns", "duplicate", "partition", "test_fold", "missing", "fractional", "nan"])
def test_invalid_assignments(prepared_directory: Path, change: str) -> None:
    frame = pd.read_csv(prepared_directory / "split_assignments.csv")
    if change == "columns":
        frame = frame.drop(columns="validation_fold")
    elif change == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif change == "partition":
        frame.loc[0, "partition"] = "other"
    elif change == "test_fold":
        frame.loc[frame["partition"].eq("test"), "validation_fold"] = 0
    elif change == "missing":
        frame = frame.loc[~frame["subject"].eq(frame.loc[frame["partition"].eq("train"), "subject"].iloc[0])]
    else:
        frame.loc[frame["partition"].eq("train"), "validation_fold"] = np.nan if change == "nan" else 0.5
    rewrite_manifest(prepared_directory, "split_assignments.csv", frame.to_csv(index=False).encode())
    with pytest.raises(ValueError):
        load_training_snapshot(prepared_directory)


@pytest.mark.parametrize("actual, reason", [([1.0], "fewer_than_two_measurements"), ([1.0, 1.0], "constant_target")])
def test_undefined_r2(actual: list[float], reason: str) -> None:
    metrics = regression_metrics(np.array(actual), np.zeros(len(actual)))
    assert metrics.r2 is None
    assert metrics.r2_reason == reason


@pytest.mark.parametrize("prediction", [np.array([np.nan, 0]), np.array([1.0]), np.array([1e308, -1e308])])
def test_invalid_predictions(prediction: np.ndarray) -> None:
    with pytest.raises(ValueError):
        regression_metrics(np.array([1.0, 2.0]), prediction)


def test_grouped_evaluation_and_fold_imputation(prepared_directory: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = load_training_snapshot(prepared_directory)
    snapshot.train.loc[snapshot.train.index[::3], "age"] = np.nan
    original = training.build_pipeline
    fits: list[set[int]] = []

    def build(features: tuple[str, ...]) -> training.Pipeline:
        pipeline = original(features)
        fit = pipeline.fit

        def checked_fit(frame: pd.DataFrame, target: object) -> training.Pipeline:
            fit(frame, target)
            fits.append(set(frame["subject"]))
            imputation = pipeline.named_steps["imputation"]
            np.testing.assert_allclose(
                imputation.named_transformers_["age"].statistics_, [np.nanmedian(frame["age"].to_numpy(dtype=float))]
            )
            return pipeline

        monkeypatch.setattr(pipeline, "fit", checked_fit)
        return pipeline

    monkeypatch.setattr(training, "build_pipeline", build)
    evaluation = evaluate(snapshot, BASELINE_FEATURES)
    actual = snapshot.train["motor_updrs"].to_numpy(dtype=float)
    assert evaluation.pooled.mae == pytest.approx(np.mean(np.abs(actual - evaluation.predictions)))
    assert evaluation.fold_summary["fold_mae_mean"] == pytest.approx(np.mean([m.mae for m in evaluation.folds]))
    for fold, patients in enumerate(fits):
        assert patients == set(snapshot.train.loc[snapshot.folds != fold, "subject"])
        assert not patients & set(snapshot.train.loc[snapshot.folds == fold, "subject"])
    assert np.isfinite(evaluation.predictions).all()


def test_constant_fold_summary_and_invalid_grouping(prepared_directory: Path) -> None:
    snapshot = load_training_snapshot(prepared_directory)
    snapshot.train["motor_updrs"] = 1.0
    assert evaluate(snapshot, BASELINE_FEATURES).fold_summary["fold_r2_mean"] is None
    invalid = replace(snapshot, folds=np.zeros(len(snapshot.train), dtype=np.int64))
    with pytest.raises(ValueError, match="patient-disjoint"):
        evaluate(invalid, BASELINE_FEATURES)
    snapshot.train.loc[0, "subject"] = snapshot.train.loc[
        np.flatnonzero(snapshot.folds != snapshot.folds[0])[0], "subject"
    ]
    with pytest.raises(ValueError, match="patient-disjoint"):
        evaluate(snapshot, BASELINE_FEATURES)


def test_incomplete_oof_rejected(prepared_directory: Path) -> None:
    snapshot = load_training_snapshot(prepared_directory)
    # Invalid fold metadata supplied directly, bypassing the loader.
    invalid = replace(snapshot, folds=cast(np.ndarray, np.full(len(snapshot.train), np.nan)))
    with pytest.raises(ValueError, match="Incomplete"):
        evaluate(invalid, BASELINE_FEATURES)


def test_tie_policy() -> None:
    def result(name: str, features: tuple[str, ...], mae: float = 1) -> ConfigurationResult:
        return ConfigurationResult(name, features, "run", "uri", Metrics(mae, 1, None, "constant_target"))

    baseline = result("baseline", BASELINE_FEATURES)
    small = result("a", ("age",))
    assert select_winner([small, baseline]) == baseline
    assert select_winner([result("b", ("age",)), small]) == small
    assert select_winner([result("b", BASELINE_FEATURES), small]) == small
    better = result("better", CANDIDATE_FEATURES, 0.5)
    assert select_winner([baseline, better]) == better


def test_pooled_metrics_use_measurements_not_fold_means(prepared_directory: Path) -> None:
    snapshot = load_training_snapshot(prepared_directory)
    folds = np.array([0] * 4 + [1] * 14, dtype=np.int64)
    snapshot.train["motor_updrs"] = np.arange(18, dtype=float) ** 2
    evaluation = evaluate(replace(snapshot, folds=folds), ("age",))
    errors = snapshot.train["motor_updrs"].to_numpy(dtype=float) - evaluation.predictions
    assert evaluation.pooled.mae == pytest.approx(np.mean(np.abs(errors)))
    assert evaluation.pooled.rmse == pytest.approx(np.sqrt(np.mean(errors**2)))
    assert evaluation.pooled.mae != pytest.approx(evaluation.fold_summary["fold_mae_mean"])
    assert evaluation.pooled.rmse != pytest.approx(evaluation.fold_summary["fold_rmse_mean"])
    assert evaluation.fold_summary["fold_mae_std"] == pytest.approx(np.std([item.mae for item in evaluation.folds]))


def test_undefined_r2_not_logged_as_finite_scalar(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict[str, float]] = []

    def log(metrics: dict[str, float], step: int) -> None:
        assert step == 2
        captured.append(metrics)

    monkeypatch.setattr(training.mlflow, "log_metrics", log)
    training.log_metrics(Metrics(1.0, 2.0, None, "constant_target"), "validation_", 2)
    assert captured == [{"validation_mae": 1.0, "validation_rmse": 2.0}]


@pytest.mark.parametrize("uri", ["", "file:///tmp/mlruns", "sqlite:///db", "http://", "http://user:secret@localhost"])
def test_no_tracking_fallback(uri: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="HTTP/HTTPS"):
        run_experiment(training_config({"experiment_name": "test"}), tmp_path, uri)


def test_runtime_config_validation() -> None:
    with pytest.raises(ValueError, match="canonical"):
        TrainConfig("test", (FeatureSet("baseline", ("age",)),))


def test_direct_config_missing_or_duplicate_names() -> None:
    baseline = FeatureSet("baseline", tuple(sorted(BASELINE_FEATURES)))
    candidate = FeatureSet("candidate", tuple(sorted(CANDIDATE_FEATURES)))
    with pytest.raises(ValueError, match="Both baseline"):
        TrainConfig("test", (baseline,))
    with pytest.raises(ValueError, match="Duplicate feature-set name"):
        TrainConfig("test", (baseline, candidate, baseline))
    with pytest.raises(ValueError, match="canonical"):
        TrainConfig("test", (baseline, candidate, FeatureSet("voice", ("sex", "ppe"))))
    assert TrainConfig("test", (candidate, baseline)).feature_sets == (baseline, candidate)


@pytest.mark.parametrize("uri", ["http://example.org", "http://192.168.1.2:5000", "http://[::]", "http://0.0.0.0"])
def test_remote_plaintext_rejected(uri: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="require HTTPS"):
        run_experiment(training_config({"experiment_name": "test"}), tmp_path, uri)


@pytest.mark.parametrize(
    "uri",
    ["http://localhost:5000", "http://127.0.0.1:5000", "http://127.0.0.2", "http://[::1]:5000", "https://example.org"],
)
def test_allowed_transport_reaches_repository_validation(
    uri: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def validated() -> None:
        raise RuntimeError("transport accepted")

    monkeypatch.setattr(training, "validate_repository", validated)
    with pytest.raises(RuntimeError, match="transport accepted"):
        run_experiment(training_config({"experiment_name": "test"}), tmp_path, uri)


@pytest.mark.parametrize("problem", ["wheel", "missing_files", "no_git", "not_git", "wrong_git_root"])
def test_repository_requirements_fail_before_mlflow(
    problem: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(training, "ROOT", tmp_path)
    if problem != "wheel":
        package = tmp_path / "src/parkinsons_ml"
        package.mkdir(parents=True)
        monkeypatch.setattr(training, "__file__", str(package / "training.py"))
    if problem not in {"wheel", "missing_files"}:
        (tmp_path / "uv.lock").write_text("lock")
        (tmp_path / "pyproject.toml").write_text("project")

    def git(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        if problem == "no_git":
            raise FileNotFoundError("git")
        if problem == "not_git":
            raise subprocess.CalledProcessError(128, "git")
        return subprocess.CompletedProcess("git", 0, stdout=str(tmp_path.parent))

    def contact(*args: object, **kwargs: object) -> None:
        pytest.fail("Repository failures must occur before MLflow state changes or network access")

    monkeypatch.setattr(training.subprocess, "run", git)
    monkeypatch.setattr(training.mlflow, "set_tracking_uri", contact)
    with pytest.raises(ValueError, match="source checkout|Git checkout"):
        run_experiment(training_config({"experiment_name": "test"}), tmp_path, "http://localhost")


@pytest.mark.parametrize("predictions", [np.array([1.0]), np.array([1.0, np.nan]), np.array([[1.0, 2.0]])])
def test_refit_prediction_validation(predictions: np.ndarray) -> None:
    with pytest.raises(ValueError, match="prediction shape or nonfinite"):
        training.validate_predictions(predictions, (2,))


def test_inherited_run_is_not_resumed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_RUN_ID", "caller-run")
    with pytest.raises(ValueError, match="unrelated"):
        run_experiment(training_config({"experiment_name": "test"}), tmp_path, "http://localhost")


def test_cli(prepared_directory: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    path = prepared_directory / "params.yaml"
    path.write_text("train:\n  experiment_name: test\n")
    monkeypatch.setattr(sys, "argv", ["training", "--params", str(path), "--data", str(prepared_directory)])
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    def run(config: TrainConfig, directory: Path, uri: str) -> training.ComparisonResult:
        assert config.experiment_name == "test" and directory == prepared_directory
        assert uri == "http://localhost:5000"
        child = ConfigurationResult("baseline", BASELINE_FEATURES, "child", "model", Metrics(1, 1, 0, None))
        return training.ComparisonResult("parent", (child,), child, "dataset", "split")

    monkeypatch.setattr(training, "run_experiment", run)
    training.main()
    assert json.loads(capsys.readouterr().out)["winner"]["name"] == "baseline"
    path.write_text("[]")
    with pytest.raises(ValueError, match="train mapping"):
        training.main()


def test_json_mapping_rejects_nonmapping() -> None:
    with pytest.raises(ValueError, match="JSON mapping"):
        training.json_mapping(b"[]")


def test_incomplete_preparation_parameters(prepared_directory: Path) -> None:
    path = prepared_directory / "manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["parameters"]["seed"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Incomplete preparation"):
        load_training_snapshot(prepared_directory)


@pytest.mark.parametrize(
    "payload",
    [
        "train:\n  experiment_name: first\n  experiment_name: second\n",
        "train:\n  experiment_name: test\n  extra_feature_sets:\n    voice: [ppe]\n    voice: [age]\n",
        "1: value\n",
    ],
)
def test_duplicate_yaml_keys_rejected(tmp_path: Path, payload: str, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "params.yaml"
    path.write_text(payload)
    monkeypatch.setattr(sys, "argv", ["training", "--params", str(path)])
    with pytest.raises(ValueError, match="unique strings"):
        training.main()
