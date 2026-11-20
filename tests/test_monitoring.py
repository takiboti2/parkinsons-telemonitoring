"""Bounded inference cohorts, real Evidently calculations, and required report logging."""

import json
from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from mlflow.entities import Run
from mlflow.store.entities.paged_list import PagedList
from mlflow.tracking import MlflowClient

from parkinsons_ml import monitoring
from parkinsons_ml.schema import POLICY_VERSION, SCHEMA_VERSION

CUTOFF = 1800000000000
IDENTITIES = {
    "dataset_id": "dataset",
    "split_id": "split",
    "schema_version": SCHEMA_VERSION,
    "policy_version": POLICY_VERSION,
    "evaluation_id": "evaluation",
}
FEATURES = ("age", "sex", "test_time")


def inference_run(run_id: str = "batch") -> Run:
    run = MagicMock(spec=Run)
    run.info.run_id, run.info.start_time = run_id, CUTOFF - 1000
    run.data.tags = {
        "purpose": "inference",
        "registered_model_name": "model",
        "model_version": "1",
        "training_run_id": "training",
        "reference_run_id": "training",
        "reference_artifact": "monitoring/reference.csv",
        "representation": monitoring.REPRESENTATION,
        **IDENTITIES,
    }
    run.data.params = {"features": ",".join(FEATURES)}
    run.data.metrics = {"batch_size": 100.0}
    return cast(Run, run)


@pytest.fixture
def reference() -> pd.DataFrame:
    # Synthetic data is confined to tests; no demonstration records are submitted.
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "age": rng.integers(50, 80, 100).astype(float),
            "sex": np.tile([0.0, 1.0], 50),
            "test_time": rng.uniform(0, 100, 100),
        }
    )


@pytest.fixture
def client(reference: pd.DataFrame) -> MagicMock:
    client = MagicMock(spec=MlflowClient)
    model = client.get_model_version.return_value
    model.run_id, model.status, model.tags = "training", "READY", IDENTITIES.copy()
    source = client.get_run.return_value
    source.info.status, source.info.experiment_id = "FINISHED", "experiment"
    source.data.tags, source.data.params = IDENTITIES.copy(), {"features": json.dumps(FEATURES)}
    client.get_experiment.return_value.artifact_location = "mlflow-artifacts:/experiment"
    raw = reference.to_csv(index=False).encode()
    files = {
        ("training", "monitoring/reference.csv"): raw,
        ("training", "monitoring/identity.json"): json.dumps(
            {
                "sha256": monitoring.digest(raw),
                "representation": monitoring.REPRESENTATION,
                "features": list(FEATURES),
                "rows": len(reference),
            }
        ).encode(),
    }
    for offset, run_id in enumerate(("batch", "new", "old"), start=1):
        current = reference.copy()
        current["test_time"] += offset
        files[run_id, "inputs.csv"] = current.to_csv(index=False).encode()
        files[run_id, "predictions.csv"] = (
            pd.DataFrame({"row": range(100), "prediction": np.arange(100) / 10 + offset}).to_csv(index=False).encode()
        )

    def download(run_id: str, name: str, directory: str) -> str:
        path = Path(directory) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(files[run_id, name])
        return str(path)

    client.download_artifacts.side_effect = download
    client.search_runs.return_value = PagedList([inference_run()], None)
    client.create_run.return_value.info.run_id = "monitoring"
    return client


@pytest.fixture
def window(client: MagicMock) -> monitoring.MonitoringWindow:
    return monitoring.load_window(client, "model", "1", monitoring.WindowConfig(), cutoff_ms=CUTOFF)


def test_cohort_snapshot_and_manifest(window: monitoring.MonitoringWindow, client: MagicMock) -> None:
    assert len(window.current) == 100 and len(window.batches) == 1
    assert window.scanned == 1 and not window.truncated
    np.testing.assert_allclose(window.current["test_time"], window.reference["test_time"] + 1)
    assert window.predictions.iloc[0]["prediction"] == 1.0
    manifest = window.manifest()
    assert manifest["reference_rows"] == 100 and manifest["cutoff_ms"] == CUTOFF
    assert manifest["actual_start_ms"] == CUTOFF - 1000
    assert manifest["actual_end_ms"] == CUTOFF - 1000
    assert window.batches[0].inputs_sha256 != window.reference_sha256
    assert "FINISHED" in client.search_runs.call_args.kwargs["filter_string"]
    assert "end_time" in client.search_runs.call_args.kwargs["filter_string"]
    for call in client.download_artifacts.call_args_list:
        assert not Path(call.args[2]).exists()


@pytest.mark.parametrize("field", ["model_version", "reference_run_id", "schema_version", "representation"])
def test_incompatible_batches_are_counted_not_mixed(client: MagicMock, field: str) -> None:
    run = inference_run()
    run.data.tags[field] = "different"
    client.search_runs.return_value = PagedList([run], None)
    window = monitoring.load_window(client, "model", "1", monitoring.WindowConfig(), cutoff_ms=CUTOFF)
    assert window.current.empty and window.incompatible == 1
    assert len(client.download_artifacts.call_args_list) == 2
    summary = monitoring.build_report(window).summary
    assert summary["status"] == "empty"


def test_empty_and_bounded_pagination(client: MagicMock) -> None:
    client.search_runs.side_effect = [
        PagedList([inference_run("new")], "next"),
        PagedList([inference_run("old")], "more"),
    ]
    window = monitoring.load_window(client, "model", "1", monitoring.WindowConfig(max_batches=2), cutoff_ms=CUTOFF)
    assert [batch.run_id for batch in window.batches] == ["new", "old"]
    np.testing.assert_allclose(window.current.iloc[:100]["test_time"], window.reference["test_time"] + 2)
    np.testing.assert_allclose(window.current.iloc[100:]["test_time"], window.reference["test_time"] + 3)
    assert window.predictions["prediction"].iloc[[0, 100]].tolist() == [2.0, 3.0]
    assert window.batches[0].inputs_sha256 != window.batches[1].inputs_sha256
    assert window.truncated and window.scanned == 2
    assert client.search_runs.call_args.kwargs["page_token"] == "next"
    client.search_runs.side_effect = None
    client.search_runs.return_value = PagedList([], None)
    empty = monitoring.load_window(client, "model", "1", monitoring.WindowConfig(), cutoff_ms=CUTOFF)
    assert empty.current.empty and not empty.truncated and empty.manifest()["actual_start_ms"] is None


def test_row_limit_never_takes_partial_batch(client: MagicMock) -> None:
    window = monitoring.load_window(client, "model", "1", monitoring.WindowConfig(max_rows=99), cutoff_ms=CUTOFF)
    assert window.current.empty and window.truncated and window.scanned == 1


@pytest.mark.parametrize(
    "problem",
    [
        "version",
        "name",
        "not_ready",
        "no_run",
        "unfinished",
        "identities",
        "schema",
        "local_artifacts",
        "features",
    ],
)
def test_invalid_cohort_fails_closed(client: MagicMock, problem: str) -> None:
    model = client.get_model_version.return_value
    source = client.get_run.return_value
    name, version = "model", "1"
    if problem == "version":
        version = "champion"
    elif problem == "name":
        name = "unsafe'name"
    elif problem == "not_ready":
        model.status = "PENDING"
    elif problem == "no_run":
        model.run_id = None
    elif problem == "unfinished":
        source.info.status = "FAILED"
    elif problem == "identities":
        model.tags["dataset_id"] = "different"
    elif problem == "schema":
        model.tags["schema_version"] = source.data.tags["schema_version"] = "old"
    elif problem == "local_artifacts":
        client.get_experiment.return_value.artifact_location = "file:///not-allowed"
    else:
        source.data.params["features"] = '["subject"]'
    with pytest.raises(ValueError):
        monitoring.load_window(client, name, version, monitoring.WindowConfig(), cutoff_ms=CUTOFF)


@pytest.mark.parametrize("problem", ["identity", "corrupt_csv", "unclipped", "selection", "count", "oversize"])
def test_corrupt_compatible_artifact_blocks_report(
    client: MagicMock, problem: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = client.download_artifacts.side_effect

    def download(run_id: str, name: str, directory: str) -> str:
        path = Path(original(run_id, name, directory))
        if problem == "identity" and name.endswith("identity.json"):
            path.write_text("{}")
        elif problem == "corrupt_csv" and name == "inputs.csv":
            path.write_text("age,sex,subject\n50,0,1\n")
        elif problem == "unclipped" and name == "inputs.csv":
            path.write_text("age,sex,test_time\n50,0,-1\n")
        return str(path)

    client.download_artifacts.side_effect = download
    if problem in {"selection", "count"}:
        run = inference_run()
        if problem == "selection":
            run.data.params["features"] = "age"
        else:
            run.data.metrics["batch_size"] = 999
        client.search_runs.return_value = PagedList([run], None)
    if problem == "oversize":
        monkeypatch.setattr(monitoring, "MAX_ARTIFACT_BYTES", 1)
    with pytest.raises(ValueError):
        monitoring.load_window(client, "model", "1", monitoring.WindowConfig(), cutoff_ms=CUTOFF)
    assert not Path(client.download_artifacts.call_args.args[2]).exists()


@pytest.mark.parametrize("artifact", ["inputs.csv", "monitoring/reference.csv", "predictions.csv"])
@pytest.mark.parametrize("problem", ["extra", "missing", "blank", "quote"])
def test_csv_records_cannot_silently_lose_or_gain_fields(client: MagicMock, artifact: str, problem: str) -> None:
    original = client.download_artifacts.side_effect
    changed_reference: bytes | None = None

    def download(run_id: str, name: str, directory: str) -> str:
        nonlocal changed_reference
        path = Path(original(run_id, name, directory))
        if name == artifact:
            lines = path.read_text().splitlines()
            if problem == "extra":
                lines[1:] = ["999," + line for line in lines[1:]]
            elif problem == "missing":
                lines[1:] = [line.rsplit(",", 1)[0] for line in lines[1:]]
            elif problem == "blank":
                lines[1] = ""
            else:
                lines[1] = '"unterminated'
            path.write_text("\n".join(lines) + "\n")
            if name == "monitoring/reference.csv":
                changed_reference = path.read_bytes()
        elif name == "monitoring/identity.json" and changed_reference is not None:
            identity = json.loads(path.read_bytes())
            identity["sha256"] = monitoring.digest(changed_reference)
            path.write_text(json.dumps(identity))
        return str(path)

    client.download_artifacts.side_effect = download
    with pytest.raises(ValueError, match="CSV"):
        monitoring.load_window(client, "model", "1", monitoring.WindowConfig(), cutoff_ms=CUTOFF)
    assert not Path(client.download_artifacts.call_args.args[2]).exists()


def test_single_feature_rows_are_not_silently_skipped() -> None:
    with pytest.raises(ValueError):
        monitoring.feature_frame(b"age\n50\n   \n60\n", ("age",))
    frame = monitoring.feature_frame(b'age\n50\n""\n60\n', ("age",))
    assert len(frame) == 3 and pd.isna(frame["age"].iloc[1])


@pytest.mark.parametrize("problem", ["nan", "infinity", "nonnumeric", "duplicate_row", "reordered_rows", "row_count"])
def test_prediction_validation_each_invariant_independently(client: MagicMock, problem: str) -> None:
    original = client.download_artifacts.side_effect

    def download(run_id: str, name: str, directory: str) -> str:
        path = Path(original(run_id, name, directory))
        if name == "predictions.csv":
            output = pd.read_csv(path)
            if problem == "nan":
                output.loc[10, "prediction"] = np.nan
            elif problem == "infinity":
                output.loc[10, "prediction"] = np.inf
            elif problem == "nonnumeric":
                output = output.astype({"prediction": object})
                output.loc[10, "prediction"] = "invalid"
            elif problem == "duplicate_row":
                output.loc[10, "row"] = 9
            elif problem == "reordered_rows":
                output["row"] = output["row"].to_numpy()[::-1]
            else:
                output = output.iloc[:-1]
            output.to_csv(path, index=False)
        return str(path)

    client.download_artifacts.side_effect = download
    with pytest.raises(ValueError):
        monitoring.load_window(client, "model", "1", monitoring.WindowConfig(), cutoff_ms=CUTOFF)


@pytest.mark.parametrize("field", ["sha256", "representation", "features", "rows"])
def test_reference_metadata_each_invariant_independently(client: MagicMock, field: str) -> None:
    original = client.download_artifacts.side_effect

    def download(run_id: str, name: str, directory: str) -> str:
        path = Path(original(run_id, name, directory))
        if name == "monitoring/identity.json":
            identity = json.loads(path.read_bytes())
            identity[field] = "incorrect"
            path.write_text(json.dumps(identity))
        return str(path)

    client.download_artifacts.side_effect = download
    with pytest.raises(ValueError, match="Reference"):
        monitoring.load_window(client, "model", "1", monitoring.WindowConfig(), cutoff_ms=CUTOFF)


@pytest.mark.parametrize("rows, expected", [(49, "insufficient"), (50, "ready")])
def test_drift_minimum_nonmissing_sample_boundary(
    window: monitoring.MonitoringWindow, rows: int, expected: str
) -> None:
    report = monitoring.build_report(replace(window, current=window.current.iloc[:rows]))
    assert report.summary["status"] == expected
    assert (report.html is not None) == (rows == 50)


def test_categorical_only_shift_uses_jensen_shannon(window: monitoring.MonitoringWindow) -> None:
    cohort = replace(window.cohort, features=("sex",))
    categorical = replace(
        window, cohort=cohort, current=window.current[["sex"]].copy(), reference=window.reference[["sex"]]
    )
    stable = monitoring.build_report(categorical)
    assert stable.summary["dataset_drift"] is False
    categorical.current["sex"] = 1.0
    shifted = monitoring.build_report(categorical)
    assert shifted.summary["dataset_drift"] is True
    assert shifted.summary["categorical_method"] == "jensenshannon"
    assert shifted.summary["unavailable"] == {}


def test_real_evidently_stable_and_shifted(window: monitoring.MonitoringWindow) -> None:
    stable = monitoring.build_report(window)
    assert stable.summary["status"] == "ready" and stable.summary["dataset_drift"] is False
    assert stable.html is not None and "<html" in stable.html.lower()
    assert stable.evidently_json is not None and json.loads(stable.evidently_json)["metrics"]
    shifted = window.current.copy()
    shifted["age"] += 50
    shifted["test_time"] += 100
    report = monitoring.build_report(replace(window, current=shifted))
    assert report.summary["dataset_drift"] is True
    assert report.summary["numerical_method"] == "wasserstein"
    assert "ground-truth" in str(report.summary["performance"])


def test_missingness_constant_and_insufficient_are_explicit(window: monitoring.MonitoringWindow) -> None:
    missing = window.current.copy()
    missing["age"] = np.nan
    reference = window.reference.copy()
    reference["test_time"] = 1.0
    report = monitoring.build_report(replace(window, current=missing, reference=reference))
    assert report.summary["status"] == "partial"
    assert report.summary["unavailable"] == {
        "age": "insufficient_non_missing_rows",
        "test_time": "constant_reference",
    }
    insufficient = monitoring.build_report(replace(window, current=window.current.head(2)))
    assert insufficient.summary["status"] == "insufficient" and insufficient.html is None


@pytest.mark.parametrize("failure", ["none", "log_dict", "log_text", "html_upload", "log_metric", "set_terminated"])
def test_required_report_logging(window: monitoring.MonitoringWindow, client: MagicMock, failure: str) -> None:
    report = monitoring.build_report(window)
    if failure != "none":
        if failure == "html_upload":

            def fail_html(run_id: str, text: str, artifact_file: str) -> None:
                if artifact_file == "evidently.html":
                    raise ConnectionError("private outage")

            client.log_text.side_effect = fail_html
        else:
            getattr(client, failure).side_effect = ConnectionError("private outage")
        with pytest.raises(ConnectionError):
            monitoring.save_report(client, window, report)
        client.set_terminated.assert_any_call("monitoring", "FAILED")
    else:
        assert monitoring.save_report(client, window, report) == "monitoring"
        client.set_terminated.assert_called_once_with("monitoring", "FINISHED")
        assert client.log_text.call_count == 2
        assert client.create_run.call_args.kwargs["tags"]["purpose"] == "monitoring"
    client.log_dict.assert_any_call("monitoring", window.manifest(), "manifest.json")


def test_save_insufficient_quality_without_evidently(window: monitoring.MonitoringWindow, client: MagicMock) -> None:
    small = replace(window, current=window.current.head(2), batches=(replace(window.batches[0], rows=2),))
    report = monitoring.build_report(small)
    monitoring.save_report(client, small, report)
    client.log_text.assert_not_called()


@pytest.mark.parametrize(
    "problem",
    [
        "none",
        "unfinished",
        "wrong_version",
        "manifest",
        "summary",
        "full",
        "rows",
        "reference_rows",
        "min_rows",
        "binding",
        "batch_counts",
    ],
)
def test_open_saved_report_is_read_only_and_cohort_bound(
    window: monitoring.MonitoringWindow, client: MagicMock, problem: str
) -> None:
    window = (
        window
        if problem == "full"
        else replace(window, current=window.current.head(2), batches=(replace(window.batches[0], rows=2),))
    )
    report = monitoring.build_report(window)
    monitoring.save_report(client, window, report)
    run = client.get_run.return_value
    run.data.tags = client.create_run.call_args.kwargs["tags"].copy()
    if problem == "unfinished":
        run.info.status = "FAILED"
    elif problem == "wrong_version":
        run.data.tags["model_version"] = "2"
    manifest = window.manifest()
    summary = report.summary
    if problem == "manifest":
        manifest["representation"] = "other"
    elif problem == "summary":
        summary = {"policy": "unsupported"}
    elif problem in {"rows", "reference_rows", "min_rows"}:
        summary[problem] = 999
    elif problem == "binding":
        manifest["cutoff_ms"] = window.cutoff_ms + 1
    elif problem == "batch_counts":
        manifest["batches"] = []
    files = {
        "manifest.json": json.dumps(manifest),
        "summary.json": json.dumps(summary),
        "evidently.json": report.evidently_json or "",
        "evidently.html": report.html or "",
    }
    downloads: list[Path] = []

    def download(run_id: str, name: str, directory: str) -> str:
        downloads.append(Path(directory))
        path = Path(directory) / name
        path.write_text(files[name])
        return str(path)

    client.download_artifacts.side_effect = download
    client.create_run.reset_mock()
    if problem in {"none", "full"}:
        stored_manifest, stored = monitoring.open_saved_report(client, window.cohort, "monitoring")
        assert stored_manifest["cutoff_ms"] == window.cutoff_ms and stored.summary == summary
        assert stored.html == report.html
    else:
        with pytest.raises(ValueError):
            monitoring.open_saved_report(client, window.cohort, "monitoring")
    client.create_run.assert_not_called()
    assert all(not path.exists() for path in downloads)


@pytest.mark.parametrize(
    "problem",
    [
        "rows",
        "reference_rows",
        "min_rows",
        "digest",
        "configuration",
        "cutoff",
        "batch_counts",
        "quality",
        "drift_results",
        "html",
    ],
)
def test_report_mismatch_rejected_before_creating_run(
    window: monitoring.MonitoringWindow, client: MagicMock, problem: str
) -> None:
    report = monitoring.build_report(window)
    if problem in {"rows", "reference_rows", "min_rows"}:
        report.summary[problem] = 999
    elif problem == "digest":
        report.summary["window_sha256"] = "incorrect"
    elif problem == "configuration":
        window = replace(window, config=monitoring.WindowConfig(hours=48))
    elif problem == "cutoff":
        window = replace(window, cutoff_ms=window.cutoff_ms + 1)
    elif problem == "batch_counts":
        window = replace(window, batches=(replace(window.batches[0], rows=99),))
    elif problem == "quality":
        report.summary.pop("quality")
    elif problem == "drift_results":
        report.summary.pop("drift")
    else:
        report = replace(report, html=None)
    with pytest.raises(ValueError):
        monitoring.save_report(client, window, report)
    client.create_run.assert_not_called()


@pytest.mark.parametrize("problem", ["none", "model", "artifact"])
def test_promotion_decision_identity(window: monitoring.MonitoringWindow, client: MagicMock, problem: str) -> None:
    run = client.get_run.return_value
    run.data.tags = {
        "purpose": "champion-promotion",
        "registered_model_name": "different" if problem == "model" else "model",
    }

    def download(run_id: str, name: str, directory: str) -> str:
        assert name == "decision.json"
        path = Path(directory) / name
        path.write_text(
            json.dumps(
                {
                    "registered_model_name": "different" if problem == "artifact" else "model",
                    "reason": "lower_oof_mae",
                }
            )
        )
        return str(path)

    client.download_artifacts.side_effect = download
    if problem == "none":
        assert monitoring.promotion_decision(client, window.cohort, "audit")["reason"] == "lower_oof_mae"
    else:
        with pytest.raises(ValueError):
            monitoring.promotion_decision(client, window.cohort, "audit")
    client.create_run.assert_not_called()
