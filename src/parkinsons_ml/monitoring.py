"""On-demand, bounded monitoring of immutable MLflow inference cohorts."""

import csv
import json
import math
import time
from contextlib import suppress
from dataclasses import asdict, dataclass
from importlib.metadata import version as package_version
from io import BytesIO, StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.core.metric_types import SingleValue
from evidently.metrics import ValueDrift
from mlflow.entities import Run
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, ConfigDict, Field

from parkinsons_ml.experimentation import PromotionConfig, digest
from parkinsons_ml.promotion import IDENTITIES
from parkinsons_ml.schema import POLICY_VERSION, SCHEMA_VERSION
from parkinsons_ml.training import FeaturePreparation, validate_features

REPRESENTATION = "selected-pre-imputation-v1"
REPORT_POLICY = "development-reference-drift-v2"
MAX_ARTIFACT_BYTES = 8 * 1024 * 1024
DRIFT_THRESHOLD = 0.1
DRIFT_SHARE = 0.5


class WindowConfig(BaseModel):
    """Limits apply to scanned batches and whole included batches, never partial rows."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    hours: int = Field(default=24, ge=1, le=168)
    max_batches: int = Field(default=100, ge=1, le=100)
    max_rows: int = Field(default=10000, ge=1, le=10000)
    min_rows: int = Field(default=50, ge=2, le=10000)


@dataclass(frozen=True)
class Cohort:
    model_name: str
    model_version: str
    training_run_id: str
    experiment_id: str
    features: tuple[str, ...]
    identities: dict[str, str]


@dataclass(frozen=True)
class Batch:
    run_id: str
    start_ms: int
    rows: int
    inputs_sha256: str
    predictions_sha256: str


@dataclass(frozen=True)
class MonitoringWindow:
    cohort: Cohort
    config: WindowConfig
    cutoff_ms: int
    reference_sha256: str
    reference: pd.DataFrame
    current: pd.DataFrame
    predictions: pd.DataFrame
    batches: tuple[Batch, ...]
    scanned: int
    incompatible: int
    truncated: bool

    def manifest(self) -> dict[str, object]:
        return {
            "policy": REPORT_POLICY,
            "cohort": asdict(self.cohort),
            "representation": REPRESENTATION,
            "config": self.config.model_dump(),
            "cutoff_ms": self.cutoff_ms,
            "window_start_ms": self.cutoff_ms - self.config.hours * 3600000,
            "actual_start_ms": min((batch.start_ms for batch in self.batches), default=None),
            "actual_end_ms": max((batch.start_ms for batch in self.batches), default=None),
            "reference_sha256": self.reference_sha256,
            "reference_rows": len(self.reference),
            "current_rows": len(self.current),
            "batches": [asdict(batch) for batch in self.batches],
            "scanned": self.scanned,
            "incompatible": self.incompatible,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class MonitoringReport:
    summary: dict[str, object]
    evidently_json: str | None
    html: str | None


def artifact_bytes(client: MlflowClient, run_id: str, name: str, directory: str) -> bytes:
    """Download only server-owned artifact names; cap memory before parsing."""
    path = Path(client.download_artifacts(run_id, name, directory))
    if path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ValueError("Monitoring artifact exceeds the size limit")
    return path.read_bytes()


def csv_frame(payload: bytes, columns: tuple[str, ...]) -> pd.DataFrame:
    """Reject ragged records before pandas can infer an index or pad missing fields."""
    try:
        records = csv.reader(StringIO(payload.decode("utf-8"), newline=""), strict=True)
        if tuple(next(records, ())) != columns:
            raise ValueError("Monitoring CSV columns do not match the expected schema")
        if any(len(record) != len(columns) for record in records):
            raise ValueError("Monitoring CSV record width does not match its header")
    except (csv.Error, UnicodeDecodeError) as error:
        raise ValueError("Monitoring CSV is malformed") from error
    frame = pd.read_csv(
        BytesIO(payload), keep_default_na=False, na_values=[""], index_col=False, skip_blank_lines=False
    )
    if frame.empty:
        raise ValueError("Monitoring CSV must contain rows")
    return frame


def feature_frame(payload: bytes, features: tuple[str, ...]) -> pd.DataFrame:
    frame = csv_frame(payload, features)
    values = frame.to_numpy(dtype=np.float64)
    # Validate without silently repairing corrupt encodings or re-clipping time.
    prepared = FeaturePreparation(features).transform(frame)
    if not np.array_equal(prepared, values, equal_nan=True):
        raise ValueError("Monitoring inputs are not in the recorded representation")
    return pd.DataFrame(values, columns=pd.Index(features))


def load_cohort(client: MlflowClient, name: str, version: str) -> Cohort:
    PromotionConfig(registered_model_name=name)
    if not version.isdecimal() or int(version) < 1:
        raise ValueError("Select a positive immutable model version")
    model = client.get_model_version(name, version)
    if model.status != "READY" or not model.run_id:
        raise ValueError("Model version is not ready or has no training provenance")
    source = client.get_run(model.run_id)
    identities = {key: model.tags.get(key, "") for key in IDENTITIES}
    if (
        source.info.status != "FINISHED"
        or not all(identities.values())
        or any(source.data.tags.get(key) != value for key, value in identities.items())
        or identities["schema_version"] != SCHEMA_VERSION
        or identities["policy_version"] != POLICY_VERSION
    ):
        raise ValueError("Model/training identities are incomplete, incompatible, or unsupported")
    features = validate_features(json.loads(source.data.params.get("features", "null")))
    experiment = client.get_experiment(source.info.experiment_id)
    if not experiment.artifact_location.startswith("mlflow-artifacts:/"):
        raise ValueError("Monitoring requires HTTP-served artifacts")
    return Cohort(name, version, model.run_id, source.info.experiment_id, features, identities)


def compatible(run: Run, cohort: Cohort) -> bool:
    expected = {
        "registered_model_name": cohort.model_name,
        "model_version": cohort.model_version,
        "training_run_id": cohort.training_run_id,
        "reference_run_id": cohort.training_run_id,
        "reference_artifact": "monitoring/reference.csv",
        "representation": REPRESENTATION,
        **cohort.identities,
    }
    return all(run.data.tags.get(key) == value for key, value in expected.items())


def load_window(
    client: MlflowClient, name: str, version: str, config: WindowConfig, *, cutoff_ms: int | None = None
) -> MonitoringWindow:
    """Read a fixed snapshot; incompatible batches are counted, corruption fails closed."""
    cohort = load_cohort(client, name, version)
    cutoff = cutoff_ms if cutoff_ms is not None else int(time.time() * 1000)
    start = cutoff - config.hours * 3600000
    inputs: list[pd.DataFrame] = []
    outputs: list[pd.DataFrame] = []
    batches: list[Batch] = []
    scanned = incompatible = rows = 0
    truncated = False
    with TemporaryDirectory(prefix="parkinsons-monitoring-") as directory:
        reference_bytes = artifact_bytes(client, cohort.training_run_id, "monitoring/reference.csv", directory)
        identity = json.loads(artifact_bytes(client, cohort.training_run_id, "monitoring/identity.json", directory))
        reference = feature_frame(reference_bytes, cohort.features)
        if (
            not isinstance(identity, dict)
            or identity.get("sha256") != digest(reference_bytes)
            or identity.get("representation") != REPRESENTATION
            or identity.get("features") != list(cohort.features)
            or identity.get("rows") != len(reference)
        ):
            raise ValueError("Reference checksum or metadata does not match")
        token: str | None = None
        while scanned < config.max_batches:
            page = client.search_runs(
                [cohort.experiment_id],
                filter_string=(
                    "tags.purpose = 'inference' AND attributes.status = 'FINISHED' "
                    f"AND attributes.start_time >= {start} AND attributes.start_time <= {cutoff} "
                    f"AND attributes.end_time <= {cutoff}"
                ),
                order_by=["attributes.start_time DESC", "attributes.run_id DESC"],
                max_results=min(25, config.max_batches - scanned),
                page_token=token,
            )
            for run in page:
                scanned += 1
                if not compatible(run, cohort):
                    incompatible += 1
                    continue
                if run.data.params.get("features") != ",".join(cohort.features):
                    raise ValueError("Inference feature selection differs from cohort")
                raw = artifact_bytes(client, run.info.run_id, "inputs.csv", directory)
                frame = feature_frame(raw, cohort.features)
                predictions_bytes = artifact_bytes(client, run.info.run_id, "predictions.csv", directory)
                predictions = csv_frame(predictions_bytes, ("row", "prediction"))
                if (
                    not 1 <= len(frame) <= 1000
                    or run.data.metrics.get("batch_size") != len(frame)
                    or tuple(predictions.columns) != ("row", "prediction")
                    or len(predictions) != len(frame)
                    or predictions["row"].tolist() != list(range(len(frame)))
                    or not np.isfinite(predictions["prediction"].to_numpy(dtype=np.float64)).all()
                ):
                    raise ValueError("Inference predictions/count/alignment is invalid")
                if rows + len(frame) > config.max_rows:
                    truncated = True
                    break
                rows += len(frame)
                inputs.append(frame)
                outputs.append(predictions.assign(run_id=run.info.run_id, start_ms=run.info.start_time))
                batches.append(
                    Batch(run.info.run_id, run.info.start_time, len(frame), digest(raw), digest(predictions_bytes))
                )
            if truncated:
                break
            token = page.token
            if not token:
                break
        truncated = truncated or bool(token)
    current = (
        pd.concat(inputs, ignore_index=True) if inputs else pd.DataFrame(columns=pd.Index(cohort.features), dtype=float)
    )
    predictions = (
        pd.concat(outputs, ignore_index=True)
        if outputs
        else pd.DataFrame(columns=pd.Index(["row", "prediction", "run_id", "start_ms"]))
    )
    return MonitoringWindow(
        cohort,
        config,
        cutoff,
        digest(reference_bytes),
        reference,
        current,
        predictions,
        tuple(batches),
        scanned,
        incompatible,
        truncated,
    )


def build_report(window: MonitoringWindow) -> MonitoringReport:
    """Quality and explicitly configured drift, never unlabelled performance."""
    reference, current = window.reference, window.current
    quality: list[dict[str, object]] = []
    available: list[str] = []
    unavailable: dict[str, str] = {}
    for name in window.cohort.features:
        before, after = reference[name], current[name]
        quality.append(
            {
                "feature": name,
                "reference_missing_share": float(before.isna().mean()),
                "current_missing_share": float(after.isna().mean()) if len(current) else None,
                "reference_mean": float(before.mean()) if bool(before.notna().any()) else None,
                "current_mean": float(after.mean()) if bool(after.notna().any()) else None,
            }
        )
        if min(before.notna().sum(), after.notna().sum()) < window.config.min_rows:
            unavailable[name] = "insufficient_non_missing_rows"
        elif name != "sex" and before.nunique() < 2:
            unavailable[name] = "constant_reference"
        else:
            available.append(name)
    summary: dict[str, object] = {
        "status": "empty" if current.empty else "insufficient",
        "rows": len(current),
        "reference_rows": len(reference),
        "window_sha256": manifest_digest(window.manifest()),
        "quality": quality,
        "unavailable": unavailable,
        "performance": "Unavailable: inference runs contain no ground-truth labels",
        "policy": REPORT_POLICY,
        "evidently_version": package_version("evidently"),
        "numerical_method": "wasserstein",
        "categorical_method": "jensenshannon",
        "threshold": DRIFT_THRESHOLD,
        "dataset_drift_share": DRIFT_SHARE,
        "min_rows": window.config.min_rows,
    }
    if not available:
        return MonitoringReport(summary, None, None)
    definition = DataDefinition(
        numerical_columns=[name for name in available if name != "sex"],
        categorical_columns=["sex"] if "sex" in available else [],
    )
    metrics = [
        ValueDrift(column=name, method="jensenshannon" if name == "sex" else "wasserstein", threshold=DRIFT_THRESHOLD)
        for name in available
    ]
    snapshot = Report(metrics).run(
        current_data=Dataset.from_pandas(current.loc[:, available], data_definition=definition),
        reference_data=Dataset.from_pandas(reference.loc[:, available], data_definition=definition),
    )
    scores: list[dict[str, object]] = []
    for name, metric in zip(available, metrics, strict=True):
        score = float(cast(SingleValue, snapshot.context.get_metric_result(metric)).value)
        if not math.isfinite(score):
            raise ValueError("Evidently returned a nonfinite drift score")
        scores.append({"feature": name, "score": score, "drift": score >= DRIFT_THRESHOLD})
    share = sum(row["drift"] is True for row in scores) / len(scores)
    summary.update(
        status="partial" if unavailable else "ready",
        drift=scores,
        drifted_share=share,
        dataset_drift=share >= DRIFT_SHARE,
    )
    return MonitoringReport(summary, snapshot.json(), snapshot.get_html_str(as_iframe=False))


def manifest_digest(manifest: dict[str, object]) -> str:
    """A stable binding to identities, configuration, cutoff, batches, and checksums."""
    return digest(json.dumps(manifest, sort_keys=True, allow_nan=False).encode())


def validate_report(manifest: dict[str, object], summary: dict[str, object]) -> None:
    config = WindowConfig.model_validate(manifest.get("config"))
    batches = manifest.get("batches")
    if (
        type(manifest.get("current_rows")) is not int
        or type(manifest.get("reference_rows")) is not int
        or cast(int, manifest["reference_rows"]) < 1
        or not isinstance(batches, list)
        or len(batches) > config.max_batches
        or any(
            not isinstance(batch, dict) or type(batch.get("rows")) is not int or not 1 <= batch["rows"] <= 1000
            for batch in batches
        )
        or sum(batch["rows"] for batch in batches) != manifest.get("current_rows")
        or not 0 <= cast(int, manifest.get("current_rows")) <= config.max_rows
    ):
        raise ValueError("Report manifest batch counts are inconsistent")
    status = summary.get("status")
    if (
        manifest.get("policy") != REPORT_POLICY
        or summary.get("policy") != REPORT_POLICY
        or summary.get("window_sha256") != manifest_digest(manifest)
        or type(summary.get("rows")) is not int
        or type(summary.get("reference_rows")) is not int
        or type(summary.get("min_rows")) is not int
        or summary.get("rows") != manifest.get("current_rows")
        or summary.get("reference_rows") != manifest.get("reference_rows")
        or summary.get("min_rows") != config.min_rows
        or summary.get("numerical_method") != "wasserstein"
        or summary.get("categorical_method") != "jensenshannon"
        or summary.get("threshold") != DRIFT_THRESHOLD
        or summary.get("dataset_drift_share") != DRIFT_SHARE
        or status not in {"ready", "partial", "empty", "insufficient"}
        or (status == "empty") != (summary.get("rows") == 0)
        or not isinstance(summary.get("quality"), list)
        or not isinstance(summary.get("unavailable"), dict)
        or not isinstance(summary.get("performance"), str)
        or not isinstance(summary.get("evidently_version"), str)
    ):
        raise ValueError("Report summary disagrees with its window manifest")
    if status in {"ready", "partial"} and (
        not isinstance(summary.get("drift"), list)
        or not isinstance(summary.get("drifted_share"), float)
        or not isinstance(summary.get("dataset_drift"), bool)
    ):
        raise ValueError("Evaluated report is missing drift results")


def save_report(client: MlflowClient, window: MonitoringWindow, report: MonitoringReport) -> str:
    """A saved success requires every upload; ordinary dashboard reads never write."""
    manifest = window.manifest()
    validate_report(manifest, report.summary)
    evaluated = report.summary["status"] in {"ready", "partial"}
    if evaluated != (report.evidently_json is not None) or evaluated != (report.html is not None):
        raise ValueError("Report artifacts do not match its evaluation status")
    cohort = window.cohort
    run = client.create_run(
        cohort.experiment_id,
        tags={
            "purpose": "monitoring",
            "mlflow.runName": "on-demand-monitoring",
            "registered_model_name": cohort.model_name,
            "model_version": cohort.model_version,
            "reference_run_id": cohort.training_run_id,
            "representation": REPRESENTATION,
            "report_policy": REPORT_POLICY,
            "cutoff_ms": str(window.cutoff_ms),
            **cohort.identities,
        },
    )
    run_id = run.info.run_id
    try:
        client.log_dict(run_id, manifest, "manifest.json")
        client.log_dict(run_id, report.summary, "summary.json")
        if report.evidently_json is not None:
            client.log_text(run_id, report.evidently_json, "evidently.json")
            assert report.html is not None
            client.log_text(run_id, report.html, "evidently.html")
        client.log_metric(run_id, "current_rows", len(window.current))
        client.log_metric(run_id, "batches", len(window.batches))
        if "drifted_share" in report.summary:
            client.log_metric(run_id, "drifted_share", cast(float, report.summary["drifted_share"]))
        client.set_terminated(run_id, "FINISHED")
    except Exception:
        with suppress(Exception):
            client.set_terminated(run_id, "FAILED")
        raise
    return run_id


def open_saved_report(client: MlflowClient, cohort: Cohort, run_id: str) -> tuple[dict[str, object], MonitoringReport]:
    """Read a completed report for this exact cohort, including after a restart."""
    run = client.get_run(run_id)
    expected = {
        "purpose": "monitoring",
        "registered_model_name": cohort.model_name,
        "model_version": cohort.model_version,
        "reference_run_id": cohort.training_run_id,
        "representation": REPRESENTATION,
        "report_policy": REPORT_POLICY,
        **cohort.identities,
    }
    if run.info.status != "FINISHED" or any(run.data.tags.get(key) != value for key, value in expected.items()):
        raise ValueError("Saved report is unfinished or belongs to another cohort/policy")
    with TemporaryDirectory(prefix="parkinsons-monitoring-") as directory:
        manifest = json.loads(artifact_bytes(client, run_id, "manifest.json", directory))
        summary = json.loads(artifact_bytes(client, run_id, "summary.json", directory))
        if (
            not isinstance(manifest, dict)
            or not isinstance(summary, dict)
            or manifest.get("cohort") != json.loads(json.dumps(asdict(cohort)))
            or manifest.get("policy") != REPORT_POLICY
            or summary.get("policy") != REPORT_POLICY
            or manifest.get("representation") != REPRESENTATION
            or summary.get("status") not in {"ready", "partial", "empty", "insufficient"}
        ):
            raise ValueError("Saved report artifacts disagree with the cohort/policy")
        validate_report(manifest, summary)
        raw = html = None
        if summary.get("status") in {"ready", "partial"}:
            raw = artifact_bytes(client, run_id, "evidently.json", directory).decode()
            html = artifact_bytes(client, run_id, "evidently.html", directory).decode()
    return manifest, MonitoringReport(summary, raw, html)


def promotion_decision(client: MlflowClient, cohort: Cohort, run_id: str) -> dict[str, object]:
    """Read only an audit selected from this model's promotion history."""
    run = client.get_run(run_id)
    if (
        run.info.experiment_id != cohort.experiment_id
        or run.data.tags.get("purpose") != "champion-promotion"
        or run.data.tags.get("registered_model_name") != cohort.model_name
    ):
        raise ValueError("Promotion audit belongs to another model/experiment")
    with TemporaryDirectory(prefix="parkinsons-monitoring-") as directory:
        decision = json.loads(artifact_bytes(client, run_id, "decision.json", directory))
    if not isinstance(decision, dict) or decision.get("registered_model_name") != cohort.model_name:
        raise ValueError("Promotion decision identity is invalid")
    return decision
