"""Track complete comparisons in HTTP MLflow and expose the experimentation CLI."""

import argparse
import ipaddress
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from pathlib import Path
from urllib.parse import urlsplit

import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow.models import infer_signature
from mlflow.sklearn import load_model, log_model
from sklearn.pipeline import Pipeline

from parkinsons_ml.experimentation import (
    EVALUATION_VERSION,
    ComparisonResult,
    ConfigurationResult,
    FeatureSet,
    Metrics,
    TrainConfig,
    TrainingSnapshot,
    UniqueKeyLoader,
    digest,
    evaluate,
    load_training_snapshot,
    select_winner,
    training_config,
    validate_predictions,
)
from parkinsons_ml.schema import POLICY_VERSION, SCHEMA_VERSION
from parkinsons_ml.promotion import promote_winner
from parkinsons_ml.training import build_pipeline

# Training is deliberately repository-only. Validate this source-layout assumption
# before contacting MLflow rather than misidentifying a wheel's site-packages root.
ROOT = Path(__file__).resolve().parents[2]
PROMOTION_LOCK = ROOT / "storage/mlflow/promotion.lock"


def validate_repository() -> None:
    """Fail early and explicitly when repository-only experimentation is unavailable."""
    if (
        Path(__file__).resolve().parent != ROOT / "src/parkinsons_ml"
        or not (ROOT / "uv.lock").is_file()
        or not (ROOT / "pyproject.toml").is_file()
    ):
        raise ValueError(
            "Training requires a source checkout containing src/parkinsons_ml, uv.lock, and pyproject.toml"
        )
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], cwd=ROOT, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(
            "Training requires Git and a Git source checkout; installed wheels are not supported"
        ) from error
    if Path(result.stdout.strip()).resolve() != ROOT:
        raise ValueError("Training source must belong to its own Git checkout")


def lineage() -> dict[str, object]:
    """Allowlisted hashes only; never upload working-tree contents or environment."""

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()

    # HEAD alone cannot describe uncommitted development code. Include source and
    # dependency hashes plus dirty status without uploading unrelated changed files.
    paths = [ROOT / "uv.lock", ROOT / "pyproject.toml", *sorted((ROOT / "src/parkinsons_ml").glob("*.py"))]
    return {
        "git_revision": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "sha256": {str(path.relative_to(ROOT)): digest(path.read_bytes()) for path in paths},
    }


def log_metrics(metrics: Metrics, prefix: str, step: int = 0) -> None:
    # MLflow scalars must be numeric; preserve undefined-score reasons in JSON
    # reports instead. Prefixes distinguish fold validation from pooled OOF scores.
    mlflow.log_metrics(
        {
            f"{prefix}{name}": value
            for name, value in asdict(metrics).items()
            if name != "r2_reason" and value is not None
        },
        step=step,
    )


def log_configuration(snapshot: TrainingSnapshot, selection: FeatureSet) -> ConfigurationResult:
    """Run inside a child context so any required failure marks that child failed."""
    mlflow.log_params(
        {
            "features": json.dumps(selection.features),
            "feature_count": len(selection.features),
            "estimator": "LinearRegression",
            "continuous_imputation": "median",
            "sex_imputation": "most_frequent",
            "test_time": "zero_clip",
        }
    )
    evaluation = evaluate(snapshot, selection.features)
    # Saved folds are contiguous and zero-based, so metric steps match fold IDs.
    # Keep all folds in one child run rather than creating another run hierarchy.
    for fold, metrics in enumerate(evaluation.folds):
        log_metrics(metrics, "validation_", fold)
        mlflow.log_metrics(
            {
                "validation_rows": int((snapshot.folds == fold).sum()),
                "validation_patients": int(snapshot.train.loc[snapshot.folds == fold, "subject"].nunique()),
            },
            step=fold,
        )
    log_metrics(evaluation.pooled, "oof_")
    mlflow.log_metrics({key: value for key, value in evaluation.fold_summary.items() if value is not None})
    # Unlike scalar metrics, this report preserves null scores and their reasons.
    mlflow.log_dict(
        {
            "pooled": asdict(evaluation.pooled),
            "folds": [asdict(item) for item in evaluation.folds],
            "fold_summary": evaluation.fold_summary,
        },
        "evaluation.json",
    )
    train = snapshot.train
    # Use snapshot-local measurement positions for alignment, not patient IDs.
    # Grouping assignments stay in separate preparation/audit artifacts.
    predictions = pd.DataFrame(
        {
            "measurement": np.arange(len(train)),
            "fold": snapshot.folds,
            "actual": train["motor_updrs"],
            "prediction": evaluation.predictions,
        }
    )
    mlflow.log_text(predictions.to_csv(index=False), "predictions/oof.csv")
    # Validation is complete for this configuration. Its saved model is a separate
    # all-train refit; fitted values below must never replace held-out selection scores.
    pipeline = build_pipeline(selection.features)
    pipeline.fit(train, train["motor_updrs"])
    # Examples, signatures, and references expose only canonical selected predictors,
    # even though pipeline selection also protects fits on the wider training table.
    inputs = train.loc[:, list(selection.features)]
    expected = np.asarray(pipeline.predict(inputs), dtype=np.float64)
    validate_predictions(expected, (len(train),))
    # Serialize preprocessing together with LinearRegression and package custom code.
    # Keep the skops trust list narrow rather than trusting arbitrary loaded types.
    info = log_model(
        pipeline,
        name="pipeline",
        input_example=inputs.head(5),
        signature=infer_signature(inputs, expected),
        code_paths=[str(ROOT / "src/parkinsons_ml")],
        skops_trusted_types=["numpy.dtype", "parkinsons_ml.training.FeaturePreparation"],
    )
    # Verify the uploaded model URI, not a local pickle or the in-memory estimator.
    # A serialization/loading mismatch fails the child and therefore the comparison.
    loaded = load_model(info.model_uri)
    if not isinstance(loaded, Pipeline):
        raise ValueError("Reloaded model must be a complete pipeline")
    np.testing.assert_allclose(loaded.predict(inputs), expected, rtol=1e-12, atol=1e-12)
    # Retain natural missingness before learned imputation for later drift analysis.
    # Hash the same CSV representation that is uploaded, with no patient identifier.
    reference_csv = inputs.to_csv(index=False)
    mlflow.log_text(reference_csv, "monitoring/reference.csv")
    mlflow.log_dict(
        {
            "representation": "selected-pre-imputation-v1",
            "features": list(selection.features),
            "sha256": digest(reference_csv.encode()),
            "rows": len(inputs),
        },
        "monitoring/identity.json",
    )
    mlflow.log_text(pd.DataFrame({"prediction": expected}).to_csv(index=False), "predictions/refit.csv")
    # Describe our fixed pipeline explicitly instead of filtering a generic deep
    # parameter tree, which silently drops non-primitive settings.
    mlflow.log_dict(
        {
            "estimator": {
                "class": "LinearRegression",
                "parameters": pipeline.named_steps["linearregression"].get_params(deep=False),
            },
            "preprocessing": {
                "features": list(selection.features),
                "continuous_imputation": "median",
                "sex_imputation": "most_frequent",
                "test_time": "zero_clip",
            },
        },
        "pipeline_parameters.json",
    )
    run = mlflow.active_run()
    assert run is not None
    return ConfigurationResult(selection.name, selection.features, run.info.run_id, info.model_uri, evaluation.pooled)


def run_experiment(config: TrainConfig, directory: Path, tracking_uri: str) -> ComparisonResult:
    """Complete a comparison, then optionally promote its winner without final-test access."""
    # TrainConfig validates itself at construction, including direct Python use.
    # Require an explicit server and keep embedded credentials/query values out of
    # URIs that MLflow may persist or display. No local tracking fallback is allowed.
    parsed = urlsplit(tracking_uri)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("An explicit HTTP/HTTPS tracking URI without embedded credentials is required")
    # Plain HTTP is suitable only for same-host loopback traffic. Parse IP literals
    # directly rather than resolving arbitrary DNS names that could later change.
    if parsed.scheme == "http" and parsed.hostname != "localhost":
        try:
            loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            loopback = False
        if not loopback:
            raise ValueError("Remote MLflow servers require HTTPS; HTTP is allowed only for loopback hosts")
    # Fluent MLflow can resume an environment-specified run even without an active
    # context. Refuse both cases rather than taking ownership of a caller's run.
    if mlflow.active_run() is not None or os.environ.get("MLFLOW_RUN_ID"):
        raise ValueError("Cannot start comparison inside an unrelated active run")
    # Repository/Git errors must not leave newly created experiments or failed runs.
    validate_repository()
    # URI setters affect process state and environment variables. Save their prior
    # values so successful comparisons and failures both restore caller routing.
    old_tracking, old_registry = mlflow.get_tracking_uri(), mlflow.get_registry_uri()
    environment = {key: os.environ.get(key) for key in ("MLFLOW_TRACKING_URI", "MLFLOW_REGISTRY_URI")}
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(tracking_uri)
        client = mlflow.MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
        experiment = client.get_experiment_by_name(config.experiment_name)
        experiment_id = experiment.experiment_id if experiment else client.create_experiment(config.experiment_name)
        # HTTP tracking alone is insufficient: an existing experiment could still
        # point clients at a local artifact directory instead of the server proxy.
        if not client.get_experiment(experiment_id).artifact_location.startswith("mlflow-artifacts:/"):
            raise ValueError("Experiment must use HTTP-served mlflow-artifacts storage")
        # Open the diagnostic parent before validating prepared inputs. Exceptions
        # propagate through the contexts, marking runs failed when MLflow is reachable.
        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name="comparison",
            tags={"purpose": "training-comparison", "evaluation_id": EVALUATION_VERSION},
        ) as parent:
            mlflow.log_dict(asdict(config), "configuration.json")
            snapshot = load_training_snapshot(directory)
            # Attach the same comparability identities to the parent and every child.
            tags = {
                "dataset_id": snapshot.dataset_id,
                "split_id": snapshot.split_id,
                "schema_version": SCHEMA_VERSION,
                "policy_version": POLICY_VERSION,
                "evaluation_id": EVALUATION_VERSION,
            }
            mlflow.set_tags(tags)
            mlflow.log_dict(lineage(), "lineage.json")
            # Grouping/audit metadata is intentionally separate from prediction artifacts.
            for name, payload in snapshot.artifacts.items():
                if name != "train.csv":
                    mlflow.log_text(payload.decode(), f"preparation/{name}")
            results: list[ConfigurationResult] = []
            # Evaluate/refit/log sequentially against one shared snapshot. Completed
            # children may remain for diagnosis if any later configuration fails.
            for selection in config.feature_sets:
                with mlflow.start_run(experiment_id=experiment_id, nested=True, run_name=selection.name, tags=tags):
                    mlflow.log_metrics(
                        {"train_rows": len(snapshot.train), "train_patients": int(snapshot.train["subject"].nunique())}
                    )
                    results.append(log_configuration(snapshot, selection))
            # Publish a winner only after every required child operation succeeds;
            # never select opportunistically from an incomplete comparison.
            winner = select_winner(results)
            result = ComparisonResult(
                parent.info.run_id, tuple(results), winner, snapshot.dataset_id, snapshot.split_id
            )
            mlflow.log_dict(asdict(result), "comparison.json")
            mlflow.set_tags(
                {"winner": winner.name, "winner_run_id": winner.run_id, "winner_model_uri": winner.model_uri}
            )
        # Promotion sees only FINISHED comparisons. Its outcome is separate from
        # training success and has its own audit run; failures preserve training records.
        promotion = promote_winner(client, result, config.promotion, PROMOTION_LOCK)
        return replace(result, promotion=promotion)
    finally:
        # Restore routing even if server access, fitting, or artifact operations fail.
        # Preserve unset environment variables as unset, not as getter defaults.
        mlflow.set_tracking_uri(old_tracking)
        mlflow.set_registry_uri(old_registry)
        for key, value in environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    # Keep this adapter thin: configuration checks and orchestration are also used
    # directly by Python callers, without depending on argparse or printed output.
    settings = yaml.load(args.params.read_text(), Loader=UniqueKeyLoader)
    if not isinstance(settings, dict):
        raise ValueError("Parameters must contain a train mapping")
    config = training_config(settings.get("train"))
    # MLflow prints run links to stdout when contexts close. Keep those diagnostics
    # visible on stderr so the CLI's stdout remains a single machine-readable result.
    with redirect_stdout(sys.stderr):
        result = run_experiment(config, args.data, os.environ.get("MLFLOW_TRACKING_URI", ""))
    print(json.dumps(asdict(result), allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
