"""Check model serialization and report generation across pinned dependencies."""

from collections.abc import Iterator
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from evidently import Report
from evidently.presets import DataDriftPreset
from mlflow.sklearn import load_model, log_model
from mlflow.utils.thread_utils import ThreadLocalVariable
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline

from parkinsons_ml.training import build_pipeline


@pytest.fixture
def isolated_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Use a temporary backend without leaking run or configuration state."""
    assert mlflow.active_run() is None
    uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    with monkeypatch.context() as isolated:
        isolated.chdir(tmp_path)
        # Snapshot the pinned MLflow version's overrides, including an unset value.
        # Restoring only public getters would turn URI fallback into an explicit URI.
        isolated.setattr("mlflow.tracking._tracking_service.utils._tracking_uri", uri)
        isolated.setattr("mlflow.tracking._model_registry.utils._registry_uri", uri)
        isolated.setattr("mlflow.tracking.fluent._active_experiment_id", None)
        # end_run() writes this thread-local state; restore the original object
        # rather than leaving a temporary run ID attached to the old backend.
        isolated.setattr("mlflow.tracking.fluent._last_active_run_id", ThreadLocalVariable(lambda: None))
        isolated.setenv("MLFLOW_TRACKING_URI", uri)
        isolated.setenv("MLFLOW_REGISTRY_URI", uri)
        isolated.delenv("MLFLOW_EXPERIMENT_ID", raising=False)
        isolated.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
        isolated.delenv("MLFLOW_RUN_ID", raising=False)
        try:
            yield mlflow.create_experiment("dependency-compatibility")
        finally:
            mlflow.end_run()


def test_mlflow_model_round_trip(isolated_mlflow: str) -> None:
    """Validate a registered version before promoting and reloading its alias."""
    features = pd.DataFrame(
        {"age": [60.0, 62.0, np.nan, 66.0], "sex": [0.0, 1.0, 0.0, 1.0], "test_time": [0.0, 7.0, 14.0, 21.0]}
    )
    target = np.array([10.0, 12.0, 14.0, 16.0])
    pipeline = make_pipeline(SimpleImputer(strategy="median"), LinearRegression())
    pipeline.fit(features, target)
    expected = pipeline.predict(features)
    assert np.isfinite(expected).all()

    with mlflow.start_run(experiment_id=isolated_mlflow) as run:
        mlflow.log_param("estimator", "LinearRegression")
        mlflow.log_param("feature_set", "demographics_time")
        mlflow.log_metric("mae", float(np.mean(np.abs(expected - target))))
        mlflow.log_dict({"status": "ok"}, "compatibility.json")
        info = log_model(
            pipeline,
            name="pipeline",
            input_example=features,
            # The locally fitted imputer stores a NumPy dtype; no tree types are needed.
            skops_trusted_types=["numpy.dtype"],
        )

    version = mlflow.register_model(info.model_uri, "compatibility-model")
    client = mlflow.MlflowClient()
    candidate = load_model(f"models:/compatibility-model/{version.version}")
    assert isinstance(candidate, Pipeline)
    assert isinstance(candidate.named_steps["linearregression"], LinearRegression)
    np.testing.assert_allclose(candidate.predict(features), expected)
    client.set_registered_model_alias("compatibility-model", "champion", version.version)
    loaded = load_model("models:/compatibility-model@champion")
    assert isinstance(loaded, Pipeline)
    np.testing.assert_allclose(loaded.predict(features), expected)
    parameters = client.get_run(run.info.run_id).data.params
    assert parameters["estimator"] == "LinearRegression"
    assert parameters["feature_set"] == "demographics_time"
    assert client.list_artifacts(run.info.run_id)[0].path == "compatibility.json"


def test_evidently_report(tmp_path: Path) -> None:
    """Exercise Evidently with pandas and serialize a dashboard-ready report."""
    reference = pd.DataFrame({"voice": np.linspace(0.0, 1.0, 100)})
    current = reference + 5.0
    report = Report([DataDriftPreset()]).run(reference_data=reference, current_data=current)
    assert report.dict()
    output = tmp_path / "drift.html"
    report.save_html(str(output))
    assert output.stat().st_size > 0


def test_project_pipeline_mlflow_round_trip(isolated_mlflow: str) -> None:
    """The actual feature selection/clipping/imputation pipeline survives MLflow."""
    features = pd.DataFrame(
        {"age": [50.0, 60.0, np.nan, 80.0], "sex": [0.0, 1.0, np.nan, 0.0], "test_time": [-5.0, 1.0, 2.0, np.nan]}
    )
    pipeline = build_pipeline()
    pipeline.fit(features, [10.0, 12.0, 14.0, 16.0])
    with mlflow.start_run(experiment_id=isolated_mlflow):
        info = log_model(
            pipeline,
            name="prepared-pipeline",
            input_example=features,
            skops_trusted_types=["numpy.dtype", "parkinsons_ml.training.FeaturePreparation"],
        )
    loaded = load_model(info.model_uri)
    assert isinstance(loaded, Pipeline)
    np.testing.assert_allclose(loaded.predict(features), pipeline.predict(features))
