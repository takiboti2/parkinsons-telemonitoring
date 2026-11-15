"""Leakage prevention and complete preprocessing pipeline serialization."""

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from parkinsons_ml.schema import CANDIDATE_FEATURES
from parkinsons_ml.training import FeaturePreparation, build_pipeline


@pytest.fixture
def features() -> pd.DataFrame:
    return pd.DataFrame(
        {"age": [50.0, 60.0, np.nan, 80.0], "sex": [0.0, 1.0, np.nan, 0.0], "test_time": [-1.0, 1.0, 2.0, np.nan]}
    )


@pytest.mark.parametrize("feature_set", ["baseline", "candidate"])
def test_pipeline_fit_imputation_and_round_trip(features: pd.DataFrame, feature_set: str) -> None:
    for name in CANDIDATE_FEATURES[3:]:
        features[name] = [1.0, np.nan, 3.0, 4.0]
    features["motor_updrs"] = [1000.0] * 4
    features["total_updrs"] = [-1000.0] * 4
    features["subject"] = [999] * 4
    original = features.copy(deep=True)
    pipeline = build_pipeline(feature_set)
    pipeline.fit(features, [10.0, 12.0, 14.0, 16.0])
    assert isinstance(pipeline.named_steps["linearregression"], LinearRegression)
    imputation = pipeline.named_steps["imputation"]
    assert imputation.named_transformers_["sex"].statistics_[0] == 0.0
    assert imputation.named_transformers_["continuous"].statistics_[0] == 60.0
    prepared = pipeline[:-1].transform(features)
    assert np.isfinite(prepared).all()
    assert prepared[0, 1] == 0  # test_time follows age among continuous columns.
    assert set(prepared[:, -1]) <= {0.0, 1.0}
    predictions = pipeline.predict(features)
    loaded = pickle.loads(pickle.dumps(pipeline))
    np.testing.assert_allclose(loaded.predict(features), predictions)
    changed = features.assign(motor_updrs=-99, total_updrs=99, subject=-1)
    np.testing.assert_allclose(pipeline.predict(changed), predictions)
    assert prepared.shape[1] == (3 if feature_set == "baseline" else 19)
    pd.testing.assert_frame_equal(features, original)
    clone(pipeline).fit(features, [10.0, 12.0, 14.0, 16.0])


def test_validation_does_not_fit_on_evaluation(features: pd.DataFrame) -> None:
    pipeline = build_pipeline()
    pipeline.fit(features.iloc[:3], [1.0, 2.0, 3.0])
    evaluation = features.iloc[3:].assign(age=10000.0, sex=np.nan, test_time=-5)
    transformed = pipeline[:-1].transform(evaluation)
    assert transformed[0, -1] == 0
    assert transformed[0, 1] == 0
    assert pipeline.named_steps["imputation"].named_transformers_["continuous"].statistics_[0] == 55.0
    assert pipeline.named_steps["imputation"].named_transformers_["sex"].statistics_[0] == 0.0


@pytest.mark.parametrize("column", ["age", "sex", "test_time"])
def test_all_missing_fitting_feature_rejected(features: pd.DataFrame, column: str) -> None:
    features[column] = np.nan
    with pytest.raises(ValueError, match="Entirely missing fitting features"):
        build_pipeline().fit(features, [1, 2, 3, 4])


def test_prediction_validation(features: pd.DataFrame) -> None:
    transformer = FeaturePreparation()
    invalid = [
        (features.drop(columns="age"), "Missing prediction features"),
        (pd.concat([features, features[["age"]]], axis=1), "Duplicate prediction columns"),
        (features.assign(age="50"), "must be numeric"),
        (features.assign(sex=True), "must be numeric"),
        (features.assign(age=np.inf), "Infinite"),
        (features.assign(sex=2.0), "Invalid sex"),
        (features.assign(age=50.5), "Noninteger age"),
    ]
    for frame, message in invalid:
        with pytest.raises(ValueError, match=message):
            transformer.transform(frame)
    with pytest.raises(ValueError, match="feature_set"):
        build_pipeline("unknown")
    nullable = features.astype("Float64")
    np.testing.assert_allclose(transformer.transform(nullable), transformer.transform(features), equal_nan=True)
