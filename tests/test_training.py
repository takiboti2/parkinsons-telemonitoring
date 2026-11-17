"""Leakage prevention and complete preprocessing pipeline serialization."""

import pickle
from collections.abc import Callable
from itertools import permutations
from typing import cast

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from parkinsons_ml.schema import BASELINE_FEATURES, CANDIDATE_FEATURES
from parkinsons_ml.training import FeaturePreparation, build_pipeline


@pytest.fixture
def features() -> pd.DataFrame:
    frame = pd.DataFrame(
        {"age": [50.0, 60.0, np.nan, 80.0], "sex": [0.0, 1.0, np.nan, 0.0], "test_time": [-1.0, 1.0, 2.0, np.nan]}
    )
    for name in CANDIDATE_FEATURES[3:]:
        frame[name] = [1.0, np.nan, 3.0, 4.0]
    return frame


@pytest.mark.parametrize("selected", [BASELINE_FEATURES, CANDIDATE_FEATURES], ids=["baseline", "candidate"])
def test_pipeline_fit_imputation_and_round_trip(
    features: pd.DataFrame,
    selected: tuple[str, ...],
) -> None:
    features["motor_updrs"] = [1000.0] * 4
    features["total_updrs"] = [-1000.0] * 4
    features["subject"] = [999] * 4
    original = features.copy(deep=True)
    pipeline = build_pipeline(selected)
    pipeline.fit(features, [10.0, 12.0, 14.0, 16.0])
    assert isinstance(pipeline.named_steps["linearregression"], LinearRegression)
    imputation = pipeline.named_steps["imputation"]
    assert imputation.named_transformers_["sex"].statistics_[0] == 0.0
    assert imputation.named_transformers_["age"].statistics_[0] == 60.0
    prepared = pipeline[:-1].transform(features)
    assert np.isfinite(prepared).all()
    canonical = tuple(sorted(selected))
    assert prepared[0, canonical.index("test_time")] == 0
    assert set(prepared[:, canonical.index("sex")]) <= {0.0, 1.0}
    predictions = pipeline.predict(features)
    loaded = pickle.loads(pickle.dumps(pipeline))
    np.testing.assert_allclose(loaded.predict(features), predictions)
    changed = features.assign(motor_updrs=-99, total_updrs=99, subject=-1)
    np.testing.assert_allclose(pipeline.predict(changed), predictions)
    assert prepared.shape[1] == len(selected)
    pd.testing.assert_frame_equal(features, original)
    cloned = clone(pipeline)
    assert isinstance(cloned, Pipeline)
    cloned.fit(features, [10.0, 12.0, 14.0, 16.0])


def test_validation_does_not_fit_on_evaluation(features: pd.DataFrame) -> None:
    pipeline = build_pipeline()
    pipeline.fit(features.iloc[:3], [1.0, 2.0, 3.0])
    evaluation = features.iloc[3:].assign(age=10000.0, sex=np.nan, test_time=-5)
    transformed = pipeline[:-1].transform(evaluation)
    assert transformed[0, BASELINE_FEATURES.index("sex")] == 0
    assert transformed[0, BASELINE_FEATURES.index("test_time")] == 0
    assert pipeline.named_steps["imputation"].named_transformers_["age"].statistics_[0] == 55.0
    assert pipeline.named_steps["imputation"].named_transformers_["sex"].statistics_[0] == 0.0


@pytest.mark.parametrize("column", CANDIDATE_FEATURES)
def test_all_missing_fitting_feature_rejected(features: pd.DataFrame, column: str) -> None:
    features[column] = np.nan
    with pytest.raises(ValueError, match="Entirely missing fitting features"):
        build_pipeline([column]).fit(features, [1, 2, 3, 4])


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
    nullable = features.astype("Float64")
    np.testing.assert_allclose(transformer.transform(nullable), transformer.transform(features), equal_nan=True)


@pytest.mark.parametrize("column", CANDIDATE_FEATURES)
def test_complex_predictors_rejected_before_casting(features: pd.DataFrame, column: str) -> None:
    features[column] = [60 + 4j] * len(features)
    original = features.copy(deep=True)
    with pytest.raises(ValueError, match="must be real, not complex"):
        build_pipeline(CANDIDATE_FEATURES).fit(features, [1.0, 2.0, 3.0, 4.0])
    pd.testing.assert_frame_equal(features, original)


@pytest.mark.parametrize(
    "selection, message",
    [
        (None, "ordered sequence"),
        ("baseline", "ordered sequence"),
        ("candidate", "ordered sequence"),
        (b"age", "ordered sequence"),
        ({"age", "sex"}, "ordered sequence"),
        ({"age": 1}, "ordered sequence"),
        ((name for name in ["age"]), "ordered sequence"),
        ([], "must not be empty"),
        ((), "must not be empty"),
        (["age", 1], "only string"),
        (["age", ["sex"]], "only string"),
        (["age", "age"], "Duplicate features"),
        (["sex", "age", "sex"], "Duplicate features"),
        (["motor_updrs"], "outside CANDIDATE_FEATURES"),
        (["total_updrs"], "outside CANDIDATE_FEATURES"),
        (["subject"], "outside CANDIDATE_FEATURES"),
        (["subject#"], "outside CANDIDATE_FEATURES"),
        (["Jitter(%)"], "outside CANDIDATE_FEATURES"),
        (["Age"], "outside CANDIDATE_FEATURES"),
        (["age "], "outside CANDIDATE_FEATURES"),
        (["unknown"], "outside CANDIDATE_FEATURES"),
    ],
)
@pytest.mark.parametrize("entry_point", ["builder", "transformer"])
def test_invalid_feature_selections_rejected(
    features: pd.DataFrame,
    selection: object,
    message: str,
    entry_point: str,
) -> None:
    # Deliberately invalid runtime input is isolated at this test boundary; the
    # normal builder and transformer interfaces retain their precise annotations.
    unchecked_builder = cast(Callable[..., Pipeline], build_pipeline)
    unchecked_transformer = cast(Callable[..., FeaturePreparation], FeaturePreparation)
    with pytest.raises(ValueError, match=message):
        if entry_point == "builder":
            unchecked_builder(selection)
        else:
            unchecked_transformer(selection).fit(features)


@pytest.mark.parametrize(
    "selected",
    [
        ("age",),
        ("sex",),
        ("test_time",),
        ("ppe",),
        ("ppe", "sex", "age"),
        ("jitter", "ppe"),
    ],
)
def test_custom_subsets_canonicalize_order_and_preserve_imputation_and_serialization(
    features: pd.DataFrame,
    selected: tuple[str, ...],
) -> None:
    # Only selected columns are supplied, deliberately in reverse DataFrame order.
    inputs = features.loc[:, list(reversed(selected))].copy()
    original = inputs.copy(deep=True)
    pipeline = build_pipeline(list(selected))
    pipeline.fit(inputs, [10.0, 12.0, 14.0, 16.0])
    prepared = pipeline[:-1].transform(inputs)
    expected: dict[str, list[float]] = {
        "age": [50.0, 60.0, 60.0, 80.0],
        "sex": [0.0, 1.0, 0.0, 0.0],
        "test_time": [0.0, 1.0, 2.0, 1.0],
        "ppe": [1.0, 3.0, 3.0, 4.0],
        "jitter": [1.0, 3.0, 3.0, 4.0],
    }
    canonical = tuple(sorted(selected))
    for index, name in enumerate(canonical):
        np.testing.assert_allclose(prepared[:, index], expected[name])
    assert prepared.shape == (len(inputs), len(selected))
    selection_step = pipeline.named_steps["features"]
    assert isinstance(selection_step, FeaturePreparation)
    assert selection_step.features == canonical
    loaded = pickle.loads(pickle.dumps(pipeline))
    np.testing.assert_allclose(loaded.predict(inputs), pipeline.predict(inputs))
    cloned = clone(pipeline)
    assert isinstance(cloned, Pipeline)
    cloned.fit(inputs, [10.0, 12.0, 14.0, 16.0])
    np.testing.assert_allclose(cloned.predict(inputs), pipeline.predict(inputs))
    pd.testing.assert_frame_equal(inputs, original)


def test_builder_snapshots_mutable_feature_list(features: pd.DataFrame) -> None:
    selected = ["sex", "ppe"]
    pipeline = build_pipeline(selected)
    selected[0] = "motor_updrs"
    selected.append("subject")
    pipeline.fit(features[["ppe", "sex"]], [10.0, 12.0, 14.0, 16.0])
    selection_step = pipeline.named_steps["features"]
    assert isinstance(selection_step, FeaturePreparation)
    assert selection_step.features == ("ppe", "sex")
    assert np.isfinite(pipeline.predict(features[["ppe", "sex"]])).all()


def test_unselected_values_and_columns_are_not_required(features: pd.DataFrame) -> None:
    pipeline = build_pipeline(["ppe"])
    # The all-missing check must not reject unused fitting features.
    fitting = features.assign(age=np.nan, sex=np.nan, test_time=np.nan)
    pipeline.fit(fitting, [10.0, 12.0, 14.0, 16.0])
    expected = pipeline.predict(features[["ppe"]])
    invalid_extras = features.assign(
        age="malformed",
        sex=7,
        test_time=np.inf,
        motor_updrs="excluded",
        total_updrs=np.inf,
        subject="excluded",
    )
    np.testing.assert_allclose(pipeline.predict(invalid_extras), expected)
    with pytest.raises(ValueError, match="Missing prediction features:.*ppe"):
        pipeline.predict(features.drop(columns="ppe"))


def test_builder_defaults_to_baseline() -> None:
    selection_step = build_pipeline().named_steps["features"]
    assert isinstance(selection_step, FeaturePreparation)
    assert selection_step.features == BASELINE_FEATURES


def test_feature_permutations_produce_identical_models(features: pd.DataFrame) -> None:
    canonical = ("age", "ppe", "sex")
    reference = build_pipeline(canonical)
    reference.fit(features, [10.0, 12.0, 14.0, 16.0])
    for selection in permutations(canonical):
        pipeline = build_pipeline(list(selection))
        selection_step = pipeline.named_steps["features"]
        assert isinstance(selection_step, FeaturePreparation)
        assert selection_step.features == canonical
        # Neither configuration order nor the incoming DataFrame's column order
        # may change the matrix passed to LinearRegression or its coefficients.
        inputs = features.loc[:, list(reversed(selection))]
        pipeline.fit(inputs, [10.0, 12.0, 14.0, 16.0])
        np.testing.assert_allclose(pipeline[:-1].transform(inputs), reference[:-1].transform(features))
        np.testing.assert_allclose(
            pipeline.named_steps["linearregression"].coef_,
            reference.named_steps["linearregression"].coef_,
        )
        np.testing.assert_allclose(pipeline.predict(inputs), reference.predict(features))


def test_direct_transformer_uses_canonical_order_and_correct_missing_feature_names(features: pd.DataFrame) -> None:
    transformer = FeaturePreparation(("sex", "age"))
    # Constructor parameters stay unchanged for sklearn cloning, but transforms
    # and missing-feature diagnostics must use the actual canonical matrix order.
    prepared = transformer.transform(features)
    np.testing.assert_allclose(prepared[:, 0], features["age"], equal_nan=True)
    np.testing.assert_allclose(prepared[:, 1], features["sex"], equal_nan=True)
    with pytest.raises(ValueError, match=r"Entirely missing fitting features: \['age'\]"):
        transformer.fit(features.assign(age=np.nan))
