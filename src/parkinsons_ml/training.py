"""Build complete preprocessing pipelines and retain legacy training entry points."""

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from parkinsons_ml.schema import BASELINE_FEATURES, CANDIDATE_FEATURES, clip_test_time

if TYPE_CHECKING:
    from parkinsons_ml.experimentation import ComparisonResult, TrainConfig


def validate_features(features: object) -> tuple[str, ...]:
    """Validate a runtime selection and return its canonical alphabetical order."""
    # Accept ordered inputs so duplicates remain detectable. Strings, sets, and
    # generators are not supported selections; check types before hashing names.
    if isinstance(features, (str, bytes)) or not isinstance(features, Sequence):
        raise ValueError("features must be an ordered sequence of column names, not a string")
    selected: list[str] = []
    for name in features:
        if not isinstance(name, str):
            raise ValueError("features must contain only string column names")
        selected.append(name)
    if not selected:
        raise ValueError("features must not be empty")
    duplicates = sorted({name for name in selected if selected.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate features: {duplicates}")
    unknown = sorted(set(selected) - set(CANDIDATE_FEATURES))
    if unknown:
        raise ValueError(f"Features outside CANDIDATE_FEATURES: {unknown}")
    # Sort only after validation: converting to a set would silently hide duplicate
    # configuration entries. Equivalent lists now have identical saved selections.
    return tuple(sorted(selected))


class FeaturePreparation(TransformerMixin, BaseEstimator):
    """Select approved features, validate numeric inputs and clip times consistently."""

    def __init__(self, features: tuple[str, ...] = BASELINE_FEATURES) -> None:
        # Store constructor arguments unchanged so scikit-learn can clone an
        # unfitted instance for each validation fold.
        self.features = features

    def fit(self, X: pd.DataFrame, y: object = None) -> "FeaturePreparation":
        values = self.transform(X)
        # An all-missing fitting feature has no training-derived imputation value.
        # Reject it instead of allowing the imputer to silently drop that column.
        selected = validate_features(self.features)
        empty = [name for index, name in enumerate(selected) if np.isnan(values[:, index]).all()]
        if empty:
            raise ValueError(f"Entirely missing fitting features: {empty}")
        return self

    def transform(self, X: pd.DataFrame) -> NDArray[np.float64]:
        # Also validate direct transformer use and sklearn parameter updates;
        # constructor arguments stay unchanged so cloning remains supported.
        selected = validate_features(self.features)
        if X.columns.has_duplicates:
            raise ValueError("Duplicate prediction columns")
        missing = set(selected) - set(X.columns)
        if missing:
            raise ValueError(f"Missing prediction features: {sorted(missing)}")
        # Select and order only approved predictors. Extra target/identifier
        # columns in prepared training tables cannot leak into the estimator.
        frame = X.loc[:, list(selected)]
        # Complex columns are numeric to pandas, but casting them to float would
        # silently discard imaginary components. Reject them before conversion.
        if any(pd.api.types.is_complex_dtype(frame[name]) for name in selected):
            raise ValueError("Prediction features must be real, not complex")
        if any(
            not pd.api.types.is_numeric_dtype(frame[name]) or pd.api.types.is_bool_dtype(frame[name])
            for name in selected
        ):
            raise ValueError("Prediction features must be numeric, not strings or booleans")
        # Work on a copy so clipping cannot mutate a caller's DataFrame. Convert
        # pandas nullable numeric values into the NaN representation sklearn uses.
        values = frame.to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
        if np.isinf(values).any():
            raise ValueError("Infinite prediction values")
        # Missing values are allowed, but observed sex/age values must still obey
        # their encodings. Imputation must not disguise invalid observed values.
        if "sex" in selected:
            sex = values[:, selected.index("sex")]
            if not np.isin(sex[~np.isnan(sex)], [0.0, 1.0]).all():
                raise ValueError("Invalid sex encoding")
        if "age" in selected:
            age = values[:, selected.index("age")]
            if not np.equal(age[~np.isnan(age)], np.floor(age[~np.isnan(age)])).all():
                raise ValueError("Noninteger age")
        if "test_time" in selected:
            index = selected.index("test_time")
            # This idempotent rule also runs for standalone serialized-model calls,
            # keeping inference consistent with prepared train/test representation.
            values[:, index] = [clip_test_time(float(value)) for value in values[:, index]]
        return values


def build_pipeline(features: Sequence[str] = BASELINE_FEATURES) -> Pipeline:
    """Construct an unfitted pipeline suitable for fold fitting and serialization."""
    selected = validate_features(features)
    # Learn continuous medians and the categorical mode separately. A median of
    # binary sex values could otherwise produce the invalid category 0.5.
    imputation = ColumnTransformer(
        [
            (name, SimpleImputer(strategy="most_frequent" if name == "sex" else "median"), [index])
            for index, name in enumerate(selected)
        ]
    )
    # Return unfitted steps: fold-specific fit calls learn imputation values only
    # from that fold's training patients, and serialization retains every step.
    return Pipeline(
        [
            # One imputer per column preserves canonical alphabetical order
            # through to the estimator, including a sex-only selection.
            ("features", FeaturePreparation(selected)),
            ("imputation", imputation),
            ("linearregression", LinearRegression()),
        ]
    )


def training_config(value: object) -> "TrainConfig":
    """Compatibility entry point; new callers should import from experimentation."""
    # Import only when called so pipeline/model loading does not pull in the
    # experimentation module or create an import cycle through its pipeline builder.
    from parkinsons_ml.experimentation import training_config as configure

    return configure(value)


def run_experiment(config: "TrainConfig", directory: Path, tracking_uri: str) -> "ComparisonResult":
    """Compatibility entry point; MLflow orchestration now belongs to tracking."""
    # Keep tracking dependencies out of inference-only imports of this module.
    from parkinsons_ml.tracking import run_experiment as run

    return run(config, directory, tracking_uri)


def main() -> None:
    """Preserve python -m parkinsons_ml.training as a thin CLI delegate."""
    from parkinsons_ml.tracking import main as tracking_main

    tracking_main()


if __name__ == "__main__":
    main()
