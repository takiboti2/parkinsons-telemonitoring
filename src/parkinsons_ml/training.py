"""Build complete preprocessing pipelines; training orchestration comes later."""

from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from parkinsons_ml.schema import BASELINE_FEATURES, CANDIDATE_FEATURES, clip_test_time


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
        empty = [name for index, name in enumerate(self.features) if np.isnan(values[:, index]).all()]
        if empty:
            raise ValueError(f"Entirely missing fitting features: {empty}")
        return self

    def transform(self, X: pd.DataFrame) -> NDArray[np.float64]:
        if X.columns.has_duplicates:
            raise ValueError("Duplicate prediction columns")
        missing = set(self.features) - set(X.columns)
        if missing:
            raise ValueError(f"Missing prediction features: {sorted(missing)}")
        # Select and order only approved predictors. Extra target/identifier
        # columns in prepared training tables cannot leak into the estimator.
        frame = X.loc[:, list(self.features)]
        # Complex columns are numeric to pandas, but casting them to float would
        # silently discard imaginary components. Reject them before conversion.
        if any(pd.api.types.is_complex_dtype(frame[name]) for name in self.features):
            raise ValueError("Prediction features must be real, not complex")
        if any(
            not pd.api.types.is_numeric_dtype(frame[name]) or pd.api.types.is_bool_dtype(frame[name])
            for name in self.features
        ):
            raise ValueError("Prediction features must be numeric, not strings or booleans")
        # Work on a copy so clipping cannot mutate a caller's DataFrame. Convert
        # pandas nullable numeric values into the NaN representation sklearn uses.
        values = frame.to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
        if np.isinf(values).any():
            raise ValueError("Infinite prediction values")
        # Missing values are allowed, but observed sex/age values must still obey
        # their encodings. Imputation must not disguise invalid observed values.
        sex = values[:, self.features.index("sex")]
        if not np.isin(sex[~np.isnan(sex)], [0.0, 1.0]).all():
            raise ValueError("Invalid sex encoding")
        age = values[:, self.features.index("age")]
        if not np.equal(age[~np.isnan(age)], np.floor(age[~np.isnan(age)])).all():
            raise ValueError("Noninteger age")
        index = self.features.index("test_time")
        # This idempotent rule also runs for standalone serialized-model calls,
        # keeping inference consistent with the prepared train/test representation.
        values[:, index] = [clip_test_time(float(value)) for value in values[:, index]]
        return values


def build_pipeline(feature_set: Literal["baseline", "candidate"] = "baseline") -> Pipeline:
    """Construct an unfitted pipeline suitable for fold fitting and serialization."""
    if feature_set not in ("baseline", "candidate"):
        raise ValueError("feature_set must be baseline or candidate")
    features = BASELINE_FEATURES if feature_set == "baseline" else CANDIDATE_FEATURES
    numeric = [index for index, name in enumerate(features) if name != "sex"]
    # Learn continuous medians and the categorical mode separately. A median of
    # binary sex values could otherwise produce the invalid category 0.5.
    imputation = ColumnTransformer(
        [
            ("continuous", SimpleImputer(strategy="median"), numeric),
            ("sex", SimpleImputer(strategy="most_frequent"), [features.index("sex")]),
        ]
    )
    # Return unfitted steps: fold-specific fit calls learn imputation values only
    # from that fold's training patients, and serialization retains every step.
    return Pipeline(
        [
            ("features", FeaturePreparation(features)),
            ("imputation", imputation),
            ("linearregression", LinearRegression()),
        ]
    )
