"""Versioned raw schema and deterministic transformations shared with inference."""

import math
import re
import unicodedata
from collections.abc import Sequence

# Manifests record these identities so consumers can distinguish schema changes
# from changes to cleaning rules, even when the raw dataset is unchanged.
SCHEMA_VERSION = "1"
POLICY_VERSION = "3"
# Preserve UCI's exact spellings at ingestion; canonical names apply only to
# derived data and model inputs, never to the original CSV.
RAW_COLUMNS = (
    "subject#",
    "age",
    "sex",
    "test_time",
    "motor_UPDRS",
    "total_UPDRS",
    "Jitter(%)",
    "Jitter(Abs)",
    "Jitter:RAP",
    "Jitter:PPQ5",
    "Jitter:DDP",
    "Shimmer",
    "Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "Shimmer:APQ11",
    "Shimmer:DDA",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "PPE",
)
# Exact, case-sensitive tokens after stripping surrounding whitespace. Other strings
# (including pandas' additional default NA tokens) are malformed numeric values.
MISSING_TOKENS = frozenset({"", "NA", "N/A", "NaN", "nan", "NULL", "null"})


def to_snake_case(name: str) -> str:
    """Use the naming convention established in notebooks/eda.ipynb."""
    # Decompose accented characters before removing combining marks, then replace
    # punctuation runs with one underscore rather than retaining units as symbols.
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# Export this mapping with prepared data so canonical names remain traceable to
# their raw counterparts (for example, Jitter(%) becomes jitter).
COLUMN_MAPPING: dict[str, str] = dict(zip(RAW_COLUMNS, map(to_snake_case, RAW_COLUMNS), strict=True))
COLUMNS = tuple(COLUMN_MAPPING.values())
# Ordered feature lists make baseline/candidate comparisons reproducible and keep
# the target, related total score, and patient identifier out of model inputs.
BASELINE_FEATURES = ("age", "sex", "test_time")
VOICE_FEATURES = COLUMNS[6:]
CANDIDATE_FEATURES = BASELINE_FEATURES + VOICE_FEATURES
# Prepared data still needs the target and patient ID for evaluation; feature
# selection in the serialized pipeline removes them before estimator fitting.
EXPORT_COLUMNS = tuple(name for name in COLUMNS if name != "total_updrs")


def validate_headers(headers: Sequence[str]) -> None:
    """Reject ambiguous or unexpected raw schemas before normalizing names."""
    missing = sorted(set(RAW_COLUMNS) - set(headers))
    unexpected = sorted(set(headers) - set(RAW_COLUMNS))
    duplicates = sorted({name for name in headers if headers.count(name) > 1})
    # Normalization can collapse distinct spellings into the same canonical name.
    # Such ambiguity is a schema error, not something row cleaning can repair.
    normalized = [to_snake_case(name) for name in headers]
    if missing or unexpected or duplicates or len(set(normalized)) != len(normalized):
        raise ValueError(
            f"Invalid raw schema: missing={missing}, unexpected={unexpected}, "
            f"duplicates={duplicates}, normalization_collision={len(set(normalized)) != len(normalized)}"
        )


def clip_test_time(value: float) -> float:
    """Clip finite negative times, preserving missingness; reject infinities."""
    # Clipping is an explicit project policy, not an inference about source units.
    # Keep NaN as the first argument to max so missing values remain imputable.
    if math.isinf(value):
        raise ValueError("Infinite test_time is invalid")
    return max(value, 0.0)
