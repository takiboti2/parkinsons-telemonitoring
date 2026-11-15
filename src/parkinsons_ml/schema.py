"""Versioned raw schema and deterministic transformations shared with inference."""

import math
import re
import unicodedata
from collections.abc import Sequence

SCHEMA_VERSION = "1"
POLICY_VERSION = "1"
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
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


COLUMN_MAPPING: dict[str, str] = dict(zip(RAW_COLUMNS, map(to_snake_case, RAW_COLUMNS), strict=True))
COLUMNS = tuple(COLUMN_MAPPING.values())
BASELINE_FEATURES = ("age", "sex", "test_time")
VOICE_FEATURES = COLUMNS[6:]
CANDIDATE_FEATURES = BASELINE_FEATURES + VOICE_FEATURES
EXPORT_COLUMNS = tuple(name for name in COLUMNS if name != "total_updrs")


def validate_headers(headers: Sequence[str]) -> None:
    """Reject ambiguous or unexpected raw schemas before normalizing names."""
    missing = sorted(set(RAW_COLUMNS) - set(headers))
    unexpected = sorted(set(headers) - set(RAW_COLUMNS))
    duplicates = sorted({name for name in headers if headers.count(name) > 1})
    normalized = [to_snake_case(name) for name in headers]
    if missing or unexpected or duplicates or len(set(normalized)) != len(normalized):
        raise ValueError(
            f"Invalid raw schema: missing={missing}, unexpected={unexpected}, "
            f"duplicates={duplicates}, normalization_collision={len(set(normalized)) != len(normalized)}"
        )


def clip_test_time(value: float) -> float:
    """Clip finite negative times, preserving missingness; reject infinities."""
    if math.isinf(value):
        raise ValueError("Infinite test_time is invalid")
    return max(value, 0.0)
