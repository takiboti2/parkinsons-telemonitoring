"""Submit an inclusive, 1-based range of development rows to the local prediction API."""

import argparse
import json
from typing import cast

import httpx
import pandas as pd
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from parkinsons_ml.schema import CANDIDATE_FEATURES
from parkinsons_ml.tracking import ROOT

URL = "http://127.0.0.1:8000/predict"


class ClientSettings(BaseSettings):
    """Read only the client credential; environment overrides the root dotenv file."""

    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )
    api_key: SecretStr = Field(validation_alias="PARKINSONS_API_KEY", min_length=32)


class RowSelectionError(ValueError):
    """A fixed operator-safe row-selection diagnostic."""


def predict_rows(start_row: int, end_row: int) -> dict[str, object]:
    """Send exactly the requested development rows, excluding targets and identifiers."""
    if start_row < 1 or end_row < start_row:
        raise RowSelectionError("Rows must satisfy 1 <= start-row <= end-row (both inclusive)")
    count = end_row - start_row + 1
    if count > 1000:
        raise RowSelectionError("Select at most 1000 rows per request")
    settings = ClientSettings.model_validate({})
    frame = pd.read_csv(
        ROOT / "data/processed/train.csv",
        usecols=pd.Index(CANDIDATE_FEATURES),
        skiprows=lambda row: isinstance(row, int) and 0 < row < start_row,
        nrows=count,
        dtype=float,
    )
    if len(frame) != count:
        raise RowSelectionError("Requested range extends beyond the available development rows")
    records = frame.astype(object).where(frame.notna(), None).to_dict(orient="records")
    # Reject nonfinite numbers rather than sending nonstandard JSON.
    payload = json.dumps({"records": records}, allow_nan=False).encode()
    response = httpx.post(
        URL,
        content=payload,
        headers={"Content-Type": "application/json", "X-API-Key": settings.api_key.get_secret_value()},
        timeout=120,
        trust_env=False,
    )
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict):
        raise ValueError("Prediction API returned an unexpected response")
    return cast(dict[str, object], result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-row", type=int, required=True, help="First development data row, starting at 1")
    parser.add_argument("--end-row", type=int, required=True, help="Last development data row, inclusive")
    args = parser.parse_args()
    try:
        result = predict_rows(args.start_row, args.end_row)
    except RowSelectionError as error:
        parser.exit(1, f"{error}\n")
    except httpx.HTTPStatusError as error:
        parser.exit(1, f"Prediction request failed (HTTP {error.response.status_code}); no automatic retry.\n")
    except (OSError, ValueError, httpx.HTTPError) as error:
        parser.exit(
            1, f"Prediction request failed ({type(error).__name__}); check data, API key and API availability.\n"
        )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
