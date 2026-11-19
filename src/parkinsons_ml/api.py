"""Thin, authenticated single-worker FastAPI adapter for training and prediction."""

import hmac
import logging
from threading import Lock
from typing import Annotated

import yaml
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from pydantic import BaseModel, ConfigDict, Field, SecretStr, StrictFloat, StrictInt, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from parkinsons_ml.experimentation import ComparisonResult, UniqueKeyLoader, training_config
from parkinsons_ml.inference import (
    InputError,
    ModelError,
    PredictionResult,
    UnavailableError,
    predict_batch,
    resolve_champion,
)
from parkinsons_ml.tracking import ROOT, run_experiment, validate_tracking_uri

LOGGER = logging.getLogger(__name__)
PARAMS_PATH = ROOT / "params.yaml"
DATA_DIRECTORY = ROOT / "data/processed"
MAX_BODY_BYTES = 1024 * 1024


class APISettings(BaseSettings):
    """Only environment/server-owned settings; dotenv files are not loaded."""

    model_config = SettingsConfigDict(populate_by_name=True, frozen=True, hide_input_in_errors=True)
    tracking_uri: str = Field(validation_alias="MLFLOW_TRACKING_URI", repr=False)
    api_key: SecretStr = Field(validation_alias="PARKINSONS_API_KEY", min_length=32)
    max_batch_size: int = Field(default=1000, validation_alias="PARKINSONS_MAX_BATCH_SIZE", ge=1, le=1000)

    @field_validator("tracking_uri")
    @classmethod
    def supported_uri(cls, value: str) -> str:
        validate_tracking_uri(value)
        return value


class TrainRequest(BaseModel):
    """No client-controlled paths, features, registry names, or training overrides."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class PredictRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        hide_input_in_errors=True,
        json_schema_extra={
            "examples": [
                {
                    "records": [
                        {
                            "age": 65,
                            "sex": 0,
                            "test_time": 10,
                            "jitter": 0.004,
                            "jitter_abs": 0.00003,
                            "jitter_rap": 0.002,
                            "jitter_ppq5": 0.002,
                            "jitter_ddp": 0.006,
                            "shimmer": 0.03,
                            "shimmer_db": 0.3,
                            "shimmer_apq3": 0.02,
                            "shimmer_apq5": 0.02,
                            "shimmer_apq11": 0.025,
                            "shimmer_dda": 0.06,
                            "nhr": 0.02,
                            "hnr": 20,
                            "rpde": 0.5,
                            "dfa": 0.65,
                            "ppe": 0.2,
                        }
                    ]
                }
            ]
        },
    )
    records: list[dict[str, StrictFloat | StrictInt | None]] = Field(
        min_length=1,
        max_length=1000,
        description="Canonical predictors only. Supply every selected feature; null is allowed, omission is not.",
    )


class BodyLimit:
    """Bound actual received bytes before JSON parsing, including chunked bodies."""

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        chunks: list[bytes] = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body = message.get("body", b"")
            size += len(body)
            if size > self.max_bytes:
                await JSONResponse({"detail": "Request body is too large"}, status_code=413)(scope, receive, send)
                return
            chunks.append(body)
            if not message.get("more_body", False):
                break
        # Buffer at most the configured cap. Replay once so FastAPI retains its
        # normal JSON parsing, validation, and OpenAPI-generated request schemas.
        payload = b"".join(chunks)
        sent = False

        async def replay() -> Message:
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": payload, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


def create_app(settings: APISettings | None = None) -> FastAPI:
    """Factory validates configuration but never trains or requires a live server."""
    # BaseSettings reads environment variables through its validated constructor.
    # model_validate also keeps the dynamically supplied required fields type-safe.
    settings = settings or APISettings.model_validate({})
    parameters = yaml.load(PARAMS_PATH.read_text(), Loader=UniqueKeyLoader)
    if not isinstance(parameters, dict):
        raise ValueError("Server parameters must contain a train mapping")
    config = training_config(parameters.get("train"))
    model_name = config.promotion.registered_model_name
    operation = Lock()
    app = FastAPI(
        title="Parkinson's Telemonitoring API",
        version="0.1.0",
        description="Research motor UPDRS prediction; not a clinical diagnostic service.",
    )
    app.add_middleware(BodyLimit)
    header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def authenticate(key: Annotated[str | None, Security(header)]) -> None:
        if key is None or not hmac.compare_digest(key.encode(), settings.api_key.get_secret_value().encode()):
            raise HTTPException(401, "Invalid API key", headers={"WWW-Authenticate": "ApiKey"})

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: object, error: RequestValidationError) -> JSONResponse:
        # Pydantic error details can echo submitted measurements/unknown fields.
        # Keep them out of responses and application logs, not just secret fields.
        return JSONResponse({"detail": "Request does not match the input schema"}, status_code=422)

    @app.exception_handler(Exception)
    async def unexpected_error(request: object, error: Exception) -> JSONResponse:
        LOGGER.error("API operation failed (%s)", type(error).__name__)
        return JSONResponse({"detail": "Internal operation failed"}, status_code=500)

    @app.post("/train", response_model=ComparisonResult, dependencies=[Depends(authenticate)])
    def train(body: TrainRequest) -> ComparisonResult:
        if not operation.acquire(blocking=False):
            raise HTTPException(409, "Another training or prediction operation is running")
        try:
            return run_experiment(config, DATA_DIRECTORY, settings.tracking_uri)
        except (MlflowException, OSError, ValueError, yaml.YAMLError) as error:
            LOGGER.error("Training failed (%s)", type(error).__name__)
            raise HTTPException(503, "Training dependency, configuration, or logging failed") from error
        except Exception as error:
            LOGGER.error("Training failed (%s)", type(error).__name__)
            raise HTTPException(500, "Internal training operation failed") from error
        finally:
            operation.release()

    @app.post("/predict", response_model=PredictionResult, dependencies=[Depends(authenticate)])
    def predict(body: PredictRequest) -> PredictionResult:
        if not operation.acquire(blocking=False):
            raise HTTPException(409, "Another training or prediction operation is running")
        try:
            return predict_batch(body.records, settings.tracking_uri, model_name, settings.max_batch_size)
        except InputError as error:
            raise HTTPException(422, str(error)) from error
        except UnavailableError as error:
            raise HTTPException(503, str(error)) from error
        except ModelError as error:
            LOGGER.error("Champion inference failed (%s)", type(error).__name__)
            raise HTTPException(500, "Champion model could not serve predictions") from error
        except Exception as error:
            LOGGER.error("Prediction failed (%s)", type(error).__name__)
            raise HTTPException(500, "Internal prediction operation failed") from error
        finally:
            operation.release()

    @app.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "alive"}

    @app.get("/health/ready")
    def ready() -> dict[str, str]:
        # Explicit client routing avoids fluent-state interference with a running
        # training operation. Readiness never downloads the model or grabs the guard.
        try:
            client = MlflowClient(tracking_uri=settings.tracking_uri, registry_uri=settings.tracking_uri)
            resolve_champion(client, model_name)
        except Exception as error:
            raise HTTPException(503, "Champion registry is not ready") from error
        return {"status": "ready"}

    return app
