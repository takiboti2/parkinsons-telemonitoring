"""Thin, authenticated single-worker FastAPI adapter for training and prediction."""

import asyncio
import hmac
import logging
from collections.abc import AsyncGenerator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from threading import Lock

import yaml
from fastapi import FastAPI, HTTPException, Security
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
BODY_READ_TIMEOUT = 10.0
READINESS_TIMEOUT = 3.0


class APISettings(BaseSettings):
    """Server-owned settings; environment variables override the root dotenv file."""

    model_config = SettingsConfigDict(
        populate_by_name=True,
        frozen=True,
        hide_input_in_errors=True,
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        # The shared file may also contain Compose-only variables.
        extra="ignore",
    )
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


class RequestGuard:
    """Authenticate POSTs before reading them, then bound upload bytes and time."""

    def __init__(
        self,
        app: ASGIApp,
        api_key: SecretStr,
        max_bytes: int = MAX_BODY_BYTES,
        read_timeout: float = BODY_READ_TIMEOUT,
    ) -> None:
        self.app = app
        self.api_key = api_key.get_secret_value().encode()
        self.max_bytes = max_bytes
        self.read_timeout = read_timeout

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        keys = [value for name, value in scope["headers"] if name.lower() == b"x-api-key"]
        if len(keys) != 1 or not hmac.compare_digest(keys[0], self.api_key):
            await JSONResponse({"detail": "Invalid API key"}, status_code=401, headers={"WWW-Authenticate": "ApiKey"})(
                scope, receive, send
            )
            return
        # A bytearray also bounds buffering overhead when a client sends many tiny
        # chunks. The deadline covers the entire upload, not each individual chunk.
        payload = bytearray()
        try:
            async with asyncio.timeout(self.read_timeout):
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    body = message.get("body", b"")
                    if len(payload) + len(body) > self.max_bytes:
                        await JSONResponse({"detail": "Request body is too large"}, status_code=413)(
                            scope, receive, send
                        )
                        return
                    payload.extend(body)
                    if not message.get("more_body", False):
                        break
        except TimeoutError:
            await JSONResponse({"detail": "Request body upload timed out"}, status_code=408)(scope, receive, send)
            return
        # Replay once so FastAPI retains its normal JSON validation and schemas.
        sent = False

        async def replay() -> Message:
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": bytes(payload), "more_body": False}
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
    probe = Lock()
    probe_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="champion-readiness")

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncGenerator[None]:
        try:
            yield
        finally:
            # This only keeps ASGI shutdown nonblocking: Python still joins a
            # running executor thread at process exit. SDK network timeouts bound
            # normal failures; a supervisor must enforce any hard exit deadline.
            probe_worker.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(
        title="Parkinson's Telemonitoring API",
        version="0.1.0",
        description="Research motor UPDRS prediction; not a clinical diagnostic service.",
        lifespan=lifespan,
    )
    app.add_middleware(RequestGuard, api_key=settings.api_key, read_timeout=BODY_READ_TIMEOUT)
    header = APIKeyHeader(name="X-API-Key", auto_error=False)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: object, error: RequestValidationError) -> JSONResponse:
        # Pydantic error details can echo submitted measurements/unknown fields.
        # Keep them out of responses and application logs, not just secret fields.
        return JSONResponse({"detail": "Request does not match the input schema"}, status_code=422)

    @app.exception_handler(Exception)
    async def unexpected_error(request: object, error: Exception) -> JSONResponse:
        LOGGER.error("API operation failed (%s)", type(error).__name__)
        return JSONResponse({"detail": "Internal operation failed"}, status_code=500)

    # Security declares the header in OpenAPI. The middleware is the single auth
    # boundary for every POST, including malformed JSON and unknown POST paths.
    @app.post("/train", response_model=ComparisonResult, dependencies=[Security(header)])
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

    @app.post("/predict", response_model=PredictionResult, dependencies=[Security(header)])
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
    async def live() -> dict[str, str]:
        return {"status": "alive"}

    def check_registry() -> None:
        # Explicit client routing avoids fluent-state interference with a running
        # training operation. A private worker keeps blocked probes out of the
        # shared pool used for authentication, prediction, and training handlers.
        try:
            client = MlflowClient(tracking_uri=settings.tracking_uri, registry_uri=settings.tracking_uri)
            resolve_champion(client, model_name)
        finally:
            probe.release()

    def release_cancelled_probe(future: Future[None]) -> None:
        # Cancellation can win before the worker starts; in that case its finally
        # block will never run. A running probe releases the lease itself.
        if future.cancelled():
            probe.release()

    @app.get("/health/ready")
    async def ready() -> dict[str, str]:
        if not probe.acquire(blocking=False):
            raise HTTPException(503, "A registry readiness probe is already running")
        try:
            future = probe_worker.submit(check_registry)
            future.add_done_callback(release_cancelled_probe)
        except RuntimeError as error:
            probe.release()
            raise HTTPException(503, "Champion registry is not ready") from error
        try:
            await asyncio.wait_for(asyncio.wrap_future(future), timeout=READINESS_TIMEOUT)
        except (UnavailableError, MlflowException, OSError, TimeoutError) as error:
            raise HTTPException(503, "Champion registry is not ready") from error
        except Exception as error:
            LOGGER.error("Readiness failed (%s)", type(error).__name__)
            raise HTTPException(500, "Internal readiness operation failed") from error
        return {"status": "ready"}

    return app
