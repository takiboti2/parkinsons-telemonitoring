"""HTTP contracts, authentication, bounded bodies, and single-worker serialization."""

from concurrent.futures import ThreadPoolExecutor
import asyncio
from pathlib import Path
from threading import Event

import pytest
import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError
from starlette.types import Message, Scope

from parkinsons_ml import api
from parkinsons_ml.experimentation import ComparisonResult, ConfigurationResult, Metrics, PromotionResult, TrainConfig
from parkinsons_ml.inference import InputError, ModelError, PredictionResult, UnavailableError

KEY = "isolated-api-test-key-not-a-real-secret"
AUTH = {"X-API-Key": KEY}


@pytest.fixture
def application(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    params = tmp_path / "params.yaml"
    params.write_text("train:\n  experiment_name: api-test\n")
    monkeypatch.setattr(api, "PARAMS_PATH", params)
    return api.create_app(
        api.APISettings(tracking_uri="http://localhost:5000", api_key=SecretStr(KEY), max_batch_size=1000)
    )


def test_required_environment_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PARKINSONS_API_KEY", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(ValidationError):
        api.APISettings.model_validate({})
    monkeypatch.setenv("PARKINSONS_API_KEY", KEY)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("PARKINSONS_MAX_BATCH_SIZE", "12")
    settings = api.APISettings.model_validate({})
    assert settings.max_batch_size == 12 and KEY not in repr(settings)
    params = tmp_path / "params.yaml"
    params.write_text("train:\n  experiment_name: api-test\n")
    monkeypatch.setattr(api, "PARAMS_PATH", params)
    assert api.create_app().title == "Parkinson's Telemonitoring API"
    params.write_text("[]")
    with pytest.raises(ValueError, match="train mapping"):
        api.create_app(settings)


def test_dotenv_settings_authenticate_and_environment_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("PARKINSONS_API_KEY", "MLFLOW_TRACKING_URI", "PARKINSONS_MAX_BATCH_SIZE"):
        monkeypatch.delenv(name, raising=False)
    dotenv = tmp_path / "api.env"
    dotenv.write_text(
        f"PARKINSONS_API_KEY={KEY}\n"
        "MLFLOW_TRACKING_URI=http://localhost:5000\n"
        "SOURCE_REVISION=compose-only\n"
        "DATA_BUNDLE_DIR=./data-bundle\n"
    )
    # The configured file is absolute; a dotenv in the caller's directory is ignored.
    caller = tmp_path / "caller"
    caller.mkdir()
    (caller / ".env").write_text("PARKINSONS_API_KEY=short\n")
    monkeypatch.chdir(caller)
    settings = api.APISettings.model_validate({})
    assert settings.api_key.get_secret_value() == KEY
    assert KEY not in repr(settings)
    assert settings.tracking_uri == "http://localhost:5000"
    params = tmp_path / "params.yaml"
    params.write_text("train:\n  experiment_name: dotenv-test\n")
    monkeypatch.setattr(api, "PARAMS_PATH", params)
    with TestClient(api.create_app()) as client:
        # An authenticated invalid body reaches schema validation, not the registry.
        assert client.post("/predict", json={}, headers=AUTH).status_code == 422
        assert client.post("/predict", json={}).status_code == 401
    override = "environment-override-key-not-a-real-secret"
    monkeypatch.setenv("PARKINSONS_API_KEY", override)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    settings = api.APISettings.model_validate({})
    assert settings.api_key.get_secret_value() == override
    assert settings.tracking_uri == "http://localhost:5001"
    with TestClient(api.create_app()) as client:
        assert client.post("/predict", json={}, headers=AUTH).status_code == 401
        assert client.post("/predict", json={}, headers={"X-API-Key": override}).status_code == 422
    assert dotenv.read_text().startswith(f"PARKINSONS_API_KEY={KEY}\n")


@pytest.mark.parametrize("key", [None, "", "short"])
def test_dotenv_does_not_allow_missing_or_weak_keys(
    key: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PARKINSONS_API_KEY", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    content = "MLFLOW_TRACKING_URI=http://localhost:5000\n"
    if key is not None:
        content += f"PARKINSONS_API_KEY={key}\n"
    (tmp_path / "api.env").write_text(content)
    with pytest.raises(ValidationError) as caught:
        api.APISettings.model_validate({})
    assert "input_value" not in str(caught.value)
    assert "PARKINSONS_API_KEY" in str(caught.value)


@pytest.mark.parametrize(
    "uri,key",
    [
        ("http://remote.example", KEY),
        ("file:///tmp/mlruns", KEY),
        ("http://user:secret@localhost", KEY),
        ("http://localhost", "short"),
    ],
)
def test_invalid_settings(uri: str, key: str) -> None:
    with pytest.raises(ValidationError):
        api.APISettings(tracking_uri=uri, api_key=SecretStr(key))


@pytest.mark.parametrize("path", ["/train", "/predict"])
@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "incorrect"}])
def test_mutating_routes_require_authentication(application: FastAPI, path: str, headers: dict[str, str]) -> None:
    with TestClient(application) as client:
        body = {} if path == "/train" else {"records": [{"age": 60}]}
        response = client.post(path, json=body, headers=headers)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "ApiKey"


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"records": []},
        {"records": [{"age": True}]},
        {"records": [{"age": "60"}]},
        {"records": [{"age": [60]}]},
        {"records": [{}] * 1001},
        {"records": [{"age": 60}], "model_uri": "file:///unapproved"},
    ],
)
def test_prediction_schema_errors_are_sanitized(application: FastAPI, body: object) -> None:
    with TestClient(application) as client:
        response = client.post("/predict", json=body, headers=AUTH)
    assert response.status_code == 422
    assert response.json() == {"detail": "Request does not match the input schema"}


def test_no_training_overrides_and_no_invalid_json_echo(application: FastAPI) -> None:
    with TestClient(application) as client:
        response = client.post("/train", json={"data": "/unapproved"}, headers=AUTH)
        assert response.status_code == 422 and "unapproved" not in response.text
        response = client.post("/predict", content=b'{"records": [private-invalid-json', headers=AUTH)
        assert response.status_code == 422 and "private" not in response.text


def test_body_limit_counts_actual_chunked_bytes(application: FastAPI) -> None:
    def chunks():
        yield b" " * (api.MAX_BODY_BYTES // 2)
        yield b" " * (api.MAX_BODY_BYTES // 2 + 1)

    with TestClient(application) as client:
        response = client.post("/predict", content=chunks(), headers=AUTH)
    assert response.status_code == 413


@pytest.mark.parametrize("disconnect", [False, True])
def test_body_limit_accumulates_stream_messages(application: FastAPI, disconnect: bool) -> None:
    messages: list[Message] = [
        {"type": "http.request", "body": b"12345", "more_body": True},
        {"type": "http.disconnect"} if disconnect else {"type": "http.request", "body": b"67890", "more_body": False},
    ]
    sent: list[Message] = []

    async def receive() -> Message:
        return messages.pop(0)

    async def send(message: Message) -> None:
        sent.append(message)

    scope: Scope = {"type": "http", "method": "POST", "headers": [(b"x-api-key", KEY.encode())]}
    asyncio.run(api.RequestGuard(application, SecretStr(KEY), max_bytes=8)(scope, receive, send))
    if disconnect:
        assert not sent
    else:
        assert sent[0]["status"] == 413


@pytest.mark.parametrize(
    "headers", [[], [(b"x-api-key", b"wrong")], [(b"x-api-key", KEY.encode()), (b"x-api-key", KEY.encode())]]
)
def test_authentication_precedes_any_body_read(application: FastAPI, headers: list[tuple[bytes, bytes]]) -> None:
    sent: list[Message] = []

    async def receive() -> Message:
        pytest.fail("Unauthenticated requests must not be read or JSON-parsed")

    async def send(message: Message) -> None:
        sent.append(message)

    scope: Scope = {"type": "http", "method": "POST", "headers": headers}
    asyncio.run(api.RequestGuard(application, SecretStr(KEY))(scope, receive, send))
    assert sent[0]["status"] == 401


def test_malformed_unauthenticated_json_is_rejected_before_parsing(application: FastAPI) -> None:
    with TestClient(application) as client:
        for path in ("/train", "/predict"):
            response = client.post(path, content=b'{"private": malformed')
            assert response.status_code == 401 and "private" not in response.text


def test_body_read_has_an_overall_deadline(application: FastAPI) -> None:
    sent: list[Message] = []

    async def receive() -> Message:
        await asyncio.Event().wait()
        raise AssertionError("A stalled upload must be cancelled")

    async def send(message: Message) -> None:
        sent.append(message)

    async def exercise() -> None:
        scope: Scope = {"type": "http", "method": "POST", "headers": [(b"x-api-key", KEY.encode())]}
        guard = api.RequestGuard(application, SecretStr(KEY), read_timeout=0.01)
        await asyncio.wait_for(guard(scope, receive, send), timeout=1)

    asyncio.run(exercise())
    assert sent[0]["status"] == 408


def test_nonfinite_json_and_configured_batch_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    params = tmp_path / "params.yaml"
    params.write_text("train:\n  experiment_name: api-test\n")
    monkeypatch.setattr(api, "PARAMS_PATH", params)
    app = api.create_app(
        api.APISettings(tracking_uri="http://localhost:5000", api_key=SecretStr(KEY), max_batch_size=2)
    )
    with TestClient(app) as client:
        for value in ("NaN", "Infinity", "-Infinity"):
            response = client.post(
                "/predict",
                content='{"records": [{"age": ' + value + "}]}",
                headers={**AUTH, "Content-Type": "application/json"},
            )
            assert response.status_code == 422
        response = client.post("/predict", json={"records": [{"age": 60}] * 3}, headers=AUTH)
        assert response.status_code == 422  # rejected before registry access


def test_predict_success_and_failures_release_guard(application: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    def predict(records: list[dict[str, float | int | None]], uri: str, name: str, limit: int) -> PredictionResult:
        assert records == [{"age": 60}] and uri == "http://localhost:5000"
        assert name == "motor-updrs" and limit == 1000
        return PredictionResult("request", "run", name, "3", (12.5,))

    with TestClient(application) as client:
        for error, status in [
            (InputError("Invalid predictors"), 422),
            (UnavailableError("Logging unavailable"), 503),
            (ModelError("private artifact details"), 500),
            (RuntimeError("private failure"), 500),
        ]:

            def fail(*args: object) -> PredictionResult:
                raise error

            monkeypatch.setattr(api, "predict_batch", fail)
            response = client.post("/predict", json={"records": [{"age": 60}]}, headers=AUTH)
            assert response.status_code == status and "private" not in response.text
            monkeypatch.setattr(api, "predict_batch", predict)
            response = client.post("/predict", json={"records": [{"age": 60}]}, headers=AUTH)
            assert response.status_code == 200
            assert response.json()["model_version"] == "3"
            assert response.json()["predictions"] == [12.5]


def test_training_reuses_server_config_and_separates_promotion(
    application: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    def train(config: TrainConfig, directory: Path, uri: str) -> ComparisonResult:
        # Only the server-owned params/data paths can reach the shared function.
        assert directory == api.DATA_DIRECTORY and uri == "http://localhost:5000"
        assert config.experiment_name == "api-test"
        child = ConfigurationResult(
            "baseline",
            ("age", "sex", "test_time"),
            "child",
            "models:/m-test",
            Metrics(1.0, 1.0, None, "constant_target"),
        )
        return ComparisonResult(
            "parent",
            (child,),
            child,
            "dataset",
            "split",
            promotion=PromotionResult("failed", "promotion_busy", "motor-updrs"),
        )

    monkeypatch.setattr(api, "run_experiment", train)
    with TestClient(application) as client:
        response = client.post("/train", json={}, headers=AUTH)
        assert response.status_code == 200
        assert response.json()["promotion"]["decision"] == "failed"
        for error, status in [(OSError("private path"), 503), (RuntimeError("private exception"), 500)]:

            def fail(*args: object) -> ComparisonResult:
                raise error

            monkeypatch.setattr(api, "run_experiment", fail)
            response = client.post("/train", json={}, headers=AUTH)
            assert response.status_code == status and "private" not in response.text
        monkeypatch.setattr(api, "run_experiment", train)
        assert client.post("/train", json={}, headers=AUTH).status_code == 200


@pytest.mark.parametrize("running", ["/train", "/predict"])
def test_busy_operations_do_not_block_health(
    application: FastAPI, monkeypatch: pytest.MonkeyPatch, running: str
) -> None:
    entered, release = Event(), Event()

    def blocking(*args: object) -> object:
        entered.set()
        assert release.wait(10)
        raise UnavailableError("test operation finished")

    monkeypatch.setattr(api, "run_experiment" if running == "/train" else "predict_batch", blocking)
    with TestClient(application) as client, ThreadPoolExecutor(max_workers=1) as pool:
        body = {} if running == "/train" else {"records": [{"age": 60}]}
        future = pool.submit(client.post, running, json=body, headers=AUTH)
        try:
            assert entered.wait(10)
            assert client.post("/train", json={}, headers=AUTH).status_code == 409
            assert client.post("/predict", json={"records": [{"age": 60}]}, headers=AUTH).status_code == 409
            assert client.get("/health/live").json() == {"status": "alive"}
        finally:
            release.set()
        assert future.result(timeout=10).status_code in {500, 503}


def test_readiness_does_not_load_models_and_swagger_has_auth(
    application: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    def ready(client: object, name: str) -> None:
        assert name == "motor-updrs"

    monkeypatch.setattr(api, "resolve_champion", ready)
    with TestClient(application) as client:
        assert client.get("/health/ready").status_code == 200
        schema = client.get("/openapi.json").json()
        assert schema["paths"]["/predict"]["post"]["security"]
        assert schema["components"]["securitySchemes"]["APIKeyHeader"]["name"] == "X-API-Key"

        def fail(*args: object) -> None:
            raise UnavailableError("private registry detail")

        monkeypatch.setattr(api, "resolve_champion", fail)
        response = client.get("/health/ready")
        assert response.status_code == 503 and "private" not in response.text


def test_stalled_readiness_does_not_fill_the_shared_thread_pool(
    application: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered, release = Event(), Event()
    calls = 0

    def stalled(client: object, name: str) -> None:
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(10)

    def predict(*args: object) -> PredictionResult:
        return PredictionResult("request", "run", "motor-updrs", "1", (12.0,))

    monkeypatch.setattr(api, "resolve_champion", stalled)
    monkeypatch.setattr(api, "predict_batch", predict)
    monkeypatch.setattr(api, "READINESS_TIMEOUT", 0.05)
    with TestClient(application) as client, ThreadPoolExecutor(max_workers=1) as requests:
        first = requests.submit(client.get, "/health/ready")
        try:
            assert entered.wait(5)
            for _ in range(45):
                assert client.get("/health/ready").status_code == 503
            assert first.result(timeout=2).status_code == 503
            assert calls == 1  # no probe queue or per-request blocked worker
            assert client.get("/health/live").json() == {"status": "alive"}
            response = client.post("/predict", json={"records": [{"age": 60}]}, headers=AUTH)
            assert response.status_code == 200
        finally:
            release.set()


def test_readiness_programming_error_is_not_reported_as_an_outage(
    application: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: object) -> None:
        raise RuntimeError("private programming detail")

    monkeypatch.setattr(api, "resolve_champion", broken)
    with TestClient(application) as client:
        response = client.get("/health/ready")
    assert response.status_code == 500 and "private" not in response.text


def test_asgi_shutdown_does_not_wait_for_running_readiness(
    application: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, finished = Event(), Event()

    def stalled(client: object, name: str) -> None:
        try:
            assert release.wait(5)
        finally:
            finished.set()

    monkeypatch.setattr(api, "resolve_champion", stalled)
    monkeypatch.setattr(api, "READINESS_TIMEOUT", 0.01)

    async def exercise() -> None:
        async with application.router.lifespan_context(application):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=application), base_url="http://test"
            ) as client:
                assert (await client.get("/health/ready")).status_code == 503
        # ASGI shutdown returned, but it did not kill the running SDK call.
        assert not finished.is_set()

    async def bounded_shutdown() -> None:
        await asyncio.wait_for(exercise(), timeout=1)

    try:
        asyncio.run(bounded_shutdown())
    finally:
        release.set()
        assert finished.wait(5)
