# Shared runtime: one image, separate MLflow/API/dashboard processes.
FROM ghcr.io/astral-sh/uv:0.10.9@sha256:10902f58a1606787602f303954cea099626a4adb02acbac4c69920fe9d278f82 AS uv
FROM python:3.13.9-slim-bookworm@sha256:b685a4fa58bb19d1814d78a1ec0f0208f351452724f78b20212c984d6e124a34 AS build
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy
COPY pyproject.toml uv.lock requirements.txt README.md params.yaml dvc.yaml dvc.lock ./
COPY Dockerfile .dockerignore compose.yaml ./
COPY src ./src
COPY documents/data-bundle.json ./documents/data-bundle.json
COPY data/raw/parkinsons_updrs.csv.dvc ./dataset.dvc
RUN uv sync --frozen --no-dev
ARG SOURCE_REVISION
RUN .venv/bin/python -m parkinsons_ml.provenance --revision "$SOURCE_REVISION"

FROM python:3.13.9-slim-bookworm@sha256:b685a4fa58bb19d1814d78a1ec0f0208f351452724f78b20212c984d6e124a34
WORKDIR /app
COPY --from=build /app /app
ENV PATH="/app/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/tmp
RUN useradd --uid 10001 --no-create-home app \
    && mkdir -p /app/data /app/storage/mlflow /var/lib/mlflow/artifacts \
    && chown -R app:app /app/data /app/storage /var/lib/mlflow
USER 10001:10001
CMD ["python", "-m", "parkinsons_ml.deployment", "bootstrap"]
