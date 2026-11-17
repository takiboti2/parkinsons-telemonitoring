#!/usr/bin/env bash
set -euo pipefail

# Resolve paths from this script, not the terminal's working directory.
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_root"

# Refuse alternate server roles and read routing inherited from another project.
# Report variable names only: their values can contain connection credentials.
for name in MLFLOW_READ_REPLICA_BACKEND_STORE_URI MLFLOW_ARTIFACTS_ONLY MLFLOW_ARTIFACTS_PRESIGNED_ONLY; do
  if [[ -n "${!name:-}" ]]; then
    printf 'Unset %s: this launcher requires one tracking/registry store with HTTP artifact serving.\n' "$name" >&2
    exit 2
  fi
done

# Tests use an isolated directory/port but execute this same launcher. Relative
# storage overrides, like the default, resolve against the repository root.
storage="${PARKINSONS_MLFLOW_STORAGE_DIR:-$project_root/storage/mlflow}"
port="${PARKINSONS_MLFLOW_PORT:-5000}"
mkdir -p -- "$storage/artifacts"
storage="$(cd -- "$storage" && pwd)"
database="sqlite:///$storage/mlflow.db"

# Replace the shell so foreground signals reach the server command. Keep SQLite
# metadata and HTTP-served artifacts persistent, with localhost-only access.
exec uv run --frozen mlflow server \
  --backend-store-uri "$database" \
  --registry-store-uri "$database" \
  --default-artifact-root "mlflow-artifacts:/" \
  --serve-artifacts \
  --artifacts-destination "$storage/artifacts" \
  --host 127.0.0.1 \
  --port "$port" \
  --workers 1
