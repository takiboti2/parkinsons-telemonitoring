# Parkinson's Telemonitoring

Predict `motor_UPDRS` with scikit-learn's `LinearRegression`. The baseline uses
`age`, `sex`, and `test_time`; the candidate adds 16 voice features. Exclude
`total_UPDRS` and use `subject#` only for patient-disjoint evaluation.

**Status:** dependencies, hooks, notebook Git tools, DVC preparation, reusable model
pipelines, MLflow-tracked experimentation, a local persistent MLflow setup, CI,
and the authenticated FastAPI service are implemented. Streamlit and Docker deployment remain.
This is not a clinical diagnostic service.

## Development setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.13:

```bash
uv sync --frozen
uv run --frozen pre-commit install
```

Versions are pinned in `pyproject.toml` and `uv.lock`. Keep pandas at 2.3.3 for the
current Evidently version. Regenerate the requirements export after dependency changes:

```bash
uv export --frozen --all-groups --no-hashes --no-emit-project --output-file requirements.txt
```

## Dataset

The raw file is `data/raw/parkinsons_updrs.csv`; see the
[data dictionary](documents/data-dictionary.md) for its source and columns.

DVC's default `localstorage` remote is a sibling directory, `../parkinsons-dvc-storage`
from the repository root. Its URL in `.dvc/config` is relative to that config file.
Ensure the populated remote is available, then retrieve data:

```bash
uv run --frozen dvc pull
uv run --frozen dvc status
uv run --frozen dvc status --cloud
```

For a different local location:

```bash
uv run --frozen dvc remote modify --local localstorage url /absolute/path/to/parkinsons-dvc-storage
uv run --frozen dvc pull
```

The override is ignored by Git. A new machine needs a copy of the populated remote;
Git contains only the dataset pointer. Back up the remote separately. Keep raw data
unchanged during processing and write derived data to `data/processed/`.

### Prepare train/test data

After `uv sync --frozen` and retrieving the raw dataset:

```bash
uv run --frozen dvc repro prepare
```

`params.yaml` controls the patient split seed, test fraction, grouped validation
fold count, and descriptive IQR multiplier. Preparation exports `train.csv`,
`test.csv`, patient/fold assignments, cleaning events, a validation summary, and
a checksum manifest under `data/processed/`. The raw CSV is never modified.
Files use the versioned snake_case mapping in `src/parkinsons_ml/schema.py`.
Preparation reads the raw CSV directly with `pandas.read_csv`, preserving literal
tokens and the header row for validation before naming columns. Cleaning,
deduplication, clipping, splits, statistics, and exports operate on DataFrames.
Truncated records are dropped and audited; unparseable CSV syntax or extra fields
stop preparation rather than risking silently skipped data.
`train` is the development pool for grouped validation and refitting; `test` is
reserved for final assessment, never tuning or promotion.

Schema errors stop preparation. Invalid rows and duplicate copies are dropped
with warnings and audited reasons; missing predictors are retained. Negative
finite `test_time` is clipped to zero as an explicit project policy, not a source
correction. Extreme-value flags are descriptive only, using train-derived bounds.
See the [processing policy](documents/data-dictionary.md#project-processing-policy).

Configuration, validation, processing, and staging failures leave existing outputs
untouched. All artifacts are written to a temporary directory before publication.
The final step removes the old processed directory and renames the staged directory
into place. This replacement is not atomic: interruption during it can leave
outputs missing or incomplete. Treat those outputs as unavailable and rerun
preparation, or restore a version available in the DVC cache/remote. There is no persistent backup or
automatic rollback/recovery mechanism; the checksum manifest remains for lineage.

Failures write diagnostics to `documents/reports/prepare-failure.json`; an older
failure report is historical, not the status of a later successful run. Successful
status is recorded in the published validation summary, but after a publication
error an old summary alone is not evidence of complete outputs. Preparation is a
single-writer operation: do not run it concurrently or consume outputs during it.

`parkinsons_ml.training.build_pipeline(features)` accepts a nonempty ordered list
or tuple of unique canonical names from `CANDIDATE_FEATURES`. It defaults to
`BASELINE_FEATURES`; the old `"baseline"`/`"candidate"` string arguments are not
accepted. For example:

```python
from parkinsons_ml.schema import BASELINE_FEATURES, CANDIDATE_FEATURES
from parkinsons_ml.training import build_pipeline

baseline = build_pipeline(BASELINE_FEATURES)
candidate = build_pipeline(CANDIDATE_FEATURES)
custom = build_pipeline(["ppe", "sex", "age"])
```

The unfitted pipeline rejects duplicates, sorts selected names alphabetically,
snapshots the canonical tuple, and excludes unselected columns. Lists containing
the same names in different orders therefore build the same feature configuration;
incoming DataFrame column order does not matter either. Only selected predictors
are required at fit and prediction time. Selected time is zero-clipped, selected sex uses most-frequent
imputation, and other selected features use median imputation before
`LinearRegression`. Fit only on training partitions; an entirely missing selected
fitting feature is rejected. Rebuild the pipeline to change its selection rather
than changing the selector alone with `set_params`.
Rebuild/refit pipelines created under the previous caller-order behaviour before
reusing them with the new transformer; their fitted coefficients may expect a
different feature order.
Preparation does not contact MLflow; experimentation logs its reports and
manifest along with the complete model pipeline.

To upload generated datasets (writes to the configured data remote):

```bash
uv run --frozen dvc push
```

Review `dvc.lock` alongside code and parameters before committing. The data remote
and any populated cache must be backed up separately.

## Training and experimentation

Start MLflow as described below, then run from the repository root:

```bash
export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
uv run --frozen python -m parkinsons_ml.training
```

An explicit HTTP/HTTPS URI is required; no filesystem fallback is used. Plain HTTP
is accepted only for `localhost` or loopback IP addresses (including IPv6 `::1`);
remote hosts require HTTPS and appropriately configured server access control.
Tracking and model loading use that same server. The CLI prints a JSON result with
parent and child run IDs, model URIs, validation metrics, identities, the winner,
and a separate promotion decision.
`--params` and `--data` override the parameter file and prepared directory.
Do not run comparisons concurrently within one process, with preparation, or
inside an existing MLflow run. Separate single-host processes can train independently;
only their registry promotion operations share a nonblocking local lock.

The implementation is split by responsibility:

| Module | Responsibility |
| --- | --- |
| `src/parkinsons_ml/training.py` | Feature validation, serialized `FeaturePreparation`, pipeline construction |
| `src/parkinsons_ml/experimentation.py` | Configuration, prepared snapshot validation, grouped evaluation, metrics, result types |
| `src/parkinsons_ml/tracking.py` | MLflow orchestration, artifacts, lineage, repository checks, CLI |
| `src/parkinsons_ml/promotion.py` | Registry eligibility, immutable-version validation, guarded alias updates and rollback |
| `src/parkinsons_ml/inference.py` | Champion resolution, input validation, prediction and required inference logging |
| `src/parkinsons_ml/api.py` | API settings, authentication, bounded schemas, routes and busy responses |
| `src/parkinsons_ml/monitoring.py` | Bounded compatible inference cohorts, Evidently reports, monitoring audit artifacts |
| `src/parkinsons_ml/dashboard.py` | Streamlit controls, provenance, quality/drift displays, saved reports and promotion history |

For Python callers, use the owning modules directly:

```python
from pathlib import Path

from parkinsons_ml.experimentation import training_config
from parkinsons_ml.tracking import run_experiment

config = training_config({"experiment_name": "motor-updrs-training"})
result = run_experiment(config, Path("data/processed"), "http://127.0.0.1:5000")
```

The previous `training.training_config()` and `training.run_experiment()` entry
points remain as thin delegates, as does the documented training command.
`python -m parkinsons_ml.tracking` also invokes the CLI. The transformer retains
its `parkinsons_ml.training.FeaturePreparation` import path for saved models.
Pipeline construction and evaluation do not import MLflow; tracking is the outer layer.

Experimentation is deliberately **source-checkout-only**, not supported from an
installed wheel. Git must be available, and the package must reside under its own
Git checkout's `src/parkinsons_ml/`, alongside `uv.lock` and `pyproject.toml` at the
repository root. These requirements are checked before contacting MLflow or creating
runs, independently of the caller's working directory. For packaged deployment,
explicit build provenance remains future work; no missing Git metadata is silently ignored.

`params.yaml` always adds the baseline and full candidate. Optional configurations
extend them, for example:

```yaml
train:
  experiment_name: motor-updrs-training
  promotion:
    enabled: true
    registered_model_name: motor-updrs
  extra_feature_sets:
    voice_subset: [jitter, ppe]
```

Names must be lowercase identifiers (letters, digits, `_`, `-`, starting with a
letter; at most 64 characters). Reserved names, duplicate YAML keys, duplicate
features, unknown features/keys, and canonically redundant selections are rejected.
Training settings do not invalidate DVC's preparation stage.

The loop verifies artifact checksums, schema/policy, summary counts, patient
assignments, and saved folds before fitting. It snapshots training inputs with
pandas and **never opens `test.csv`**, including for checksum verification. Invalid
inputs fail rather than being cleaned again or triggering preparation.

Selection uses pooled, measurement-weighted out-of-fold MAE on identical saved
patient folds. Exact ties prefer baseline, then fewer features, then name.
Per-fold/pooled MAE, RMSE, R², fold means/population standard deviations, and
counts are logged. Undefined R² is `null` with a reason in reports and is omitted
from scalar metrics. Fold R² means/standard deviations are null if any fold R²
is undefined; pooled R² is calculated independently.

One parent comparison run contains one child per configuration. Every child
refits on all train data, logs and reload-verifies a complete pipeline, and saves
selected-feature-only signatures/examples, identifier-free prediction artifacts,
and a checksummed pre-imputation monitoring reference. The parent records
preparation/audit metadata separately, code/dependency hashes, Git revision/dirty
status, comparison results, and selection. It does not upload the full training
table or arbitrary working-tree files.

Required fit, prediction, logging, artifact upload, or reload errors fail the
comparison; completed children can remain for diagnosis but cannot yield a partial
winner. MLflow context managers mark failed runs when the server is reachable;
an outage can leave remote status uncertain. There is no training retry/resume.
Final-test assessment remains a separate operation and never drives promotion.

### Automatic champion promotion

The repository's `params.yaml` enables promotion to the registered model
`motor-updrs`, using the fixed **`champion`** alias. If the `promotion` block is
omitted, promotion defaults to disabled for compatibility with programmatic callers
and older parameter files. Set `enabled: false` for experimentation without registry changes.
The current file compares 17 configurations, including the baseline/full candidate,
individual voice families, and age-plus-time combinations of two voice families.

After the entire comparison has finished successfully:

1. Resolve the incumbent champion and read authoritative completed-run metadata.
2. Promote the first valid winner when no champion exists. Otherwise require
   **strictly lower pooled `oof_mae`** with matching dataset, split, schema,
   preparation-policy, and evaluation-policy identities. Exact ties retain the
   incumbent. Missing or incompatible provenance rejects automatic promotion.
3. Register only an eligible winner. Load its exact registry version, verify the
   pipeline/signature/features, and compare predictions with its saved refit artifact.
4. Record the intended previous/new versions, update `champion`, and read back
   the alias. A failed update automatically restores the previous alias; failed
   bootstrap removes the new alias instead. Older versions remain available.

Each completed registry assessment has a separate MLflow run tagged `purpose=champion-promotion`, with
the comparison run ID, registered model name, policy version, and `decision.json`.
The returned `promotion` object reports `promoted`, `retained`, `rejected`,
`disabled`, or `failed`, along with available version IDs, scores, and confirmation status.
Disabled promotion and failure to acquire the guard do not create an audit run.
A promotion failure does not mark the completed training comparison failed; inspect
the separate promotion result rather than assuming training success means deployment.

The guard is `storage/mlflow/promotion.lock`; it covers registry work, not model
fitting. All automatic writers must run on this one host using the same guard.
External/manual registry writers do not honor this lock; unexpected alias changes
are not overwritten and require operator review.

If rollback cannot be confirmed, the registry retains a `pending_promotion_run_id`
tag referencing the audit intent. Before the next automatic promotion, the code
reconciles that one operation: preserve a fully audited success, or roll back an
interrupted update. An unresolved or inconsistent intent blocks another automatic
update. Do not clear the marker merely to bypass a failure; first verify the alias
and recorded old/new versions. There is no queue, scheduler, second deployment
pointer, additional database, or general workflow-recovery framework.

The alias is the only deployment selection. The API resolves it once per prediction
batch and loads that exact version, without an application model cache. Registered
artifacts and writers must be trusted.

## FastAPI service

Start MLflow first in its own terminal. In another terminal, set the connection and
a secret API key of at least 32 characters, then start **one worker**:

```bash
export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
export PARKINSONS_API_KEY="$(uv run --frozen python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export MLFLOW_HTTP_REQUEST_MAX_RETRIES=0
export MLFLOW_HTTP_REQUEST_TIMEOUT=10
uv run --frozen uvicorn parkinsons_ml.api:create_app --factory \
  --host 127.0.0.1 --port 8000 --workers 1
```

Keep that key private and reuse it for client requests. The service requires an
explicit MLflow URI and API key; it does not load `.env`. Plain HTTP MLflow URIs
are restricted to loopback as with CLI training. Do not expose the API publicly
without HTTPS and appropriate deployment controls. CORS is not enabled.

Open <http://127.0.0.1:8000/docs> for Swagger schemas, a full-feature prediction
example, and the **Authorize** control for the `X-API-Key` header.

| Endpoint | Authentication | Behavior |
| --- | --- | --- |
| `POST /train` | API key | Synchronously run the server's configured comparison and promotion policy |
| `POST /predict` | API key | Predict one bounded batch with the current immutable champion version |
| `GET /health/live` | None | Lightweight process check; no MLflow calls |
| `GET /health/ready` | None | Check champion registry metadata in one bounded probe; does not download artifacts |

The application reads repository-owned `params.yaml` once when it starts; restart
after changing settings or feature selections. Prepared data comes from the fixed
`data/processed/` directory. Clients cannot override paths, feature selections,
registered model names, aliases, or model URIs. Startup never trains automatically,
and a missing champion leaves prediction/readiness unavailable while training remains usable.
An unconfirmed promotion intent likewise blocks prediction and readiness until a later
promotion reconciles it; the service never serves a transient alias.

Run training with an empty JSON request:

```bash
curl --fail-with-body http://127.0.0.1:8000/train \
  -H "X-API-Key: $PARKINSONS_API_KEY" -H 'Content-Type: application/json' -d '{}'
```

The response contains the normal comparison result and a separate `promotion`
outcome. Completed training returns success even when promotion is retained,
rejected, or fails; inspect that field before assuming the champion changed.
Do not automatically retry a timed-out training request: first inspect MLflow,
because the original comparison may have completed.

Submit canonical predictor names, for example all 19 predictors so the same
client payload remains valid when the champion's selected subset changes:

```bash
curl --fail-with-body http://127.0.0.1:8000/predict \
  -H "X-API-Key: $PARKINSONS_API_KEY" -H 'Content-Type: application/json' -d '{
    "records": [{
      "age": 65, "sex": 0, "test_time": 10,
      "jitter": 0.004, "jitter_abs": 0.00003, "jitter_rap": 0.002,
      "jitter_ppq5": 0.002, "jitter_ddp": 0.006,
      "shimmer": 0.03, "shimmer_db": 0.3, "shimmer_apq3": 0.02,
      "shimmer_apq5": 0.02, "shimmer_apq11": 0.025, "shimmer_dda": 0.06,
      "nhr": 0.02, "hnr": 20, "rpde": 0.5, "dfa": 0.65, "ppe": 0.2
    }]
  }'
```

Every record must contain all selected features. Explicit `null` values are
allowed for learned imputation, but omitted selected fields are rejected. Additional
canonical predictors are validated and discarded if unused. Strings, booleans,
nonfinite/unrepresentable numbers, invalid sex/age encodings, targets, subject IDs,
and unknown keys are rejected. Time is zero-clipped using the pipeline's policy.
No measurements are silently dropped; predictions retain request order.

The default limit is 1,000 records; `PARKINSONS_MAX_BATCH_SIZE` can lower it to any
integer from 1 to 1,000. Actual POST body bytes, including chunked bodies, are capped
at 1 MiB before JSON parsing. Authentication happens first, so unauthenticated
requests are rejected without reading their bodies. Responses are `401` for invalid
keys, `408` for a stalled upload, `413` for oversized bodies, `422` for invalid inputs,
`409` when another operation is busy,
`503` for unavailable champion/MLflow dependencies, and `500` for invalid model
artifacts or unexpected internal failures. Error responses do not echo request data
or internal tracebacks.

One nonblocking in-process guard serializes training and prediction; overlapping
operations fail immediately rather than queueing. Health endpoints remain available:
liveness does no I/O, and readiness allows one probe at a time with a three-second
response deadline so a stalled registry cannot exhaust request workers.
That deadline does not cancel the underlying SDK call. Shutdown does not wait in
the event loop, but Python still waits for a running probe thread before process
exit. Keep the MLflow network timeout/retry settings above; use a service supervisor's
termination deadline if a hard shutdown bound is required. No hard process-exit
deadline is enforced by the API itself.
Use one worker/replica: the guard does not coordinate multiple API workers. Champion
reads share the existing promotion file lock, which also protects registry updates
across single-host processes.
No background jobs, automatic request retries, or model cache are added. MLflow SDK
network retry/timeout behavior follows its environment settings; the startup example
uses zero retries and a ten-second per-request timeout, not a total training deadline.

Each successful prediction response contains a request ID, inference run ID,
registered model name, exact version, and predictions. One MLflow run per valid
batch is tagged `purpose=inference`, with model/training/reference identities,
batch size, timing, selected clipped **pre-imputation** inputs (`inputs.csv`), and
row-aligned outputs (`predictions.csv`). Unused predictors, headers, and API keys
are not saved. These artifacts are not anonymous merely because subject IDs are absent.
Required logging must finish before a successful response; failed operations are
marked failed where MLflow is reachable. Readiness checks registry availability,
not artifact integrity or spare capacity; a ready service can still report busy or
an artifact-loading error on a prediction request.

## Streamlit monitoring dashboard

Start MLflow and the API first and create actual prediction batches through `/predict`.
The dashboard reads MLflow directly; it does not train, serve predictions, or need the
API key. In another terminal:

```bash
export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
export MLFLOW_HTTP_REQUEST_MAX_RETRIES=0
export MLFLOW_HTTP_REQUEST_TIMEOUT=10
uv run --frozen streamlit run src/parkinsons_ml/dashboard.py \
  --server.address 127.0.0.1 --server.port 8501
```

Open <http://127.0.0.1:8501>. Tracking settings come from the environment without
dotenv loading; the registered model name comes from `params.yaml`. Keep the dashboard
local: it has no separate authentication and can display sensitive measurements and
write monitoring reports. External deployment requires access controls and TLS.

Select an immutable model version (champion by default), then **Refresh window**.
The selector lists up to 100 recent versions plus the independently resolved champion,
so an older champion remains available even when it falls outside that history.
The default lookback is 24 hours, adjustable to 1–168 hours. A fixed UTC cutoff
selects completed inference runs in the selected training experiment. At most 100
recent batches are scanned, using bounded pagination; at most 10,000 rows are
included, always as whole batches. The dashboard shows the actual included time
range, incompatible-batch count, and truncation. A busy model's traffic can consume
the scan limit before older traffic for another model version is reached.

Only batches with matching model version, training/reference run, dataset/split,
schema/policy/evaluation identities, feature selection, and representation are combined.
CSV record widths are checked before parsing; missing/extra fields cannot be silently
padded or interpreted as an index. Inputs and predictions are checked for row alignment
and valid values. The reference
is the selected model's saved **development-data** reference; its checksum and
metadata are verified. Corrupt compatible artifacts block reporting rather than
being silently skipped. Artifact files are limited to 8 MiB before parsing.
Neither the dashboard nor monitoring reads `test.csv` or loads a serialized model.

**Generate and save report** runs Evidently and creates a separate MLflow run tagged
`purpose=monitoring`. Normal refreshes, widget reruns, and opening saved reports do
not create runs. Reports contain a cohort/window manifest with exact batch IDs and
checksums, a quality/drift summary, and Evidently JSON/HTML when drift is evaluable.
Required uploads must complete before success is displayed. Raw input CSVs are not
duplicated into monitoring runs. Saved reports remain readable after restart.
HTML is offered as a download, not embedded in the dashboard.
Report policy `development-reference-drift-v2` binds the summary to the exact window
manifest with a deterministic SHA-256 digest and validates row counts and sample
configuration before saving or opening. This detects mismatched artifacts, not a
malicious trusted-store writer. Older unbound reports remain in MLflow but are not
accepted or listed by this reader; generate a new report with the current policy.

Monitoring uses selected, clipped **pre-imputation** inputs. Missingness and
distributions are reported separately from drift. Numerical features use normalized
Wasserstein distance; `sex`, when selected, uses Jensen–Shannon distance. Both use
a fixed 0.1 threshold; a drift signal requires at least 50% of evaluated features.
Each feature needs at least 50 nonmissing rows in both datasets. Constant numerical
reference columns and insufficient/all-missing columns are marked unavailable.
Partial reports identify the unevaluated features; a low drift share is not proof
that those features are stable. Methods, thresholds, library version, and sample
requirements are saved with each report.

The dashboard also shows recent prediction summaries, model provenance, pending
promotions, promotion decisions/reasons, and saved reports. Measurement previews are
opt-in and capped at 100 rows. Empty/insufficient windows are explicit, not “no drift.”
Current performance is unavailable because inference batches have no ground-truth
labels. Drift is a distribution signal, not evidence of performance degradation;
repeated patient measurements and population changes can affect it. Nothing in
monitoring triggers retraining or promotion. Synthetic stable/shifted/missing-data
fixtures exist **only in tests**, never as demonstration traffic.

## MLflow tracking server

From the repository root, start the pinned server in a dedicated terminal:

```bash
./scripts/start_mlflow.sh
```

The Bash script creates storage directories, runs the frozen uv environment, and
resolves all paths relative to the repository even when invoked from elsewhere.
Stop any server already using port 5000 before starting another instance.

Both tracking and registry metadata are explicitly routed to the same SQLite
database, overriding `MLFLOW_REGISTRY_STORE_URI`. The launcher rejects nonempty
`MLFLOW_READ_REPLICA_BACKEND_STORE_URI`, `MLFLOW_ARTIFACTS_ONLY`, and
`MLFLOW_ARTIFACTS_PRESIGNED_ONLY` settings; unset them first (including explicit
`false` values). Other environment settings are retained.

For isolated tests or another local store, only storage and port are configurable:

```bash
PARKINSONS_MLFLOW_STORAGE_DIR=/absolute/path/to/mlflow-test \
PARKINSONS_MLFLOW_PORT=5001 ./scripts/start_mlflow.sh
```

Relative storage paths resolve against the repository root. Changing the directory
selects a different database/artifact store; changing the port requires updating
client URIs. Only the default `storage/mlflow/` location is automatically Git-ignored.

Open <http://127.0.0.1:5000>. When working through Remote SSH, forward port 5000
instead of exposing the server publicly. This command runs in the foreground;
stop with Ctrl+C and rerun it to restart against the same persistent store. It is
not a boot-time service. No authentication is configured: keep it localhost-only.

`storage/mlflow/mlflow.db` contains run/registry metadata and
`storage/mlflow/artifacts/` contains models and reports. Both are ignored by Git
and are separate from DVC datasets. Absolute paths prevent working-directory
changes from selecting a different store. Keep both locations together; for a
simple consistent backup, stop the server and copy the entire `storage/mlflow/`
directory before restarting. Do not back up just the database or just artifacts,
or edit MLflow's internal tables.

In each client terminal, configure the same HTTP server explicitly:

```bash
export MLFLOW_TRACKING_URI="http://127.0.0.1:5000"
export MLFLOW_REGISTRY_URI="$MLFLOW_TRACKING_URI"
```

These exports apply to that shell; `.env` is not automatically loaded. Training
requires an explicit tracking URI and uses that same server for model loading;
there is no untracked-training/local-store fallback.

HTTP artifact serving means clients do not need a shared artifact-directory mount.
New experiments should have `mlflow-artifacts:/...` locations, not server-local
`file:` paths. Changing server options does not rewrite existing experiments'
artifact locations. This setup uses one SQLite-backed server and one worker;
multi-host deployment and high concurrency are outside the project scope.

The default test suite includes real HTTP integration tests for both project
pipelines, artifact bytes, model registry versions, fresh-process model loading,
restart persistence, and failure behavior. They start isolated temporary MLflow
servers through the same launcher with isolated storage/ports; they do not use
port 5000 or modify this development store. Run the full `uv run --frozen pytest`
for the normal quality gate, including
the 95% whole-project coverage requirement. Localhost subprocess/network
access is required. The tests use separate client/server working directories and
verify HTTP routing; they do not constitute a container mount-isolation test.

## Notebooks

Commit `.ipynb` files; nbstripout clears outputs on commit. Review and restage hook
changes. Configure nbdime once per clone for local notebook diffs and merges:

```bash
git config --local diff.jupyternotebook.command 'uv run --frozen git-nbdiffdriver diff'
git config --local merge.jupyternotebook.driver 'uv run --frozen git-nbmergedriver merge %O %A %B %L %P'
```

## Checks

```bash
uv run --frozen pytest
uv run --frozen pre-commit run --all-files
uv pip check --python .venv/bin/python
```

Pytest enforces 95% coverage over `src/` and tests. Hooks format/lint Python, check
source and test types, strip notebook outputs, and check file hygiene. Stage new files before
running `--all-files`; review and restage automatic fixes.

### Continuous integration

`.github/workflows/ci.yml` runs on pull requests, pushes to `main`, and manual
dispatch. Its single Ubuntu job uses Python 3.13 and pinned uv/actions to check
lockfile consistency, frozen installation, dependency compatibility, the requirements
export, all pre-commit hooks, and the full pytest suite with the existing 95%
coverage gate. Hook fixes or a stale export fail CI; the workflow does not commit fixes.

Tests use generated data and isolated temporary MLflow servers, so CI needs neither
the sibling DVC remote nor a development tracking server or project secrets.
Docker builds and deployment smoke tests will extend this same workflow later.

After the first successful GitHub run, configure the repository's `main` branch
protection/ruleset to require the **Quality gates** status check before merging.
The workflow file alone does not enable that enforcement. A manual dispatch is
available after the workflow exists on the default branch. For a first feature
branch, open a pull request to exercise it before merging.

## Project documents

- [Implementation plan](documents/implementation-plan.md): remaining work and acceptance criteria.
- [AGENTS.md](AGENTS.md): development rules.

AI assistance has been used for planning, configuration, dependency compatibility
checks, and test implementation; review generated changes before committing.
