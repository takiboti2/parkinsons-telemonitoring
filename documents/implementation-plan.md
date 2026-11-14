# Parkinson's Telemonitoring — Implementation Plan

**Updated:** 14 November 2026  
**Target:** `motor_UPDRS` regression  
**Runtime:** Python 3.13  
**Storage profile:** SQLite + local filesystem, single host  
**Deployment entry point:** `docker compose up`

## 1. Objectives and boundaries

Build a reproducible system for predicting motor symptom severity from voice measurements and patient information. Prioritize reliable automation, traceability, and prevention of model degradation rather than a fixed performance score.

The system will provide:

- Reproducible, versioned data preparation and patient-disjoint evaluation.
- MLflow experiment tracking, artifacts, model registration, and guarded promotion.
- FastAPI training and prediction endpoints with auditable requests and predictions.
- Evidently drift reporting presented through Streamlit.
- Automated quality checks, tests, and container deployment.
- A minimal default stack without PostgreSQL, MinIO, cloud credentials, or S3 plugins.
- Optional Airflow orchestration in a separate profile after the required services are working.

This system is not a clinical diagnostic service. Feature drift alone must not be described as evidence of reduced predictive performance.

## 2. Current implementation status

| Area | Current state | Remaining work |
| --- | --- | --- |
| Git | Repository initialized | Feature-branch workflow, reviews where applicable, remote protections |
| Dependencies | Exact application/development pins, `uv.lock`, exported `requirements.txt` | Preserve consistency as functionality is added |
| Local quality checks | Installed hooks; Black/Flake8/Pyright; central pytest coverage over `src` and tests | Apply gates continuously to new source code |
| Compatibility tests | Dependency smoke suite includes SQLite audit persistence and filesystem DVC recovery | Application, integration, failure-path, and deployment tests |
| Raw dataset | Available at `data/raw/parkinsons_updrs.csv` | DVC initialization, provenance, validation, and preprocessing |
| Application | No Python implementation in `src/` yet | Packaging, modeling, API, tracking, and monitoring |
| Airflow | Separate pinned requirements; isolated DAG-construction smoke test passed | Optional operational orchestration |
| Deployment and CI | Not implemented | Dockerfiles, Compose, GitHub Actions, operational evidence |

Compatibility-test results are not evidence that the full application or deployed system works. No running API, dashboard, database, or Compose deployment is claimed by this plan.

## 3. Architecture and responsibilities

```text
Versioned raw data (DVC)
        |
        v
Validate -> Prepare -> Patient-disjoint split
        |
        v
Training pipeline -> Development evaluation -> Candidate registry version
        |                                      |
        v                                      v
MLflow runs and artifacts              Guarded champion promotion
                                               |
                                               v
Client -> FastAPI /predict -> Active pipeline -> Predictions
                 |                                  |
                 +------ Request/prediction log -----+
                                    |
                                    v
                       Evidently reports -> Streamlit
```

| Component | Responsibility |
| --- | --- |
| Git | Code, configuration, documentation, dependency locks, DVC metadata |
| uv | Python environment, exact dependency resolution, frozen installation |
| DVC | Dataset versions and reproducible validation/preparation/splitting stages |
| scikit-learn | Training/inference pipeline and grouped evaluation |
| MLflow | Runs, metrics, artifacts, packaged models, registry, champion alias |
| FastAPI | Validated HTTP contracts, training access control, inference |
| SQLite | Separate local files for MLflow metadata/registry and application audit records |
| Local filesystem | Persistent artifact volume served by MLflow; separate filesystem DVC remote |
| Evidently / Streamlit | Drift analysis and monitoring presentation |
| Docker Compose | Service lifecycle, networking, persistence, initialization |
| Airflow, optional | Scheduled monitoring and conditional retraining orchestration |

MLflow tracks and packages workflows; it is not the general-purpose scheduler. Use DVC for data stages and Airflow, if implemented, for orchestration.

### Minimal local storage contract

| State | Planned location | Owner and access |
| --- | --- | --- |
| MLflow metadata/registry | `/var/lib/mlflow/mlflow.db` on a named local volume | MLflow service only |
| Application audit | `/var/lib/parkinsons/audit.db` on a separate named local volume | Single API process; dashboard reads through API |
| MLflow artifacts | `/var/lib/mlflow/artifacts` on a separate named local volume | MLflow serves/proxies HTTP artifact access |
| DVC remote | `/var/lib/dvc-remote`, mapped to a documented persistent host directory | DVC commands, separate from cache and artifacts |

These paths are planned defaults, not existing deployment configuration. Do not mount MLflow database or artifact files into API/dashboard containers. A local DVC remote still needs an explicit first-use data source or offline bundle; an empty remote cannot restore the dataset by itself.

The default deployment is single-host and low-concurrency. Use one API worker/replica, short SQLite transactions, bounded busy timeouts, and WAL mode for the application audit database. Keep database files on local disks, not network filesystems. Migration to PostgreSQL/object storage is a future scaling option, not a default requirement.

## 4. Repository and package structure

Create directories when adding real files; avoid empty placeholder trees.

```text
src/parkinsons_ml/
  api/                   # Routes, schemas, authentication, dependencies
  data/                  # Validation, preparation, split construction
  modeling/              # Training, evaluation, acceptance policy, promotion
  monitoring/            # Inference windows, drift reports, decisions
  tracking/              # MLflow and registry adapters
  config.py              # Typed environment configuration
tests/
  test_dependency_compatibility.py
  unit/
  integration/
  e2e/
dashboards/              # Thin Streamlit entry point
dags/                    # Optional thin Airflow adapters
data/raw/                # Source dataset, preserved unchanged
data/processed/          # Versioned derived outputs
docker/                  # Images, initialization, separate Airflow requirements
documents/               # Technical planning documents
scripts/                 # Operational entry points as needed
screenshots/             # Genuine running-system evidence
.github/workflows/       # Automated checks and deployment smoke tests
```

Configure a build backend and an installable `src` package before implementing application imports. Keep business logic in `src/`, including logic called by Streamlit and Airflow.

## 5. Data and modeling design

### 5.1 Feature contract

| Role | Columns |
| --- | --- |
| Target | `motor_UPDRS` |
| Excluded predictor | `total_UPDRS` |
| Grouping/audit only | `subject#` |
| Predictors | Voice measurements, `age`, `sex`, `test_time` |

Normalize API field names deliberately and maintain an explicit mapping to original dataset columns. Freeze the feature order and schema with the model artifacts.

### 5.2 Versioning and validation

1. Initialize DVC and track the raw CSV without committing its contents to Git.
2. Document dataset origin, license, checksum, and schema.
3. Configure a persistent filesystem remote outside the working data and cache, with documented host/container paths and an accessible pinned-data retrieval path or offline bundle.
4. Validate required columns, numeric types, finite values, missing targets, invalid values, and duplicates.
5. Make cleaning decisions explicit; do not silently delete repeated patient measurements as duplicates.
6. Define `validate`, `prepare`, and `split` stages in `dvc.yaml`; commit DVC pointers and lock metadata.
7. Export validated/prepared data, a schema, and a reproducible split manifest to `data/processed/`.

Keep raw data unchanged. Git ignore rules must exclude data payloads while allowing `.dvc` files and directory-level `.gitignore` files.

### 5.3 Leakage prevention and preprocessing

- Reserve a fixed patient-disjoint test set.
- Use grouped cross-validation within development data, with no patient shared between fitting and validation partitions.
- Record seeds, patient assignments, dataset identity, and split-policy identity.
- Fit imputation and any learned transformations only inside the training pipeline and training folds.
- Serialize preprocessing and estimator as one scikit-learn pipeline.
- Do not use the test set for tuning, acceptance thresholds, or automated promotion.

This protocol measures generalization to unseen patients. A future task predicting later visits for known patients would need a separate temporal evaluation design.

### 5.4 Baseline and candidate evaluation

Every fitted experiment must use scikit-learn's `LinearRegression` (ordinary least squares), including compatibility tests. Do not fit dummy regressors, random forests, Ridge, Lasso, or other estimator families.

| Experiment | Estimator | Predictors | Purpose |
| --- | --- | --- | --- |
| Baseline | `LinearRegression` | `age`, `sex`, `test_time` | Establish predictive value from patient information and elapsed trial time without voice measurements |
| Full-feature candidate | `LinearRegression` | Baseline features plus all 16 voice measurements | Measure the incremental predictive value of voice features |

Evaluate both on identical patient-disjoint development splits and grouped cross-validation folds using MAE, RMSE, and R². Fit imputation separately within each training fold, using an explicit feature list followed by the estimator in one pipeline. Do not scale, clip, transform, or select features by default; any later preprocessing experiment must remain inside the training pipeline and retain `LinearRegression` as its estimator. Neither experiment may use `subject#` or `total_UPDRS` as predictors. Keep the final test set out of selection and promotion.

Ordinary least squares requires no estimator random seed; preserve split seeds and fold assignments for reproducible comparisons. Log the estimator name and feature-set identity with every run. The baseline is an evaluation reference, not a guaranteed deployment choice: promote only a candidate that passes the common acceptance policy.

The exported preprocessed dataset must be documented separately from learned model transformations: exporting data must not cause preprocessing to be fitted on held-out observations.

## 6. Tracking and model lifecycle

### Training artifacts

Every training run must log:

- Hyperparameters, seeds, feature schema, and dependency versions.
- Git revision, DVC dataset identity, split identity, and acceptance-policy version.
- Data-quality reports, baseline/candidate metrics, and evaluation summaries.
- Complete inference pipeline, model signature, and representative input example.
- Registration details and a promotion/rejection decision with reasons.

Provide a reproducible training entry point and an MLflow Project definition where appropriate. Verify required artifacts before permitting promotion.

### Guarded promotion and rollback

1. Resolve the approved dataset and evaluation protocol.
2. Train and evaluate candidates on development data.
3. Evaluate the current champion comparably; do not compare unrelated historical scores.
4. Reject invalid data, non-finite predictions, failed evaluation, missing artifacts, or unacceptable validation degradation.
5. Register valid candidates without changing `champion`.
6. Load the candidate by immutable registry version and smoke-test its schema and predictions. Failure leaves the alias and active model unchanged.
7. Under the promotion guard, capture the previous alias/active pipeline, update `champion`, and activate the already-validated pipeline. If activation fails, restore both and record the rollback.
8. Retain previous registry versions and reconcile alias/active-model state after a crash or interrupted promotion; the alias and in-memory swap are not one database transaction.

Use one API worker/replica and guard competing training/promotion operations within that process. Bootstrap and optional scheduled jobs must not introduce concurrent promotion writers; Airflow should call the API. Before adding API processes or replicas, redesign locking and reassess the SQLite storage choice.

Drift can trigger investigation or candidate training when suitable labels exist; it must never directly trigger unconditional model replacement.

## 7. FastAPI and audit logging

| Endpoint | Contract |
| --- | --- |
| `POST /train` | Authenticated training, evaluation, registration, conditional promotion; returns run ID, version, metrics, decision |
| `POST /predict` | Bounded validated batch; returns predictions, request ID, model version, inference run ID |
| `GET /health/live` | Process liveness |
| `GET /health/ready` | Required dependency and active-model readiness |
| `/docs` | Interactive Swagger documentation |

Initially use synchronous training with a single-training-job guard. Introduce a durable queue only if execution time requires it; avoid untracked critical background tasks.

### Request handling

- Allow approved dataset versions and bounded hyperparameters, not arbitrary paths or executable configuration.
- Reject missing/extra features as defined by the contract, invalid values, and oversized batches.
- Define stable error responses for unavailable dependencies, missing models, invalid input, and conflicting training requests.
- Store secrets outside source control; protect training and avoid unnecessary patient identifiers in logs.

### Prediction artifacts

Use one MLflow inference run per request batch initially. Save validated inputs and predictions as artifacts, with request ID, timestamp, schema version, model identity, and latency.

Persist structured records in a dedicated local SQLite audit file for monitoring windows, using parameterized statements and short transactions. Do not reuse MLflow's tables. The API owns database writes and exposes a read-only monitoring interface for Streamlit; avoid sharing SQLite files across containers. Define failure semantics for partial logging, SQLite lock timeouts, disk exhaustion, and tracking outages: do not report successful audit persistence unless it occurred. Test correlation and consistency between API responses, database records, and MLflow runs.

## 8. Monitoring and automatic safeguards

Use versioned training data as the Evidently reference and recent logged requests as current data. Record report windows, reference identity, model version, and sample counts.

Monitor feature drift, missingness, invalid values, and prediction distribution changes. Calculate performance metrics only when trustworthy ground-truth labels are available and joined correctly to predictions.

Streamlit will present:

- Active model version and training run.
- Request volume and recent predictions.
- Data-quality summaries and monitoring-window sample counts.
- Embedded Evidently reports, drift status, and latest promotion decision.

Keep report generation and policy decisions in `src/`. Add deterministic stable-data and simulated-drift scenarios for tests and demonstrations; label synthetic data clearly. Define minimum sample requirements so sparse windows are not reported as conclusive drift findings.

## 9. Deployment and dependency management

### Compose services

| Service | Startup responsibility |
| --- | --- |
| `mlflow` | Own persistent SQLite metadata; serve registry and local artifact access over HTTP |
| `bootstrap` | Initialize local directories and retrieve pinned data; create initial champion only if absent |
| `api` | Serve training/inference with one worker/replica; own persistent SQLite audit records |
| `streamlit` | Display monitoring state and reports |
| Airflow services, optional profile only | Schedule monitoring and conditional retraining through the API |

The default stack must start with `docker compose up`, with health checks, bounded retries, persistent volumes, and no undocumented manual setup. Pin image versions when creating Dockerfiles and Compose definitions.

Bootstrap must tolerate restarts without retraining unnecessarily or overwriting model state. Validate a fresh deployment and a restart with persisted volumes. Initialize artifact serving before creating experiments so clients receive proxied artifact URIs rather than container-only file paths. Bootstrap/model loading must use the MLflow server, not direct database access.

Bind published ports to localhost by default. Document disk capacity, permissions, and backups for both SQLite databases, artifact files, and the DVC remote. Use SQLite's backup API or quiesce writers for consistent backups; copying a live database without its WAL is not sufficient. Never bake credentials into images.

Airflow remains optional and outside plain `docker compose up`. If implemented, document an explicit profile such as `docker compose --profile airflow up`, a compatible executor/database choice, and any profile-only services. Its upstream dependencies and operational requirements must not expand the default application stack.

### Dependency constraints

- Install application/development dependencies using `uv sync --frozen`.
- Keep direct exact pins in `pyproject.toml`, transitive pins in `uv.lock`, and generated exports synchronized.
- Use base `dvc` for the filesystem remote and standard-library `sqlite3` for audit storage; retain SQLAlchemy for MLflow/SQLite integration.
- Remove direct PostgreSQL drivers, `dvc-s3`, and `boto3` from the default requirements. Some libraries may still bring unrelated optional integrations transitively; do not require those services in the local workflow.
- Preserve pandas **2.3.3** with Evidently **0.7.23**; pandas 3.1.1 failed runtime import checks.
- Allow the verified source build of DVC's ANTLR dependency; do not assume all dependencies have wheels.
- Review MLflow/skops trusted types explicitly for locally trained artifacts; never blindly trust external serialized models.
- Keep Airflow **3.4.0** in its separate pinned environment using official Python 3.13 constraints. Communicate through API contracts rather than installing both dependency sets together.
- Keep upstream deprecation warnings visible and rerun compatibility checks after dependency changes.

## 10. Quality gates and Git workflow

| Gate | Acceptance criterion |
| --- | --- |
| Tests | Unit, local SQLite/filesystem integration, real MLflow HTTP integration, failure-path, and end-to-end tests pass |
| Coverage | At least 95% across Python code in `src/` and the test code itself |
| Style | Aim for zero Flake8 violations; maximum 120-character lines |
| Typing | Zero Pyright errors in `src/` under the configured mild-strict policy |
| Dependencies | Lockfile/export consistency and installed-package checks pass |
| Containers | Docker image builds and clean-start Compose smoke tests pass |

Keep Streamlit/Airflow adapters thin and their logic covered in `src/`. Coverage omissions and suppression comments require explicit local justification; do not use them to hide untested behavior.

Pytest discovery, measured sources (`src` and `tests`), and the 95% threshold are configured centrally in `pyproject.toml`. Run `uv run --frozen pytest`; avoid `--cov=tests`, which would narrow measurement and leave future application code out of the gate. Both `.venv` and `.venv-airflow` are excluded from root-level Flake8 discovery.

MLflow tests must isolate tracking/registry overrides, environment variables, experiment selection, and active runs. Fixture cleanup must restore previous state even if the test fails. Narrow test-only patches to the pinned MLflow version's internal overrides are documented in the fixture and must be reviewed during dependency upgrades. Compatibility tests validate a version before assigning the champion alias; production activation and rollback remain future application work.

Local pre-commit hooks provide fast formatting, linting, source typing, file hygiene, configuration validation, large-file checks, and private-key detection. The existing private-key check is not a comprehensive secret scanner.

GitHub Actions must explicitly run the hooks, full tests, coverage, image builds, and Compose API smoke tests. Upload reports and logs as CI artifacts. Configure required checks and branch protection where permissions permit.

Use meaningful incremental commits, at least two preserved feature branches, and pull-request merges. Except for permitted typo fixes, changes should reach `main` through merges. Follow the established commit convention and use recorded reviews for collaborative changes. Do not manufacture history or discard user changes.

## 11. Implementation sequence and completion criteria

| Phase | Deliverables | Completion criterion |
| --- | --- | --- |
| 1. Foundation | Complete packaging and basic CI; maintain central pytest/coverage configuration | Frozen install, hooks, tests, typing, and coverage checks reproducible |
| 2. Data | Filesystem DVC remote, provenance, validation, grouped splits, processed export | Data recoverable with empty cache; stages reproducible; no patient leakage |
| 3. Modeling | Baseline/candidate pipeline, MLflow runs, registry | Artifacts complete; model reload preserves predictions |
| 4. API | Training/prediction contracts, authentication, SQLite audit logging | Success, validation, lock/timeout, and dependency-failure paths tested |
| 5. Safeguards | Comparable evaluation, promotion policy, rollback | Rejected candidate cannot replace champion; rollback verified |
| 6. Monitoring | Evidently reports, inference windows, Streamlit | Stable/shifted scenarios tested and report visible in dashboard |
| 7. Deployment | Minimal Compose services, SQLite/artifact volumes, bootstrap, end-to-end CI | Single-command clean startup and persisted restart both pass without PostgreSQL/MinIO |
| 8. Optional orchestration | Isolated Airflow profile, thin DAG, scheduled workflow | DAG executes monitoring and conditional retraining safely through API |
| 9. Operational documentation | Complete README, screenshots, reproducible exports | Another developer can start, exercise, and reproduce the system |

Implement tests alongside each phase rather than postponing them to the end. Add early deployment scaffolding when needed for real-service integration tests, even though full deployment acceptance comes later.

## 12. Operational evidence and documentation

Keep operational instructions in `README.md`; this document describes implementation intent. Document architecture, data provenance, leakage controls, configuration, startup, endpoints, reproduction, monitoring, promotion, rollback, security limitations, known debt, and AI assistance.

Commit `documents/implementation-plan.md` as versioned technical documentation. Ignore only disposable agent-generated reports under `documents/reports/`, not the entire documents directory. Keep this plan, `AGENTS.md`, and README synchronized as decisions change.

Capture genuine screenshots under `screenshots/` showing:

1. MLflow experiments and registry.
2. Successful training and prediction requests through Swagger, Postman, or Python.
3. Running Docker containers.
4. Streamlit displaying an Evidently report.
5. Test results and coverage.
6. Airflow dashboard, if implemented.

Exports must identify the code revision, dataset version, model version, schema, and dependency lockfile. Provide documented offline reproduction where required, without bundling secrets or unnecessary sensitive records.

### Final acceptance checklist

- [ ] Fresh checkout starts with `docker compose up` using documented data access.
- [ ] Bootstrap is idempotent and model/artifact state survives restart.
- [ ] Default stack uses separate local SQLite databases and local MLflow artifacts without PostgreSQL, MinIO, or S3 credentials.
- [ ] DVC can recover pinned data from its filesystem remote with an empty local cache.
- [ ] Database locking, disk/permission failures, and consistent backup procedures are addressed.
- [ ] Training and prediction both produce traceable MLflow artifacts.
- [ ] Requests and predictions are available for reproducible monitoring windows.
- [ ] Patient-disjoint evaluation and held-out-test isolation are enforced.
- [ ] Automatic promotion cannot bypass validation or acceptance policy.
- [ ] Streamlit displays a genuine Evidently report.
- [ ] Tests, coverage, style, typing, dependency checks, and image builds pass.
- [ ] CI enforces required checks and Git history remains auditable.
- [ ] README, screenshots, and reproducible data/model exports match actual behavior.
