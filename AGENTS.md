# Project instructions

## Purpose and scope

Build a production-ready, reproducible Parkinson's telemonitoring MLOps system. Prioritize reproducible pipelines, observable services, and reliable automation over a fixed model-score target. Automatically improve performance where feasible and prevent potential degradation through validation and guarded model promotion.

Do not claim that checks passed, screenshots were captured, or services ran unless they actually did. Document failures, fixes, limitations, and justified deviations.

## Dataset and modeling

- The raw dataset is `data/raw/parkinsons_updrs.csv`. Keep it in place and preserve the source data; write derived outputs to `data/processed/`.
- Predict **`motor_UPDRS`**, using that exact dataset column name.
- Exclude **`total_UPDRS`** from model inputs.
- Use `subject#` only for patient grouping, split construction, and auditing, never as a predictor.
- Initial predictors are voice measurements, age, sex, and `test_time`.
- Use patient-disjoint train/validation/test partitions because patients have repeated measurements. Use grouped cross-validation on development data; keep the fixed test set out of candidate selection and promotion.
- Fit imputation and other learned preprocessing only on training data. Package preprocessing and the estimator together in a scikit-learn pipeline for identical training and inference behavior.
- Validate schema, types, missing targets, missing values, invalid values, and duplicates. Make cleaning decisions explicit and reproducible.
- Start with a `DummyRegressor` baseline and a random forest regression candidate. Log MAE, RMSE, and R² without promising a particular score.
- Preserve a versioned split manifest, data schema, and preprocessed dataset for reproducibility and export.
- The system is not a clinical diagnostic service; document that limitation in the API and README.

## Python and dependency management

- Use **uv** for environments and dependency management.
- Maintain `pyproject.toml` and commit `uv.lock`.
- Generate and commit `requirements.txt` with exact versions from the lockfile; do not maintain a conflicting dependency list manually.
- Use frozen dependency resolution in CI and Docker.
- Select and document a Python version compatible with the chosen MLflow, Evidently, and optional Airflow versions.
- Follow PEP8, use type hints, and configure a maximum line length of **120**.
- Keep business logic under `src/parkinsons_ml/`. Streamlit and optional Airflow code must be thin adapters to tested business logic in `src/`.

## Suggested layout

```text
src/parkinsons_ml/
  api/
  data/
  modeling/
  monitoring/
  tracking/
  config.py
tests/
  unit/
  integration/
  e2e/
dashboards/
dags/                     # Optional Airflow
data/raw/
data/processed/
docker/
scripts/
screenshots/
.github/workflows/
```

Adapt this layout to existing code rather than replacing established structures unnecessarily.

## Data versioning and reproducibility

- Use **Git** for code, configuration, documentation, and DVC metadata.
- Use **DVC** for raw and processed datasets. Do not commit datasets or model binaries directly to Git.
- Define reproducible validation, preparation, and splitting stages in `dvc.yaml`; commit the corresponding lockfile and metadata.
- Configure a persistent DVC remote and document access. A clean checkout must not depend on an undocumented local cache.
- Support an accessible pinned dataset download or a documented offline data bundle for deployment and reproducibility.
- Record seeds, dataset identities, split identities, and code revisions for reproducibility.

## MLflow and model lifecycle

- Use **MLflow Tracking** for experiments and artifacts and **MLflow Model Registry** for model versions.
- For training runs, log parameters, seeds, dependencies, feature schema, Git revision, DVC dataset identity, split metadata, metrics, evaluation reports, and promotion decisions.
- Save the complete inference pipeline with its input example and model signature.
- Use a registry alias such as `champion` for the deployed version and retain earlier versions for rollback.
- Track and package pipelines with MLflow. MLflow is not a general workflow scheduler: use DVC for data stages and optionally Airflow for orchestration. Provide an MLflow Project training entry point where appropriate.
- Compare candidates against the champion using comparable development validation data and a versioned acceptance policy.
- Reject candidates with invalid data, non-finite predictions, failed evaluation, incomplete required artifacts, or unacceptable validation degradation.
- Never use the held-out test set for automatic tuning or promotion.
- Load and validate a promoted model before replacing the active API model.
- Drift may trigger investigation or candidate training, not unconditional promotion. Retraining requires suitable labeled data.

## FastAPI requirements

Provide a REST API with interactive Swagger documentation and these endpoints:

- `POST /train`: train, evaluate, register, and conditionally promote a candidate. Return the run ID, model version, metrics, and promotion decision.
- `POST /predict`: validate a bounded batch, run the deployed pipeline, and return predictions, request ID, model version, and tracking run ID.
- `GET /health/live`: process liveness.
- `GET /health/ready`: required dependency and model readiness.

For the initial small dataset, synchronous training is acceptable with a one-training-job-at-a-time guard. Add a durable job queue only if needed; do not run critical jobs as untracked in-process background tasks.

- Accept only approved dataset versions and bounded configuration, not arbitrary paths or executable input.
- Protect training with authentication and keep secrets out of source control and logs.
- Save artifacts with MLflow for **both training and prediction**.
- Initially, use one MLflow inference run per request batch, storing validated inputs, predictions, timestamps, schema version, model identity, and latency.
- Persist structured incoming-request and prediction records suitable for Evidently reports.
- Define and test tracking-failure behavior; never silently claim an unlogged result was recorded.
- Avoid unnecessary patient identifiers in request and prediction logs.

## Monitoring and dashboard

- Use **Evidently** to compare versioned training reference data with recent logged inference data.
- Monitor feature drift, missingness, invalid values, and prediction distribution changes.
- Report model-performance changes only when ground-truth labels are available; feature drift alone does not demonstrate performance degradation.
- Build a **Streamlit** dashboard showing active model identity, training run, recent predictions, request volume, data quality, Evidently reports, drift status, and promotion decisions.
- Keep report generation and decision logic in `src/`, not in the Streamlit entry point.
- Provide a deterministic simulated drift scenario for tests and demonstrations, clearly labeled as synthetic.

## Docker deployment

- The system must start with a single **`docker compose up`** command.
- Use Docker images and Compose services for FastAPI, MLflow, PostgreSQL, MinIO, Streamlit, and required initialization/bootstrap tasks.
- Use separate PostgreSQL databases for MLflow metadata and application audit records.
- Persist database contents, artifacts, and other required state with volumes.
- Create storage buckets automatically. Add health checks, dependency readiness checks, and bounded retries.
- Bootstrap the pinned dataset and an initial champion when needed. Bootstrap must be idempotent and must not retrain or overwrite existing models on every restart.
- Do not require undocumented manual setup, local caches, or pre-existing model artifacts.
- Do not bake secrets into images or commit them to Git. Document safe local defaults and production configuration separately.

## Optional Airflow orchestration

- Implement Airflow only after the required core system is working and tested.
- Keep DAGs thin; put all business logic in `src/`.
- A DAG may collect an inference window, validate it, generate a drift report, determine labeled-data availability, train/evaluate a candidate, and conditionally promote it.
- If implemented, Airflow must start with the same default Compose command and have dashboard evidence.

## Tests and mandatory quality gates

- Use pytest and **coverage**. Enforce at least **95% Python test coverage** across all Python code in `src/` **and the Python test code itself**.
- Configure measurement to include test modules; unexecuted test code is also a defect.
- Streamlit and Airflow adapters may remain thin, but their business logic belongs in covered `src/` code.
- Do not game coverage with exclusions. `# pragma: no cover` or coverage omissions require an explicit adjacent justification; for configuration omissions, place the explanation beside the configuration.
- Aim for zero Flake8 violations, with an upper limit of 1% of all Python lines; `noqa` requires an adjacent justification.
- Require zero Pyright errors in `src/`. Target this strict configuration from the outset:

```toml
[tool.pyright]
include = ["src"]
typeCheckingMode = "strict"
reportMissingTypeStubs = false
reportUnknownMemberType = false
reportUnknownArgumentType = false
reportUnknownVariableType = false
```

- At minimum, standard-mode Pyright must pass with zero errors if strict checking is deferred and documented.
- Test preprocessing, validation, patient-disjoint splitting, reproducibility, tracking artifacts, registry integration, promotion/rejection/rollback, API contracts, authentication, training concurrency, tracking failures, unavailable dependencies, and stable versus shifted monitoring inputs.
- Use mocks for isolated unit tests and real services for integration tests.
- Include a clean-start Compose smoke test exercising readiness, training, prediction, and required artifacts.
- Maintain a passing Docker build and check lockfile/requirements consistency.

## Pre-commit and CI

- Configure **pre-commit** for fast formatting, Flake8, Pyright, file hygiene, configuration validation, large-file checks, and secret detection where practical.
- Local setup and full checks should be available through:

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

- Hooks may modify formatting; review and stage those changes before retrying the commit.
- Do not put slow Docker builds or the full integration suite on every local commit. Keep local feedback fast to support frequent meaningful commits.
- GitHub Actions must explicitly run pre-commit checks: GitHub does not automatically execute local Git hooks.
- CI must also run the full test suite, enforce coverage, build Docker images, start the Compose stack, and execute API smoke tests.
- Publish test and coverage reports as CI artifacts.
- Configure required status checks and branch protection where repository permissions allow; local hooks alone can be bypassed.

## Git workflow

- Initialize Git before implementation if it is not initialized already.
- Make frequent, meaningful commits throughout development. Preserve incremental changes and fixes; do not fabricate history or split changes artificially to inflate commit counts.
- Follow the repository's established commit-message format; confirm the convention if none exists.
- Keep `main` clean and passing. Except for allowed typo fixes, changes must reach `main` through merges.
- Use at least two feature branches and preserve their history, preferably through pull requests and merge commits rather than squash merges.
- For collaborative development, request reviews and record review notes for each feature branch before merging.
- Never discard existing user changes or publish to a remote without authorization.

## README and evidence

Keep project documentation in **README.md**, including:

- Architecture, responsibilities of tools, and prerequisites.
- Dataset provenance/license, target, excluded features, and leakage controls.
- One-command startup, endpoints, Swagger, and example requests.
- uv development commands, DVC access, reproduction instructions, and dependency management.
- Tracking, registry, evaluation, promotion, rollback, and monitoring policies.
- Test, coverage, lint, typing, and CI commands.
- Configuration, secrets, persistence, and known limitations or technical debt.
- A short description of AI assistance and how it was used.
- Links to actual screenshots committed under `screenshots/`.

Capture genuine screenshots of:

1. MLflow dashboard, experiments, and registry.
2. Working training and prediction requests using Swagger, Postman, or Python.
3. Running Docker containers.
4. Streamlit displaying an Evidently report.
5. Test results and coverage.
6. Airflow dashboard, if implemented.

Do not substitute fabricated images or claims for running-system evidence.

## Reproducible exports

- Provide a versioned preprocessed dataset export with its schema and provenance.
- Document the repository revision, dataset version, model version, and dependency lockfile needed to reproduce a release.
- Document offline setup when external storage or private repository access is unavailable.
- Keep credentials and sensitive records out of exported bundles.

## Implementation priorities

1. Git, uv, code-quality configuration, and initial CI.
2. DVC, dataset validation, and leakage-safe splitting.
3. Training pipeline, MLflow tracking, and model registry.
4. FastAPI endpoints, inference logging, and integration tests.
5. Guarded automated model selection and promotion.
6. Evidently reporting and Streamlit.
7. Reliable one-command Docker Compose deployment and end-to-end CI.
8. Optional Airflow.
9. Verified screenshots, complete README, and reproducible exports.

Build incrementally, keep tests alongside implementation, and document deviations rather than silently dropping requirements.
