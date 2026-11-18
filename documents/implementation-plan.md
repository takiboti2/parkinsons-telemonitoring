# Implementation plan

## 1. Establish the application foundation

- The installable `src/parkinsons_ml/` package, preprocessing, experimentation, and
  tracking modules exist. Add API and monitoring modules as those services are implemented.
- Define typed settings for MLflow, dataset paths, API authentication, and batch limits.
- The initial GitHub Actions workflow covers hooks, tests, coverage, lock/export
  consistency, and dependency compatibility. Verify its first hosted run and enable
  the required merge check; extend the same workflow with Docker/deployment checks.

## 2. Reproducible data preparation

The single `prepare` stage, canonical schema, cleaning audit, train/test assignments,
grouped folds, descriptive extreme flags, and staged artifact publication are implemented.
Training now logs preparation reports/identities and selected-feature monitoring
references. See README for commands and the data dictionary for policy. Remaining:

- Review and commit DVC metadata with code; publish data to the approved remote and
  verify empty-cache recovery whenever dataset or preparation identities change.

## 3. Start MLflow as the authoritative store

- Localhost MLflow setup with persistent SQLite metadata and HTTP-served local
  artifacts is documented in README. Real HTTP artifact/model round-trip,
  registry, restart, and failure integration tests cover the existing pipelines.
- Remaining: configure clients in the API/dashboard paths and
  migrate the same storage design to persistent Docker volumes during deployment.
- Use MLflow for training records, model registry, inference batches, and monitoring
  reports. Do not introduce a separate audit database or monitoring API.
- Tag records with their purpose, dataset/schema identity, and relevant timestamps
  so training, inference, and monitoring runs can be queried separately.

## 4. Integrate the shared experimentation function

The synchronous `tracking.run_experiment` implements saved-fold comparison,
pooled out-of-fold selection, all-configuration refits, HTTP MLflow logging/reload
checks, and typed results. `experimentation` owns evaluation and `training` owns
pipeline construction. See README for the module boundaries, CLI, and policy.

- Reuse this function from `/train` and bootstrap without invoking the CLI.
- Define explicit build provenance before supporting training from installed wheels
  or deployment images without their own Git source checkout.
- Implement a separate final-test assessment operation; keep test data out of
  experimentation, tuning, and promotion decisions.
- Capture genuine development-system run evidence when running real-data experiments;
  integration tests use generated datasets and temporary tracking servers.

## 5. Integrate champion selection with serving

Automatic strict-MAE promotion, immutable registry validation, a promotion-only
single-host guard, alias rollback, and limited interrupted-update reconciliation
are implemented. See README for configuration, audit records, and limitations.

- Each future prediction must resolve `champion` once and load that exact version;
  do not introduce a second active-model pointer. Add an immutable-version cache
  only if measured latency requires it.
- Expose promotion decisions separately from completed training in the API/UI.
- Capture genuine operational evidence and document reviewed manual alias changes
  without bypassing unresolved promotion intents or comparability checks.

## 6. Expose training and logged prediction through FastAPI

- Implement authenticated `POST /train`, bounded `POST /predict`, liveness/readiness
  endpoints, and Swagger examples. Accept approved configuration, not arbitrary paths.
- Use synchronous training and one API worker/replica. Return run/request IDs,
  model identity, and training metrics/promotion decisions where applicable.
- Create one inference run per request batch. Save validated pre-imputation inputs
  and predictions as MLflow artifacts, with timestamp, latency, schema version,
  and resolved model version. Avoid unnecessary patient identifiers or secrets.
- Complete required logging before returning success. Define validation, missing-model,
  dependency/logging-failure, and concurrent-training responses; do not silently
  succeed with an unlogged prediction. Mark incomplete runs as failed.

## 7. Build monitoring directly from MLflow

- Let Streamlit query MLflow and download completed inference-run artifacts for a
  bounded recent window. Filter by compatible schema/model/reference identity.
- Compare requests with a versioned development-data reference in the same
  pre-imputation representation. Generate Evidently reports and save them to MLflow.
- Show model identity, recent predictions, data quality, drift, and promotion decisions.
  Handle empty/insufficient windows explicitly; label synthetic drift demonstrations.
- Do not equate feature drift with performance degradation. Calculate performance
  only with suitable ground truth; never promote solely because drift was detected.

## 8. Make deployment reproducible with Compose

- Use three long-running services: MLflow, FastAPI, and Streamlit. Add a small one-shot
  bootstrap task using the shared training function to create a champion only if absent.
- Pin Docker images, persist MLflow metadata/artifacts, configure health checks and
  bounded startup retries, and bind published ports to localhost. Keep secrets out of images.
- Make `docker compose up` obtain pinned data and start the full system without manual
  service setup. Supply an accessible data source or documented offline bundle for a
  machine without the local DVC remote; do not assume an existing cache.
- Verify clean startup, MLflow HTTP artifact access, train/predict behavior, and restart
  persistence. Document manual consistent SQLite/artifact/remote backups; no backup service.

## 9. Verify and document the operational system

- Add tests alongside application behavior: data validation/leakage prevention,
  training/serialization, promotion rejection, endpoint contracts, logging failures,
  and stable/shifted monitoring data. Use real-service integration tests where needed.
- Enforce at least 95% coverage across `src/` and tests, zero Pyright errors in both,
  and the configured Flake8/formatting gates. Do not re-test standard tool behavior.
- Extend the same CI workflow with image builds and a Compose train/predict smoke test.
- Keep README to working setup/usage and material limitations; keep development rules
  in `AGENTS.md`. Record dataset provenance and a brief AI-assistance disclosure.
- Capture genuine MLflow, API, Docker, Streamlit/Evidently, and test/coverage screenshots
  under `screenshots/`. Provide the preprocessed dataset export with version identities.
