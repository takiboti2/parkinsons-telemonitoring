# Development instructions

## Scope and simplicity

- Make the smallest change that meets the request. Prefer standard configuration and existing commands over wrappers, frameworks, or speculative infrastructure.
- Ask before expanding scope or adding unnecessary dependencies, workflows, or abstractions. Keep this a single-host, low-concurrency system.
- Keep documentation separate: README for current setup/usage, this file for development rules, and `documents/implementation-plan.md` for remaining work. Do not repeat whole sections across them.
- Never claim checks passed or services ran without evidence. Document material limitations and keep credentials out of code, logs, and screenshots.

## Code and dependencies

- Use Python 3.13 and uv. Pin direct dependencies exactly, commit `uv.lock`, and export `requirements.txt` from it. Use frozen installs in development, CI, and Docker.
- Follow PEP8 with a 120-character line limit and type hints. Require zero Pyright errors in `src/` and `tests/` under the configuration in `pyproject.toml`.
- Keep business logic in `src/parkinsons_ml/` and Streamlit as a thin adapter. Start with simple modules; create packages/directories only as needed.
- Use SQLite and local files, not additional database/object-storage services. Keep application audit records separate from MLflow's internal tables.

## Data and model correctness

- Preserve `data/raw/parkinsons_updrs.csv`; write derived data to `data/processed/`. Track datasets with DVC and metadata/configuration with Git.
- Predict `motor_UPDRS`. Never use `total_UPDRS`, the target, or `subject#` as predictors; use the subject identifier only for grouping/auditing.
- Every fitted estimator must be scikit-learn's `LinearRegression`. Baseline features are `age`, `sex`, and `test_time`; the candidate adds all 16 voice features.
- Use identical patient-disjoint development/test assignments and grouped validation folds for comparisons. Keep the final test set out of tuning and promotion.
- Validate schema, invalid values, missing targets, and duplicates explicitly. Fit learned preprocessing only on training partitions, inside the serialized estimator pipeline.
- Track lineage, parameters, metrics, reports, model signature, and complete pipelines in MLflow. Save artifacts for both training and prediction; log requests/predictions without unnecessary identifiers.
- Evaluate candidates comparably, load and validate immutable versions before changing `champion`, and preserve the previous model on rejection. Guard promotion, roll back failed activation, and reconcile interrupted updates.
- Treat drift as a signal, not proof of performance degradation. Performance reporting/retraining requires suitable labels. The system is not a clinical diagnostic service.

## Tests and quality gates

- Test application behavior and concrete integration risks, not standard library functions, tool CLI startup, or test helpers for their own sake.
- Preserve pytest's default 95% coverage gate across `src/` and tests. Do not narrow coverage or add unjustified exclusions/suppressions. Keep fixture state isolated and restored on failure.
- Aim for zero Flake8 errors and require zero Pyright errors in source and tests. Use the existing hooks for fast checks; put full tests, coverage, image builds, and deployment smoke tests in CI.
- Add success/failure tests alongside implemented functionality. Do not add dummy services or notebooks just to produce tests or screenshots.

## Git and notebooks

- Preserve user changes. Do not stage, commit, publish, or discard work without authorization; follow the established commit convention.
- Use meaningful incremental commits and preserved feature branches/merges for feature work. Keep `main` passing; bootstrap configuration may be committed directly when explicitly approved.
- Track `.ipynb` files, ignore checkpoints, and invoke nbstripout directly from pre-commit. Use nbdime's repository-local diff/merge drivers, not global settings or custom scripts.
- Do not use notebook metadata/tags to retain outputs. Review source/Markdown/attachments for secrets. Keep notebook execution tools separate from Git tooling.
- Version the implementation plan; ignore only disposable reports under `documents/reports/`. Capture genuine running-system evidence when the services exist.
