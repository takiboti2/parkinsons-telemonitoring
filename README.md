# Parkinson's Telemonitoring

## Minimal local architecture

The planned default deployment uses FastAPI, MLflow, Streamlit, and an idempotent
bootstrap task on one host. No PostgreSQL, MinIO, cloud credentials, or S3 plugin is
required. The API, dashboards, and Compose deployment are not implemented yet.

| State | Storage | Access |
| --- | --- | --- |
| MLflow metadata and registry | Dedicated SQLite file on a persistent local volume | MLflow service |
| Application audit records | Separate SQLite file on its own persistent local volume | Single API worker/replica |
| Models, batches, reports | Persistent local artifact volume | Served/proxied by MLflow over HTTP |
| Versioned datasets | Separate filesystem DVC remote | Documented local host directory/container mount |

Use Python's standard-library `sqlite3` for audit storage. Dashboard clients will use
a read-only API interface rather than sharing database files. MLflow clients will
not need the artifact directory mounted. The DVC remote must be populated from a
pinned source or offline bundle; an empty remote is not a dataset bootstrap source.

This configuration targets local, low-concurrency use. Keep SQLite databases on local
disks, use short transactions and bounded busy timeouts, and use WAL mode for audit
records where appropriate. Back up databases consistently using SQLite's backup API
or stopped writers, and preserve artifacts and DVC remote contents separately.
Reassess storage and locking before introducing multiple API workers or hosts.

## Pinned dependency environments

The application targets Python **3.13** (`>=3.13,<3.14`). All direct application and
development dependencies use exact versions in `pyproject.toml`; `uv.lock` also pins
transitive dependencies. `requirements.txt` is the generated export, including
development tools. Use `uv sync --frozen` to reproduce the environment.

Key application versions:

| Component | Version |
| --- | --- |
| MLflow | 3.18.0 |
| scikit-learn | 1.9.1 |
| NumPy / SciPy | 2.5.4 / 1.18.1 |
| pandas | 2.3.3 |
| FastAPI / Uvicorn | 0.162.7 / 0.54.0 |
| Pydantic | 2.14.3 |
| DVC, filesystem remote | 3.67.1 |
| Evidently | 0.7.23 |
| Streamlit | 1.67.0 |
| SQLAlchemy | 2.1.8 |
| SQLite driver | Python standard library |

Compatibility constraints:

- Keep pandas at 2.3.3: Evidently 0.7.23 fails to import with pandas 3.1.1 because
  it references removed pandas internals. Dependency resolution alone did not catch this.
- Base DVC supports the filesystem remote without cloud plugins. PostgreSQL drivers,
  boto3, and DVC's S3 plugin are not part of the default dependency set.
- DVC's `antlr4-python3-runtime` dependency builds from source. Its build was verified;
  a blanket wheels-only installation is not supported by this stack.
- MLflow 3.18.0 uses skops serialization. The compatibility test explicitly trusts
  `numpy.dtype` and `sklearn.tree._tree.Tree` for its locally trained random forest.
  Never automatically trust all types found in an external model artifact.

### Optional Airflow environment

Airflow **3.4.0** has an independent dependency set in
`docker/airflow/requirements.txt`. Do not install it into the application environment:
its upstream constraints pin different versions of shared packages. Orchestration
should communicate with the application through its API.

Airflow is not part of the minimal default deployment. If implemented, it will use
an explicit optional Compose profile, with an executor/database combination supported
by that Airflow release. Any additional services belong to that profile only.

The input file uses Airflow's official Python 3.13 constraints. Regenerate its fully
pinned Linux x86-64 requirements with:

```bash
uv pip compile docker/airflow/requirements.in --python-version 3.13 \
  --python-platform x86_64-manylinux_2_28 --only-binary :all: \
  --output-file docker/airflow/requirements.txt
```

For an isolated local installation:

```bash
uv venv --python 3.13 .venv-airflow
uv pip install --python .venv-airflow/bin/python -r docker/airflow/requirements.txt
uv pip check --python .venv-airflow/bin/python
```

### Compatibility validation

Run the application checks with:

```bash
uv sync --frozen
uv pip check --python .venv/bin/python
uv run --frozen pytest
```

Validated on CPython **3.13.9**, Linux x86-64:

- All 15 dependency smoke and isolation regression tests passed, with 100% coverage of the compatibility test module.
- scikit-learn preprocessing, fitting, and prediction; MLflow tracking, artifact
  logging, registration, alias resolution, and model reload passed.
- Evidently HTML reporting, FastAPI request validation, dependency imports, and
  DVC/Streamlit CLI startup passed.
- Separate SQLite audit persistence with WAL mode and filesystem DVC push/pull using
  a fresh cache passed without S3 or PostgreSQL packages.
- Both environments passed `uv pip check`; the isolated Airflow environment also
  passed SDK/provider imports and DAG construction.

These checks do not validate MLflow's HTTP artifact proxy, running dashboards,
an Airflow scheduler, or a Compose deployment. Dependencies emit upstream deprecation
warnings, including Evidently's use of NumPy internals, Litestar dependency APIs,
SQLAlchemy's `noload` strategy, and Starlette's HTTPX test-client support. The warnings
remain visible; rerun the tests when changing any pins.

Pytest discovery and coverage are configured in `pyproject.toml`. The default test
command measures both `src/` and test code and fails below 95%. The current coverage
result covers the compatibility tests only because application source is not yet
implemented. Do not narrow the quality gate with `--cov=tests` when adding source code.

MLflow tests use an isolated fixture that restores URI overrides, experiment
selection, environment variables (including inherited `MLFLOW_RUN_ID`), active runs,
and the previous last-active-run reference. Regression tests cover successful and
exceptional exits. They validate a registered
version before changing its champion alias; application promotion/activation rollback
will be implemented separately.

## Development checks

Use Python 3.13 and install [uv](https://docs.astral.sh/uv/getting-started/installation/).
From the repository root, install the locked development dependencies and Git hooks:

```bash
uv sync --frozen
uv run pre-commit install
```

Hooks run automatically on staged files during `git commit`. Run them manually across
all tracked files with:

```bash
uv run pre-commit run --all-files
```

The checks include:

- Black formatting and Flake8 linting, with a 120-character line limit.
- Strict Pyright checking of the entire `src/` tree when source Python files change.
- Trailing whitespace, final newlines, YAML/TOML/JSON syntax, and merge conflict markers.
- Files larger than 1 MiB, to help keep datasets and model artifacts out of Git.
- Private key detection. This is not a comprehensive secret scanner; never commit credentials.

Root-level Flake8 checks exclude both `.venv` and `.venv-airflow`.

All hooks use the development tools pinned in `uv.lock`. uv must be on `PATH`, including
when committing through an editor. Each clone needs its own `pre-commit install`.
After a hook fixes files, review and stage the changes before committing again.

The default `--all-files` command checks tracked files, not untracked files. Stage new
files before using it, or check specific files with `uv run pre-commit run --files PATH`.

Full tests, coverage enforcement, and Docker builds belong in CI rather than the
commit hook. CI should also run `uv run pre-commit run --all-files`, because local
hooks can be bypassed.

Regenerate the pinned dependency export after changing dependencies:

```bash
uv export --frozen --all-groups --no-hashes --no-emit-project --output-file requirements.txt
```

## Technical plan

The versioned [implementation plan](documents/implementation-plan.md) describes the
remaining work and acceptance criteria. `AGENTS.md` holds repository-wide development
instructions. Only disposable generated reports under `documents/reports/` are ignored.
