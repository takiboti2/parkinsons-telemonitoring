# Parkinson's Telemonitoring

Predict `motor_UPDRS` with scikit-learn's `LinearRegression`. The baseline uses
`age`, `sex`, and `test_time`; the candidate adds 16 voice features. Exclude
`total_UPDRS` and use `subject#` only for patient-disjoint evaluation.

**Status:** dependencies, hooks, notebook Git tools, DVC preparation, and reusable
model-preprocessing pipelines are implemented. Training orchestration, application
services, CI, and Docker deployment remain. This is not a clinical diagnostic service.

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

`parkinsons_ml.training.build_pipeline("baseline" | "candidate")` creates an
unfitted pipeline with feature selection, time clipping, median continuous-feature
imputation, most-frequent sex imputation, and `LinearRegression`. Fit it only on
training partitions. The model rejects an entirely missing fitting feature.
Preparation does not contact MLflow; future training will log its reports and
manifest along with the complete model pipeline.

To upload generated datasets (writes to the configured data remote):

```bash
uv run --frozen dvc push
```

Review `dvc.lock` alongside code and parameters before committing. The data remote
and any populated cache must be backed up separately.

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

## Project documents

- [Implementation plan](documents/implementation-plan.md): remaining work and acceptance criteria.
- [AGENTS.md](AGENTS.md): development rules.

AI assistance has been used for planning, configuration, dependency compatibility
checks, and test implementation; review generated changes before committing.
