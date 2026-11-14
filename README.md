# Parkinson's Telemonitoring

Predict `motor_UPDRS` with scikit-learn's `LinearRegression`. The baseline uses
`age`, `sex`, and `test_time`; the candidate adds 16 voice features. Exclude
`total_UPDRS` and use `subject#` only for patient-disjoint evaluation.

**Status:** dependencies, hooks, notebook Git tools, and DVC raw-data tracking are
configured. Data processing, application services, CI, and Docker deployment remain
to be implemented. This is not a clinical diagnostic service.

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
source types, strip notebook outputs, and check file hygiene. Stage new files before
running `--all-files`; review and restage automatic fixes.

## Project documents

- [Implementation plan](documents/implementation-plan.md): remaining work and acceptance criteria.
- [AGENTS.md](AGENTS.md): development rules.

AI assistance has been used for planning, configuration, dependency compatibility
checks, and test implementation; review generated changes before committing.
