# Notebooks

Track `.ipynb` files without outputs; checkpoints are ignored. nbstripout runs on
commit, and nbdime handles local Git diffs/merges after the
[one-time setup](../README.md#jupyter-notebooks-in-git).

Keep reusable logic in `src/`, data in DVC, and experiment artifacts in MLflow.
Use only `LinearRegression` for fitted models. Do not embed secrets or private data.
