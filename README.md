# Futsal

A reproducible foundation for futsal computer vision research. This first delivery
focuses on source registration and read-only dataset auditing. Model training and
sports statistics belong to subsequent stages.

## Development

Use Python 3.13 and uv. From the project root, on Windows or Linux:

```text
uv sync --locked
uv run python -m futsal --version
uv run ruff check .
uv run pytest
```

The environment is isolated from the previous implementation. The dependency lockfile
is part of the project. GPU dependencies are not installed in this stage; the UTFPR
server environment will be checked before training.

## Conventions

Code, configuration, documentation, and commit messages use English. Owned names use
camelCase; classes use PascalCase. Tool-required filenames and external APIs retain
their required spelling. Original data filenames and labels are preserved.

Source code lives in `src/futsal`, configuration in `configs`, tests in `tests`, and
technical documentation in `docs`. Experimental variants will use configuration files
and a shared implementation. Commits describe coherent changes with their validation.

Datasets, model weights, environments, and generated results are excluded from Git.
The legacy source data must remain unchanged. Source licensing is independent of this
implementation and must be checked before redistribution.
