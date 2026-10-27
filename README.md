# Futsal

A reproducible foundation for futsal computer vision research, with source
registration, read-only auditing and curriculum dataset preparation. The first
experiment targets one detector for ball, four athlete classes and twenty court
landmarks. Training and sports statistics belong to subsequent stages.

## Development

Use Python 3.13 and uv. From the project root, on Windows or Linux:

```text
uv sync --locked
uv run python -m futsal --version
uv run ruff check .
uv run pytest
```

If pytest's system temporary directory is not writable, use a fresh project-local
directory: `uv run pytest --basetemp .pytest_cache/localTestRun`.

The environment is isolated from the previous implementation. The dependency lockfile
is part of the project. GPU dependencies are not installed in this stage; the UTFPR
server environment will be checked before training.

## Dataset inventory

From the project root, point `dataRoot` at the directory containing the original
registered sources. For the current sibling reference directory:

```text
uv run python -m futsal inventory --dataRoot ../Python --outputDir outputs/initialInventory
```

Use `--config <path>` for a different catalog; the default is
`configs/dataSources.yaml`, relative to the working directory. Sources are read in
place. The output directory must be outside `dataRoot` and cannot overwrite the
catalog or label map. Generated outputs stay outside Git.

The command writes `inventory.json` and `report.md`, including source provenance,
reviewed-frame records, per-match class coverage, media metadata, and findings.

| Exit code | Meaning |
|---|---|
| 0 | Audit completed with no error findings; warnings may exist. |
| 1 | Audit completed and reports were written; source findings require review. |
| 2 | The command could not complete because of an input or output failure. |

After exit code 2, existing outputs must not be treated as results of that run.
Outputs are replaced atomically per file, rather than as a single transaction.

Read [source registration](docs/dataSources.md), [annotation formats](docs/annotationFormats.md),
and the [initial audit](docs/initialAudit.md) for dataset-specific decisions and limitations.

## Curriculum datasets

Prepare the 2-to-3-to-25 class curriculum using the frozen original-match split:

```text
uv run python -m futsal planDataset --dataRoot ../Python --outputDir outputs/curriculumPlan
uv run python -m futsal prepareDataset --dataRoot ../Python --plan outputs/curriculumPlan/plan.json --outputDir data/prepared/curriculumV1
uv run python -m futsal verifyDataset --datasetDir data/prepared/curriculumV1
```

Planning and preparation require new output directories. Expected quarantine is
recorded per task. All 25 canonical classes have eligible training and test
annotations; the test images and labels remain reserved. Validation has no
lowerCornerSideB support. Each stage has a portable `dataset.yaml` with training
and validation only. Public annotations retain their original semantics and
enter only compatible public training stages.

The bundle includes source hashes, native observations, transformed labels,
quarantine previews and a verifiable output contract. Read
[dataset preparation](docs/dataPreparation.md) for source eligibility, split
proportions, geometry rules, commands and limitations. No GPU packages are needed.

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
