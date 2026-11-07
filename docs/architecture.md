# Package architecture

`src` is the source directory; `futsal` is the installable Python package. Hatch
includes `src/futsal` and its subpackages in the wheel. The package name remains
stable for `python -m futsal`, the installed `futsal` command and explicit imports.

Only `__init__.py`, `__main__.py` and `cli.py` remain at the package root.
`__init__.py` declares the version, `__main__.py` delegates module execution, and
`cli.py` parses commands and dispatches workflows. Domain initializers are empty;
importing a subpackage does not eagerly load its workflows or optional backends.

## Responsibilities

| Subpackage | Responsibility | Modules |
| --- | --- | --- |
| `annotations` | Read annotation formats, audit source geometry and convert detection targets. | `cvatReader`, `yoloAnnotations`, `annotationAudit`, `detectionGeometry` |
| `inventory` | Register sources, inspect media and audit reviewed own/public records. | `catalog`, `mediaMetadata`, `inventory`, `ownInventory`, `yoloInventory` |
| `datasets` | Apply recipes and quarantine, freeze match splits, extract eligible frames, export and verify derived datasets. | `recipes`, `matchSplits`, `datasetSamples`, `datasetPlan`, `datasetExport`, `datasetVerify`, `datasetFiles`, `frameExtraction`, `reviewImages` |
| `training` | Inspect the environment, prepare weights, select development data, audit transfer and orchestrate native training/continuation. | `trainingEnvironment`, `modelWeights`, `trainingConfig`, `trainingData`, `checkpointTransfer`, `trainingBackend`, `trainingContinuation`, `trainingRun` |
| `evaluation` | Validate selections and predictions, run archived checkpoints and calculate detection/native-point metrics. | `evaluationData`, `evaluationBackend`, `evaluationMetrics`, `pointMetrics`, `evaluationRun` |
| `reporting` | Render inventory/research reports, select examples, produce figures and coordinate research CLI processing. | `report`, `researchExamples`, `reportExamples`, `researchFigures`, `researchReport`, `researchCli` |

Keep additions in the subpackage that owns their behavior. Format readers belong
in `annotations`; dataset eligibility and export policy belong in `datasets`;
research measurement belongs in `evaluation`; presentation belongs in `reporting`.
`reviewImages` stays with dataset quarantine because it visualizes preparation
findings. Research example selection stays with reporting even though evaluation
archives the chosen source images as portable presentation evidence.

## Imports and dependencies

Use explicit module paths, for example:

```python
from futsal.annotations.cvatReader import readCvat
from futsal.datasets.datasetPlan import buildDatasetPlan
from futsal.training.trainingRun import runTraining
from futsal.evaluation.evaluationRun import evaluateRun
from futsal.reporting.researchReport import renderReport
```

These groups organize existing responsibilities rather than impose independent
layers. Cross-domain imports remain explicit. Shared file/evidence helpers stay
in `datasets.datasetFiles`, source-path/hash helpers in `inventory.catalog`, and
Markdown/atomic-write helpers in `reporting.report`. Those modules do not load a
model or plotting backend. Backend imports remain deferred until their operations
are requested; inventory and dataset commands do not require the training extra,
and evidence-only rendering does not require model packages.

`scripts` contains checkout-level operational tools. `datasetTransfer`,
`resultTransfer` and their shared `transferArchive` helper use only the standard
library, allowing verification/extraction before the model environment exists.
`runServerTraining` invokes the project CLI through its current prepared interpreter
and records launch evidence. These tools are not imported by the package or
included in its wheel; server execution requires a real Git checkout.

Independent detection adds dataset-owned projection planning/materialization,
evaluation-owned canonical two-model matching and sealed quality comparisons,
and reporting-owned joint/paired presentations. `projectionTransfer` transports
only the incremental data supplement; resultTransfer can close both component
runs and their joint/comparison seals. Package/backend imports remain deferred so
comparison and presentation regeneration do not require model packages.

The former flat internal imports, such as `futsal.trainingRun`, are replaced by
their domain paths. There are no forwarding modules or broad initializer exports.
The restructuring preserved CLI names, arguments, configuration, data formats and
research metric definitions. Later scientific schema/CLI additions are documented
in [training](training.md) and [server training](serverTraining.md).

## Tests and archived source

`tests` mirrors the six domain directories. Shared fixtures remain in the root
`tests/conftest.py`; general CLI tests remain in `tests/testCli.py`. The existing
test markers and default collection rules still apply. Run a domain explicitly
with, for example, `uv run pytest tests/datasets`.

Training and evaluation snapshots copy the complete source tree with relative
paths. Renderer snapshots likewise preserve every nested module and subpackage
initializer under `presentation/src/futsal`, retaining distinct files with the
same basename. Project-relative runtime paths are anchored to the top-level
package rather than the depth of the calling module.

Existing run/evaluation archives keep their original code snapshots and hashes.
Regeneration reads sealed evidence and writes a new presentation directory; it
does not migrate or rewrite historical archives. See [evaluation](evaluation.md)
for evidence contracts and [training](training.md) for execution provenance.

## Restructuring verification

On 2026-11-02, Ruff, all 164 regular tests and all five optional integration tests
passed with this structure. After the path/archive preparation change, the 39
existing module bodies remained equivalent when imports were excluded from the
comparison. Six empty package initializers were added.

The wheel contained all 45 package Python files. An isolated installation outside
the repository imported all 44 discovered submodules, exercised the help commands
for all nine CLI workflows and checked the installed entry point. That environment
used only locally copied base dependencies; model and plotting packages were absent.
No dependency download was needed.

An existing sealed real evaluation was regenerated through `reportRun` into
`outputs/verification/packageLayout/reportReplay`. Evidence contracts, source
presentation artifacts and training metadata retained their hashes; detection
tables remained byte-identical for all three stages. The new renderer snapshot
preserved all 45 source files with their subdirectories and hashes. Local check
summaries are `outputs/verification/packageLayout/installedPackage.json` and
`outputs/verification/packageLayout/reportCompatibility.json`. The compatibility
check used archived validation evidence; original data and reserved test data were
not changed or newly evaluated.
