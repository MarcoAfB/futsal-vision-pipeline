# Validation research reports

Research evaluation is separate from training. Reports identify their checkpoint,
dataset contract, exact record IDs, protocol and purpose. The technical smoke is
not a scientific model comparison. Test images and labels remain reserved.

`configs/evaluation/researchValidation.yaml` freezes confidence floor 0.001,
operational confidence 0.25, operational IoU 0.50, AP IoU thresholds 0.50:0.95,
and point tolerances 0.25%, 0.5%, 1%, 2% and 5% of the original image diagonal.
The primary point tolerance is 1%. These defaults are diagnostic thresholds, not
automatically calibrated or model-quality acceptance thresholds.

`runValidation` uses exactly the saved training validation selection, verified
against the prepared manifest. `fullValidation` uses all eligible own validation
records for each task. The two selections produce separate evaluations. They do
not change task quarantine or the original-match split.

References retain original image dimensions, derived pixel boxes, source annotation
indices and native point coordinates. A prediction JSONL contains one complete
record per expected image, including empty detection lists. Missing, duplicate,
unexpected, incorrectly sized or incorrectly classified records fail validation.
Zero-area predicted boxes remain false-positive candidates rather than disappearing.

## Metric definitions

Operational matching is greedy in descending confidence, class-agnostic to expose
class confusions, with one reference per prediction and IoU >= 0.50. Confidence
>= 0.25 is included. A wrong-class association is both an FP for the predicted
class and an FN for the reference class. The confusion matrix has true classes on
rows, predicted classes on columns, and an explicit background row/column.
AP matching is class-aware at each IoU threshold and uses the pinned native
`ap_per_class` integration. Backend best-F1 precision/recall is not substituted
for the fixed operational scores. Group averages are macro averages over supported
classes, with both declared and supported class counts. Matches retain raw
prediction indices, including when low-confidence predictions precede them.

Landmark localization selects the highest-confidence prediction of each class,
never the closest prediction to the reference. Missing points are unsuccessful
in the tolerance success rate. Median and p90 error are conditional on an available
prediction and are accompanied by the localized count. Errors use native points
and the original image diagonal. Extra predictions, unsupported classes and the
native-to-artificial-box center offset are retained separately. No correction of
that representation offset is applied. Per-match values are descriptive; frames
are not treated as independent statistical replicates.

## Archived evaluation

The local validation adapter loads each archived best checkpoint as FP32 on CPU,
with the training image size, batch configuration and explicit YOLO26 end-to-end
mode. It uses native validation through a read-only trainer context, without an
optimizer or the generic dataset downloader. Native matching masks and image order
are captured alongside original-pixel predictions. Retaining native order also
preserves AP semantics when scores tie. Final and per-match AP reuse those masks;
rescaling/clipping boxes for display cannot change the recorded native AP matches.

The evaluation binds the training run, checkpoint hashes, exact selection, protocol,
code snapshot, native references, predictions, metrics, and small example source
archive. A sealed evaluation contract protects those inputs. Working image/label
copies and caches live in short `outputs/evaluationWork` directories, outside the
report bundle and archived training run. The short workspace avoids Windows path
limits for long SHA-256 image filenames. Failure traces and partial results remain;
later stages do not execute after a failed stage. Bundle and runtime hashes are
checked afterward, including after a backend error.

## Figures and visual examples

The optional `reporting` extra provides the already-locked Matplotlib version
without installing model packages. Figures use the Agg backend and export SVG and
PNG with stable SVG identifiers and no embedded generation date. Tiny positive AP
values use an explicitly labelled zoomed axis; unsupported classes remain N/A.
Single-epoch histories and uninformative PR/confusion plots are omitted. Point
success plots retain the annotated and missing counts, including when all rates
are zero. Class, match and point plots are separate within each task.

Qualitative panels place native references and operational predictions side by
side. Point references are crosses; predictions are box centers with displacement
segments. Stable class colours and numeric class IDs reference the class table.
An annotated ball receives a separate enlarged detail strip below the full image,
so the crop does not hide the court. Up to three seed-42 shared cases and three
explicitly ranked difficult cases are retained per stage. Archived source examples
are deduplicated by image hash; the full validation image gallery is not exported.

## Markdown report bundles

Each evaluation has one `report.md`, equally prominent task chapters, task-specific
`tables`, `figures` and `examples`, plus sealed `evidence`. The report includes
original-match counts, actual training sources, support in images and instances,
task-specific quarantine and frozen own split proportions, transfer/optimizer
audits, operational confusions, native-point errors and traceable examples.
Unsupported values display as N/A and remain empty cells in CSV rather than zeros.
Conditional localization error is always accompanied by the detected/missing counts.

Presentation files are separate from the sealed evaluation inputs. Regeneration
verifies the contract, copies the portable evidence into a new directory and uses
only archived numbers and selected source images. It never reruns inference.
`reportState.json` records independent presentation status, rendering package
versions, source-code snapshots and artifact hashes. Failed or unevaluated training
runs can receive execution-only reports without invented research scores. The
catalogue groups presentations by training run and creation time, without rankings;
failed presentations remain discoverable through their state file.

## Commands and lifecycle

From the project root, install the CPU evaluation and lightweight presentation extras:

```text
uv sync --locked --extra training --extra reporting
uv run --offline --extra training python -m futsal evaluateRun --runDir outputs/training/yolo26SmokeSeed42 --datasetDir data/prepared/curriculumV1 --selection runValidation --outputDir outputs/reports/yolo26SmokeSeed42/runValidationReview
```

That command evaluates all completed stages and renders the report. An explicit
`--selection fullValidation` evaluates the task-specific complete own validation
sets in a separate new output directory. Full validation is never run implicitly.
Neither command accepts a test selection. Existing destinations are refused.

```text
uv run --offline --extra reporting python -m futsal reportRun --evaluationDir outputs/reports/yolo26SmokeSeed42/runValidationReview --outputDir outputs/reports/yolo26SmokeSeed42/presentationReplay
uv run --offline --extra reporting python -m futsal reportRun --runDir outputs/training/interruptedRun --outputDir outputs/reports/interruptedRun/executionReview
```

The first command regenerates from sealed evidence without a model backend. The
second creates an execution-only report, including stages that were never run.
Open `outputs/reports/index.md` for the catalogue, then a presentation's `report.md`.
CSV tables and vector figures are linked from each equally prominent stage chapter.

New `train` CLI runs automatically evaluate their recorded validation subset and
publish a report. `--evaluationConfig` selects a different explicit evaluation
protocol. Training, evaluation and presentation retain separate statuses. If
research processing fails, the training archive stays unchanged and a failure
record remains discoverable in the catalogue. Successful complete commands return
0, retained partial/failed/interrupted processing returns 1, and research input or
rendering errors return 2. A presentation of a failed training run can itself
complete successfully; its report still identifies the failed training status.

Optional checks are explicit and stay offline:

```text
uv run --offline --extra training pytest -m trainingIntegration
uv run --offline --extra reporting pytest -m reportingIntegration
```

## Verified local delivery

On 2026-10-29 (local time), archived-checkpoint evaluation completed on the real
CPU technical smoke run, without retraining. Each of the three stages evaluated
the same 16 own validation records: six from match01, five from match11 and five
from match12. All 24 supported unified classes were retained; lowerCornerSideB
has no validation references and remains N/A. Each derived macro AP50-95 exactly
matched its recorded native validator value. Six qualitative examples per stage
include the same three fixed records across the curriculum.

The sealed evaluation is in
`outputs/reports/yolo26SmokeSeed42/runValidation-20261030T003633987272Z`.
Training used clean commit `d565501`; evaluation used clean commit `8dffbd0`.
The dataset contract, complete bundle membership/hashes, validation runtime copies
and archived training inputs retained their integrity. No test images were used.
These one-epoch results establish execution and report compatibility, not a
scientific performance baseline.

`outputs/reports/yolo26SmokeSeed42/presentationReplay/report.md` is an updated
presentation of that same sealed evidence. A second independent regeneration in
`outputs/verification/reportingOnly` succeeded with torch, torchvision and
ultralytics physically absent from the environment. The locked training and
reporting environment was restored afterward. All three bundles retained identical
evaluation contracts and evidence; artifact hashes, Markdown file links, PNG
decoding and SVG parsing passed. Class/support, point-localization and qualitative
panels were also inspected visually. A summary is retained locally in
`outputs/verification/researchDelivery.json`.

Ruff passed and the regular suite passed 160 tests. Three optional native tests
passed, including synthetic run/full validation, offline execution and native AP
parity. Two optional presentation tests passed again in the reporting-only
environment. Full real validation was not requested or run; final test evaluation,
threshold calibration, scientific training and UTFPR GPU compatibility remain
subsequent work.
