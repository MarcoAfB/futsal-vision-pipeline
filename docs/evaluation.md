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
