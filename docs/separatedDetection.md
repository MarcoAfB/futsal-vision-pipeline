# Independent object and court detectors

The separated detection recipe defines detailedObjects (ball and four athlete
classes) and courtPoints (twenty detection classes). Each model starts independently
from the official YOLO26n checkpoint. Court landmarks retain the existing 1/64 by
1/36 boxes and native-point references. Only registered real own data are selected.

Projection recomputes eligibility from native annotations, keeping the canonical
unified25 task as the unchanged frozen split coverage basis. A landmark-only issue
does not exclude a valid object example, and an object-only issue does not exclude
a valid landmark example. Expected real train/validation counts are 2553/541 and
2579/548. The unsupported lowerCornerSideB validation score remains N/A. Test image
files remain reserved; training/test coverage is checked from eligibility metadata.

## Incremental data preparation

Prepare labels and only missing reviewed development images locally:

```text
python -m futsal prepareProjection --sourceDatasetDir data/prepared/curriculumV1 --dataRoot ../Python --config configs/separatedDetection.yaml --outputDir data/projections/separatedV1
python scripts/projectionTransfer.py pack --projectionDir data/projections/separatedV1 --projectionId <projectionId> --archive data/transfers/separatedV1.tar.gz --receipt data/transfers/separatedV1.receipt.json
```

Transfer the incremental archive/receipt and the updated code to the server. The
projection is bound to its parent contract and native plan. It contains task labels,
development quarantine previews and the missing PNGs, without parent image copies.
The current real bundle requires47 supplemental records (38 training,9 validation).
Source video/annotation hashes are checked before and after extraction. No test
frame is retrieved or exported. Ownership of new staging directories is inherited.

On a target volume with enough room, unpack and materialize a new complete bundle:

```text
python scripts/projectionTransfer.py unpack --archive incoming/separatedV1.tar.gz --receipt incoming/separatedV1.receipt.json --outputDir data/projections/separatedV1
python -m futsal materializeProjection --sourceDatasetDir data/prepared/curriculumV1 --projectionDir data/projections/separatedV1 --outputDir data/prepared/separatedV1
```

Materialization checks disk space, copies image bytes from the verified parent or
supplement and runs complete native-label/image verification before publication.
The new bundle is self-contained and compatible with existing training/transport.
Original bundles and raw sources remain unchanged. Destinations must be new;
corruption, another parent, missing/extra files or insufficient space fail explicitly.

## Independent training

Use the existing train CLI twice, sequentially, with the same official YOLO26n
checkpoint and sidecar. `configs/training/detailedObjects.yaml` and
`configs/training/courtPoints.yaml` each contain exactly one training stage.
The initial protocol is seed42, image960, physical batch2/nbs64, FP32 SGD,
LR0.001, momentum0.937, weight decay0.0005, warmup3, linear final LR fraction0.01,
100 maximum epochs and patience20. Workers2, cache and all augmentations disabled.
There is no mandatory pilot or checkpoint transfer between these two executions.
Independent native CPU fixture training verifies5/20-class checkpoints, the common
initial SHA, fresh optimizer state and preserved native labels/loaders offline.

Each run archives its own input weights, source/resolved profile, seed, environment,
effective parameters, epochs, best/last checkpoints and task-specific validation.
Standalone evaluation/reporting preserves the existing full per-task cohorts and
native point N/A semantics. Single-stage reports identify independent training.

## Joint validation

`evaluateConfiguration --objectsRunDir <objects> --courtRunDir <court> --datasetDir
<projectedDataset> --referenceDatasetDir <curriculumDataset> --outputDir <newEvaluation>`
verifies both completed scientific single-stage runs, their official initialization,
code/configuration compatibility, full selections and parent dataset lineage.
`--config` selects the frozen research validation protocol; `--workRoot` controls
the isolated runtime copies. Requested GPU failure never falls back to CPU.

Both models see the same canonical validation images (539 real records common to
the three views). Outputs are mapped by class name, stably ordered by confidence
(objects then court for ties), capped globally at300 and matched by the native
validator in letterboxed input coordinates after composition. Original-pixel
predictions retain component provenance. Per-component AP values are not averaged
or concatenated to manufacture the system score. Native masks/order are archived.

The joint seal binds both run/checkpoint hashes and exact canonical references.
Its report focuses on group/class support, native-point success/error and a few
deterministic examples. Rendering/regeneration through reportRun needs no model
packages. Individual task reports still use all541/548 validation records. Runtime
and bundle hashes are checked after evaluation; partial failures remain explicit.

## Comparing sealed quality results

Import the completed scientific curriculum with resultTransfer before comparing:
its run inputs/artifacts, final checkpoint, selection and evaluation seals must be
verified. A user-reported completion is distinct from locally verified artifacts.

`compareConfigurations --baselineEvaluationDir <curriculumEvaluation>
--candidateEvaluationDir <jointEvaluation> --outputDir <newComparison>` compares
only completed scientific evaluations with identical reference contract, canonical
references/record order, protocol, seed, model scale, resolution and precision.
Wrong/missing input, unsupported support changes and corrupted seals are refused.
It reads sealed metrics/native masks without importing a model or running inference.

The report shows absolute candidate-minus-baseline differences for ball, athletes,
court detection and native-point success/error/missing counts. Detailed CSVs and a
few paired examples from the sealed galleries retain N/A support and exact inputs.
The two source evaluations are archived and sealed inside the comparison, allowing
standalone reportRun regeneration with reporting dependencies only.

Detector partition, public curriculum exposure and task-eligible training records
differ. This is a comparison of complete configurations, not an isolated curriculum
ablation or a statistically established winner. No FPS, latency, inference-memory
benchmark or final test is added. Single-seed results remain explicitly labelled.
