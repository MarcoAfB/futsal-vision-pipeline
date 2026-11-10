# Independent object and court detectors

For new runs over existing curriculum/projection media, prefer
[shared task views](sharedData.md). Register both providers and prepare a lightweight
view with this recipe; full projection materialization is unnecessary in that workflow.

For optional synthetic supplementation of the same court detector, see
[synthetic court training](syntheticCourt.md). Generation runs independently;
the profiles and real-only workflow below remain the defaults.

The separated detection recipe defines detailedObjects (ball and four athlete
classes) and courtPoints (twenty detection classes). Each model starts independently
from the official checkpoint of its selected YOLO26 size (default `n`). Court landmarks retain the existing 1/64 by
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
python -m futsal prepareProjection --sourceDatasetDir data/prepared/curriculumV1 --dataRoot data/raw --config configs/separatedDetection.yaml --outputDir data/projections/separatedV1
python scripts/projectionTransfer.py pack --projectionDir data/projections/separatedV1 --projectionId <projectionId> --archive data/transfers/separatedV1.tar.gz --receipt data/transfers/separatedV1.receipt.json
```

Transfer the incremental archive/receipt and the updated code to the server. The
projection is bound to its parent contract and native plan. It contains task labels,
development quarantine previews and the missing PNGs, without parent image copies.
The current real bundle requires 47 supplemental records (38 training, 9 validation).
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
These are the default examples. Each run can independently select `n/s/m/l/x`
through YAML or `--modelSize`, with matching `--weights`; see [model sizes](modelSizes.md).
The initial protocol is seed 42, image size 960, physical batch 2/nbs 64, FP32 SGD,
LR 0.001, momentum 0.937, weight decay 0.0005, warmup 3, linear final LR fraction 0.01,
100 maximum epochs and patience 20. Workers 2, cache and all augmentations disabled.
There is no mandatory pilot or checkpoint transfer between these two executions.
Independent native CPU fixture training verifies 5/20-class checkpoints, the common
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

Object and court sizes may differ, for example objects `s` with court `m`.
Each component must use its own corresponding official initialization and its
trained checkpoint must match the recorded scale. Components of the same scale
must share official initialization. The other shared training parameters, seed,
device, epoch budget and code requirements still apply, including equal batch.

Both models see the same canonical validation images (539 real records common to
the three views). Outputs are mapped by class name, stably ordered by confidence
(objects then court for ties), capped globally at 300 and matched by the native
validator in letterboxed input coordinates after composition. Original-pixel
predictions retain component provenance. Per-component AP values are not averaged
or concatenated to manufacture the system score. Native masks/order are archived.

The joint seal binds both run/checkpoint hashes and exact canonical references.
Its report focuses on group/class support, native-point success/error and a few
deterministic examples. Rendering/regeneration through reportRun needs no model
packages. Individual task reports still use all 541/548 validation records. Runtime
and bundle hashes are checked after evaluation; partial failures remain explicit.

## Comparing sealed quality results

Import the completed scientific curriculum with resultTransfer before comparing:
its run inputs/artifacts, final checkpoint, selection and evaluation seals must be
verified. A user-reported completion is distinct from locally verified artifacts.

`compareConfigurations --baselineEvaluationDir <curriculumEvaluation>
--candidateEvaluationDir <jointEvaluation> --outputDir <newComparison>` compares
only completed scientific evaluations with identical reference contract, canonical
references/record order, protocol, seed, model scale, resolution and precision.
This is the default `--comparisonMode controlled`. To compare different sizes,
explicitly use `--comparisonMode modelVariants`; only model size and its official
initialization may additionally differ. Both modes check every candidate component.
The report records each size/checkpoint and identifies changes in model capacity.
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

## Server execution

Receive the updated Git checkout/bundle and the incremental archive/receipt. Keep
the checkout clean. Run in the server-approved GPU job or a persistent tmux session;
tmux protects the session against SSH disconnects but does not extend job limits.
Use the installed Python 3.12 explicitly and prepare the locked trainingGpu/reporting
environment before launching. No mandatory pilot is introduced.

Follow the [server environment guide](serverTraining.md) for the locked GPU
installation and readiness checks. If the server permits tmux, start the session
inside the checkout with `tmux new -s futsalConfig2`, then run the commands below.
Detach with Ctrl+B followed by D and reconnect with `tmux attach -t futsalConfig2`.
Use the existing FUTSAL_WORK value; it names the data/run area and can be outside
the checkout. The checkout and this working area do not need duplicate datasets.

The following Bash example assumes the checkout is the current directory and
FUTSAL_WORK points to the existing server data/run area:

```bash
export FUTSAL_WORK="${FUTSAL_WORK:-$HOME/futsalWork}"
export FUTSAL_PYTHON="$(command -v python3.12)"
futsalRun() {
  uv run --locked --offline --no-sync --python "$FUTSAL_PYTHON" --extra trainingGpu --extra reporting "$@"
}

"$FUTSAL_PYTHON" scripts/projectionTransfer.py unpack \
  --archive "$FUTSAL_WORK/incoming/separatedV1.tar.gz" \
  --receipt "$FUTSAL_WORK/incoming/separatedV1.receipt.json" \
  --outputDir "$FUTSAL_WORK/data/projections/separatedV1"

futsalRun python -m futsal materializeProjection \
  --sourceDatasetDir "$FUTSAL_WORK/data/prepared/curriculumV1" \
  --projectionDir "$FUTSAL_WORK/data/projections/separatedV1" \
  --outputDir "$FUTSAL_WORK/data/prepared/separatedV1"

futsalRun python -m futsal train \
  --datasetDir "$FUTSAL_WORK/data/prepared/separatedV1" \
  --config configs/training/detailedObjects.yaml \
  --weights "$FUTSAL_WORK/data/modelWeights/yolo26n/yolo26n.pt" --seed 42 \
  --outputDir "$FUTSAL_WORK/runs/objectsSeed42" \
  --reportRoot "$FUTSAL_WORK/reports" --workRoot "$FUTSAL_WORK/scratch/evaluationWork"

futsalRun python -m futsal train \
  --datasetDir "$FUTSAL_WORK/data/prepared/separatedV1" \
  --config configs/training/courtPoints.yaml \
  --weights "$FUTSAL_WORK/data/modelWeights/yolo26n/yolo26n.pt" --seed 42 \
  --outputDir "$FUTSAL_WORK/runs/courtSeed42" \
  --reportRoot "$FUTSAL_WORK/reports" --workRoot "$FUTSAL_WORK/scratch/evaluationWork"

futsalRun python -m futsal evaluateConfiguration \
  --objectsRunDir "$FUTSAL_WORK/runs/objectsSeed42" --courtRunDir "$FUTSAL_WORK/runs/courtSeed42" \
  --datasetDir "$FUTSAL_WORK/data/prepared/separatedV1" \
  --referenceDatasetDir "$FUTSAL_WORK/data/prepared/curriculumV1" \
  --outputDir "$FUTSAL_WORK/reports/configuration2Seed42" --workRoot "$FUTSAL_WORK/scratch/evaluationWork"

futsalRun python -m futsal compareConfigurations \
  --baselineEvaluationDir "$FUTSAL_WORK/reports/seed42/<completedCurriculumEvaluation>" \
  --candidateEvaluationDir "$FUTSAL_WORK/reports/configuration2Seed42" \
  --outputDir "$FUTSAL_WORK/reports/comparisonSeed42"
```

Run each command only after the preceding one succeeds. Replace the baseline
evaluation placeholder with its exact sealed full scientific evaluation directory.
Each output/receipt must be new. Preserve failures and choose a new attempt name;
never restart the court detector from the object detector checkpoint.

The real projection materializes 6221 task images (12032844444 image bytes, plus
labels/metadata). The two complete training runtime copies require approximately
the same total image bytes again. Also allow validation workspaces, checkpoints,
reports, environment/cache and reserve space beyond existing curriculum artifacts.
The current local disk cannot hold this new full bundle with adequate reserve;
only the incremental package is prepared locally. No originals are deleted.

## Returning both models and reports

Use the extended standard-library results tool after joint evaluation. Repeated
--reportDir may add the sealed comparison and individual component reports:

```text
python scripts/resultTransfer.py pack --configurationDir <jointEvaluation> --reportDir <comparison> --archive outgoing/configuration2Seed42.tar.gz --receipt outgoing/configuration2Seed42.receipt.json
python scripts/resultTransfer.py unpack --archive incoming/configuration2Seed42.tar.gz --receipt incoming/configuration2Seed42.receipt.json --outputDir outputs/imported/configuration2Seed42
```

The import contains runs/detailedObjects, runs/courtPoints, configuration/ and
numbered reports/. Both run/checkpoint hashes, joint seal and attached comparison
links are verified. Runtime images, backend caches and evaluationWork are excluded;
their original evidence is retained without claiming receiver verification of
omitted bytes. Original server absolute paths and source archives are preserved.
The earlier --runDir transport remains compatible. Use short import roots on Windows.

## Prepared real supplement

The source curriculum bundle retained contract
`3d7e7fe9ba0e45ee2bc4a0ac3f3a435ccd74c01285a26fa4eac6e595b4b1f0e1`.
Projection ID:
`cf44e114eab73e06a2a7b156f163265b51962212c219414ac0a469379a84ca5d`.
The real 47 supplemental records were recovered from two source videos, preserving
their video/annotation evidence and the frozen split. The verified incremental
archive has 6323 files and 116866155 compressed bytes, SHA-256
`d855c1eab2c7af28651542970e0f543ad0259c60e16464bd86498a1445c35ffb`.
Local archive/receipt are data/transfers/separatedV1.tar.gz and
data/transfers/separatedV1.receipt.json, excluded from Git along with projection data.
Local software/native fixture success is distinct from the pending real two-model
training and scientific comparison. The completed server curriculum is user-reported
until its run/checkpoint/evaluation artifacts are received and verified locally.
