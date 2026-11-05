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
