# Tracking on validation for any trained detector

`trackRun` applies the native Ultralytics trackers to the own validation frames of
any completed YOLO26 training run: ball-only, generic player, ball + player or a
detailed-class model. It reads the run's final stage, builds the matching contiguous
validation sequences and reports detection quality and exploratory ID stability for a
detector-only baseline and each tracker. It trains nothing and never opens test data.
The ball-specific [compareBallTrackers](ballTracking.md) remains available and unchanged
in behavior.

## Inputs

- One or more completed scientific training roots (`run.json`, `selection.json`, best
  checkpoint). All runs of one campaign must share the same final task, ontology and
  complete validation cohort, so each campaign needs one prepared view. A run is
  `PATH` or `LABEL=PATH`; the label defaults to the folder name and appears in condition
  IDs, so it must be unique and filename-safe.
- The shared view that contains the run's final task (for example the player or ball + player
  view used for training) and its media store. Court-only tasks are refused.
- `configs/evaluation/tracking.yaml`: 960 px, batch 1, FP32, seed 42, detector confidence
  0.10, operational confidence 0.25, IoU 0.50, up to 300 detections, six native trackers
  with ReID disabled and no tuning.

The environment is the one described in [ballTracking.md](ballTracking.md) (`tracking`
extra with pinned `lap`, and the `trainingGpu` or `training` extra).

## Run

```text
uv run --no-sync python -m futsal trackRun --runDir player=/existing/work/runs/playerMixed/yolo26mRealAttempt02 --datasetDir /existing/work/data/views/playerMixedV1 --mediaStoreDir /existing/work/data/mediaStore --device 0 --outputDir /existing/work/reports/trackingPlayerM01
```

Repeat `--runDir` to compare several models of the same task. `--resumeFrom` reuses only
fully identical, verified completed conditions of an earlier sealed campaign. Exit 0 means
every condition completed; 1 failed, interrupted or partial; 2 invalid input. Replay the
report without models or GPU:

```text
uv run --no-sync python -m futsal reportRun --evaluationDir /existing/work/reports/trackingPlayerM01 --outputDir outputs/trackingReplay
```

## Cohort and state

Frames come from the run's final task: all eligible own validation records of the view,
which must equal each run's saved validation selection. They are ordered by match, source,
reviewed interval and frame. A new sequence (and tracker state) starts at a change of
video, match, interval or non-adjacent frame number. Every eligible frame is processed,
including negatives and short blocks. Task-quarantined frames create block boundaries, so a
task with fewer quarantined frames (the player cohort has 550 frames) has fewer cuts than
the 541-frame ball cohort. Internal cuts inside a reviewed interval remain unverified.

## Metrics

Detection (per condition, per match and per class): frames, reference objects, TP, FP, FN,
precision, recall and F1 at operational confidence 0.25 and IoU 0.50 with one-to-one
confidence-ordered matching that requires the same class. With a single class and at most
one object per frame this equals the ball tracking scoring. These are fixed-threshold
counts, not AP, MOTA, IDF1 or HOTA. The detector-only condition uses identical inference
settings and supplies the baseline.

Stability is exploratory because annotation track IDs are not verified athlete
identities. Reference objects are chained across adjacent frames of one sequence by greedy
same-class IoU linking (same IoU as the operational threshold). A chain is a pseudo-identity,
and fragmentation is read from the tracker IDs on its matched detections:

| Field | Meaning |
| --- | --- |
| `trackCount`, `meanTrackLength`, `medianTrackLength`, `singleFrameTracks` | Tracker output above the operational confidence, IDs namespaced by sequence |
| `referenceChains`, `matchedChains` | Chains of at least two frames, and those with at least one matched frame |
| `chainsWithMultipleIds`, `meanIdsPerMatchedChain` | Chains whose matched detections carry more than one ID |
| `adjacentMatchedPairs`, `adjacentIdChanges` | Consecutive frames both matched, and ID changes between them |
| `gapRecoveries`, `gapIdChanges` | Matched again after a missed gap, and ID changes across it |

Linking can break at occlusions and crowded frames, so chain counts are neither a count of
players nor a certified ID-switch count. A low `adjacentIdChanges` does not prove stable
identity, and a high count can come from linking errors. Interpret differences between
trackers together with recall and the loss of matched frames. No automatic winner is chosen.

## Evidence and limits

The sealed folder contains the selection (with classes and task), code and dependency
snapshot, training records and selections, native tracker YAMLs, worker arguments and logs,
frame predictions and metrics. Checkpoint hashes identify weights; reports do not copy
weights or images. Limits: three validation matches, one seed per model, validation also
selected the checkpoints, trackers keep native parameters, and execution time includes
setup and is not a speed benchmark. Match11 is a deliberately difficult amateur-broadcast
match and should be inspected separately.

## Verification

```text
uv run --no-sync ruff check .
uv run --no-sync pytest tests/tracking tests/evaluation/testTrackingMetrics.py tests/evaluation/testTrackingRun.py tests/evaluation/testBallTrackingRun.py -o addopts=
```

The tests use synthetic detections and fixture checkpoints; they establish software
behavior, not tracking quality. A real run requires the server GPU and a checkpoint, and must
not be described as executed without that evidence.
