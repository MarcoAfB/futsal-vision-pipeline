# Court landmarks: box versus Pose representation

Court landmarks are normally trained as twenty detection classes. Each class has a
proxy box of 1/64 x 1/36 of the image around the native point. This document
describes the optional `representation: courtPose` profile field. It trains one
`court` instance with the twenty landmarks as YOLO26 Pose keypoints. Profiles
without the field keep the box representation and their configuration identity.

The study compares the two representations on the court component only. It does
not change the configurations 2 to 4. Whether Pose becomes their court
representation is decided after this study.

## Controlled comparison

| Profile | Representation | Everything else |
| --- | --- | --- |
| `courtPointsMAugmented.yaml` (baseline, `yolo26mAugmentedAttempt01`) | twenty proxy-box classes | as below |
| `courtPoseMAugmented.yaml` | one `court` instance, twenty keypoints | identical |

The settings shared by both profiles are:

- **Data:** the `separatedV1` court view and contract, plus the verified
  1,000-image synthetic subset.
- **Augmentation:** `courtSideSwapLight`, with synthetic images exempt.
- **Checkpoint selection:** macro native point success at confidence 0.25 and
  tolerance 0.01.
- **Training:** YOLO26m initialized from the official detection weights
  `yolo26m.pt`, seed 42, 960 px, FP32, physical batch 4.

The profile files differ only in the `representation` line.
`courtPosePilot.yaml` is the same Pose profile as a one-epoch `scientificPilot`.
It is an execution check, not a result.

## Pose labels

- **Keypoint order.** Keypoints follow `landmarkNames` in `geometry/courtModel.py`,
  and `kpt_shape` is `[20, 3]`.
- **Visibility.** An annotated landmark gets visibility 2 at its native point.
  Real frames read that point from `plan.json`, never from the centre of a
  truncated box. Synthetic frames use the exact centre of their boxes, which are
  never truncated; a truncated synthetic target is refused. Landmarks that are not
  annotated are `(0, 0, 0)`.
- **Instance box.** The `court` instance box is the full frame `[0.5, 0.5, 1, 1]`,
  so the box-area normalization of the keypoint loss is constant across frames.
  This matches tolerances expressed as fractions of the image diagonal. A frame
  with no annotated landmark has no instance.
- **Runtime files.** Labels are written to
  `runtimeData/courtPoints/poseLabels.json` and recorded in `runtimeEvidence.json`.
  The loader audit checks every frame's keypoints and instance against them.
- **Horizontal flip.** `flip_idx` is `mirroredClassIndices(landmarkNames)`. It is
  the same Side A/B and substitution-point swap as the box representation's
  class-swapping flip. Here Ultralytics' native flip applies it, and the
  augmentation audit records `horizontalFlipKeypointMap`.
- **Transfer.** The architecture is `yolo26m-pose.yaml` with one class. Backbone,
  neck and box-head tensors transfer from the detection weights through the
  audited loader. The keypoint and keypoint-sigma heads start new.
- **Loss settings.** Pose loss gains and OKS sigmas keep the Ultralytics defaults
  (uniform sigma 1/20). They are not profile keys and are recorded in
  `effectiveArguments.json`.

Pose training and evaluation require a shared task view.

## Predictions

The evaluation and the point-based checkpoint selection convert the most
confident `court` instance of each frame into native landmark detections:

- class `k` is `landmarkNames[k]`;
- confidence is the instance confidence times the keypoint visibility;
- the box has the reference proxy size and is centred exactly on the keypoint,
  shrunk symmetrically at the image border, so its centre is the keypoint;
- keypoints outside the image and confidences below the protocol floor are not
  detections.

The prediction archive therefore has the same format as for box runs. Point
success, F1 and the confusion matrix, AP, homography and the side vote all read
it unchanged. AP matches use the converted boxes against the reference proxy
boxes in original-image pixels; the stage records
`apMatchingSpace: originalImagePixels`. Box AP measures only centre offset here,
because every box has the reference size. The raw best instance of each frame is
sealed in `evidence/courtPoints/poseKeypoints.jsonl`.

## Decision rule

The rule is fixed before results.

**Primary metrics:**

- macro native point success at confidence 0.25 and 1% of the diagonal;
- the share of verified-reference frames that receive a `verified` predicted
  homography, using the frozen `homography.yaml` and the same athlete archive as
  the baseline homography evaluation.

**Secondary metrics**, each reported per original match:

- frames with at least four correct points;
- the false-valid rate;
- athletes within 1 m;
- median and p90 point error, which detects a loss of precision.

A single seed is descriptive. If both primary metrics improve by at least five
points, with no worse p90 error or false-valid rate, a second seed is run before
Pose is proposed for configurations 2 to 4. Otherwise the result is recorded as
inconclusive or negative. `compareCourtTraining` compares real-only against
real-plus-synthetic runs and does not apply. The comparison uses both sealed
prediction archives and both homography evaluations on the same validation
frames. The reserved test set is not used.

## Commands

```bash
futsal train --datasetDir data/views/separatedV1 --mediaStoreDir data/mediaStore \
  --config configs/training/courtPoseMAugmented.yaml --modelSize m \
  --weights data/modelWeights/yolo26m/yolo26m.pt --seed 42 \
  --syntheticDir <verified 1,000-image subset> --batchRecovery fixed \
  --outputDir <new run> --reportRoot <reports> --workRoot <scratch>
futsal evaluateHomography --courtEvaluationDir <Pose run validation> \
  --playerEvaluationDir <same athlete evaluation as the baseline> \
  --mediaStoreDir data/mediaStore --outputDir <new homography report>
```

## Not covered

- Joint configuration evaluation (`evaluateConfiguration`) still predicts court
  components as detection models.
- Tracking and team assignment do not read Pose output directly, but the side
  vote consumes the converted candidates.
- Heatmap keypoint models, tuned OKS sigmas and loss gains are not part of the
  study.
