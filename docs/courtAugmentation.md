# Court landmarks: augmentation and point-based checkpoint selection

Earlier court runs trained without augmentation and selected the checkpoint by
validation box mAP50:95 of the derived landmark boxes. This document describes two
optional schema 2 profile fields and the YOLO26m ablation that uses them. Profiles
without these fields keep their earlier behavior and configuration identity.

## Two-run ablation

| Profile | Augmentation | Checkpoint selection |
| --- | --- | --- |
| `courtPointsMPointSelection.yaml` (A0) | none | macro native point success |
| `courtPointsMAugmented.yaml` (A1) | `courtSideSwapLight` | macro native point success |

Both profiles train YOLO26m from the official weights with seed 42, 960 px, FP32,
SGD, lr0 0.001, nominal batch 64, physical batch 4, at most 100 epochs and
patience 20. Both use the real `separatedV1` court view plus the verified
1,000-image synthetic subset that `yolo26mMixed1000Attempt01` used. The only
difference between A0 and A1 is augmentation. Neither run is directly comparable
with older court runs, which used a different selection rule, batch and code.

## Augmentation policy

```yaml
augmentation:
  policy: courtSideSwapLight
  exemptOrigins: [synthetic]
```

The policy allows only these Ultralytics keys to be nonzero, within the listed
bounds. Every other augmentation key, including mosaic, mixup, flipud, shear and
copy-paste, must remain zero.

| Key | A1 value | Upper bound |
| --- | ---: | ---: |
| `hsv_h` / `hsv_s` / `hsv_v` | 0.015 / 0.5 / 0.4 | 0.1 / 0.9 / 0.9 |
| `translate` / `scale` | 0.1 / 0.3 | 0.3 / 0.5 |
| `degrees` / `perspective` | 3.0 / 0.0001 | 10 / 0.001 |
| `fliplr` | 0.5 | 1.0 |

Policies are accepted only for independent `courtPoints` training. Perspective
is kept small because at 960 px a value of 0.0003 already changes the
homogeneous scale by about 14% at the image border.

Records whose origin is listed in `exemptOrigins` pass through the plain
pipeline. That pipeline is the same Ultralytics pipeline with every augmentation
key set to zero, so synthetic images are only resized and letterboxed, as before.
Each stage writes `augmentationAudit.json` with the policy, the effective values,
the flip class map and the number of exempt training records. The same audit is
stored in the stage metrics and stated in the report.

Training fails if Ultralytics' Albumentations step is active. That step applies
blur, CLAHE and grayscale at a small probability whenever the package is
installed. It is not in `uv.lock`, so it is only possible in a drifted environment.

### Horizontal flip with side swap

Court coordinates run in x from Side A to Side B and in y from the upper (far)
touchline to the lower (near) touchline. A horizontal image mirror maps x to
length − x, so mirrored landmark classes are relabeled together with the image:

| Before flip | After flip |
| --- | --- |
| any `…SideA` | the same landmark `…SideB` (and the reverse) |
| `substitutionPoint1` / `2` | `substitutionPoint4` / `3` (and the reverse) |
| center, halfway, upper and lower identities | unchanged |

After the relabeling, x still increases from left to right in the image, so the
homography orientation gate remains valid. Ultralytics' own flip swaps only pose
keypoints, never detection classes, so the project replaces the native horizontal
flip with a class-swapping flip. A unit test checks the map against the metric
court model.

## Point-based checkpoint selection

```yaml
checkpointSelection:
  metric: pointSuccessMacro
  confidence: 0.25
  tolerance: 0.01
```

After each epoch, the training validator keeps its predictions in original
pixels. For every class it takes the highest-confidence box with confidence of
at least 0.25 and compares its center with the native annotated point. The
selection score is the mean over supported classes of the share of points within
1% of the image diagonal. This is the "point 1% macro" value of the evaluation
report. The score selects `best.pt` and drives early stopping. `epochs.jsonl`
also records the box fitness, the micro point success and the number of frames
with at least four correct points. The archived evidence therefore shows which
epoch box AP would have chosen.

The selection score comes from the training validator. The sealed evaluation of
the best checkpoint remains the reported result. As before, the validation
matches both select the checkpoint and report the result. Reported validation
values are therefore optimistic; the reserved test set is not used.

## Training commands

Each run trains `courtPoints` from the official m weights on the real court view
and a verified 1,000-image synthetic directory. It is evaluated automatically:

```bash
python -m futsal train --datasetDir "$work/data/views/separatedV1"   --mediaStoreDir "$work/data/mediaStore"   --config configs/training/courtPointsMAugmented.yaml --modelSize m   --weights "$work/data/modelWeights/yolo26m/yolo26m.pt" --seed 42   --syntheticDir "$synthetic" --batchRecovery fixed   --outputDir "$runs/yolo26mAugmentedAttempt01"   --reportRoot "$work/reports/<campaign>" --workRoot "$work/scratch/evaluationWork"
```

Run A0 the same way with `courtPointsMPointSelection.yaml`. Use the same
synthetic directory, so the only difference is augmentation. With
`--batchRecovery fixed`, a memory failure at batch 4 stops the run instead of
silently lowering the batch. Retry both runs the same way with
`--memoryRetryFrom`.
