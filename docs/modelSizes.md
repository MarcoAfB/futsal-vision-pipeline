# YOLO26 model sizes

Model size is a property of each training run, independent of its task. All
existing profiles accept `n`, `s`, `m`, `l` and `x`; their default remains `n`.
Choose it in YAML with `modelSize`, or override that value with `--modelSize`.

| Case | Profile | Size selection |
| --- | --- | --- |
| Unified curriculum | `yolo26Curriculum.yaml` | One size for all three stages |
| Detailed athletes and ball | `detailedObjects.yaml` | Independent per run |
| Generic player and ball | `genericObjects.yaml` | Independent per run |
| Generic player | `playerOnly.yaml` | Independent per run |
| Ball | `ballOnly.yaml` | Independent per run |
| Twenty court landmarks | `courtPoints.yaml` | Independent per real-only or supplemented run |
| Pilot / CPU smoke | `yolo26CurriculumPilot.yaml` / `yolo26Smoke.yaml` | One size for all declared stages |

The profiles live under `configs/training/`. Size means model scale, while `imgsz`
is image resolution. Batch, resolution, device and optimizer settings do not change
automatically with scale. Resource changes require an explicit training profile.

## Training

Reuse existing shared data and one official weights directory per size. For an
existing separated task view, a new detailed-object `s` run is:

```text
python -m futsal prepareWeights --modelSize s --outputDir data/modelWeights/yolo26s
python -m futsal train --datasetDir data/views/separatedV1 --mediaStoreDir data/mediaStore --config configs/training/detailedObjects.yaml --modelSize s --weights data/modelWeights/yolo26s/yolo26s.pt --outputDir outputs/training/objectsSSeed42
```

Skip preparation when verified official weights and `weights.json` already exist.
For player/ball tasks, use the corresponding profile with a view prepared from
`configs/sharedDetection.yaml`. Curriculum profiles need the curriculum dataset/view.
No dataset or image copies are needed for another size. Synthetic court training
continues to accept the optional existing `--syntheticDir` supplement.

The original profile remains unchanged in `inputs/training.yaml`. Effective size
and overrides are recorded in `run.json` and `inputs/resolvedTraining.json`; they
participate in configuration and method identities. Weights must match that size.
Continuation cannot switch scales between stages. The server launcher supports
the same override and retains its matching-pilot requirement.

## Evaluation and comparison

`evaluateRun` reads size from the archived training record. The existing
`evaluateConfiguration` accepts different object/court sizes and records each
component's initial and trained checkpoint hashes. Its other shared-protocol
checks remain unchanged: differing batch or code snapshots still prevent joint
evaluation. Training profiles for the other task cases do not add new joint
compositions in this delivery.

The two existing comparison commands offer:

| Mode | Behavior |
| --- | --- |
| `controlled` (default) | Requires equal model scales and official initializations |
| `modelVariants` | Also permits different scales and their corresponding official initializations |

```text
python -m futsal compareConfigurations --baselineEvaluationDir outputs/reports/curriculum --candidateEvaluationDir outputs/reports/objectsSCourtM --comparisonMode modelVariants --outputDir outputs/reports/modelVariantComparison
python -m futsal compareCourtTraining --baselineEvaluationDir outputs/reports/courtRealN --candidateEvaluationDir outputs/reports/courtMixedS --comparisonMode modelVariants --outputDir outputs/reports/courtVariantComparison
```

Both commands keep their existing comparison purposes and semantic/reference
checks. Variant mode only relaxes scale and associated weights; it does not permit
different seed, resolution or precision. Court real/mixed comparison also retains
identical real membership, code and training hyperparameters. Same-scale runs must
still agree on initialization in either mode.

Comparison evidence archives the selected mode, model sizes, checkpoint hashes
and changed factors. Reports distinguish complete variants from comparisons that
hold model size fixed. `reportRun` regenerates either mode without inference;
older comparisons without a mode field are treated as `controlled`.
