# Ball tracking comparison

`compareBallTrackers` compares completed standalone ball-only YOLO26 runs with
native trackers on the complete eligible own validation cohort. With all five
model sizes and the default six trackers, it evaluates 35 conditions, including
one detector-only baseline per size. It does not train models or open test data.

## Environment and inputs

Reuse the existing checkout, workspace, shared view and media providers.
Synchronize the environment with the appropriate backend:

```text
uv sync --locked --extra trainingGpu --extra tracking --extra reporting
```

For CPU development, replace `trainingGpu` with `training`. These extras are
mutually exclusive. The `tracking` extra pins `lap==0.5.12`; an ad-hoc installation
is not needed. Execution disables auto-installation and downloads. Preflight
checks the pinned backend, device, a real LAP assignment, native tracker configs,
checkpoint ontologies and integrity. A failed global preflight produces sealed
diagnostic evidence without starting any condition. Requested GPUs never fall
back to CPU.

Create an ignored `localSettings.ballTracking.yaml`, using actual completed
training roots rather than report directories or failed attempts:

```yaml
schemaVersion: 1
runs:
  n: /existing/work/runs/ballCampaign/yolo26nRealAttempt01
  s: /existing/work/runs/ballCampaign/yolo26sRealAttempt01
  m: /existing/work/runs/ballMStandalone/yolo26mRealAttempt01
  l: /existing/work/runs/ballCampaign/yolo26lRealAttempt01
  x: /existing/work/runs/ballXStandalone/yolo26xRealAttempt01
```

Paths may be absolute or relative to that YAML. Runs must have completed
scientific `ballOnly` training, real own data, the configured seed, an unchanged
selection and best checkpoint, and the identical complete validation cohort.
Training records must obey the current frozen own-match split. Checkpoints must
retain the declared YOLO26 scale, one ball class and end-to-end head. A subset of
sizes/trackers can be configured for explicit technical checks; such a run is not
a complete 35-condition comparison.

## Run and resume

From the checkout, supply existing inputs and a new output directory:

```text
uv run --no-sync python -m futsal compareBallTrackers --runsConfig localSettings.ballTracking.yaml --datasetDir /existing/work/data/views/allTasksV1 --mediaStoreDir /existing/work/data/mediaStore --device 0 --outputDir /existing/work/reports/ballTrackingValidation01
```

The default device is `cpu`; CUDA requires an explicit index. `--config` defaults
to `configs/evaluation/ballTracking.yaml`: 960px, batch 1, FP32, seed 42, detector
confidence 0.10, operational confidence 0.25, matching IoU 0.50 and max 300 detections.
Native tracker YAMLs are archived with hashes, ReID disabled and no parameter
tuning. Trackers are ByteTrack, BoT-SORT, OC-SORT, Deep OC-SORT, FastTracker and
TrackTrack. Detector-only baselines use the same inference settings. Each
condition runs in a separate child process, retaining native tracking callbacks
including TrackTrack's predictor integration.

The current view selects 541 validation records: 200 from match01, 164 from match11
and 177 from match12, including 527 positives and 14 negatives. Nine task-quarantined
records yield eight contiguous blocks. These counts describe the current view,
not hard-coded selection limits. State resets on video, original match, reviewed
interval or frame discontinuity. All eligible frames, including short blocks and
negatives, are processed. Images reuse existing media bindings without copies.
No interpolation or additional smoothing is applied.

To retry, use a new output directory and add `--resumeFrom` pointing to the old
comparison. Its seal is verified. Only complete conditions with identical code
content, data, protocol, model inputs, environment/device and tracker YAML hashes
are reused. Failed/interrupted conditions restart from their first frame in the
new output. Original comparisons remain unchanged. Identity changes are refused;
omit `--resumeFrom` to deliberately start an independent campaign. An execution
killed before sealing cannot be resumed through this interface.

Exit 0 means all conditions completed; exit 1 means failed, interrupted or partial
execution; exit 2 indicates invalid command/input before campaign creation.
Individual failures retain partial JSONL/logs without complete metrics. Integrity
failures stop the campaign and invalidate numerical results. Temporary settings
and requests live separately in `.ballTrackingWork` beside the report roots.

## Outputs and interpretation

`report.md` links to overall/per-match CSV results, frame outcomes, loss runs,
sequence boundaries and exclusions. Sealed evidence includes selection,
code/dependency snapshot, training metadata/selections, native tracker YAMLs,
worker arguments/logs, predictions and metrics. Checkpoint hashes identify the
original weights; reports duplicate neither weights nor source images.

TP/FP/FN use the existing confidence-ordered one-to-one matcher at the operational
thresholds, with at most one reference ball per frame. Global precision, recall
and F1 derive from summed counts. Undefined denominators are N/A. Extra boxes
count as false positives. Tracker output without IDs is not confirmed tracking;
detector-only IDs remain null.

Loss runs contain consecutive false negatives on ball-positive frames within one
block. Negatives, gaps and boundaries end a run. Recovery requires a hit before
and after the missed run; initial acquisition is not recovery. Adjacent-match ID
changes compare only consecutive true-positive frames in one block. IDs are
namespaced by segment; detector-only ID diagnostics are N/A. Internal cuts are
unverified, so temporal diagnostics remain exploratory, not MOTA, IDF1 or HOTA.
The report computes neither detection AP nor an automatic winner.

Three validation matches and one training seed limit inference. Requested
training batches are recorded; effective historical batches are not inferred
from directory names. Execution duration includes setup/integrity checks and is
not a speed/memory benchmark.

Regenerate presentation without models, original images or GPU access:

```text
uv run --no-sync python -m futsal reportRun --evaluationDir /existing/work/reports/ballTrackingValidation01 --outputDir outputs/ballTrackingReplay
```

The reporting extra is sufficient. Replay verifies and copies evidence into a
new directory while retaining the original evaluation seal.

## Verification

```text
uv run --no-sync ruff check .
uv run --no-sync pytest
uv run --no-sync pytest tests/tracking tests/evaluation/testBallTrackingRun.py -o addopts=
```

Native tests use isolated synthetic images/detections and locally constructed
test checkpoints with real association, persistence, resets and prediction
callbacks. They establish software behavior, not scientific quality. GPU/server
campaign execution requires separate observed execution evidence.
