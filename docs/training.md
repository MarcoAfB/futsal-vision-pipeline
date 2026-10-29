# YOLO26 training

The training extra pins the technical CPU reference environment: PyTorch 2.9.1,
torchvision 0.24.1 and ultralytics-opencv-headless 8.4.190 on Python 3.13.
PyTorch wheels come from the official CPU index. The default installation does
not require training packages; dataset commands import no training backend.

```text
uv sync --locked --extra training
uv run --extra training python -m futsal inspectEnvironment --device cpu --outputDir outputs/environment/localCpu
```

Environment inspection records installed versions, CPU/disk information, available
CUDA devices and a small tensor operation on the explicitly requested device.
Missing dependencies or an unavailable GPU produce a report with ready=false and
exit code 1. Input/output failures return 2. No device fallback is performed.
The report can later be collected on UTFPR; its CUDA compatibility is not assumed.

Ordinary tests stay offline and exclude the trainingIntegration marker. Native
training tests are explicitly selected with that marker and the training extra.

## Official initial weights

```text
uv run python -m futsal prepareWeights --modelSize n --outputDir data/modelWeights/yolo26n
```

This explicitly downloads the requested detection checkpoint from the official
ultralytics/assets v8.4.0 release. A sidecar records origin, size and observed
SHA-256. Existing destinations are refused and failed downloads are not published.
Training requires the checkpoint and its matching sidecar and performs no implicit
weight download. The SHA-256 binds the observed file; it is not an external signature.

## Development subsets

`configs/training/yolo26Smoke.yaml` declares a CPU-only, FP32, one-epoch technical
smoke run with 32 training records per stage and 16 shared validation records.
Selection first covers available classes greedily, breaking ties and filling slots
by SHA-256 of the seed, selection namespace, and record ID. Validation candidates
are the intersection of eligible records across the requested stages. Every
available validation class is retained; `lowerCornerSideB` remains unsupported.
All declared training classes must be supported. Infeasible subset sizes fail.

The run records full selected provenance and per-class support in `selection.json`.
Selected images and labels are byte copies in `runtimeData`, with absolute runtime
dataset roots. Runtime hashes detect backend repairs or changes. Source partitions,
task quarantine, and the immutable prepared bundle are preserved. Backend label
caches belong only to the run directory. No test images or labels are selected.

Transfer uses the pinned backend's native class-name remapping. Target names are
assigned before loading. A structured pre-batch audit compares compatible backbone
and hidden-head tensors and hashes every shared classification row in both `cv3`
and `one2one_cv3`. Only exact normalized names match: `ball` survives both project
transitions; `player` does not become a role or team by index. COCO `sports ball`
is not aliased to `ball`. New class weights must not equal an unrelated source row.
Each stage starts a new optimizer from the previous stage's `best.pt`, without resume.

The native adapter supplies the verified runtime-only detection configuration
directly, avoiding the backend's generic dataset downloader and font download.
Only native base callbacks and local audit callbacks execute; external platform
and analytics callbacks are excluded even if a platform API key is configured.
Final validation reloads `best.pt` as FP32 and reuses the verified runtime loaders
through the trainer context, avoiding the generic standalone dataset loader.
Explicit `nms=false` preserves YOLO26 end-to-end inference during validation and
in both saved checkpoints; the head class count and mode are checked after saving.
It checks exact image membership and label rows in both native dataloaders before
the first batch. Silent duplicate removal, corrupt-image exclusion or JPEG repair
fails the stage. CPU execution uses two threads and nominal batch size two so each
batch can update the optimizer. Actual optimizer groups and effective backend
arguments are recorded. Batch losses, model tensors, and saved checkpoints must
remain finite. Both checkpoint ontologies are checked after every stage.

The optional native test trains all three stages on a tiny synthetic bundle with
no downloaded weights. Run it explicitly:

```text
uv run --offline --extra training pytest -m trainingIntegration --basetemp .pytest_cache/nativeRun
```

## Running and inspecting the technical smoke

```text
uv run --offline --extra training python -m futsal train --datasetDir data/prepared/curriculumV1 --weights data/modelWeights/yolo26n/yolo26n.pt --config configs/training/yolo26Smoke.yaml --outputDir outputs/training/yolo26SmokeSeed42
```

Every run requires a new directory. `run.json` is updated atomically from `running`
to `completed`, `failed`, or `interrupted`. Configuration, source-code snapshot,
Git commit/dirty state, lockfile, initial checkpoint, environment and input hashes
are retained alongside selected record IDs and the prepared contract ID. The full
bundle is independently verified before training; all its file hashes and exact
membership are checked again afterward. Runtime images, labels and configuration
are hashed before and after native use. Logs, batch losses, partial checkpoints,
and failure details remain available after an error. Subsequent stages are stopped;
there is no automatic resume or overwrite. Use a fresh process for each CLI run.

Stage evidence is in `stages/stage01GenericObjects`, `stage02RoleObjects`, and
`stage03Unified25`. Each contains `transferAudit.json`, `loaderAudit.json`,
`effectiveArguments.json`, `optimizer.json`, `losses.jsonl`, native `results.csv`,
`metrics.json`, and `weights/best.pt` and `last.pt`. Native messages are retained
under `logs`. The next stage receives the prior `best.pt` hash, selected by the
pinned backend's validation mAP50–95 fitness. Checkpoint tensors serialize as FP16
by backend convention even though forward/backward computation uses FP32.

Per-class metrics with zero validation support are JSON null, including
`lowerCornerSideB`. Supported classes without detections have measured zero scores.
These tiny development subsets establish execution and transfer compatibility;
they provide no model-quality estimate or scientific comparison. No AP threshold
defines success. Final test records remain reserved and are never evaluated.

Exit code 0 means completed, 1 means a retained failed/interrupted run, and 2 means
the command could not start because an input or output boundary was invalid.

## Local validation status

On 2026-10-28 the Windows CPU environment inspection succeeded with Python 3.13.5,
PyTorch 2.9.1+cpu, torchvision 0.24.1+cpu and the pinned headless backend.
Ruff passed and the ordinary suite passed all 145 tests. The optional native suite
passed both integration tests, including a complete 2 -> 3 -> 25 run on synthetic
data with one training image per stage and two shared validation images. It checked
fresh optimizers, both checkpoint heads, saved class counts, end-to-end inference,
unsupported metrics, runtime preservation and the synthetic bundle's final hashes.
Requests and urllib network entry points were blocked in that native run, and an
injected external platform callback was prevented from executing.

The official pretrained checkpoint downloaded successfully and loaded as YOLO26n
with 80 source classes, 2,572,280 parameters and 12 classification tensors across
the two branches. Its 5,544,453 bytes have observed SHA-256
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.
An additional load-only probe using the actual project class names verified 696
compatible tensors per transition and all 12 `ball` rows in each project-to-project
transition. Probe evidence is in
`outputs/environment/officialCanonicalTransferProbe/summary.json`. This probe did
not read dataset samples or train a model.

The planned real-data 32/16 smoke remains pending. The command above was attempted,
but Windows denied reading `data/prepared/curriculumV1/plan.json`, including an
elevated attempt. It stopped before creating a training run or consuming samples.
Restore read access to the existing verified bundle, or provide an accessible copy
with the same contract, then rerun the command with a new output directory. Real
subset selection, training metrics and post-run bundle integrity are not yet
verified by this delivery. Final test evaluation and UTFPR GPU compatibility remain
outside this local technical validation.
