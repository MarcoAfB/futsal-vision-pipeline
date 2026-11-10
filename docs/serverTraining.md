# First scientific curriculum on a server

For an existing server workspace and completed curriculum, use the incremental
[shared data workflow](sharedData.md). It registers existing inputs without image
copies and supports independent task training. The initial setup below is for
first-time provisioning and the legacy self-contained bundle workflow.

This workflow prepares one YOLO26n detector through genericObjects (2 classes),
roleObjects (3), then unified25 (25). Only the final unified25 best checkpoint is
the resulting detector. Stage scores describe different ontologies; they do not
rank configurations or establish that curriculum learning improves accuracy.
Scientific training and the resource pilot use registered real data only.

YOLO26n is the default profile. All sizes `n/s/m/l/x` are supported through
`modelSize` in YAML or `--modelSize` in both training and the server launcher.
Use the matching official weights and a matching resource pilot; see
[model sizes](modelSizes.md). Reuse existing work/data paths for subsequent runs.

## Files and storage

Keep a real, clean Git checkout for code snapshots. Transfer code independently
through the repository remote, or create an offline bundle from committed main:

```text
git bundle create ../futsalCode.bundle main
git clone /incoming/futsalCode.bundle futsal
git -C futsal rev-parse HEAD
```

Retain the expected full commit hash at the sender. Follow [data transport](dataTransfer.md)
for dataset archive/receipt verification, safe extraction and full semantic
verification after receipt. Transfer official `yolo26n.pt` with its `weights.json`
sidecar separately. Existing trained smoke checkpoints are not initial weights.
Data, weights, machine settings, receipts and generated runs stay outside Git.

A suggested layout separates immutable input, per-attempt byte copies and reports:

```text
futsal/                         clean code checkout and locked environment
data/prepared/curriculumV1/      immutable task images, labels and native provenance
data/modelWeights/yolo26n/       official checkpoint and provenance
incoming/                       archive and external receipt
runs/pilot42/                   pilot runtime copies, inputs, stages and logs
runs/seed42/                    complete scientific attempt
reports/pilot42/<evaluation>/   sealed evidence, tables, selected figures/examples
reports/seed42/<evaluation>/
scratch/evaluationWork/         disposable isolated validation copies and caches
launches/<attempt>/             command, environment, stdout/stderr and exit record
outgoing/                       selective results packages
```

All paths below are examples passed as arguments. Use storage/quota approved for
the server. The current bundle occupies 8,857,066,554 bytes; one full curriculum
runtime copy needs 8,765,999,096 image/label bytes. Budget the archive, extracted
bundle, runtime bytes for every retained attempt, environment/cache,
checkpoints/reports and headroom. Rough allocations of 60 GB for one retained complete run or 100 GB for
three are planning estimates; measure actual available space with `df -h` and the
archive receipt. Pilots and continuations retained beside runs add copies. Scripts
do not remove originals, old attempts or transfer archives to make space.

## Environment and driver

Use the actual installed Python 3.12 interpreter explicitly; `.python-version`
describes local development and must not select a different server interpreter.
Python 3.12 and 3.13 are supported. Prepare the environment before launching:

```bash
uv sync --locked --python /usr/bin/python3.12 --extra trainingGpu --extra reporting
uv run --locked --no-sync --python /usr/bin/python3.12 --extra trainingGpu --extra reporting python -m futsal inspectEnvironment --device 0 --outputDir outputs/environment/serverCheck
```

The GPU extra pins torch 2.9.1+cu126, torchvision 0.24.1+cu126 and headless
Ultralytics 8.4.190. CPU training uses the mutually exclusive `training` extra.
No packages are installed during launch/training. A shared driver/NVML mismatch
must be corrected by the server administrator; these scripts do not alter drivers
or reboot the machine. Proceed only after `nvidia-smi`, CUDA tensor execution and
the requested-device inspection succeed. Record GPU memory, OS/glibc, quota,
scheduler and job wall-time limit before allocating the scientific run.

## Pilot, profile freeze and complete runs

The committed pilot uses all real task data, image size 960, physical batch 2,
nominal batch 64, FP32 SGD and two epochs per stage. Pilot warmup/early stopping
are disabled to exercise two complete epochs. The scientific profile proposes
maximum epochs 80/60/100, LR 0.001/0.0005/0.0002, warmup 3 and patience 15/15/20.
All augmentations are disabled. The nominal batch controls native accumulation;
actual optimizer groups/accumulation, LR history, epoch counts, best epoch and
peak CUDA memory are saved. These starting values are not measured optima.

Launch within the server-approved job/session mechanism. The launcher uses its
current interpreter, requires the expected clean commit, verifies official weights,
source policy, disk space and devices, then invokes the audited CLI offline:

```bash
uv run --locked --offline --no-sync --python /usr/bin/python3.12 --extra trainingGpu --extra reporting python scripts/runServerTraining.py \
  --expectedCommit <fullCommitHash> --datasetDir /work/data/prepared/curriculumV1 \
  --config configs/training/yolo26CurriculumPilot.yaml --weights /work/data/modelWeights/yolo26n/yolo26n.pt \
  --outputDir /work/runs/pilot42 --reportRoot /work/reports --workRoot /work/scratch/evaluationWork \
  --logDir /work/launches/pilot42
```

Inspect `launch.json`, stdout/stderr, run.json, all stage audits and the report.
Check peak memory, finite losses, exact loader membership, both detection heads,
best/last checkpoints and full validation. Fit the estimated complete stage within
the allocated job limit, including warmup, validation and integrity overhead. A
session manager does not bypass scheduler limits. If a stage cannot fit, native
optimizer/RNG resume or an explicitly revised protocol is required first.

Freeze the resource-feasible profile before science. Any batch, image size,
precision, optimizer, augmentation or loader-resource change requires a new pilot.
If the profile changes, commit it and repeat the pilot on the final clean commit.
The scientific launcher checks a completed pilot with matching code, data, official
weights, full task cohorts, devices and effective parameters except the deliberately
different epoch/patience/warmup settings. It does not choose architecture from
pilot scores. Then launch the scientific profile with a new attempt directory:

```bash
uv run --locked --offline --no-sync --python /usr/bin/python3.12 --extra trainingGpu --extra reporting python scripts/runServerTraining.py \
  --expectedCommit <sameFullCommitHash> --datasetDir /work/data/prepared/curriculumV1 \
  --config configs/training/yolo26Curriculum.yaml --weights /work/data/modelWeights/yolo26n/yolo26n.pt \
  --pilotRun /work/runs/pilot42 --seed 42 --outputDir /work/runs/seed42 \
  --reportRoot /work/reports --workRoot /work/scratch/evaluationWork --logDir /work/launches/seed42
```

If budget permits, repeat the entire curriculum with explicit seeds 43 and 44 in
new directories and the same frozen profile/seed list. A single seed remains a
single-seed result. The source and resolved configs, overrides, method/config IDs,
code/lock hashes, environment and input checkpoint hashes are archived per attempt.
No architecture substitution or CPU/OOM fallback occurs automatically.

On interruption, preserve the attempt and add `--continueFrom /work/runs/seed42`
with new output/log directories and the identical profile/seed/code. Only a verified
completed stage prefix is copied; an interrupted stage restarts at epoch zero
from its original input with a fresh optimizer. No within-stage resume is claimed.
See [training contracts](training.md) for exact conditions.

## Validation, interpretation and delivery status

Full train/validation image counts are 3398/541, 1989/541 and 2547/539. Task-specific
quarantine is preserved; the common validation intersection (539) supports shared
examples and does not restrict metrics. runValidation replays those saved full
cohorts, so automatic postprocessing evaluates once. The evaluation device is
explicit. Box AP, operational detection and native-point localization remain
separate diagnostics, with fixed thresholds and unsupported classes N/A.

The frozen own split uses the original 3,684 reviewed records: 2585/550/549,
approximately 70.17/14.93/14.90%. After unified25 quarantine, 2547/539/540 remain
(70.24/14.86/14.89% of 3,626 eligible records). All 25 classes retain eligible train
and test support. lowerCornerSideB is absent from validation and remains N/A.
Public training sources are outside that own-data denominator. Test images/labels
remain reserved; final test evaluation and later multi-model variants are separate.

Use [selective results transport](dataTransfer.md#returning-results) to return the
attempt and sealed reports. Preserve the server original. Recreate presentations
locally without inference; reconstruct new evaluation runtime from the same bundle
when needed. Scientific training success and postprocessing failures remain separate.

Maintain a local campaign table with evidence links for codeReady, dataPacked,
dataReceived, environmentReady, pilotCompleted, seedRunCompleted,
validationCompleted, reportCompleted and resultsReceived. Local software/native
fixture checks establish engineering readiness, not server CUDA compatibility or
a completed scientific campaign. No actual server execution is asserted by this guide.

## Local implementation verification

On 2026-11-03, Ruff and 199 regular tests passed. The complete regular suite also
passed on Python 3.12.3, including 201 tests when the two presentation tests were
enabled in a reporting-only environment with torch, torchvision and Ultralytics
physically absent. Both Python 3.12.3 and 3.13.5 passed all six optional native/
presentation tests, including one-epoch and three-epoch 2 -> 3 -> 25 curricula,
native AP parity, exact loader/checkpoint checks and report regeneration.
Native training tests used isolated generated fixtures and fixture weights only.

The real prepared bundle passed full semantic verification of 9,555 development
images/labels. Its 19,199 files were streamed into a verified archive, retaining
the original dataset contract. Existing real smoke results also passed selective
pack/import/report regeneration: 270 included files and 29 byte-identical CSV
tables, with unchanged source run metadata. No real scientific training or server
GPU execution was performed. Driver correction, target receipt/extraction,
environment verification, resource pilot and campaign remain pending there.
