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
# Development subsets

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
