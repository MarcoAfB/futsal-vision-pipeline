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
