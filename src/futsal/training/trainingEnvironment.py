import importlib.metadata
import os
import platform
import shutil
import subprocess
from pathlib import Path

import futsal
from futsal.datasets.datasetFiles import stagedDirectory, writeJson


def packageVersions():
    versions = {}
    for name in ("torch", "torchvision", "ultralytics-opencv-headless", "opencv-python-headless",
                 "opencv-python", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def inspectEnvironment(device="cpu"):
    if device != "cpu" and not device.isdecimal():
        raise ValueError("device must be cpu or a nonnegative CUDA device index.")
    report = {"schemaVersion": 1, "pythonVersion": platform.python_version(),
              "platform": platform.platform(), "cpuCount": os.cpu_count(),
              "libc": list(platform.libc_ver()),
              "freeDiskBytes": shutil.disk_usage(Path.cwd()).free,
              "packages": packageVersions(), "requestedDevice": device, "ready": False,
              "cudaAvailable": False, "gpus": [], "findings": []}
    gpuTool = shutil.which("nvidia-smi")
    if gpuTool:
        try:
            result = subprocess.run([gpuTool, "--query-gpu=name,memory.total,driver_version",
                                     "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
            report["nvidiaSmi"] = {"exitCode": result.returncode, "output": result.stdout.strip(),
                                   "error": result.stderr.strip()}
            if device != "cpu" and result.returncode:
                report["findings"].append("NVIDIA driver inspection failed; regularize the driver before GPU training.")
        except (OSError, subprocess.TimeoutExpired) as error:
            report["nvidiaSmi"] = {"error": str(error)}
            if device != "cpu":
                report["findings"].append("NVIDIA driver inspection could not complete.")
    elif device != "cpu":
        report["findings"].append("nvidia-smi is required to verify the requested NVIDIA driver.")
    kernelVersion = Path("/proc/driver/nvidia/version")
    if platform.system() == "Linux" and kernelVersion.is_file():
        report["nvidiaKernelModule"] = kernelVersion.read_text(encoding="utf-8")
    if not report["packages"]["torch"]:
        report["findings"].append("Install the optional training extra: uv sync --locked --extra training.")
        return report
    try:
        import torch

        report["cudaBuild"] = torch.version.cuda
        report["cudnnVersion"] = torch.backends.cudnn.version()
        report["cudaAvailable"] = torch.cuda.is_available()
        report["gpus"] = [{"index": index, "name": torch.cuda.get_device_name(index),
                           "memoryBytes": torch.cuda.get_device_properties(index).total_memory}
                          for index in range(torch.cuda.device_count())]
        if device != "cpu" and (not report["cudaAvailable"] or int(device) >= len(report["gpus"])):
            report["findings"].append("The requested CUDA device is unavailable; no CPU fallback was used.")
            return report
        target = "cpu" if device == "cpu" else f"cuda:{device}"
        value = torch.ones((2, 2), device=target) @ torch.ones((2, 2), device=target)
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError("The device probe returned nonfinite values.")
        report["probe"] = {"device": str(value.device), "finite": True}
        if device != "cpu":
            report["computeCapability"] = list(torch.cuda.get_device_capability(int(device)))
            free, total = torch.cuda.mem_get_info(int(device))
            report["gpuMemory"] = {"freeBytes": free, "totalBytes": total}
        if not report["packages"]["torchvision"] or not report["packages"]["ultralytics-opencv-headless"]:
            report["findings"].append("The torchvision and headless Ultralytics training packages are required.")
        if report["packages"]["opencv-python"]:
            report["findings"].append("Both GUI and headless OpenCV providers must not be installed together.")
        try:
            import psutil

            report["memoryBytes"] = psutil.virtual_memory().total
        except ImportError:
            report["memoryBytes"] = None
        report["ready"] = not report["findings"]
    except (ImportError, OSError, RuntimeError) as error:
        report["findings"].append(f"{type(error).__name__}: {error}")
    return report


def writeEnvironmentReport(report, outputDir):
    with stagedDirectory(outputDir, Path(futsal.__file__).resolve().parent) as staging:
        writeJson(staging / "environment.json", report)
        lines = ["# Training environment", "", f"Ready: {report['ready']}",
                 f"Device: {report['requestedDevice']}", f"Python: {report['pythonVersion']}", ""]
        lines += [f"- {name}: {value or 'not installed'}" for name, value in report["packages"].items()]
        lines += ["", *[f"- {finding}" for finding in report["findings"]], ""]
        (staging / "report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
