import importlib.metadata
import os
import platform
import shutil
import subprocess
from pathlib import Path

from futsal.datasetFiles import stagedDirectory, writeJson


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
              "freeDiskBytes": shutil.disk_usage(Path.cwd()).free,
              "packages": packageVersions(), "requestedDevice": device, "ready": False,
              "cudaAvailable": False, "gpus": [], "findings": []}
    gpuTool = shutil.which("nvidia-smi")
    if gpuTool:
        result = subprocess.run([gpuTool, "--query-gpu=name,memory.total,driver_version",
                                 "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
        report["nvidiaSmi"] = {"exitCode": result.returncode, "output": result.stdout.strip()}
    if not report["packages"]["torch"]:
        report["findings"].append("Install the optional training extra: uv sync --locked --extra training.")
        return report
    try:
        import torch

        report["cudaBuild"] = torch.version.cuda
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
    with stagedDirectory(outputDir, Path(__file__).resolve().parent) as staging:
        writeJson(staging / "environment.json", report)
        lines = ["# Training environment", "", f"Ready: {report['ready']}",
                 f"Device: {report['requestedDevice']}", f"Python: {report['pythonVersion']}", ""]
        lines += [f"- {name}: {value or 'not installed'}" for name, value in report["packages"].items()]
        lines += ["", *[f"- {finding}" for finding in report["findings"]], ""]
        (staging / "report.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
