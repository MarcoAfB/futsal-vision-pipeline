import math
from pathlib import Path

from futsal.inventory.catalog import readYaml
from futsal.training.modelWeights import modelSizes

trackerNames = ("bytetrack", "botsort", "ocsort", "deepocsort", "fasttrack", "tracktrack")


def loadTrackingConfig(path):
    config, _ = readYaml(Path(path))
    required = {"schemaVersion", "trackers", "imgsz", "batch", "half", "seed", "numThreads",
                "detectorConfidence", "operationalConfidence", "operationalIou", "maxDetections"}
    if not isinstance(config, dict) or set(config) != required or config["schemaVersion"] != 1:
        raise ValueError("Tracking configuration requires schemaVersion 1 and supported fields.")
    trackers = config["trackers"]
    if not isinstance(trackers, list) or not trackers or any(
        not isinstance(name, str) or name not in trackerNames for name in trackers
    ) or len(trackers) != len(set(trackers)):
        raise ValueError("trackers must be a nonempty unique list of supported tracker names.")
    for field in ("imgsz", "seed", "numThreads", "maxDetections"):
        if type(config[field]) is not int or config[field] < (0 if field == "seed" else 1):
            raise ValueError(f"{field} must be an integer in its supported range.")
    if config["imgsz"] % 32 or type(config["batch"]) is not int or config["batch"] != 1 or config["half"] is not False:
        raise ValueError("Tracking requires a multiple-of-32 image size, batch 1 and FP32.")
    for field in ("detectorConfidence", "operationalConfidence", "operationalIou"):
        if type(config[field]) not in {int, float} or not math.isfinite(config[field]) or not 0 < config[field] <= 1:
            raise ValueError(f"{field} must be finite and in (0, 1].")
    if config["detectorConfidence"] > config["operationalConfidence"]:
        raise ValueError("Detector confidence must not exceed operational confidence.")
    return config


def loadTrackingRuns(path):
    path = Path(path).resolve()
    data, _ = readYaml(path)
    if not isinstance(data, dict) or set(data) != {"schemaVersion", "runs"} or data["schemaVersion"] != 1:
        raise ValueError("Runs configuration requires schemaVersion 1 and runs.")
    runs = data["runs"]
    if not isinstance(runs, dict) or not runs or any(size not in modelSizes for size in runs):
        raise ValueError("runs must map YOLO26 sizes n/s/m/l/x to training directories.")
    if any(not isinstance(value, str) or not value.strip() for value in runs.values()):
        raise ValueError("Every run directory must be a nonempty path.")
    resolved = {size: (path.parent / runs[size]).resolve()
                for size in modelSizes if size in runs}
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("Each model size requires a distinct training run.")
    return resolved
