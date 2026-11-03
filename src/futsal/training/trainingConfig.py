import math
from pathlib import Path

from futsal.datasets.recipes import activeTasks
from futsal.inventory.catalog import readYaml


def loadTrainingConfig(path, recipe):
    data, _ = readYaml(Path(path))
    required = {"schemaVersion", "purpose", "modelSize", "device", "seed", "numThreads",
                "trainSampleCount", "validationSampleCount", "stages", "parameters"}
    scientific = data.get("schemaVersion") == 2
    if scientific:
        required |= {"evaluationDevice", "sourcePolicy"}
    if set(data) != required or data["schemaVersion"] not in {1, 2}:
        raise ValueError("Training configuration requires supported schemaVersion 1 or 2 fields.")
    if not scientific and (data["purpose"] != "technicalSmoke" or data["modelSize"] != "n" or data["device"] != "cpu"):
        raise ValueError("This delivery supports technicalSmoke with YOLO26n on CPU.")
    if scientific:
        if data["purpose"] not in {"scientificTraining", "scientificPilot"} or data["modelSize"] != "n":
            raise ValueError("Schema 2 requires scientificTraining or scientificPilot with YOLO26n.")
        if data["sourcePolicy"] != "realOnly":
            raise ValueError("Scientific training and pilots require sourcePolicy realOnly.")
        if data["trainSampleCount"] is not None or data["validationSampleCount"] is not None:
            raise ValueError("Scientific profiles require full per-task data (sample counts null).")
        for field in ("device", "evaluationDevice"):
            if not isinstance(data[field], str) or not (data[field] == "cpu" or data[field].isdigit()):
                raise ValueError(f"{field} must be cpu or an explicit CUDA index string.")
    integerFields = ("seed", "numThreads") if scientific else (
        "seed", "numThreads", "trainSampleCount", "validationSampleCount")
    for field in integerFields:
        if type(data[field]) is not int or data[field] < (0 if field == "seed" else 1):
            raise ValueError(f"{field} must be an integer in its supported range.")
    stages = data["stages"]
    if not isinstance(stages, list) or not stages:
        raise ValueError("At least one training stage is required.")
    for stage in stages:
        if not isinstance(stage, dict) or not {"taskId", "epochs"} <= set(stage) or set(stage) - {"taskId", "epochs", "parameters"}:
            raise ValueError("Each stage requires taskId and epochs, with optional parameters.")
        if type(stage["epochs"]) is not int or stage["epochs"] < 1:
            raise ValueError("Stage epochs must be a positive integer.")
        if not scientific and (stage["epochs"] != 1 or "parameters" in stage):
            raise ValueError("The technical smoke requires one epoch per stage.")
        overrides = stage.get("parameters", {})
        if not isinstance(overrides, dict) or set(overrides) - {"lr0", "patience", "warmup_epochs"}:
            raise ValueError("Stage parameters may override only lr0, patience and warmup_epochs.")
        validateParameters(stageParameters(data, stage), scientific)
    stageIds = [stage["taskId"] for stage in stages]
    activeIds = [task["taskId"] for task in activeTasks(recipe)]
    if len(stageIds) != len(set(stageIds)) or stageIds != [i for i in activeIds if i in stageIds]:
        raise ValueError("Stages must be unique active tasks in curriculum order.")
    validateParameters(data["parameters"], scientific)
    return data


def stageParameters(config, stage):
    return {**config["parameters"], **stage.get("parameters", {})}


def validateParameters(parameters, scientific=False):
    fixed = {"workers": 0, "optimizer": "SGD", "warmup_epochs": 0.0, "patience": 0,
             "amp": False, "deterministic": True, "cache": False, "plots": False, "save": True}
    disabled = ("hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
                "perspective", "flipud", "fliplr", "mosaic", "mixup", "copy_paste", "cutmix",
                "multi_scale", "close_mosaic")
    numeric = {"imgsz", "batch", "lr0", "momentum", "weight_decay"}
    if scientific:
        for key in ("workers", "warmup_epochs", "patience"):
            del fixed[key]
        fixed["cos_lr"] = False
        numeric |= {"nbs", "lrf"}
    variable = {"workers", "warmup_epochs", "patience"} if scientific else set()
    if not isinstance(parameters, dict) or set(parameters) != set(fixed) | set(disabled) | numeric | variable:
        raise ValueError("Unexpected or missing training parameters.")
    if any(parameters[key] != value or type(parameters[key]) is not type(value)
           for key, value in fixed.items()):
        raise ValueError("Technical smoke requires deterministic FP32 training and isolated caches.")
    if any(type(parameters[key]) not in {int, float} or parameters[key] != 0 for key in disabled):
        raise ValueError("All data augmentations must be disabled for the technical smoke.")
    if any(type(parameters[key]) not in {int, float} or not math.isfinite(parameters[key])
           or parameters[key] <= 0 for key in numeric):
        raise ValueError("Training dimensions and optimizer values must be finite and positive.")
    if any(type(parameters[key]) is not int for key in ("imgsz", "batch")):
        raise ValueError("imgsz and batch must be integers.")
    if parameters["imgsz"] % 32 or parameters["momentum"] >= 1:
        raise ValueError("imgsz must be a multiple of 32 and momentum must be below one.")
    if scientific:
        if any(type(parameters[key]) is not int or parameters[key] < 0 for key in ("workers", "patience")):
            raise ValueError("workers and patience must be nonnegative integers.")
        warmup = parameters["warmup_epochs"]
        if type(warmup) not in {int, float} or not math.isfinite(warmup) or warmup < 0:
            raise ValueError("warmup_epochs must be finite and nonnegative.")
        if type(parameters["nbs"]) is not int or parameters["nbs"] < parameters["batch"] or parameters["lrf"] > 1:
            raise ValueError("nbs must be an integer at least batch; lrf must be at most one.")
