import math

from futsal.catalog import readYaml
from futsal.recipes import activeTasks


def loadTrainingConfig(path, recipe):
    data, _ = readYaml(path)
    required = {"schemaVersion", "purpose", "modelSize", "device", "seed", "numThreads",
                "trainSampleCount", "validationSampleCount", "stages", "parameters"}
    if set(data) != required or data["schemaVersion"] != 1:
        raise ValueError("Training configuration requires the supported schemaVersion 1 fields.")
    if data["purpose"] != "technicalSmoke" or data["modelSize"] != "n" or data["device"] != "cpu":
        raise ValueError("This delivery supports technicalSmoke with YOLO26n on CPU.")
    for field in ("seed", "numThreads", "trainSampleCount", "validationSampleCount"):
        if type(data[field]) is not int or data[field] < (0 if field == "seed" else 1):
            raise ValueError(f"{field} must be an integer in its supported range.")
    stages = data["stages"]
    if not isinstance(stages, list) or not stages:
        raise ValueError("At least one training stage is required.")
    for stage in stages:
        if not isinstance(stage, dict) or set(stage) != {"taskId", "epochs"}:
            raise ValueError("Each stage requires taskId and epochs.")
        if stage["epochs"] != 1 or type(stage["epochs"]) is not int:
            raise ValueError("The technical smoke requires one epoch per stage.")
    stageIds = [stage["taskId"] for stage in stages]
    activeIds = [task["taskId"] for task in activeTasks(recipe)]
    if len(stageIds) != len(set(stageIds)) or stageIds != [i for i in activeIds if i in stageIds]:
        raise ValueError("Stages must be unique active tasks in curriculum order.")
    validateParameters(data["parameters"])
    return data


def validateParameters(parameters):
    fixed = {"workers": 0, "optimizer": "SGD", "warmup_epochs": 0.0, "patience": 0,
             "amp": False, "deterministic": True, "cache": False, "plots": False, "save": True}
    disabled = ("hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
                "perspective", "flipud", "fliplr", "mosaic", "mixup", "copy_paste", "cutmix",
                "multi_scale", "close_mosaic")
    numeric = {"imgsz", "batch", "lr0", "momentum", "weight_decay"}
    if not isinstance(parameters, dict) or set(parameters) != set(fixed) | set(disabled) | numeric:
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
