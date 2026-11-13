import importlib.metadata
import math
from pathlib import Path

from futsal.inventory.catalog import readYaml
from futsal.training.trainingBackend import initializeBackend
from futsal.training.trainingEnvironment import inspectEnvironment


def preflightTracking(config, device, runtimeDir):
    initializeBackend(runtimeDir, config["numThreads"])
    environment = inspectEnvironment(device)
    if not environment["ready"]:
        raise ValueError(f"Tracking environment is not ready: {environment['findings']}")
    try:
        import lap
        import numpy as np
        import ultralytics
        from ultralytics.trackers.track import TRACKER_MAP
    except ImportError as error:
        raise ValueError("Tracking dependencies are missing; synchronize tracking and training/trainingGpu extras.") from error
    if importlib.metadata.version("lap") != "0.5.12":
        raise ValueError("Tracking requires pinned lap 0.5.12.")
    _, assignment, _ = lap.lapjv(np.array([[0.0, 1.0], [1.0, 0.0]]))
    if assignment.tolist() != [0, 1]:
        raise ValueError("lap assignment probe failed.")
    environment["packages"]["lap"] = importlib.metadata.version("lap")
    configs = {}
    for name in config["trackers"]:
        if name not in TRACKER_MAP:
            raise ValueError(f"Tracker is unavailable in the pinned backend: {name}.")
        path = Path(ultralytics.__file__).parent / "cfg" / "trackers" / f"{name}.yaml"
        values, _ = readYaml(path)
        if values["tracker_type"] != name or values.get("with_reid", False):
            raise ValueError("Tracking requires native configs with ReID disabled.")
        if config["detectorConfidence"] > values["track_low_thresh"]:
            raise ValueError("Detector confidence discards low-confidence tracker candidates.")
        configs[name] = path.read_text(encoding="utf-8")
    return environment, configs


class NativeBallTracker:
    def __init__(self, checkpoint, size, config, device, trackerPath=None):
        from ultralytics import YOLO
        from ultralytics.utils.torch_utils import init_seeds

        init_seeds(config["seed"], deterministic=True)
        self.model = YOLO(str(checkpoint))
        native = self.model.model
        if native.yaml.get("scale") != size or native.names != {0: "ball"} or not native.end2end or native.model[-1].nc != 1:
            raise ValueError("Checkpoint must be an end-to-end YOLO26 ballOnly model of the declared size.")
        native.eval().float()
        self.config, self.device, self.trackerPath = config, device, trackerPath
        self.segmentId = None

    def predict(self, image, segmentId):
        if segmentId != self.segmentId:
            self.model.predictor = None
            self.segmentId = segmentId
        config = self.config
        arguments = {"imgsz": config["imgsz"], "batch": 1, "half": False, "device": self.device,
                     "seed": config["seed"], "deterministic": True, "workers": 0, "plots": False,
                     "conf": config["detectorConfidence"], "max_det": config["maxDetections"],
                     "nms": False, "end2end": True, "rect": False, "augment": False, "verbose": False,
                     "save": False, "save_txt": False, "save_crop": False, "show": False}
        if self.trackerPath is None:
            result = self.model.predict(image, **arguments)[0]
        else:
            result = self.model.track(image, tracker=str(self.trackerPath), persist=True, **arguments)[0]
        return resultDetections(result, tracked=self.trackerPath is not None)


def resultDetections(result, *, tracked):
    boxes = result.boxes
    if boxes is None or (tracked and boxes.id is None):
        return []
    identities = boxes.id.int().cpu().tolist() if tracked else [None] * len(boxes)
    detections = []
    for box, score, classId, identity in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(),
                                            boxes.cls.cpu().tolist(), identities, strict=True):
        if classId != 0 or not math.isfinite(score):
            raise ValueError("Tracker returned an unexpected class or confidence.")
        detections.append({"classId": 0, "className": "ball", "confidence": score,
                           "box": box, "trackId": identity})
    return detections
