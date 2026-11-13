from pathlib import Path

import numpy as np
import pytest
import yaml

from futsal.tracking.trackingConfig import loadTrackingConfig, trackerNames


@pytest.mark.trainingIntegration
@pytest.mark.parametrize("name", trackerNames)
def testNativeTrackerAssociationPersistenceAndReset(tmp_path, name):
    from futsal.tracking.trackingBackend import preflightTracking

    config = loadTrackingConfig("configs/evaluation/ballTracking.yaml")
    environment, configs = preflightTracking(config, "cpu", tmp_path / "backend")
    assert environment["packages"]["lap"] == "0.5.12"
    assert set(configs) == set(trackerNames)
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.track import TRACKER_MAP
    from ultralytics.utils import IterableSimpleNamespace

    values = yaml.safe_load(configs[name])
    values["device"] = "cpu"
    tracker = TRACKER_MAP[name](args=IterableSimpleNamespace(**values))
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    detections = Boxes(np.array([[20, 20, 30, 30, 0.95, 0]], dtype=np.float32), image.shape[:2])
    tracks = []
    for _ in range(5):
        result = tracker.update(detections, image)
        assert len(result) == 1
        tracks.append(int(result[0, 4]))
    assert len(set(tracks)) == 1
    tracker.reset()
    assert tracker.frame_id == 0 and not tracker.tracked_stracks
    result = tracker.update(detections, image)
    assert len(result) == 1


@pytest.mark.trainingIntegration
@pytest.mark.parametrize("name", trackerNames)
def testNativePredictionUsesTrackCallbacksAndStartsNewSegments(tmp_path, monkeypatch, name):
    from futsal.tracking.trackingBackend import NativeBallTracker, preflightTracking

    config = {**loadTrackingConfig("configs/evaluation/ballTracking.yaml"), "imgsz": 96}
    _, configs = preflightTracking(config, "cpu", tmp_path / "backend")
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    model = YOLO("yolo26n.yaml")
    model.model = DetectionModel("yolo26n.yaml", nc=1, verbose=False)
    model.model.names = {0: "ball"}
    model.model.end2end = True
    checkpoint = tmp_path / "fixture.pt"
    model.save(checkpoint)
    if name == "bytetrack":
        with pytest.raises(ValueError, match="declared size"):
            NativeBallTracker(checkpoint, "s", config, "cpu")
    runner = NativeBallTracker(checkpoint, "n", config, "cpu", Path(tmp_path / f"{name}.yaml"))
    runner.trackerPath.write_text(configs[name], encoding="utf-8")

    def fixedForward(image, *args, **kwargs):
        return torch.tensor([[[20, 20, 30, 30, 0.95, 0]]], dtype=torch.float32, device=image.device)

    monkeypatch.setattr(runner.model.model, "forward", fixedForward)
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    first = runner.predict(image, "segmentA")
    assert first and first[0]["trackId"] is not None
    predictor = runner.model.predictor
    assert runner.predict(image, "segmentA")[0]["trackId"] == first[0]["trackId"]
    assert runner.model.predictor is predictor
    assert runner.predict(image, "segmentB")
    assert runner.model.predictor is not predictor
    assert runner.model.predictor.trackers[0].frame_id == 1
