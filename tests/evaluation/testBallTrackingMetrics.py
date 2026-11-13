import copy

import pytest

from futsal.evaluation.ballTrackingMetrics import scoreBallTracking
from futsal.tracking.trackingConfig import loadTrackingConfig


def trackingRows(events):
    references, predictions = [], []
    for frame, (positive, detections, segment) in enumerate(events):
        row = {"recordId": str(frame), "matchId": "match", "frameIndex": frame,
               "segmentId": segment, "dimensions": [100, 100]}
        references.append({**row, "targets": [{"classId": 0, "box": [10, 10, 20, 20]}] if positive else []})
        predictions.append({**row, "detections": [{"classId": 0, "className": "ball", "box": [10, 10, 20, 20],
                             "confidence": score, "trackId": identity} for score, identity in detections]})
    return references, predictions


def testOperationalThresholdsMultiplePredictionsAndTemporalLosses():
    references, predictions = trackingRows([(True, [(0.25, 1)], "a"), (True, [], "a"),
        (True, [], "a"), (True, [(0.9, 2), (0.8, 3)], "a"), (True, [(0.9, 4)], "a"),
        (False, [(0.9, 4)], "a"), (True, [], "a"), (True, [(0.9, 5)], "b")])
    result = scoreBallTracking(references, predictions, loadTrackingConfig("configs/evaluation/ballTracking.yaml"), tracked=True)
    overall = result["overall"]
    assert (overall["tp"], overall["fp"], overall["fn"]) == (4, 2, 3)
    assert overall["f1"] == pytest.approx(8 / 13)
    assert overall["lossCount"] == 2 and overall["maxLossFrames"] == 2
    assert overall["recoveredLossCount"] == 1 and overall["meanRecoveryFrames"] == 2
    assert overall["observedIdChanges"] == overall["adjacentMatchedPairs"] == 1
    assert result["byMatch"]["match"] == overall


def testDetectorIdsEmptyDenominatorsAndConfidenceFloor():
    config = loadTrackingConfig("configs/evaluation/ballTracking.yaml")
    refs, predictions = trackingRows([(False, [(0.1, None)], "a")])
    overall = scoreBallTracking(refs, predictions, config, tracked=False)["overall"]
    assert overall["f1"] is None and overall["observedIdChanges"] is None
    broken = copy.deepcopy(predictions)
    broken[0]["detections"][0]["trackId"] = 1
    with pytest.raises(ValueError, match="Detector-only"):
        scoreBallTracking(refs, broken, config, tracked=False)
    with pytest.raises(ValueError, match="track IDs"):
        scoreBallTracking(refs, predictions, config, tracked=True)
    predictions[0]["frameIndex"] = 9
    with pytest.raises(ValueError, match="temporal"):
        scoreBallTracking(refs, predictions, config, tracked=False)


def testIouBoundaryAndGapDoNotBecomeAnIdSwitch():
    config = loadTrackingConfig("configs/evaluation/ballTracking.yaml")
    refs, predictions = trackingRows([(True, [(0.8, 1)], "a"), (True, [(0.8, 2)], "b")])
    predictions[0]["detections"][0]["box"] = [10, 10, 15, 20]
    result = scoreBallTracking(refs, predictions, config, tracked=True)["overall"]
    assert result["tp"] == 2 and result["observedIdChanges"] == 0
