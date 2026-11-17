import copy

import pytest

from futsal.evaluation.ballTrackingMetrics import scoreBallTracking
from futsal.evaluation.trackingMetrics import scoreTracking
from futsal.tracking.trackingConfig import loadTrackingConfig

config = loadTrackingConfig("configs/evaluation/tracking.yaml")
classes = ["player", "ball"]


def box(x, size=10):
    return [x, 10, x + size, 10 + size]


def rows(frames, segment="a", match="match"):
    references, predictions = [], []
    for index, (targets, detections) in enumerate(frames):
        row = {"recordId": f"{segment}{index}", "matchId": match, "frameIndex": index,
               "segmentId": segment, "dimensions": [200, 100]}
        references.append({**row, "targets": [{"classId": classes.index(name), "className": name, "box": item}
                                              for name, item in targets]})
        predictions.append({**row, "detections": [
            {"classId": classes.index(name), "className": name, "box": item, "confidence": score, "trackId": identity}
            for name, item, score, identity in detections]})
    return references, predictions


def testMultiObjectCountsAreClassAwareAndThresholded():
    references, predictions = rows([
        ([("player", box(10)), ("player", box(60)), ("ball", box(120))],
         [("player", box(10), 0.9, 1), ("player", box(60), 0.8, 2), ("player", box(120), 0.9, 3),
          ("player", box(150), 0.2, 4), ("player", box(160), 0.9, 5)])])
    overall = scoreTracking(references, predictions, classes, config, tracked=True)["overall"]
    assert (overall["tp"], overall["fp"], overall["fn"]) == (2, 2, 1)
    assert overall["perClass"]["player"]["tp"] == 2 and overall["perClass"]["player"]["fp"] == 2
    assert overall["perClass"]["ball"]["fn"] == 1 and overall["perClass"]["ball"]["recall"] == 0
    assert overall["f1"] == pytest.approx(4 / 7)


def testIdChangesAreCountedAdjacentAndAcrossGapsPerChain():
    moving = [("player", box(10)), ("player", box(100))]
    frames = [(moving, [("player", box(10), 0.9, 1), ("player", box(100), 0.9, 7)]),
              (moving, [("player", box(10), 0.9, 1), ("player", box(100), 0.9, 8)]),
              (moving, [("player", box(10), 0.9, 1)]),
              (moving, [("player", box(10), 0.9, 1), ("player", box(100), 0.9, 9)])]
    references, predictions = rows(frames)
    stability = scoreTracking(references, predictions, classes, config, tracked=True)["overall"]["stability"]
    assert stability["referenceChains"] == 2 and stability["matchedChains"] == 2
    assert stability["adjacentMatchedPairs"] == 4 and stability["adjacentIdChanges"] == 1
    assert stability["gapRecoveries"] == 1 and stability["gapIdChanges"] == 1
    assert stability["chainsWithMultipleIds"] == 1 and stability["meanIdsPerMatchedChain"] == 2.0
    assert stability["trackCount"] == 4 and stability["singleFrameTracks"] == 3


def testSegmentBoundariesAndMovedObjectsBreakChainsWithoutFalseChanges():
    frames = [([("player", box(10))], [("player", box(10), 0.9, 1)]),
              ([("player", box(10))], [("player", box(10), 0.9, 2)])]
    references, predictions = rows(frames)
    references[1]["segmentId"] = predictions[1]["segmentId"] = "b"
    stability = scoreTracking(references, predictions, classes, config, tracked=True)["overall"]["stability"]
    assert stability["referenceChains"] == 0 and stability["adjacentIdChanges"] == 0
    references, predictions = rows([([("player", box(10))], [("player", box(10), 0.9, 1)]),
                                    ([("player", box(100))], [("player", box(100), 0.9, 2)])])
    stability = scoreTracking(references, predictions, classes, config, tracked=True)["overall"]["stability"]
    assert stability["referenceChains"] == 0 and stability["trackCount"] == 2


def testDetectorOnlyHasNoStabilityAndRejectsIds():
    references, predictions = rows([([("player", box(10))], [("player", box(10), 0.9, None)])])
    result = scoreTracking(references, predictions, classes, config, tracked=False)
    assert result["overall"]["stability"] is None and result["overall"]["tp"] == 1
    assert result["identityInterpretation"] == "notApplicable"
    broken = copy.deepcopy(predictions)
    broken[0]["detections"][0]["trackId"] = 3
    with pytest.raises(ValueError, match="Detector-only"):
        scoreTracking(references, broken, classes, config, tracked=False)
    with pytest.raises(ValueError, match="track IDs"):
        scoreTracking(references, predictions, classes, config, tracked=True)
    predictions[0]["frameIndex"] = 9
    with pytest.raises(ValueError, match="temporal"):
        scoreTracking(references, predictions, classes, config, tracked=False)


def testSingleClassTrackingReproducesBallScoring():
    events = [(True, [(0.25, 1)]), (True, []), (True, [(0.9, 2), (0.8, 3)]), (False, [(0.9, 4)]), (True, [(0.9, 4)])]
    references, predictions = [], []
    for index, (positive, detections) in enumerate(events):
        row = {"recordId": str(index), "matchId": "match", "frameIndex": index, "segmentId": "a", "dimensions": [100, 100]}
        references.append({**row, "targets": [{"classId": 0, "className": "ball", "box": [10, 10, 20, 20]}] if positive else []})
        predictions.append({**row, "detections": [{"classId": 0, "className": "ball", "box": [10, 10, 20, 20],
                                                   "confidence": score, "trackId": identity} for score, identity in detections]})
    general = scoreTracking(references, predictions, ["ball"], config, tracked=True)["overall"]
    ball = scoreBallTracking(references, predictions, config, tracked=True)["overall"]
    assert (general["tp"], general["fp"], general["fn"]) == (ball["tp"], ball["fp"], ball["fn"])
    assert general["f1"] == pytest.approx(ball["f1"])
