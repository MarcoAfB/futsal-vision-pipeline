import math
from pathlib import Path

import pytest

from futsal.evaluationData import loadEvaluationConfig
from futsal.evaluationMetrics import matchBoxes, nativeAveragePrecision, operationalMetrics
from futsal.pointMetrics import scorePoints


def protocol():
    return loadEvaluationConfig(Path("configs/evaluation/researchValidation.yaml"))


def target(classId, box, point=None, truncated=False):
    return {"classId": classId, "box": box, "point": point, "truncated": truncated}


def prediction(classId, box, confidence):
    return {"classId": classId, "box": box, "confidence": confidence}


def testWrongClassesAndDuplicatesHaveTraceableOperationalCounts():
    references = [{"recordId": "frame", "matchId": "match", "targets": [
        target(0, [0, 0, 10, 10]), target(1, [20, 0, 30, 20])]}]
    predictions = [{"detections": [prediction(0, [0, 0, 10, 10], 0.1),
        prediction(1, [0, 0, 10, 10], 0.9), prediction(0, [0, 0, 10, 10], 0.8),
        prediction(1, [20, 0, 30, 20], 0.7)]}]
    result = operationalMetrics(references, predictions, ["ball", "player", "absent"], protocol())
    ball, player, absent = result["perClass"]
    assert (ball["truePositives"], ball["falsePositives"], ball["falseNegatives"]) == (0, 1, 1)
    assert (player["truePositives"], player["falsePositives"]) == (1, 1)
    assert absent["precision"] is None
    assert result["confusionMatrix"][0][1] == 1
    assert result["matches"][0]["predictionIndex"] == 1
    assert result["imageErrors"][0]["ballMisses"] == 1


def testNoPredictionsAndZeroAreaBoxesCannotBecomeTruePositives():
    truth = [target(0, [0, 0, 10, 10])]
    assert matchBoxes(truth, [prediction(0, [2, 2, 2, 3], 0.9)], 0.5) == []
    result = operationalMetrics([{"recordId": "frame", "matchId": "match", "targets": truth}],
                                [{"detections": []}], ["ball"], protocol())
    assert result["perClass"][0]["recall"] == 0
    assert result["perClass"][0]["falseNegatives"] == 1


def testPointScoringUsesHighestConfidenceNativeCoordinatesAndMissingDenominator():
    references = [{"recordId": str(i), "matchId": "match", "dimensions": [100, 100],
                   "targets": [target(0, [0, 45, 10, 55], [0, 50], True)]} for i in range(2)]
    predictions = [{"detections": [prediction(0, [25, 45, 35, 55], 0.9),
                                     prediction(0, [0, 49, 0, 51], 0.8),
                                     prediction(1, [0, 0, 2, 2], 0.7)]}, {"detections": []}]
    result = scorePoints(references, predictions, ["landmark", "unsupported"],
                          {"landmark", "unsupported"}, protocol())
    point, unsupported = result["perClass"]
    assert point["support"] == 2 and point["localizedCount"] == 1 and point["missingCount"] == 1
    assert point["medianDistanceDiagonal"] == pytest.approx(30 / math.hypot(100, 100))
    assert point["medianTargetCenterOffsetDiagonal"] == pytest.approx(5 / math.hypot(100, 100))
    assert point["extraPredictions"] == 1
    assert point["successRates"]["0.01"] == 0
    assert unsupported["successRates"]["0.01"] is None
    assert unsupported["predictions"] == 1


@pytest.mark.trainingIntegration
def testAveragePrecisionMatchesPinnedNativeMatching(tmp_path, monkeypatch):
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path / "backend"))
    monkeypatch.setenv("YOLO_OFFLINE", "true")
    import numpy as np
    import torch
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils.metrics import box_iou

    truth = [target(0, [0, 0, 10, 10])]
    detected = [prediction(0, [0, 0, 10, 10], 0.9), prediction(0, [0, 0, 10, 10], 0.8),
                prediction(1, [0, 0, 10, 10], 0.7)]
    reference = {"recordId": "frame", "targets": truth}
    result = nativeAveragePrecision([reference], [{"detections": detected}], ["ball", "absent"], protocol())
    native = object.__new__(DetectionValidator)
    native.iouv = torch.tensor(protocol()["apIouThresholds"])
    matched = native.match_predictions(torch.tensor([d["classId"] for d in detected]), torch.tensor([0]),
                                       box_iou(torch.tensor([truth[0]["box"]], dtype=torch.float),
                                               torch.tensor([d["box"] for d in detected], dtype=torch.float)))
    assert np.array_equal(result["matches"][0]["correctAtIou"], matched.numpy())
    assert 0.99 <= result["perClass"]["ball"]["ap50"] <= 1
    assert result["perClass"]["absent"]["ap50"] is None
