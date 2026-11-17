import pytest

from futsal.training.pointSelection import pointSelectionScore

names = ["upperCornerSideA", "lowerCornerSideA", "upperCornerSideB"]
selection = {"metric": "pointSuccessMacro", "confidence": 0.25, "tolerance": 0.01}


def reference(recordId, points):
    return {"recordId": recordId, "matchId": "match01", "dimensions": [1920, 1080],
            "targets": [{"classId": classId, "point": point, "truncated": False,
                         "box": [point[0] - 15, point[1] - 15, point[0] + 15, point[1] + 15]}
                        for classId, point in points.items()]}


def detection(classId, center, confidence):
    return {"classId": classId, "confidence": confidence,
            "box": [center[0] - 15, center[1] - 15, center[0] + 15, center[1] + 15]}


def testMacroUsesSupportedClassesAndTheBestConfidentDetection():
    references = [reference("a", {0: [100, 100], 1: [100, 900]}), reference("b", {0: [200, 100]})]
    predictions = [
        {"recordId": "a", "detections": [detection(0, [500, 500], 0.9), detection(0, [100, 100], 0.5),
                                         detection(1, [100, 900], 0.2)]},
        {"recordId": "b", "detections": [detection(0, [205, 100], 0.8), detection(2, [900, 100], 0.9)]},
    ]
    scores = pointSelectionScore(references, predictions, names, selection)
    # Class 0: frame a picks the 0.9 detection (wrong), frame b is correct; class 1 is below 0.25.
    assert scores["macro"] == pytest.approx((0.5 + 0.0) / 2)
    assert scores["micro"] == pytest.approx(1 / 3)
    assert scores["framesWithFourCorrect"] == 0 and scores["frames"] == 2


def testMembershipMustMatchTheValidationReferences():
    references = [reference("a", {0: [100, 100]})]
    with pytest.raises(ValueError, match="exactly once"):
        pointSelectionScore(references, [], names, selection)
    duplicated = [{"recordId": "a", "detections": []}] * 2
    with pytest.raises(ValueError, match="exactly once"):
        pointSelectionScore(references, duplicated, names, selection)
