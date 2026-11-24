import cv2
import numpy as np
import pytest

from futsal.evaluation.evaluationData import loadEvaluationConfig, validatePredictions
from futsal.evaluation.homographyMetrics import predictedCandidates
from futsal.evaluation.posePredictions import keypointBox, landmarkDetections
from futsal.geometry.courtModel import landmarkNames, loadCourtGeometry, matchCourt
from futsal.training.pointSelection import pointSelectionScore

names = list(landmarkNames)
pointBoxes = {"widthFraction": 1 / 64, "heightFraction": 1 / 36}
dimensions = [1920, 1080]
protocol = loadEvaluationConfig("configs/evaluation/researchValidation.yaml")


def instance(confidence, points):
    keypoints = [[0.0, 0.0, 0.0] for _ in names]
    for name, (x, y, visibility) in points.items():
        keypoints[names.index(name)] = [x, y, visibility]
    return {"confidence": confidence, "keypoints": keypoints}


def center(box):
    return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]


def testBoxIsCentredOnTheKeypointAndShrinksSymmetricallyAtTheBorder():
    assert keypointBox(960.0, 540.0, dimensions, pointBoxes) == [945.0, 525.0, 975.0, 555.0]
    edge = keypointBox(4.0, 1079.0, dimensions, pointBoxes)
    assert edge == [0.0, 1078.0, 8.0, 1080.0] and center(edge) == [4.0, 1079.0]
    assert keypointBox(0.0, 0.0, dimensions, pointBoxes) == [0.0, 0.0, 0.0, 0.0]


def testOnlyTheMostConfidentInstanceBecomesLandmarkDetections():
    weak = instance(0.4, {"upperCornerSideA": (100.0, 100.0, 1.0)})
    strong = instance(0.8, {"upperCornerSideB": (1800.0, 120.0, 0.5), "penaltyMarkSideA": (300.0, 600.0, 0.9),
                            "lowerCornerSideA": (-3.0, 900.0, 0.99), "upperHalfwayLinePoint": (960.0, 100.0, 0.001)})
    detections = landmarkDetections([weak, strong], dimensions, names, pointBoxes, protocol["confidenceFloor"])
    # Products 0.72 and 0.40 remain; the off-image keypoint and the one below the floor are not detections.
    assert [item["className"] for item in detections] == ["penaltyMarkSideA", "upperCornerSideB"]
    assert [item["confidence"] for item in detections] == pytest.approx([0.72, 0.4])
    assert landmarkDetections([], dimensions, names, pointBoxes, protocol["confidenceFloor"]) == []


def testConvertedKeypointsFeedPointScoringAndHomographyUnchanged():
    court = matchCourt(loadCourtGeometry("configs/courtGeometry.yaml"), "match01")
    matrix = cv2.getPerspectiveTransform(np.float32([[0, 0], [40, 0], [40, 20], [0, 20]]),
                                         np.float32([[300, 150], [1620, 150], [1900, 1000], [20, 1000]]))
    visible = ["upperCornerSideA", "upperCornerSideB", "upperCenterCourtPoint", "penaltyMarkSideA",
               "penaltyMarkSideB"]
    image = cv2.perspectiveTransform(np.float32([[court["landmarks"][name] for name in visible]]), matrix)[0]
    points = {name: (float(x), float(y)) for name, (x, y) in zip(visible, image, strict=True)}
    detections = landmarkDetections([instance(0.9, {name: (*point, 0.95) for name, point in points.items()})],
                                    dimensions, names, pointBoxes, protocol["confidenceFloor"])
    reference = {"recordId": "frame", "matchId": "match01", "dimensions": dimensions,
                 "targets": [{"classId": names.index(name), "className": name, "point": list(point),
                              "truncated": False, "box": keypointBox(*point, dimensions, pointBoxes)}
                             for name, point in points.items()]}
    prediction = {"recordId": "frame", "dimensions": dimensions, "detections": detections}
    validatePredictions([reference], [prediction], names, protocol)
    candidates = predictedCandidates(prediction, court, 0.25, dimensions)
    assert {item["className"]: item["image"] for item in candidates} == pytest.approx(
        {name: list(point) for name, point in points.items()})
    selection = {"metric": "pointSuccessMacro", "confidence": 0.25, "tolerance": 0.01}
    assert pointSelectionScore([reference], [prediction], names, selection)["micro"] == 1.0
