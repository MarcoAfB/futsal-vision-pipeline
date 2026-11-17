import cv2
import numpy as np
import pytest

from futsal.datasets.datasetFiles import hashJson, readJson, writeJson
from futsal.evaluation.evaluationData import writeJsonLines
from futsal.evaluation.homographyMetrics import evaluateFrame, summarizeEvaluation
from futsal.evaluation.homographyRun import evaluateHomography, loadHomographyConfig
from futsal.geometry.courtModel import landmarkNames, loadCourtGeometry, matchCourt
from futsal.geometry.courtProjection import projectPoints
from futsal.inventory.catalog import hashFile
from futsal.reporting.researchReport import renderReport, verifyEvaluation
from futsal.training.trainingRun import fileEvidence

configPath, courtPath = "configs/evaluation/homography.yaml", "configs/courtGeometry.yaml"
# A broadcast-like trapezoid: the far touchline is short and high, the near one wide and low.
courtToImage = cv2.getPerspectiveTransform(
    np.float32([[0, 0], [40, 0], [40, 20], [0, 20]]),
    np.float32([[300, 150], [1620, 150], [1900, 1000], [20, 1000]])).astype(float)
playerFeet = [(10.0, 5.0), (20.0, 12.0), (31.0, 16.0)]


@pytest.fixture
def config():
    return loadHomographyConfig(configPath)[0]


@pytest.fixture
def court():
    return matchCourt(loadCourtGeometry(courtPath), "match01")


def box(point, width=30.0, height=30.0, bottom=False):
    x, y = point
    return [x - width / 2, y - height, x + width / 2, y] if bottom else [
        x - width / 2, y - height / 2, x + width / 2, y + height / 2]


def imagePoint(matrix, point):
    return projectPoints(matrix, point)[0][0].tolist()


def courtRecord(recordId, names, matrix=courtToImage):
    landmarks = matchCourt(loadCourtGeometry(courtPath), "match01")["landmarks"]
    targets = []
    for index, name in enumerate(names):
        point = imagePoint(matrix, landmarks[name])
        targets.append({"annotationIndex": index, "classId": landmarkNames.index(name), "className": name,
                        "box": box(point), "point": point, "truncated": False})
    return {"recordId": recordId, "matchId": "match01", "sourceId": "video01", "frameIndex": int(recordId[-1]),
            "imageSha256": recordId[-1] * 64, "dimensions": [1920, 1080], "targets": targets}


def courtPrediction(reference, matrix=courtToImage, names=None):
    landmarks = matchCourt(loadCourtGeometry(courtPath), "match01")["landmarks"]
    names = [target["className"] for target in reference["targets"]] if names is None else names
    detections = [{"box": box(imagePoint(matrix, landmarks[name])), "classId": landmarkNames.index(name),
                   "className": name, "confidence": 0.9} for name in names]
    return {"recordId": reference["recordId"], "dimensions": reference["dimensions"], "detections": detections,
            "correctAtIou": [], "nativeOrder": []}


def playerRecords(reference):
    boxes = [box(imagePoint(courtToImage, foot), 40.0, 120.0, bottom=True) for foot in playerFeet]
    targets = [{"annotationIndex": index, "classId": 0, "className": "player", "box": value, "point": None,
                "truncated": False} for index, value in enumerate(boxes)]
    detections = [{"box": value, "classId": 0, "className": "player", "confidence": 0.8} for value in boxes]
    return ({**reference, "targets": targets},
            {"recordId": reference["recordId"], "dimensions": reference["dimensions"], "detections": detections,
             "correctAtIou": [], "nativeOrder": []})


def testCorrectPredictionHasZeroCourtAndAthleteErrorAndIgnoresOutliers(config, court):
    reference = courtRecord("video01:0", landmarkNames)
    prediction = courtPrediction(reference)
    prediction["detections"] += [
        {"box": box([50.0, 50.0]), "classId": 4, "className": "penaltyMarkSideA", "confidence": 0.5},
        {"box": box([900.0, 900.0]), "classId": 0, "className": "upperCornerSideA", "confidence": 0.2}]
    row, athletes = evaluateFrame(reference, prediction, court, config, playerRecords(reference))
    chosen = {item["className"]: item for item in row["prediction"]["candidates"]}

    assert row["reference"]["status"] == row["prediction"]["status"] == "verified"
    assert chosen["penaltyMarkSideA"]["confidence"] == 0.9 and chosen["upperCornerSideA"]["inlier"]
    assert row["geometry"]["grid"]["p90Meters"] < 1e-3 and row["geometry"]["visibleIou"] == pytest.approx(1.0)
    assert len(athletes) == 3 and all(item["insideCourt"] and item["matched"] for item in athletes)
    assert np.allclose([item["truthMeters"] for item in athletes], playerFeet, atol=1e-6)
    assert max(item["endToEndMeters"] for item in athletes) < 1e-3


def testRejectedFramesCountAsFailuresAndWrongAcceptedFramesAsFalseValid(config, court):
    shifted = courtToImage @ np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 1.5], [0.0, 0.0, 1.0]])
    good, wrong, sparse = (courtRecord(f"video01:{index}", landmarkNames) for index in range(3))
    rows = [evaluateFrame(good, courtPrediction(good), court, config)[0],
            evaluateFrame(wrong, courtPrediction(wrong, shifted), court, config)[0],
            evaluateFrame(sparse, courtPrediction(sparse, names=landmarkNames[:3]), court, config)[0]]
    cohort = summarizeEvaluation(rows, [], config)["overall"]["cohorts"]["verified"]

    assert [row["prediction"]["status"] for row in rows] == ["verified", "verified", "rejected"]
    assert rows[2]["prediction"]["reasons"] == ["insufficientPoints"]
    assert cohort["frames"] == 3 and cohort["coverage"] == pytest.approx(2 / 3)
    assert cohort["successRates"]["1.0"] == pytest.approx(1 / 3)
    assert cohort["falseValid"] == {"count": 1, "rate": 0.5, "byStatus": {"verified": 1}}
    assert cohort["acceptedByStatus"]["verified"]["falseValid"] == 1
    assert cohort["verifiedOnly"]["coverage"] == pytest.approx(2 / 3)


def sealedArchive(root, taskId, classes, references, predictions):
    evidence = root / "evidence" / taskId
    evidence.mkdir(parents=True)
    writeJsonLines(evidence / "references.jsonl", references)
    writeJsonLines(evidence / "predictions.jsonl", predictions)
    writeJson(root / "evaluation.json", {
        "schemaVersion": 1, "status": "completed", "testEvaluated": False, "trainingRunId": f"{taskId}Run",
        "protocolId": "protocol", "stages": [{"taskId": taskId, "status": "completed", "classes": classes,
                                              "evidenceDirectory": f"evidence/{taskId}",
                                              "checkpoint": {"sha256": "0" * 64}}]})
    contract = {"schemaVersion": 1, "evaluationSha256": hashFile(root / "evaluation.json"),
                "artifacts": [fileEvidence(path, root) for path in sorted((root / "evidence").rglob("*"))
                              if path.is_file()]}
    contract["contractId"] = hashJson(contract)
    writeJson(root / "evaluationContract.json", contract)
    return root


@pytest.fixture
def archives(tmp_path):
    references = [courtRecord(f"video01:{index}", landmarkNames) for index in range(2)]
    references.append(courtRecord("video01:2", ["lowerHalfwayLinePoint", "upperCenterCourtPoint",
                                                "upperHalfwayLinePoint", "secondPenaltyMarkSideA"]))
    players = [playerRecords(reference) for reference in references]
    courtDir = sealedArchive(tmp_path / "court", "courtPoints", list(landmarkNames), references,
                             [courtPrediction(reference) for reference in references])
    playerDir = sealedArchive(tmp_path / "player", "playerOnly", ["player"], [item[0] for item in players],
                              [item[1] for item in players])
    return courtDir, playerDir


def testSealedEvaluationProducesVerifiableReport(archives, tmp_path):
    courtDir, playerDir = archives
    output = tmp_path / "homography"
    state = evaluateHomography(courtDir, output, configPath, courtPath, playerEvaluationDir=playerDir,
                               progress=lambda message: None)
    summary = readJson(output / "evidence" / "summary.json")["overall"]
    report = renderReport(output).read_text(encoding="utf-8")

    assert state["status"] == "completed" and verifyEvaluation(output)[0]["kind"] == "homographyEvaluation"
    assert summary["referenceStatus"] == {"verified": 2, "rejected": 1}
    assert summary["referenceReasons"] == {"degenerateCourtConfiguration": 1}
    assert summary["cohorts"]["verified"]["athletes"]["truthAvailable"] == 6
    assert state["inputs"]["athletes"]["commonRecords"] == 3
    assert "Athlete ground-position error" in report and "40 x 20 m (nominalDefault)" in report


def testWrongCourtDimensionsMakeReferencesInconsistent(archives, tmp_path):
    courtDir, _ = archives
    output = tmp_path / "homography"
    state = evaluateHomography(courtDir, output, configPath, courtPath, courtLength=38.0, courtWidth=19.0,
                               progress=lambda message: None)
    summary = readJson(output / "evidence" / "summary.json")["overall"]

    assert state["courtGeometry"]["defaultSource"] == "commandLine"
    assert summary["referenceReasons"]["inlierRmseHigh"] == 2


def testChangedSealedPredictionsAreRefused(archives, tmp_path):
    courtDir, _ = archives
    path = courtDir / "evidence" / "courtPoints" / "predictions.jsonl"
    path.write_text(path.read_text().replace("0.9", "0.8"), encoding="utf-8", newline="\n")

    with pytest.raises(ValueError, match="Input changed"):
        evaluateHomography(courtDir, tmp_path / "homography", configPath, courtPath, progress=lambda message: None)
