from pathlib import Path

import cv2
import numpy as np
import pytest

from futsal.datasets.datasetFiles import hashJson, writeJson
from futsal.evaluation.evaluationData import writeJsonLines
from futsal.evaluation.evaluationRun import sealEvaluation
from futsal.geometry.courtProjection import projectPoints
from futsal.inventory.catalog import hashFile, readYaml
from futsal.training.trainingRun import fileEvidence

dimensions = [640, 360]
courtToImage = cv2.getPerspectiveTransform(
    np.float32([[0, 0], [40, 0], [40, 20], [0, 20]]),
    np.float32([[100, 60], [540, 60], [630, 340], [10, 340]])).astype(float)
# Uniform colours (BGR) and the detailed class each one represents in the synthetic match.
uniforms = {"outfieldPlayerTeamA": (0, 0, 220), "outfieldPlayerTeamB": (220, 0, 0),
            "goalkeeperTeamA": (0, 200, 0), "goalkeeperTeamB": (0, 220, 220)}
labelIds = {"ball": 0, "outfieldPlayerTeamA": 1, "outfieldPlayerTeamB": 2, "goalkeeperTeamA": 3, "goalkeeperTeamB": 4}


def athletes(frame):
    shift = 0.3 * frame
    rows = [("outfieldPlayerTeamA", (10 + shift, 4)), ("outfieldPlayerTeamA", (24 + shift, 12)),
            ("outfieldPlayerTeamB", (17 + shift, 8)), ("outfieldPlayerTeamB", (31 + shift, 16)),
            ("goalkeeperTeamA", (1.5, 10))]
    return rows + ([("goalkeeperTeamB", (38.5, 10))] if frame < 3 else [])


def athleteBox(foot):
    u, v = projectPoints(courtToImage, foot)[0][0]
    return [float(u - 12), float(v - 60), float(u + 12), float(v)]


def frameImage(frame):
    image = np.full((dimensions[1], dimensions[0], 3), 128, dtype=np.uint8)
    for className, foot in athletes(frame):
        x0, y0, x1, y1 = (round(value) for value in athleteBox(foot))
        image[y0:y1, x0:x1] = uniforms[className]
    return image


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
    return root, contract["contractId"]


def homographyArchive(root, images):
    (root / "evidence").mkdir(parents=True)
    frames = [{"recordId": recordId, "matchId": "match01", "imageSha256": sha,
               "prediction": {"status": "verified", "courtToImage": courtToImage.tolist(), "candidates": []}}
              for recordId, sha in images]
    writeJsonLines(root / "evidence" / "frames.jsonl", frames)
    sealEvaluation(root, {"schemaVersion": 1, "kind": "homographyEvaluation", "status": "completed",
                          "testEvaluated": False, "inputs": {"athletes": None},
                          "courtGeometry": {"defaultDimensions": {"length": 40.0, "width": 20.0},
                                            "matchDimensions": {}}})
    return root


def trackingArchive(root, players):
    """A sealed trackRun campaign with a detector-only and a BoT-SORT condition over one segment.

    Athletes keep their list position across frames, so that position is their track ID.
    """
    segmentId = "video01:video01Reviewed:0"
    references = [{**reference, "taskId": "playerOnly", "segmentId": segmentId} for reference, _ in players]
    selection = {"taskId": "playerOnly", "classes": ["player"], "references": references, "frameCount": len(players)}
    conditions = []
    for tracker in ("detector", "botsort"):
        conditionId = f"player_{tracker}"
        folder = root / "evidence" / "conditions" / conditionId
        folder.mkdir(parents=True)
        writeJsonLines(folder / "predictions.jsonl", [
            {"recordId": prediction["recordId"], "frameIndex": reference["frameIndex"], "segmentId": segmentId,
             "dimensions": prediction["dimensions"],
             "detections": [{**item, "trackId": None if tracker == "detector" else index + 1}
                            for index, item in enumerate(prediction["detections"])]}
            for reference, prediction in players])
        conditions.append({"conditionId": conditionId, "label": "player", "tracker": tracker, "status": "completed",
                           "evidenceDirectory": f"evidence/conditions/{conditionId}", "frames": len(players)})
    writeJson(root / "evidence" / "selection.json", selection)
    sealEvaluation(root, {"schemaVersion": 1, "kind": "trackingComparison", "status": "completed",
                          "testEvaluated": False, "campaignId": "c" * 64, "selectionEvidence": selection,
                          "protocol": {"detectorConfidence": 0.1, "maxDetections": 300},
                          "models": {"player": {"checkpointSha256": "0" * 64}}, "conditions": conditions})
    return root


@pytest.fixture
def scene(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    assets, players, detailed = {}, [], []
    for frame in range(6):
        path = media / f"frame{frame}.png"
        cv2.imwrite(str(path), frameImage(frame))
        sha = hashFile(path)
        assets[sha] = {"root": str(media), "path": path.name}
        record = {"recordId": f"video01:{frame}", "matchId": "match01", "sourceId": "video01", "frameIndex": frame,
                  "imageSha256": sha, "dimensions": dimensions}
        boxes = [(className, athleteBox(foot)) for className, foot in athletes(frame)]
        players.append(({**record, "targets": [
            {"annotationIndex": index, "classId": 0, "className": "player", "box": box, "point": None,
             "truncated": False} for index, (_, box) in enumerate(boxes)]},
            {"recordId": record["recordId"], "dimensions": dimensions, "detections": [
                {"box": box, "classId": 0, "className": "player", "confidence": 0.9} for _, box in boxes]}))
        targets = [{"annotationIndex": index, "classId": labelIds[name], "className": name, "box": box, "point": None,
                    "truncated": False} for index, (name, box) in enumerate(boxes)]
        targets.append({"annotationIndex": len(boxes), "classId": 0, "className": "ball",
                        "box": [300.0, 200.0, 306.0, 206.0], "point": None, "truncated": False})
        # The baseline four-class detector calls every athlete team B outfield.
        detailed.append(({**record, "targets": targets}, {"recordId": record["recordId"], "dimensions": dimensions,
                         "detections": [{"box": box, "classId": 2, "className": "outfieldPlayerTeamB",
                                         "confidence": 0.8} for _, box in boxes]}))
    writeJson(media / "bindings.json", {"schemaVersion": 1, "assets": assets})
    # A two-class ball/player archive, like the mixed ball-player detector, also supplies features.
    playerDir, _ = sealedArchive(tmp_path / "player", "ballPlayer", ["ball", "player"],
                                 [item[0] for item in players], [item[1] for item in players])
    referenceDir, _ = sealedArchive(tmp_path / "detailed", "detailedObjects", list(labelIds),
                                    [item[0] for item in detailed], [item[1] for item in detailed])
    homographyDir = homographyArchive(tmp_path / "homography",
                                      [(item[0]["recordId"], item[0]["imageSha256"]) for item in players])
    return {"media": media, "player": playerDir, "reference": referenceDir, "homography": homographyDir,
            "root": tmp_path, "players": players}


def writeConfig(path, source, **changes):
    config, _ = readYaml(Path(source))
    config.update(changes)
    writeJson(path, config)
    return path


@pytest.fixture
def features(scene):
    from futsal.evaluation.teamFeatureRun import extractTeamFeatures

    config = writeConfig(scene["root"] / "teamFeatures.yaml", "configs/evaluation/teamFeatures.yaml",
                         featureKinds=["colorHistogram"])
    output = scene["root"] / "features"
    state = extractTeamFeatures(scene["player"], scene["media"], output, config, progress=lambda message: None)
    return output, state


@pytest.fixture
def assignmentConfig(scene):
    return writeConfig(scene["root"] / "teamAssignment.yaml", "configs/evaluation/teamAssignment.yaml",
                       featureKind="colorHistogram", reduction="none", minCleanCrops=4)


@pytest.fixture
def foreignHomography(scene):
    return homographyArchive(scene["root"] / "otherHomography", [("video01:0", "f" * 64)])


@pytest.fixture
def tracking(scene):
    return trackingArchive(scene["root"] / "tracking", scene["players"])


@pytest.fixture
def trackedFeatures(scene, tracking):
    from futsal.evaluation.teamFeatureRun import extractTeamFeatures

    config = writeConfig(scene["root"] / "teamFeatures.yaml", "configs/evaluation/teamFeatures.yaml",
                         featureKinds=["colorHistogram"])

    def extract(condition):
        output = scene["root"] / f"features_{condition}"
        state = extractTeamFeatures(tracking, scene["media"], output, config, condition=condition,
                                    progress=lambda message: None)
        return output, state

    return extract
