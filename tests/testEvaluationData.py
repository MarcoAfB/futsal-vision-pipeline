import copy
from pathlib import Path

import pytest

from futsal.catalog import hashFile
from futsal.datasetFiles import hashJson, readJson, writeJson
from futsal.evaluationData import (
    buildReferences,
    evaluationTasks,
    loadEvaluationConfig,
    validatePredictions,
)
from futsal.trainingData import selectTrainingData


@pytest.fixture
def evaluationFixture(trainingBundle, tmp_path):
    bundle, _, training = trainingBundle
    runDir = tmp_path / "trainingRun"
    runDir.mkdir()
    writeJson(runDir / "selection.json", selectTrainingData(bundle, training))
    run = {"configuration": training, "selectionSha256": hashFile(runDir / "selection.json")}
    config = loadEvaluationConfig(Path("configs/evaluation/researchValidation.yaml"))
    return bundle, runDir, run, config


def testReferencesKeepNativePointsAndStageSpecificValidation(evaluationFixture):
    bundle, runDir, run, _ = evaluationFixture
    plan, tasks = evaluationTasks(bundle, runDir, run, "runValidation")
    references = buildReferences(plan, tasks["unified25"])
    point = next(target for target in references[0]["targets"] if target["point"] is not None)
    assert point["point"] == [40, 30]
    assert len(references) == 2
    assert all(row["matchId"] == "matchB" for row in references)
    assert all(target["className"] != "lowerCornerSideB" for row in references for target in row["targets"])
    with pytest.raises(ValueError, match="restricted"):
        evaluationTasks(bundle, runDir, run, "test")
    saved = readJson(runDir / "selection.json")
    saved["tasks"]["unified25"]["validation"][0]["split"] = "test"
    writeJson(runDir / "selection.json", saved)
    run["selectionSha256"] = hashFile(runDir / "selection.json")
    with pytest.raises(ValueError, match="manifest"):
        evaluationTasks(bundle, runDir, run, "runValidation")


def testEmptyDetectionsAreValidButMissingOrDuplicateFramesAreNot(evaluationFixture):
    bundle, runDir, run, config = evaluationFixture
    plan, tasks = evaluationTasks(bundle, runDir, run, "runValidation")
    references = buildReferences(plan, tasks["genericObjects"])
    predictions = [{"recordId": row["recordId"], "dimensions": row["dimensions"], "detections": []}
                   for row in references]
    assert validatePredictions(references, predictions[::-1], ["ball", "player"], config) == predictions
    for invalid in (predictions[:1], predictions + predictions[:1]):
        with pytest.raises(ValueError, match="membership"):
            validatePredictions(references, invalid, ["ball", "player"], config)
    invalid = copy.deepcopy(predictions)
    invalid[0]["detections"] = [{"classId": 0, "className": "ball", "confidence": 0.9,
                                 "box": [0, 0, 10000, 40]}]
    with pytest.raises(ValueError, match="coordinates"):
        validatePredictions(references, invalid, ["ball", "player"], config)
    assert hashJson(predictions) == hashJson(validatePredictions(references, predictions, ["ball", "player"], config))
