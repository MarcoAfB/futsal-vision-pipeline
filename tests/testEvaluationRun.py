import shutil
from collections import Counter
from pathlib import Path

import pytest

from futsal.catalog import hashFile
from futsal.datasetFiles import readJson
from futsal.evaluationRun import evaluateRun
from futsal.modelWeights import prepareWeights
from futsal.trainingRun import fileEvidence, runTraining


@pytest.fixture
def evaluationRunFixture(trainingBundle, tmp_path, monkeypatch):
    bundle, configPath, _ = trainingBundle
    monkeypatch.setattr("futsal.modelWeights.downloadWeights", lambda url, path: path.write_bytes(b"fixture weights"))
    weights = tmp_path / "weights"
    prepareWeights("n", weights)
    monkeypatch.setattr("futsal.trainingRun.inspectEnvironment", lambda device: {"ready": True})
    monkeypatch.setattr("futsal.trainingRun.initializeBackend", lambda *args: None)

    def train(config, task, stage, source, stageDir, runDir, evidence):
        stageDir.mkdir(parents=True)
        best = stageDir / "best.pt"
        shutil.copyfile(source, best)
        return {"losses": {"fixtureLoss": 0.0}, "checkpoints": {"best": fileEvidence(best, runDir)
                | {"classCount": len(task["classes"]), "names": task["classes"]}}}

    monkeypatch.setattr("futsal.trainingRun.trainStage", train)
    runDir = tmp_path / "run"
    run = runTraining(bundle, configPath, weights / "yolo26n.pt", runDir, progress=lambda message: None)
    assert run["status"] == "completed"
    monkeypatch.setattr("futsal.evaluationRun.initializeBackend", lambda *args: None)

    def predictions(checkpoint, task, references, stage, trainingConfig, config, workDir):
        return ([{"recordId": row["recordId"], "dimensions": row["dimensions"], "detections": []}
                 for row in references], {})

    def noDetectionAp(references, predictions, names, config):
        counts = Counter(target["classId"] for row in references for target in row["targets"])
        return {"perClass": {name: {"ap50": 0.0 if counts[index] else None,
                                     "ap50To95": 0.0 if counts[index] else None}
                              for index, name in enumerate(names)}, "prCurves": {}, "matches": []}

    monkeypatch.setattr("futsal.evaluationRun.predictValidation", predictions)
    monkeypatch.setattr("futsal.evaluationMetrics.nativeAveragePrecision", noDetectionAp)
    return bundle, runDir, predictions


def testEvaluationSealsEvidenceAndRefusesFrozenOutputDirectories(evaluationRunFixture, tmp_path):
    bundle, runDir, _ = evaluationRunFixture
    output = tmp_path / "evaluation"
    before = hashFile(runDir / "run.json")
    state = evaluateRun(runDir, bundle, "configs/evaluation/researchValidation.yaml", "runValidation", output,
                        progress=lambda message: None)
    assert state["status"] == "completed"
    assert state["bundleAfter"]["membershipVerified"] and state["runtimeAfter"]
    assert all(len(stage["examples"]) <= 6 for stage in state["stages"])
    assert (output / "evaluationContract.json").is_file()
    assert hashFile(runDir / "run.json") == before
    for invalid in (runDir / "report", bundle / "report", output):
        with pytest.raises(ValueError):
            evaluateRun(runDir, bundle, "configs/evaluation/researchValidation.yaml", "runValidation", invalid)


def testBackendMutationRetainsFailureAndStopsLaterEvaluationStages(evaluationRunFixture, tmp_path, monkeypatch):
    bundle, runDir, original = evaluationRunFixture
    before = hashFile(bundle / "contract.json")
    calls = []

    def mutate(checkpoint, task, references, stage, trainingConfig, config, workDir):
        calls.append(task["taskId"])
        first = Path(workDir) / "runtimeData" / task["rows"][0]["imagePath"]
        first.write_bytes(b"unexpected backend repair")
        return original(checkpoint, task, references, stage, trainingConfig, config, workDir)

    monkeypatch.setattr("futsal.evaluationRun.predictValidation", mutate)
    output = tmp_path / "failedEvaluation"
    state = evaluateRun(runDir, bundle, "configs/evaluation/researchValidation.yaml", "runValidation", output,
                        progress=lambda message: None)
    assert state["status"] == "failed" and "integrityError" in state
    assert calls == ["genericObjects"]
    assert state["stages"][1]["status"] == "notEvaluated"
    assert state["bundleAfter"]["hashesVerified"]
    assert hashFile(bundle / "contract.json") == before
    assert "changed" in (output / "evidence" / "failure.txt").read_text()
    assert readJson(output / "evaluation.json")["status"] == "failed"
