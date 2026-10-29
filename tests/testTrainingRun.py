import shutil

import pytest

from futsal.catalog import hashFile
from futsal.datasetFiles import readJson, verifyEvidence, writeJson
from futsal.modelWeights import prepareWeights
from futsal.trainingRun import fileEvidence, runTraining


@pytest.fixture
def localWeights(tmp_path, monkeypatch):
    monkeypatch.setattr("futsal.modelWeights.downloadWeights",
                        lambda url, path: path.write_bytes(b"fixture weights"))
    directory = tmp_path / "weights"
    prepareWeights("n", directory)
    return directory / "yolo26n.pt"


@pytest.fixture
def fakeTraining(monkeypatch):
    monkeypatch.setattr("futsal.trainingRun.inspectEnvironment", lambda device: {"ready": True})
    monkeypatch.setattr("futsal.trainingRun.initializeBackend", lambda *arguments: None)

    def stageResult(config, task, stage, weights, stageDir, runDir, evidence):
        stageDir.mkdir(parents=True)
        best = stageDir / "best.pt"
        shutil.copyfile(weights, best)
        return {"checkpoints": {"best": {**fileEvidence(best, runDir),
                                       "classCount": len(task["classes"])}}}

    monkeypatch.setattr("futsal.trainingRun.trainStage", stageResult)
    return stageResult


def testRunChainsBestWeightsAndRecordsReproducibleInputs(trainingBundle, localWeights,
                                                         fakeTraining, tmp_path):
    bundle, configPath, _ = trainingBundle
    runDir = tmp_path / "run"
    result = runTraining(bundle, configPath, localWeights, runDir, progress=lambda message: None)
    assert result["status"] == "completed"
    assert result["testEvaluated"] is False
    assert [stage["results"]["checkpoints"]["best"]["classCount"] for stage in result["stages"]] == [2, 3, 25]
    assert result["stages"][1]["inputWeights"]["path"] == result["stages"][0]["results"]["checkpoints"]["best"]["path"]
    assert result["bundleIntegrityAfterRun"]["hashesVerified"]
    verifyEvidence(runDir, result["code"]["files"])
    verifyEvidence(runDir, result["inputEvidence"])
    assert readJson(runDir / "run.json")["status"] == "completed"
    with pytest.raises(ValueError, match="new directory"):
        runTraining(bundle, configPath, localWeights, runDir)


@pytest.mark.parametrize("failure", [ValueError("failed stage"), KeyboardInterrupt()])
def testRunPreservesFailedStageAndNeverStartsLaterStages(trainingBundle, localWeights,
                                                       fakeTraining, tmp_path, monkeypatch, failure):
    bundle, configPath, _ = trainingBundle
    calls = []

    def failSecond(config, task, stage, weights, stageDir, runDir, evidence):
        calls.append(stage["taskId"])
        if len(calls) == 2:
            stageDir.mkdir(parents=True)
            (stageDir / "partial.pt").write_bytes(b"partial")
            raise failure
        return fakeTraining(config, task, stage, weights, stageDir, runDir, evidence)

    monkeypatch.setattr("futsal.trainingRun.trainStage", failSecond)
    runDir = tmp_path / "failedRun"
    result = runTraining(bundle, configPath, localWeights, runDir, progress=lambda message: None)
    assert result["status"] == ("interrupted" if isinstance(failure, KeyboardInterrupt) else "failed")
    assert calls == ["genericObjects", "roleObjects"]
    assert result["stages"][0]["status"] == "completed"
    assert (runDir / result["stages"][1]["directory"] / "partial.pt").read_bytes() == b"partial"
    assert result["bundleIntegrityAfterRun"]["hashesVerified"]
    assert type(failure).__name__ in (runDir / "failure.txt").read_text()


def testRuntimeMutationFailsWithoutChangingPreparedBundle(trainingBundle, localWeights,
                                                        fakeTraining, tmp_path, monkeypatch):
    bundle, configPath, _ = trainingBundle
    originalContract = hashFile(bundle / "contract.json")

    def mutate(config, task, stage, weights, stageDir, runDir, evidence):
        result = fakeTraining(config, task, stage, weights, stageDir, runDir, evidence)
        (runDir / evidence[0]["path"]).write_bytes(b"backend repair")
        return result

    monkeypatch.setattr("futsal.trainingRun.trainStage", mutate)
    result = runTraining(bundle, configPath, localWeights, tmp_path / "mutatedRun",
                         progress=lambda message: None)
    assert result["status"] == "failed"
    assert len(result["stages"]) == 1
    assert result["integrityErrors"]
    assert result["bundleIntegrityAfterRun"]["hashesVerified"]
    assert hashFile(bundle / "contract.json") == originalContract


def testInvalidBundleFailsBeforeBackendInitialization(trainingBundle, localWeights,
                                                    fakeTraining, tmp_path):
    bundle, configPath, _ = trainingBundle
    contract = readJson(bundle / "contract.json")
    contract["testExported"] = True
    writeJson(bundle / "contract.json", contract)
    result = runTraining(bundle, configPath, localWeights, tmp_path / "badBundle",
                         progress=lambda message: None)
    assert result["status"] == "failed"
    assert result["stages"] == []
    assert "contract" in result["error"]["message"].lower()
