import copy
import shutil
from pathlib import Path

import pytest
import yaml

from futsal.datasets.datasetFiles import readJson
from futsal.training.objectCampaign import (
    campaignLock,
    nextAttempt,
    processModel,
    retryConfiguration,
    worker,
)
from futsal.training.trainingConfig import stageParameters
from futsal.training.trainingRun import fileEvidence


@pytest.fixture
def campaignHarness(objectCurriculumView, tmp_path, monkeypatch):
    from futsal.training.modelWeights import prepareWeights

    view, store, configFile = objectCurriculumView
    config = yaml.safe_load(configFile.read_text())
    config.update(device="cpu", evaluationDevice="cpu")
    monkeypatch.setattr("futsal.training.modelWeights.downloadWeights", lambda url, path: path.write_bytes(b"fixture"))
    identity = {"configuration": config, "datasetDir": str(view), "mediaStoreDir": str(store), "weights": {}}
    for size in ("n", "s"):
        weights = tmp_path / f"weights{size}"
        metadata = prepareWeights(size, weights)
        identity["weights"][size] = {"path": str(weights / f"yolo26{size}.pt"), "metadata": metadata}
    monkeypatch.setattr("futsal.training.trainingRun.inspectEnvironment", lambda device: {"ready": True})
    monkeypatch.setattr("futsal.training.trainingRun.initializeBackend", lambda *a: None)
    calls, actions = [], []
    control = {"failTask": None, "minimumBatch": 4, "reportFailure": False, "interrupt": False}
    control["identity"] = identity

    def train(config, task, stage, weights, stageDir, runDir, evidence, **kwargs):
        batch = stageParameters(config, stage)["batch"]
        calls.append((config["modelSize"], stage["taskId"], batch))
        if stage["taskId"] == control["failTask"] and config["modelSize"] == "n":
            if control["interrupt"]:
                control["interrupt"] = False
                raise KeyboardInterrupt
            if batch > control["minimumBatch"]:
                raise RuntimeError("CUDA out of memory")
        stageDir.mkdir(parents=True)
        checkpoint = stageDir / "best.pt"
        shutil.copyfile(weights, checkpoint)
        return {"checkpoints": {"best": fileEvidence(checkpoint, runDir)},
                "training": {"requestedBatch": batch, "effectiveBatch": batch}}

    monkeypatch.setattr("futsal.training.trainingRun.trainStage", train)

    def dispatch(payload):
        actions.append(copy.deepcopy(payload))
        if payload["action"] == "report":
            return 1 if control["reportFailure"] else 0
        return worker(payload)

    directory = tmp_path / "launch"
    directory.mkdir()

    def process(model, size="n"):
        processModel(model, size, identity, directory, tmp_path / "runs", tmp_path / "reports", tmp_path / "evaluationWork",
                     Path("configs/evaluation/researchValidation.yaml"), dispatch, lambda: None)

    return process, calls, actions, control


def testQueueReusesCompletedStagesAndReturnsToEight(campaignHarness):
    process, calls, _, control = campaignHarness
    control["failTask"] = "roleObjects"
    model = {"attempts": []}
    process(model)
    assert model["status"] == "completed"
    assert calls == [("n", "genericObjects", 8), ("n", "roleObjects", 8),
                     ("n", "roleObjects", 4), ("n", "detailedObjects", 8)]
    assert [Path(item["runDir"]).name for item in model["attempts"]] == ["yolo26nRealAttempt01", "yolo26nRealAttempt02"]


def testBatchOneExhaustionPreservesAttemptsAndAllowsNextSize(campaignHarness):
    process, calls, _, control = campaignHarness
    control.update(failTask="genericObjects", minimumBatch=0)
    first, second = {"attempts": []}, {"attempts": []}
    process(first)
    process(second, "s")
    assert first["status"] == "failed" and second["status"] == "completed"
    assert [batch for size, _, batch in calls if size == "n"] == [8, 4, 2, 1]
    assert len(first["attempts"]) == 4
    for attempt in first["attempts"]:
        assert Path(attempt["runDir"], "failure.txt").is_file()
    final = readJson(Path(first["attempts"][-1]["runDir"]) / "run.json")
    assert retryConfiguration(final) is None


def testReportOnlyResumeNeverRetrains(campaignHarness):
    process, calls, actions, control = campaignHarness
    control["reportFailure"] = True
    model = {"attempts": []}
    process(model)
    assert model["status"] == "reportFailed"
    count = len(calls)
    control["reportFailure"] = False
    process(model)
    assert len(calls) == count and len(model["attempts"]) == 1
    assert model["status"] == "completed" and actions[-1]["action"] == "report"


def testInterruptionStopsThenStrictlyContinuesPrefix(campaignHarness):
    process, calls, _, control = campaignHarness
    control.update(failTask="roleObjects", interrupt=True, minimumBatch=8)
    model = {"attempts": []}
    with pytest.raises(KeyboardInterrupt):
        process(model)
    assert calls == [("n", "genericObjects", 8), ("n", "roleObjects", 8)]
    process(model)
    assert calls[2:] == [("n", "roleObjects", 8), ("n", "detailedObjects", 8)]
    assert "continueFrom" in model["attempts"][-1]


def testCampaignLockAndAttemptAllocation(tmp_path):
    directory = tmp_path / "campaign"
    with campaignLock(directory):
        with pytest.raises(ValueError, match="locked"), campaignLock(directory):
            pytest.fail("Acquired a second campaign lock")
    assert not (directory / "campaign.lock").exists()
    (tmp_path / "runs/yolo26nRealAttempt01").mkdir(parents=True)
    assert nextAttempt(tmp_path / "runs", "n").name == "yolo26nRealAttempt02"


@pytest.mark.parametrize("presentationStatus", [None, "failed", "completed"])
def testReportWorkerRendersCompletedSealWithoutInference(tmp_path, monkeypatch, presentationStatus):
    from futsal.datasets.datasetFiles import writeJson

    reportRoot = tmp_path / "reports"
    directory = reportRoot / "yolo26nRealAttempt01/runValidation-a"
    directory.mkdir(parents=True)
    writeJson(directory / "evaluation.json", {"status": "completed"})
    writeJson(directory / "evaluationContract.json", {})
    if presentationStatus:
        writeJson(directory / "reportState.json", {"status": presentationStatus, "artifacts": []})
    calls = []
    monkeypatch.setattr("futsal.reporting.researchReport.renderReport", lambda path, output=None: calls.append((path, output)))
    monkeypatch.setattr("futsal.reporting.researchReport.verifyEvaluation", lambda path: None)
    monkeypatch.setattr("futsal.reporting.researchCli.generateRunReport", lambda *a, **kw: pytest.fail("Repeated inference"))
    assert worker({"action": "report", "runDir": "yolo26nRealAttempt01", "reportRoot": str(reportRoot)}) == 0
    if presentationStatus == "completed":
        assert calls == []
    else:
        assert calls[0][0] == directory
        assert (calls[0][1] is not None) == (presentationStatus == "failed")


def testCampaignResumeKeepsIdentityAndDoesNotDuplicateTraining(campaignHarness, tmp_path, monkeypatch):
    from futsal.inventory.catalog import hashFile
    from futsal.training.objectCampaign import trainCampaign

    _, calls, _, control = campaignHarness
    identity = control["identity"]
    identity.update(files={}, contractSha256=hashFile(Path(identity["datasetDir"]) / "contract.json"))
    monkeypatch.setattr("futsal.training.objectCampaign.campaignIdentity", lambda *args: copy.deepcopy(identity))
    monkeypatch.setattr("futsal.training.objectCampaign.runWorker", lambda script, project, directory, payload, name:
                        0 if payload["action"] in {"environment", "report"} else worker(payload))
    options = {"project": Path.cwd(), "script": Path("scripts/runObjectCurriculum.py"),
        "datasetDir": identity["datasetDir"], "mediaStoreDir": identity["mediaStoreDir"],
        "workDir": tmp_path / "work", "campaignDir": tmp_path / "campaign",
        "configPath": tmp_path / "objectTraining.yaml", "evaluationConfig": Path("configs/evaluation/researchValidation.yaml"),
        "weightsRoot": tmp_path / "weights", "modelSizes": ["n"]}
    assert trainCampaign(**options) == 0
    count = len(calls)
    assert trainCampaign(**options, resume=True) == 0
    assert len(calls) == count
    identity["configuration"]["seed"] = 43
    with pytest.raises(ValueError, match="changed"):
        trainCampaign(**options, resume=True)


def testWorkersUseIsolatedOfflineProcesses(tmp_path, monkeypatch):
    import sys

    from futsal.training.objectCampaign import runWorker

    calls = []

    class Process:
        def __init__(self, command, **kwargs):
            calls.append((command, kwargs))

        def wait(self):
            return 0

    monkeypatch.setattr("futsal.training.objectCampaign.subprocess.Popen", Process)
    payload = {"action": "environment", "device": "cpu"}
    script = Path("scripts/runObjectCurriculum.py")
    assert runWorker(script, Path.cwd(), tmp_path, payload, "worker0001") == 0
    assert calls[0][0] == [sys.executable, str(script), "worker", "--request", str(tmp_path / "worker0001.json")]
    assert calls[0][1]["env"]["YOLO_OFFLINE"] == "true"
    assert calls[0][1]["env"]["YOLO_AUTOINSTALL"] == "false"
    assert readJson(tmp_path / "worker0001.json") == payload
