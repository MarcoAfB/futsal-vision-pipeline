from types import SimpleNamespace

import pytest

from futsal.datasetFiles import readJson
from futsal.trainingBackend import initializeBackend, validationMetrics, verifyLoader
from futsal.trainingData import copyRuntimeData, selectTrainingData


def testUnsupportedMetricsAreNullEvenWhenBackendFillsAggregate():
    metrics = SimpleNamespace(ap_class_index=[0], class_result=lambda index: (0.1, 0.2, 0.3, 0.4),
                              results_dict={"mAP": 0.4})
    result = validationMetrics(metrics, ["ball", "landmark"], {"ball": 4, "landmark": 0})
    assert result["perClass"]["landmark"]["ap50To95"] is None
    assert result["perClass"]["ball"]["ap50To95"] == 0.4


def testLoaderRejectsSilentAnnotationDrops(trainingBundle, tmp_path):
    import numpy as np

    bundle, _, config = trainingBundle
    selection = selectTrainingData(bundle, config)
    runDir = tmp_path / "run"
    runDir.mkdir()
    copyRuntimeData(bundle, runDir, selection)
    task = selection["tasks"]["genericObjects"]
    row = task["training"][0]
    label = {"im_file": str(runDir / "runtimeData" / row["imagePath"]),
             "cls": np.array([[0]]), "bboxes": np.array([[0.5, 0.5, 0.1, 0.1]])}
    with pytest.raises(ValueError, match="discarded or changed"):
        verifyLoader(SimpleNamespace(labels=[label]), task, "training", runDir)


@pytest.mark.trainingIntegration
def testNativeCpuStagesProduceTwoThreeAndTwentyFiveClassCheckpoints(trainingBundle, tmp_path,
                                                                  monkeypatch):
    from futsal.modelWeights import prepareWeights
    from futsal.trainingRun import runTraining

    bundle, configFile, config = trainingBundle
    runDir = tmp_path / "nativeRun"
    initializeBackend(tmp_path / "modelSetup", config["numThreads"])
    import torch
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils import callbacks

    networkCalls = []

    def rejectNetwork(*arguments, **keywords):
        networkCalls.append(True)
        raise AssertionError("Native tests must stay offline.")

    monkeypatch.setattr("urllib.request.urlopen", rejectNetwork)
    monkeypatch.setattr("requests.sessions.Session.request", rejectNetwork)
    monkeypatch.setenv("ULTRALYTICS_API_KEY", "testKeyMustNeverBeUsed")
    originalCallbacks = callbacks.add_integration_callbacks

    def rejectPlatform(trainer):
        raise AssertionError("External callbacks must not execute.")

    rejectPlatform.__module__ = "ultralytics.utils.callbacks.platform"

    def addForbiddenCallback(instance):
        originalCallbacks(instance)
        instance.callbacks["on_pretrain_routine_start"].append(rejectPlatform)

    monkeypatch.setattr(callbacks, "add_integration_callbacks", addForbiddenCallback)

    def fixtureDownload(url, destination):
        torch.save({"model": DetectionModel("yolo26n.yaml", nc=80, verbose=False)}, destination)

    monkeypatch.setattr("futsal.modelWeights.downloadWeights", fixtureDownload)
    prepareWeights("n", tmp_path / "initial")
    run = runTraining(bundle, configFile, tmp_path / "initial" / "yolo26n.pt", runDir)
    assert run["status"] == "completed", run.get("error")
    assert run["runtimeIntegrityAfterRun"]
    assert run["bundleIntegrityAfterRun"]["hashesVerified"]
    for index, stage in enumerate(run["stages"]):
        result = stage["results"]
        stageDir = runDir / stage["directory"]
        assert result["checkpoints"]["best"]["classCount"] == (2, 3, 25)[index]
        assert result["checkpoints"]["best"]["end2end"]
        assert readJson(stageDir / "optimizer.json")["initialStateEntries"] == 0
    assert result["perClass"]["lowerCornerSideB"]["ap50"] is None
    assert networkCalls == []
