from types import SimpleNamespace

import pytest

from futsal.datasets.datasetFiles import readJson
from futsal.training.trainingBackend import initializeBackend, validationMetrics, verifyLoader
from futsal.training.trainingData import copyRuntimeData, selectTrainingData


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
@pytest.mark.parametrize("modelSize,scientific,shared", [
    ("n", False, False), ("n", True, False), ("n", False, True), ("n", True, True),
    ("s", True, True),
])
def testNativeCpuStagesProduceTwoThreeAndTwentyFiveClassCheckpoints(trainingBundle, tmp_path,
                                                                  monkeypatch, scientific, shared, preparedSources, modelSize):
    from futsal.training.modelWeights import prepareWeights
    from futsal.training.trainingRun import runTraining

    bundle, configFile, config = trainingBundle
    options = {}
    if shared:
        from futsal.datasets.mediaRegistration import registerMedia
        from futsal.datasets.sharedViews import prepareView

        store, view = tmp_path / "mediaStore", tmp_path / "sharedView"
        registerMedia(store, bundle)
        prepareView(store, preparedSources[1], view)
        bundle = view
        options["mediaStoreDir"] = store
    if scientific:
        import yaml

        from futsal.training.trainingConfig import loadTrainingConfig

        config = loadTrainingConfig("configs/training/yolo26CurriculumPilot.yaml", readJson(bundle / "plan.json")["recipe"])
        config.update(device="cpu", evaluationDevice="cpu")
        config["parameters"].update(imgsz=64, batch=1, workers=0, nbs=2)
        for stage in config["stages"]:
            stage["epochs"] = 3
        configFile.write_text(yaml.safe_dump(config))
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
        torch.save({"model": DetectionModel(f"yolo26{modelSize}.yaml", nc=80, verbose=False)}, destination)

    monkeypatch.setattr("futsal.training.modelWeights.downloadWeights", fixtureDownload)
    prepareWeights(modelSize, tmp_path / "initial")
    run = runTraining(bundle, configFile, tmp_path / "initial" / f"yolo26{modelSize}.pt", runDir,
                      modelSize=modelSize, **options)
    assert run["status"] == "completed", run.get("error")
    assert run["runtimeIntegrityAfterRun"]
    assert run["bundleIntegrityAfterRun"]["hashesVerified"]
    for index, stage in enumerate(run["stages"]):
        result = stage["results"]
        stageDir = runDir / stage["directory"]
        assert result["checkpoints"]["best"]["classCount"] == (2, 3, 25)[index]
        assert result["checkpoints"]["best"]["end2end"]
        assert readJson(stageDir / "optimizer.json")["initialStateEntries"] == 0
        assert result["training"]["completedEpochs"] == (3 if scientific else 1)
        assert 1 <= result["training"]["bestEpoch"] <= result["training"]["completedEpochs"]
        if scientific:
            import json

            history = [json.loads(line) for line in (stageDir / "epochs.jsonl").read_text().splitlines()]
            assert history[0]["learningRates"][0] > history[-1]["learningRates"][0]
            assert readJson(stageDir / "effectiveArguments.json")["nbs"] == 2
            assert result["interpretation"] == "scientificPilot"
    assert result["perClass"]["lowerCornerSideB"]["ap50"] is None
    assert networkCalls == []
    from futsal.evaluation.evaluationRun import evaluateRun

    evaluationDir = tmp_path / "researchEvaluation"
    evaluation = evaluateRun(runDir, bundle, "configs/evaluation/researchValidation.yaml",
                             "runValidation", evaluationDir, **options)
    assert evaluation["status"] == "completed", evaluation.get("error")
    from futsal.evaluation.evaluationBackend import predictValidation

    final = evaluation["stages"][-1]
    checkpoint = runDir / run["stages"][-1]["results"]["checkpoints"]["best"]["path"]
    with pytest.raises(ValueError, match="recorded model size"):
        predictValidation(checkpoint, {"taskId": "unified25", "classes": final["classes"]}, [], {},
                          {**run["configuration"], "modelSize": "x"}, {}, tmp_path / "unused")
    assert evaluation["runtimeAfter"] and evaluation["bundleAfter"]["hashesVerified"]
    for stage in evaluation["stages"]:
        evidenceDir = evaluationDir / stage["evidenceDirectory"]
        assert readJson(evidenceDir / "detection.json")["imageCount"] == 2
        scores = readJson(evidenceDir / "detection.json")["groups"]["all"]
        assert scores["ap50To95"] == pytest.approx(stage["nativeValidationMetrics"]["metrics/mAP50-95(B)"], abs=1e-12)
        assert len(stage["recordIds"]) == 2
    point = readJson(evaluationDir / "evidence" / "unified25" / "points.json")
    unsupported = next(row for row in point["perClass"] if row["className"] == "lowerCornerSideB")
    assert unsupported["successRates"]["0.01"] is None
    assert networkCalls == []
    full = evaluateRun(runDir, bundle, "configs/evaluation/researchValidation.yaml",
                       "fullValidation", tmp_path / "fullValidation", **options)
    assert full["status"] == "completed", full.get("error")
    assert full["selection"] == "fullValidation"
    assert all(stage["imageCount"] == 2 for stage in full["stages"])
    if shared:
        assert not list((runDir / "runtimeData").rglob("*.png"))
