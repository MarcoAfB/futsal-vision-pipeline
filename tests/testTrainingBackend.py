from types import SimpleNamespace

import pytest

from futsal.datasetFiles import readJson
from futsal.trainingBackend import initializeBackend, trainStage, validationMetrics, verifyLoader
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
def testNativeCpuStagesProduceTwoThreeAndTwentyFiveClassCheckpoints(trainingBundle, tmp_path):
    bundle, _, config = trainingBundle
    runDir = tmp_path / "nativeRun"
    runDir.mkdir()
    initializeBackend(runDir / "backend", config["numThreads"])
    import torch
    from ultralytics.nn.tasks import DetectionModel

    weights = runDir / "initial.pt"
    torch.save({"model": DetectionModel("yolo26n.yaml", nc=80, verbose=False)}, weights)
    selection = selectTrainingData(bundle, config)
    evidence = copyRuntimeData(bundle, runDir, selection)
    for index, stage in enumerate(config["stages"]):
        stageDir = runDir / f"stage{index + 1}"
        result = trainStage(config, selection["tasks"][stage["taskId"]], stage, weights,
                            stageDir, runDir, evidence)
        assert result["checkpoints"]["best"]["classCount"] == (2, 3, 25)[index]
        assert readJson(stageDir / "optimizer.json")["initialStateEntries"] == 0
        weights = runDir / result["checkpoints"]["best"]["path"]
    assert result["perClass"]["lowerCornerSideB"]["ap50"] is None
