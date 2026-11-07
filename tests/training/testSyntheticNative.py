from pathlib import Path

import pytest
import yaml

from futsal.datasets.datasetFiles import readJson
from futsal.datasets.datasetProjection import materializeProjection


@pytest.mark.trainingIntegration
def testNativeMixedCourtUsesExactSupplementAndFreshOptimizer(
    projectionBundle, syntheticSource, tmp_path, monkeypatch
):
    from futsal.evaluation.evaluationRun import evaluateRun
    from futsal.training.modelWeights import prepareWeights
    from futsal.training.trainingBackend import initializeBackend
    from futsal.training.trainingRun import runTraining

    parent, projection, _, _ = projectionBundle
    data = tmp_path / "data"
    materializeProjection(parent, projection, data, progress=lambda message: None)
    initializeBackend(tmp_path / "setup", 2)
    import torch
    from ultralytics.nn.tasks import DetectionModel

    def rejectNetwork(*args, **kwargs):
        raise AssertionError("Native software fixtures cannot access the network.")

    monkeypatch.setattr("requests.sessions.Session.request", rejectNetwork)
    monkeypatch.setattr("urllib.request.urlopen", rejectNetwork)
    monkeypatch.setattr(
        "futsal.training.modelWeights.downloadWeights",
        lambda url, destination: torch.save(
            {"model": DetectionModel("yolo26n.yaml", nc=80, verbose=False)}, destination
        ),
    )
    prepareWeights("n", tmp_path / "weights")
    config = yaml.safe_load(Path("configs/training/courtPoints.yaml").read_text())
    config.update(device="cpu", evaluationDevice="cpu")
    config["parameters"].update(imgsz=64, batch=1, nbs=2, workers=0, warmup_epochs=0.0, patience=0)
    config["stages"][0]["epochs"] = 2
    path = tmp_path / "court.yaml"
    path.write_text(yaml.safe_dump(config))
    runDir = tmp_path / "mixed"
    run = runTraining(
        data,
        path,
        tmp_path / "weights/yolo26n.pt",
        runDir,
        syntheticDir=syntheticSource,
        progress=lambda message: None,
    )
    assert run["status"] == "completed", run.get("error")
    stage = run["stages"][0]
    assert stage["results"]["checkpoints"]["best"]["classCount"] == 20
    assert stage["results"]["training"]["optimizerUpdates"] > 0
    assert stage["results"]["training"]["batches"] > 0
    assert readJson(runDir / stage["directory"] / "optimizer.json")["initialStateEntries"] == 0
    selection = readJson(runDir / "selection.json")["tasks"]["courtPoints"]
    audit = readJson(runDir / stage["directory"] / "loaderAudit.json")
    assert audit["training"]["imageCount"] == len(selection["training"])
    assert sum(r["origin"] == "synthetic" for r in selection["training"]) == 1
    result = evaluateRun(
        runDir,
        data,
        "configs/evaluation/researchValidation.yaml",
        "runValidation",
        tmp_path / "validation",
        progress=lambda message: None,
        workRoot=tmp_path / "work",
    )
    assert result["status"] == "completed", result.get("error")
    assert result["stages"][0]["imageCount"] == len(selection["validation"])
