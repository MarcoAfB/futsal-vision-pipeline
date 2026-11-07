from pathlib import Path

import pytest
import yaml

from futsal.datasets.datasetFiles import readJson
from futsal.datasets.datasetProjection import materializeProjection
from futsal.inventory.catalog import hashFile
from futsal.training.trainingConfig import loadTrainingConfig


def testIndependentProfilesAreSingleStageAndShareDeclaredProtocol(projectionBundle):
    parent, projectionDir, _, _ = projectionBundle
    recipe = readJson(projectionDir / "plan.json")["recipe"]
    configurations = [loadTrainingConfig(Path(f"configs/training/{taskId}.yaml"), recipe)
                      for taskId in ("detailedObjects", "courtPoints")]
    assert configurations[0]["parameters"] == configurations[1]["parameters"]
    assert all(len(config["stages"]) == 1 and config["stages"][0]["epochs"] == 100 for config in configurations)
    assert all(config["sourcePolicy"] == "realOnly" and config["seed"] == 42 for config in configurations)
    assert readJson(parent / "contract.json")["contractId"] == readJson(projectionDir / "projection.json")["parentContractId"]


@pytest.mark.trainingIntegration
@pytest.mark.parametrize("shared", [False, True])
def testNativeIndependentModelsStartFromSameOfficialInput(projectionBundle, tmp_path, monkeypatch, shared):
    from futsal.training.modelWeights import prepareWeights
    from futsal.training.trainingBackend import initializeBackend
    from futsal.training.trainingRun import runTraining

    parent, projectionDir, _, _ = projectionBundle
    bundle = tmp_path / "data"
    options = {}
    if shared:
        from futsal.datasets.mediaRegistration import registerMedia
        from futsal.datasets.sharedViews import prepareView

        store = tmp_path / "store"
        registerMedia(store, parent)
        registerMedia(store, projectionDir)
        prepareView(store, projectionBundle[2], bundle)
        options["mediaStoreDir"] = store
    else:
        materializeProjection(parent, projectionDir, bundle, progress=lambda message: None)
    initializeBackend(tmp_path / "setup", 2)
    import torch
    from ultralytics.nn.tasks import DetectionModel

    def rejectNetwork(*args, **kwargs):
        raise AssertionError("Isolated native tests must not access the network.")

    monkeypatch.setattr("requests.sessions.Session.request", rejectNetwork)
    monkeypatch.setattr("urllib.request.urlopen", rejectNetwork)
    monkeypatch.setattr("futsal.training.modelWeights.downloadWeights", lambda url, destination:
                        torch.save({"model": DetectionModel("yolo26n.yaml", nc=80, verbose=False)}, destination))
    prepareWeights("n", tmp_path / "weights")
    weights = tmp_path / "weights/yolo26n.pt"
    original = hashFile(weights)
    for taskId, count in (("detailedObjects", 5), ("courtPoints", 20)):
        config = yaml.safe_load(Path(f"configs/training/{taskId}.yaml").read_text(encoding="utf-8"))
        config.update(device="cpu", evaluationDevice="cpu")
        config["parameters"].update(imgsz=64, batch=1, nbs=2, workers=0, warmup_epochs=0.0, patience=0)
        config["stages"][0]["epochs"] = 2
        configFile = tmp_path / f"{taskId}.yaml"
        configFile.write_text(yaml.safe_dump(config), encoding="utf-8")
        runDir = tmp_path / taskId
        run = runTraining(bundle, configFile, weights, runDir, progress=lambda message: None, **options)
        assert run["status"] == "completed", run.get("error")
        stage = run["stages"][0]
        assert stage["inputWeights"]["sha256"] == original
        assert stage["results"]["checkpoints"]["best"]["classCount"] == count
        assert stage["results"]["training"]["completedEpochs"] == 2
        assert readJson(runDir / stage["directory"] / "optimizer.json")["initialStateEntries"] == 0
    assert hashFile(weights) == original
    from futsal.evaluation.configurationRun import evaluateConfiguration
    from futsal.reporting.researchReport import renderReport

    output = tmp_path / "joint"
    state = evaluateConfiguration(tmp_path / "detailedObjects", tmp_path / "courtPoints", bundle, parent,
                                  "configs/evaluation/researchValidation.yaml", output, progress=lambda message: None, **options)
    assert state["status"] == "completed", state.get("error")
    if shared:
        assert not list((tmp_path / "detailedObjects/runtimeData").rglob("*.png"))
        assert not list((Path(state["workDirectory"]) / "runtimeData").rglob("*.png"))
    assert len(state["recordIds"]) == 2
    detection = readJson(output / "evidence/joint/detection.json")
    assert detection["groups"]["all"]["ap50To95"] == pytest.approx(state["nativeValidationMetrics"]["metrics/mAP50-95(B)"], abs=1e-12)
    assert next(row for row in detection["perClass"] if row["className"] == "lowerCornerSideB")["ap50To95"] is None
    points = readJson(output / "evidence/joint/points.json")
    assert next(row for row in points["perClass"] if row["className"] == "lowerCornerSideB")["successRates"]["0.01"] is None
    assert renderReport(output).is_file()
    before = hashFile(output / "evaluationContract.json")
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    text = renderReport(output, tmp_path / "regenerated").read_text(encoding="utf-8")
    assert "Two independent detectors" in text and "N/A" in text
    assert hashFile(output / "evaluationContract.json") == before
