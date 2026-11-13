import pytest
import yaml

from futsal.datasets.datasetFiles import readJson
from futsal.training.trainingRun import runTraining


@pytest.mark.trainingIntegration
@pytest.mark.parametrize("viewFixture,counts", [("objectCurriculumView", [2, 3, 5]), ("playerCurriculumView", [1, 2, 4])])
def testNativeObjectCurriculumTransfersTwoThreeFiveWithFreshOptimizers(request, tmp_path, monkeypatch, viewFixture, counts):
    from futsal.training.modelWeights import prepareWeights
    from futsal.training.trainingBackend import initializeBackend

    view, store, profile = request.getfixturevalue(viewFixture)
    config = yaml.safe_load(profile.read_text())
    config.update(device="cpu", evaluationDevice="cpu")
    config["parameters"].update(imgsz=64, batch=1, workers=0, nbs=2)
    for stage in config["stages"]:
        stage["epochs"] = 1
    profile.write_text(yaml.safe_dump(config))
    initializeBackend(tmp_path / "setup", 2)
    import torch
    from ultralytics.nn.tasks import DetectionModel, load_checkpoint

    def offlineWeights(url, path):
        torch.save({"model": DetectionModel("yolo26n.yaml", nc=80, verbose=False)}, path)

    def rejectNetwork(*args, **kwargs):
        pytest.fail("Native curriculum must remain offline")

    monkeypatch.setattr("futsal.training.modelWeights.downloadWeights", offlineWeights)
    monkeypatch.setattr("urllib.request.urlopen", rejectNetwork)
    monkeypatch.setattr("requests.sessions.Session.request", rejectNetwork)
    prepareWeights("n", tmp_path / "weights")
    output = tmp_path / "native"
    run = runTraining(view, profile, tmp_path / "weights/yolo26n.pt", output,
                      batchRecovery="fixed", mediaStoreDir=store)
    assert run["status"] == "completed", run.get("error")
    assert [stage["results"]["checkpoints"]["best"]["classCount"] for stage in run["stages"]] == counts
    for index, stage in enumerate(run["stages"]):
        checkpoint = stage["results"]["checkpoints"]["best"]
        model, _ = load_checkpoint(output / checkpoint["path"], device="cpu", fuse=False)
        assert model.names == dict(enumerate(readJson(output / "selection.json")["tasks"][stage["taskId"]]["classes"]))
        assert model.model[-1].nc == counts[index] and model.end2end
        optimizer = readJson(output / stage["directory"] / "optimizer.json")
        assert optimizer["initialStateEntries"] == 0 and not optimizer["resume"]
        assert stage["results"]["training"]["effectiveBatch"] == 1
        audit = readJson(output / stage["directory"] / "transferAudit.json")
        assert audit["verifiedCompatibleTensorCount"] > 0
        assert set(audit["newClasses"]) == set(checkpoint["names"]) - set(audit["sourceNames"].values())
        if index:
            assert stage["inputWeights"]["sha256"] == run["stages"][index - 1]["results"]["checkpoints"]["best"]["sha256"]
    from futsal.evaluation.evaluationRun import evaluateRun

    evaluation = evaluateRun(output, view, "configs/evaluation/researchValidation.yaml", "runValidation",
                             tmp_path / "evaluation", mediaStoreDir=store)
    assert evaluation["status"] == "completed", evaluation.get("error")
    assert [len(stage["classes"]) for stage in evaluation["stages"]] == counts
