import json
import math

import pytest
import yaml

from futsal.datasets.datasetFiles import readJson
from futsal.training.trainingRun import runTraining


@pytest.mark.trainingIntegration
def testNativeMixedBallPlayerUsesEveryImageOnceWithoutDroppingPartialBatch(ballPlayerMixedView, tmp_path, monkeypatch):
    from futsal.training.modelWeights import prepareWeights
    from futsal.training.trainingBackend import initializeBackend

    view, store, profile = ballPlayerMixedView
    config = yaml.safe_load(profile.read_text())
    config.update(device="cpu", evaluationDevice="cpu")
    config["parameters"].update(imgsz=64, batch=2, workers=0, nbs=2)
    config["stages"][0]["epochs"] = 2
    profile.write_text(yaml.safe_dump(config))
    initializeBackend(tmp_path / "setup", 2)
    import torch
    from ultralytics.nn.tasks import DetectionModel, load_checkpoint

    def fixtureWeights(url, path):
        torch.save({"model": DetectionModel("yolo26n.yaml", nc=80, verbose=False)}, path)

    def rejectNetwork(*args, **kwargs):
        pytest.fail("Native mixed training must remain offline")

    monkeypatch.setattr("futsal.training.modelWeights.downloadWeights", fixtureWeights)
    monkeypatch.setattr("urllib.request.urlopen", rejectNetwork)
    monkeypatch.setattr("requests.sessions.Session.request", rejectNetwork)
    prepareWeights("n", tmp_path / "weights")
    output = tmp_path / "native"
    run = runTraining(view, profile, tmp_path / "weights/yolo26n.pt", output, batchRecovery="fixed", mediaStoreDir=store)
    assert run["status"] == "completed", run.get("error")
    assert len(run["stages"]) == 1
    stage = run["stages"][0]
    assert stage["taskId"] == "ballPlayer"
    training = stage["results"]["training"]
    assert training["sampling"] == {"policy": "naturalMixedSources", "shuffle": True, "replacement": False,
        "dropLast": False, "imagesPerEpoch": 5, "verifiedEpochs": 2}
    assert training["batches"] == 2 * math.ceil(5 / 2)
    assert training["effectiveBatch"] == 2
    epochs = [json.loads(line) for line in (output / stage["directory"] / "exposure.jsonl").read_text().splitlines()]
    assert len(epochs) == 2
    for row in epochs:
        assert row["coverageVerified"] and row["imageCount"] == row["uniqueRecordCount"] == 5
        assert sum(source["negativeImages"] for source in row["bySource"]) == 0
        assert sum(source["targetCount"] for source in row["bySource"]) == 15
        assert {source["origin"] for source in row["bySource"]} == {"own", "public"}
    checkpoint = stage["results"]["checkpoints"]["best"]
    model, _ = load_checkpoint(output / checkpoint["path"], device="cpu", fuse=False)
    assert model.names == {0: "ball", 1: "player"} and model.model[-1].nc == 2 and model.end2end
    optimizer = readJson(output / stage["directory"] / "optimizer.json")
    assert optimizer["initialStateEntries"] == 0 and not optimizer["resume"]
    from futsal.evaluation.evaluationRun import evaluateRun

    evaluated = evaluateRun(output, view, "configs/evaluation/researchValidation.yaml", "runValidation",
                            tmp_path / "evaluation", mediaStoreDir=store)
    assert evaluated["status"] == "completed", evaluated.get("error")
    assert evaluated["stages"][0]["imageCount"] == 2
    assert (tmp_path / "evaluation/evidence/ballPlayer/exposure.jsonl").is_file()
    detection = readJson(tmp_path / "evaluation/evidence/ballPlayer/detection.json")
    assert {"ball", "athletes"} <= set(detection["groups"])
