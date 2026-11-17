import json
from pathlib import Path

import pytest
import yaml

from futsal.datasets.datasetFiles import readJson
from futsal.datasets.mediaRegistration import registerMedia
from futsal.datasets.sharedViews import prepareView
from futsal.training.trainingBackend import initializeBackend


@pytest.mark.trainingIntegration
@pytest.mark.parametrize("profileName", ["courtPointsMAugmented", "courtPointsMPointSelection"])
def testCourtProfilesTrainWithDeclaredAugmentationAndPointSelection(projectionBundle, syntheticSource, tmp_path,
                                                                    monkeypatch, profileName):
    import torch
    from ultralytics.nn.tasks import DetectionModel

    from futsal.evaluation.evaluationRun import evaluateRun
    from futsal.reporting.researchReport import renderReport
    from futsal.training.modelWeights import prepareWeights
    from futsal.training.trainingRun import runTraining

    parent, supplement, recipe, _ = projectionBundle
    store, view = tmp_path / "store", tmp_path / "view"
    registerMedia(store, parent)
    registerMedia(store, supplement)
    prepareView(store, recipe, view)
    initializeBackend(tmp_path / "setup", 2)
    monkeypatch.setattr("futsal.training.modelWeights.downloadWeights", lambda url, path: torch.save(
        {"model": DetectionModel("yolo26n.yaml", nc=80, verbose=False)}, path))
    prepareWeights("n", tmp_path / "weights")
    config = yaml.safe_load(Path(f"configs/training/{profileName}.yaml").read_text(encoding="utf-8"))
    config.update(modelSize="n", device="cpu", evaluationDevice="cpu")
    config["parameters"].update(imgsz=64, batch=1, nbs=2, workers=0, warmup_epochs=0.0, patience=0)
    config["stages"][0]["epochs"] = 2
    profile = tmp_path / "court.yaml"
    profile.write_text(yaml.safe_dump(config), encoding="utf-8")
    runDir = tmp_path / "run"
    run = runTraining(view, profile, tmp_path / "weights/yolo26n.pt", runDir, mediaStoreDir=store,
                      syntheticDir=syntheticSource, batchRecovery="fixed")
    assert run["status"] == "completed", run.get("error")
    stage = run["stages"][0]
    stageDir = runDir / stage["directory"]
    training = stage["results"]["training"]
    assert training["selectionCriterion"] == "nativeValidationPointSuccessMacro"
    epochs = [json.loads(line) for line in (stageDir / "epochs.jsonl").read_text().splitlines()]
    assert len(epochs) == 2
    assert all(row["fitness"] == row["pointSelection"]["macro"] and "boxFitness" in row for row in epochs)
    audit = readJson(stageDir / "augmentationAudit.json")
    assert training["augmentation"] == audit
    if profileName == "courtPointsMAugmented":
        assert audit["policy"] == "courtSideSwapLight" and audit["exemptRecords"] == 1
        assert audit["trainingRecords"] == len(readJson(runDir / "selection.json")["tasks"]["courtPoints"]["training"])
        assert audit["horizontalFlipClassMap"]["upperCornerSideA"] == "upperCornerSideB"
        assert audit["horizontalFlipClassMap"]["substitutionPoint1"] == "substitutionPoint4"
    else:
        assert audit == {"policy": None}
    output = tmp_path / "evaluation"
    evaluation = evaluateRun(runDir, view, "configs/evaluation/researchValidation.yaml", "runValidation", output,
                             workRoot=tmp_path / "work", mediaStoreDir=store)
    assert evaluation["status"] == "completed", evaluation.get("error")
    report = renderReport(output).read_text(encoding="utf-8")
    assert "Checkpoint selection and early stopping use macro native point success" in report
    assert ("policy courtSideSwapLight" if audit["policy"] else "Training augmentation: none.") in report
