import copy
from pathlib import Path

import pytest

from futsal.datasets.datasetFiles import readJson
from futsal.datasets.datasetProjection import materializeProjection
from futsal.datasets.syntheticCourt import readSyntheticManifest, verifySyntheticCourt
from futsal.training.trainingConfig import loadTrainingConfig
from futsal.training.trainingData import copyRuntimeData, selectTrainingData


def testSupplementChangesOnlyCourtTrainingSelection(projectionBundle, syntheticSource, tmp_path):
    parent, projection, _, _ = projectionBundle
    data = tmp_path / "data"
    materializeProjection(parent, projection, data, progress=lambda message: None)
    recipe = readJson(data / "plan.json")["recipe"]
    config = loadTrainingConfig(Path("configs/training/courtPoints.yaml"), recipe)
    real = selectTrainingData(data, config)
    snapshot = verifySyntheticCourt(syntheticSource), readSyntheticManifest(syntheticSource)
    mixedConfig = {**config, "sourcePolicy": "realWithSyntheticCourt"}
    mixed = selectTrainingData(data, mixedConfig, snapshot)
    task = mixed["tasks"]["courtPoints"]
    assert task["validation"] == real["tasks"]["courtPoints"]["validation"]
    assert [r for r in task["training"] if r["origin"] != "synthetic"] == real["tasks"][
        "courtPoints"
    ]["training"]
    assert len(task["training"]) == len(real["tasks"]["courtPoints"]["training"]) + 1
    runtime = tmp_path / "run"
    runtime.mkdir()
    copyRuntimeData(data, runtime, mixed, syntheticDir=syntheticSource)
    assert len(list((runtime / "runtimeData/courtPoints/images/train").glob("*.jpg"))) == 1
    assert not (runtime / "runtimeData/debug").exists()
    with pytest.raises(ValueError, match="refuses"):
        selectTrainingData(data, config, snapshot)
    with pytest.raises(ValueError, match="previously generated"):
        selectTrainingData(data, mixedConfig)
    objects = loadTrainingConfig(Path("configs/training/detailedObjects.yaml"), recipe)
    with pytest.raises(ValueError, match="only.*courtPoints"):
        selectTrainingData(data, {**objects, "sourcePolicy": "realWithSyntheticCourt"}, snapshot)
    bad = copy.deepcopy(snapshot)
    bad[1][0]["split"] = "validation"
    with pytest.raises(ValueError, match="membership"):
        selectTrainingData(data, mixedConfig, bad)
