from pathlib import Path

import pytest

from futsal.datasets.datasetFiles import readJson, verifyEvidence, writeJson
from futsal.training.trainingConfig import loadTrainingConfig
from futsal.training.trainingData import copyRuntimeData, selectRows, selectTrainingData


def testSharedSelectionCopiesBytesWithoutReservedTestOrLinks(trainingBundle, tmp_path):
    bundle, _, config = trainingBundle
    selection = selectTrainingData(bundle, config)
    reordered = readJson(bundle / "manifest.json")
    testIds = {sample["recordId"] for sample in readJson(bundle / "plan.json")["samples"]
               if sample["origin"] == "own" and sample["matchId"] == "matchC"}
    assert not testIds & set(selection["validationRecordIds"])
    for task in selection["tasks"].values():
        assert [row["recordId"] for row in task["validation"]] == selection["validationRecordIds"]
        assert all(task["trainingSupport"].values())
    assert selection["tasks"]["unified25"]["validationSupport"]["lowerCornerSideB"] == 0
    runDir = tmp_path / "run"
    runDir.mkdir()
    evidence = copyRuntimeData(bundle, runDir, selection)
    label = next(item for item in evidence if "/labels/" in item["path"])
    runtime = runDir / label["path"]
    original = bundle / Path(label["path"]).relative_to("runtimeData")
    assert not runtime.samefile(original)
    runtime.write_text("changed")
    assert original.read_text() != "changed"
    with pytest.raises(ValueError, match="changed"):
        verifyEvidence(runDir, evidence)
    assert reordered == readJson(bundle / "manifest.json")


def testGreedySelectionIsOrderIndependentAndRejectsImpossibleCoverage():
    rows = [{"recordId": str(i), "targets": [{"classId": i}]} for i in range(3)]
    assert selectRows(rows, 3, 42, "test") == selectRows(rows[::-1], 3, 42, "test")
    with pytest.raises(ValueError, match="coverage"):
        selectRows(rows, 2, 42, "test")
    with pytest.raises(ValueError, match="fewer"):
        selectRows(rows, 4, 42, "test")


def testTrainingConfigurationRejectsUnsafeOverridesAndStageOrder(trainingBundle):
    import yaml

    bundle, configFile, config = trainingBundle
    recipe = readJson(bundle / "plan.json")["recipe"]
    config["parameters"]["resume"] = True
    configFile.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="parameters"):
        loadTrainingConfig(configFile, recipe)
    del config["parameters"]["resume"]
    config["stages"].reverse()
    configFile.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="curriculum order"):
        loadTrainingConfig(configFile, recipe)


def testScientificSelectionKeepsFullTaskCohortsAndRejectsSyntheticProvenance(trainingBundle):
    bundle, _, _ = trainingBundle
    config = loadTrainingConfig("configs/training/yolo26Curriculum.yaml", readJson(bundle / "plan.json")["recipe"])
    manifest = readJson(bundle / "manifest.json")
    removed = next(row for row in manifest["samples"] if row["taskId"] == "unified25" and row["split"] == "validation")
    manifest["samples"].remove(removed)
    writeJson(bundle / "manifest.json", manifest)
    selection = selectTrainingData(bundle, config)
    assert len(selection["tasks"]["genericObjects"]["validation"]) == 2
    assert len(selection["tasks"]["unified25"]["validation"]) == 1
    assert len(selection["commonValidationRecordIds"]) == 1
    assert selection["sourcePolicy"]["provenanceVerified"]
    manifest["samples"][0]["origin"] = "synthetic"
    writeJson(bundle / "manifest.json", manifest)
    with pytest.raises(ValueError, match="provenance"):
        selectTrainingData(bundle, config)


def testScientificConfigurationRejectsSubsetsAndUnsafeStageParameters(trainingBundle):
    import yaml

    bundle, configFile, _ = trainingBundle
    recipe = readJson(bundle / "plan.json")["recipe"]
    config = loadTrainingConfig("configs/training/yolo26CurriculumPilot.yaml", recipe)
    assert [stage["epochs"] for stage in config["stages"]] == [2, 2, 2]
    config["trainSampleCount"] = 32
    configFile.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="full per-task"):
        loadTrainingConfig(configFile, recipe)
    config["trainSampleCount"] = None
    config["stages"][0]["parameters"]["fliplr"] = 0.5
    configFile.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="override only"):
        loadTrainingConfig(configFile, recipe)
