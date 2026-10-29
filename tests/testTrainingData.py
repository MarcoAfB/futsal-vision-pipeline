from pathlib import Path

import pytest
import yaml

from futsal.datasetExport import prepareDataset
from futsal.datasetFiles import readJson, verifyEvidence
from futsal.datasetPlan import buildDatasetPlan, writeDatasetPlan
from futsal.trainingConfig import loadTrainingConfig
from futsal.trainingData import copyRuntimeData, selectRows, selectTrainingData


@pytest.fixture
def trainingBundle(preparedSources):
    dataRoot, configPath, planDir = preparedSources
    plan = buildDatasetPlan(configPath, dataRoot)
    planPath = writeDatasetPlan(plan, planDir, dataRoot)
    bundle = planDir.parent / "bundle"
    prepareDataset(planPath, dataRoot, bundle, progress=lambda message: None)
    configuration = yaml.safe_load(Path("configs/training/yolo26Smoke.yaml").read_text())
    configuration.update(trainSampleCount=1, validationSampleCount=2)
    configFile = planDir.parent / "smoke.yaml"
    configFile.write_text(yaml.safe_dump(configuration))
    config = loadTrainingConfig(configFile, plan["recipe"])
    return bundle, configFile, config


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
