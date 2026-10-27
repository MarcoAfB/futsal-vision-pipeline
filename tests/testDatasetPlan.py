import copy

import pytest

from futsal.cli import main
from futsal.datasetFiles import readJson, stagedDirectory, verifyEvidence
from futsal.datasetPlan import buildDatasetPlan, loadDatasetPlan, validatePlan, writeDatasetPlan


def testPlanBindsSourcesAndPreservesEveryTestClass(preparedSources):
    dataRoot, configPath, outputDir = preparedSources
    plan = buildDatasetPlan(configPath, dataRoot)
    assert plan["ownSplitSummaries"]["unified25"]["test"]["missingClasses"] == []
    assert plan["ownSplitSummaries"]["unified25"]["validation"]["missingClasses"] == ["lowerCornerSideB"]
    assert plan["frozenSplit"]["reviewedFrameCount"] == 6
    path = writeDatasetPlan(plan, outputDir, dataRoot, [configPath])
    assert loadDatasetPlan(path)["planId"] == plan["planId"]
    assert not list(outputDir.rglob("*.png"))
    verifyEvidence(dataRoot, plan["sourceEvidence"])
    broken = copy.deepcopy(plan)
    broken["samples"][0]["frameIndex"] = 1000
    with pytest.raises(ValueError, match="checksum"):
        validatePlan(broken)
    (dataRoot / "futsal4400/image.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Input changed"):
        verifyEvidence(dataRoot, plan["sourceEvidence"])


def testPlanCommandRefusesSourcesAndExistingDestinations(preparedSources):
    dataRoot, configPath, outputDir = preparedSources
    for destination in (dataRoot / "output", configPath.parent):
        with pytest.raises(SystemExit) as error:
            main(["planDataset", "--dataRoot", str(dataRoot), "--config", str(configPath),
                  "--outputDir", str(destination)])
        assert error.value.code == 2
    assert main(["planDataset", "--dataRoot", str(dataRoot), "--config", str(configPath),
                 "--outputDir", str(outputDir)]) == 0
    assert readJson(outputDir / "plan.json")["kind"] == "curriculumDatasetPlan"


def testFailedPublicationLeavesNoDatasetOrTemporaryDirectory(tmp_path):
    source = tmp_path / "sources"
    source.mkdir()
    destination = tmp_path / "output"
    with pytest.raises(OSError, match="interrupted"):
        with stagedDirectory(destination, source) as staging:
            (staging / "partial.txt").write_text("partial", encoding="utf-8")
            raise OSError("interrupted")
    assert not destination.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["sources"]
