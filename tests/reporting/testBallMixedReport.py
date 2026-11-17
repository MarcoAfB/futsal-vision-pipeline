import pytest

from futsal.datasets.datasetFiles import readJson, verifyEvidence, writeJson
from futsal.inventory.catalog import hashFile
from futsal.reporting.researchReport import mixedSourceMarkdown, renderTrainingReport
from futsal.training.trainingConfig import loadTrainingConfig
from futsal.training.trainingData import selectTrainingData


@pytest.mark.parametrize("viewFixture", ["ballMixedView", "playerMixedView", "ballPlayerMixedView"])
def testMixedCompositionShowsSourcesAndValidNegativesWithoutModelImports(request, tmp_path, monkeypatch, viewFixture):
    import builtins

    view, _, profile = request.getfixturevalue(viewFixture)
    config = loadTrainingConfig(profile, readJson(view / "plan.json")["recipe"])
    selection = selectTrainingData(view, config)
    runDir = tmp_path / "failed"
    runDir.mkdir()
    writeJson(runDir / "selection.json", selection)
    writeJson(runDir / "run.json", {"purpose": "scientificTraining", "status": "failed", "stages": [],
        "configuration": config, "batchRecovery": "fixed", "selectionSha256": hashFile(runDir / "selection.json")})
    originalImport = builtins.__import__

    def withoutModel(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "ultralytics"}:
            raise AssertionError("Report rendering imported a model backend")
        return originalImport(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", withoutModel)
    report = renderTrainingReport(runDir, tmp_path / "report")
    text = report.read_text()
    assert "Positive images" in text and "Negative images" in text
    assert "| own | 2 |" in text and "| public | 3 |" in text
    assert "Research evaluation: **not executed**" in text
    state = readJson(report.parent / "reportState.json")
    verifyEvidence(report.parent, state["artifacts"])
    assert readJson(report.parent / "trainingSelection.json") == selection
    assert mixedSourceMarkdown([{"sourceId": "legacy"}]) == []
