import pytest

from futsal.datasets.datasetFiles import hashJson, readJson, writeJson
from futsal.evaluation.configurationComparison import (
    compareConfigurations,
    delta,
    validateComparison,
)
from futsal.evaluation.evaluationRun import sealEvaluation
from futsal.inventory.catalog import hashFile
from futsal.reporting.researchReport import verifyEvaluation


def testComparisonUsesIdenticalSealedCohortsAndRetainsUnsupportedScores(comparisonEvaluations, tmp_path):
    baseline, candidate = comparisonEvaluations
    before = [hashFile(root / "evaluation.json") for root in (baseline, candidate)]
    output = tmp_path / "comparison"
    state = compareConfigurations(baseline, candidate, output)
    assert state["status"] == "completed" and state["singleSeed"] == 42
    assert verifyEvaluation(output)[0] == state
    metrics = readJson(output / "evidence/comparisonMetrics.json")
    unsupported = [row for row in metrics["points"] if row["className"] == "lowerCornerSideB" and row["metric"] == "successRate"]
    assert unsupported[0]["baseline"] is None and unsupported[0]["candidate"] is None and unsupported[0]["delta"] is None
    assert metrics["pointSuccess"]["delta"] == 0.0
    assert before == [hashFile(root / "evaluation.json") for root in (baseline, candidate)]
    assert delta(0.2, 0.5) == pytest.approx(0.3)
    assert delta(None, 0.5) is None


@pytest.mark.parametrize("mutation", ["protocol", "references", "seed", "incomplete", "corruption"])
def testComparisonRejectsIncompatibleOrIncompleteScientificEvidence(comparisonEvaluations, tmp_path, mutation):
    from futsal.evaluation.evaluationData import readJsonLines, writeJsonLines

    baseline, candidate = comparisonEvaluations
    state = readJson(baseline / "evaluation.json")
    if mutation == "protocol":
        state["protocol"]["operationalConfidence"] = 0.3
        state["protocolId"] = hashJson(state["protocol"])
    elif mutation == "references":
        path = baseline / "evidence/unified25/references.jsonl"
        refs = readJsonLines(path)
        refs[0]["targets"][0]["box"][0] += 1
        writeJsonLines(path, refs)
    elif mutation == "seed":
        state["training"]["configuration"]["seed"] = 43
    elif mutation == "incomplete":
        state["status"] = "failed"
    else:
        (baseline / "evidence/unified25/detection.json").write_text("changed")
    if mutation != "corruption":
        sealEvaluation(baseline, state)
    with pytest.raises(ValueError):
        validateComparison(baseline, candidate)
    with pytest.raises(ValueError):
        compareConfigurations(baseline, candidate, tmp_path / "mustNotExist")
    assert not (tmp_path / "mustNotExist").exists()


@pytest.mark.reportingIntegration
def testComparisonRegeneratesFromSealedEvidenceWithoutModelPackages(comparisonEvaluations, tmp_path, monkeypatch):
    import sys

    from futsal.reporting.comparisonReport import renderComparisonReport

    baseline, candidate = comparisonEvaluations
    output = tmp_path / "comparison"
    compareConfigurations(baseline, candidate, output)
    before = hashFile(output / "evaluationContract.json")
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    report = renderComparisonReport(output)
    text = report.read_text(encoding="utf-8")
    assert "N/A" in text and "do not isolate a curriculum effect" in text
    regenerated = renderComparisonReport(output, tmp_path / "regenerated")
    assert regenerated.is_file() and hashFile(output / "evaluationContract.json") == before
    assert list((tmp_path / "regenerated/examples").glob("*Baseline.png"))


@pytest.mark.parametrize("configurationEvaluation", [("n", "s")], indirect=True)
def testControlledComparisonChecksCourtScaleAsWellAsObjects(comparisonEvaluations, tmp_path):
    baseline, candidate = comparisonEvaluations
    state = readJson(candidate / "evaluation.json")
    court = state["components"][1]["run"]
    with pytest.raises(ValueError, match="same model scale"):
        compareConfigurations(baseline, candidate, tmp_path / "controlled")
    output = tmp_path / "variants"
    result = compareConfigurations(baseline, candidate, output, comparisonMode="modelVariants")
    assert result["status"] == "completed"
    assert result["comparisonMode"] == "modelVariants"
    assert "modelSize" in result["configurationChanges"]
    assert [item["modelSize"] for item in result["modelVariants"]["candidate"]] == ["n", "s"]
    court["configuration"]["seed"] += 1
    writeJson(candidate / "evidence/components/courtPoints/run.json", court)
    sealEvaluation(candidate, state)
    with pytest.raises(ValueError, match="same seed"):
        compareConfigurations(baseline, candidate, tmp_path / "seedMismatch", comparisonMode="modelVariants")


@pytest.mark.reportingIntegration
@pytest.mark.parametrize("mode,configurationEvaluation", [
    ("controlled", ("n", "n")), ("modelVariants", ("n", "s")),
], indirect=["configurationEvaluation"])
def testComparisonReportRestoresArchivedMode(comparisonEvaluations, tmp_path, monkeypatch, mode):
    import sys

    from futsal.reporting.researchReport import renderReport

    baseline, candidate = comparisonEvaluations
    output = tmp_path / "comparison"
    result = compareConfigurations(baseline, candidate, output, comparisonMode=mode)
    if mode == "controlled":
        for key in ("comparisonMode", "modelVariants", "configurationChanges"):
            del result[key]
        sealEvaluation(output, result)
    before = hashFile(output / "evaluationContract.json")
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    text = renderReport(output).read_text(encoding="utf-8")
    assert f"Comparison mode: **{mode}**" in text
    assert "yolo26n" in text
    if mode == "modelVariants":
        assert "yolo26s" in text and "do not isolate the effect" in text
    assert renderReport(output, tmp_path / "regenerated").is_file()
    assert hashFile(output / "evaluationContract.json") == before
