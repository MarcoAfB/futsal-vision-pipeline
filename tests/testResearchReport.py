import csv
import shutil
from pathlib import Path

import pytest

from futsal.datasetFiles import readJson, verifyEvidence, writeJson
from futsal.evaluationData import buildReferences, loadEvaluationConfig, writeJsonLines
from futsal.evaluationMetrics import aggregateGroups, operationalMetrics
from futsal.evaluationRun import sealEvaluation
from futsal.pointMetrics import scorePoints
from futsal.researchExamples import fixedExamples, selectExamples
from futsal.researchReport import renderReport, renderTrainingReport, stageTables, verifyEvaluation


def testExampleSelectionIsStableAndDoesNotLabelPerfectCasesAsFailures():
    rows = [{"recordId": str(i), "matchId": f"match{i % 3}"} for i in range(9)]
    config = loadEvaluationConfig(Path("configs/evaluation/researchValidation.yaml"))
    fixed = fixedExamples({"first": {"rows": rows}, "second": {"rows": rows[::-1]}}, config)
    assert fixed == fixedExamples({"first": {"rows": rows[::-1]}, "second": {"rows": rows}}, config)
    assert len({next(row["matchId"] for row in rows if row["recordId"] == recordId) for recordId in fixed}) == 3
    detection = {"imageErrors": [{"recordId": str(i), "falsePositives": 0, "falseNegatives": 0,
                                  "ballMisses": 0, "classConfusions": 0, "highestFalsePositiveConfidence": 0}
                                 for i in range(9)]}
    selected = selectExamples(fixed, detection, {"errors": [], "perClass": []}, config)
    assert [item["recordId"] for item in selected] == fixed


def testHardExamplesKeepAthleteFailuresVisibleBesideManyLandmarkMisses():
    config = loadEvaluationConfig(Path("configs/evaluation/researchValidation.yaml"))
    errors = [{"recordId": recordId, "falsePositives": 0, "falseNegatives": count,
               "ballMisses": 0, "classConfusions": 0, "athleteErrors": athlete,
               "highestFalsePositiveConfidence": 0}
              for recordId, count, athlete in (("fixed", 0, 0), ("athlete", 2, 2), ("points", 100, 0))]
    points = {"perClass": [{"className": "landmark"}], "errors": [
        {"recordId": "points", "distanceDiagonal": 0.2, "annotated": True}]}
    selected = selectExamples(["fixed"], {"imageErrors": errors}, points, config)
    assert selected[1] == {"recordId": "athlete", "reason": "athleteDetectionErrors"}


@pytest.fixture
def reportEvaluation(trainingBundle, tmp_path):
    bundle, _, trainingConfig = trainingBundle
    plan = readJson(bundle / "plan.json")
    rows = [row for row in readJson(bundle / "manifest.json")["samples"]
            if row["taskId"] == "genericObjects" and row["split"] == "validation"]
    task = {"taskId": "genericObjects", "classes": ["ball", "player", "unsupported"], "rows": rows}
    config = loadEvaluationConfig(Path("configs/evaluation/researchValidation.yaml"))
    references = buildReferences(plan, task)
    predictions = [{"recordId": row["recordId"], "dimensions": row["dimensions"], "detections": []}
                   for row in references]
    detection = operationalMetrics(references, predictions, task["classes"], config)
    for row in detection["perClass"]:
        row.update(ap50=0.0 if row["support"] else None, ap50To95=0.0 if row["support"] else None)
    detection.update(imageCount=2, matchCounts={"matchB": 2}, groups=aggregateGroups(detection["perClass"], []), prCurves={})
    detection["byMatch"] = {"matchB": {"imageCount": 2, "perClass": detection["perClass"], "groups": detection["groups"]}}
    points = scorePoints(references, predictions, task["classes"], [], config)
    root = tmp_path / "reports" / "fixtureRun" / "runValidation"
    evidenceDir = root / "evidence" / "genericObjects"
    evidenceDir.mkdir(parents=True)
    writeJson(evidenceDir / "detection.json", detection)
    writeJson(evidenceDir / "points.json", points)
    writeJsonLines(evidenceDir / "references.jsonl", references)
    writeJsonLines(evidenceDir / "predictions.jsonl", predictions)
    image = root / "evidence" / "example.png"
    shutil.copyfile(bundle / references[0]["imagePath"], image)
    state = {"schemaVersion": 1, "status": "completed", "purpose": "syntheticFixture", "selection": "runValidation",
             "protocol": config, "protocolId": "fixtureProtocol", "trainingRunId": "fixtureRun", "contractId": "fixtureContract",
             "training": {"status": "completed", "stages": [], "code": {"commit": "fixtureCode", "dirty": False}},
             "code": {"commit": "fixtureCode", "dirty": False}, "pointClasses": [], "reviewedOwnRecordCount": 6,
             "datasetSummary": readJson(bundle / "manifest.json")["summary"],
             "stages": [{"taskId": "genericObjects", "status": "completed", "classes": task["classes"], "imageCount": 2,
                         "evidenceDirectory": "evidence/genericObjects", "checkpoint": {"sha256": "fixtureWeightHash", "byteSize": 10},
                         "examples": [{"recordId": references[0]["recordId"], "reason": "fixedSharedExample", "image": "evidence/example.png"}]}]}
    sealEvaluation(root, state)
    return root


@pytest.mark.reportingIntegration
def testReportRegeneratesWithoutTorchAndPreservesEvidence(reportEvaluation, tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    original = (reportEvaluation / "evaluation.json").read_bytes()
    report = renderReport(reportEvaluation)
    text = report.read_text(encoding="utf-8")
    assert "N/A" in text and "No detections met" in text
    assert "fewer than two recorded epochs" in text
    assert "No informative object precision-recall" in text
    assert (reportEvaluation.parents[1] / "index.md").is_file()
    for target in (reportEvaluation / "genericObjects" / "figures").glob("*.png"):
        assert target.stat().st_size > 1000
    manifest = readJson(reportEvaluation / "reportState.json")
    verifyEvidence(reportEvaluation, manifest["artifacts"])
    regenerated = tmp_path / "regenerated"
    renderReport(reportEvaluation, regenerated)
    assert (reportEvaluation / "evaluation.json").read_bytes() == original
    assert (regenerated / "genericObjects" / "tables" / "classDetection.csv").read_bytes() == (
        reportEvaluation / "genericObjects" / "tables" / "classDetection.csv").read_bytes()
    with pytest.raises(ValueError, match="new"):
        renderReport(reportEvaluation)
    (reportEvaluation / "evidence" / "genericObjects" / "predictions.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        renderReport(reportEvaluation, tmp_path / "badRegeneration")


def testFailedTrainingReportKeepsUnexecutedStagesVisible(tmp_path):
    runDir = tmp_path / "run"
    runDir.mkdir()
    writeJson(runDir / "run.json", {"purpose": "technicalSmoke", "status": "failed", "stages": [
        {"taskId": "genericObjects", "status": "failed"}], "configuration": {"stages": [
        {"taskId": "genericObjects"}, {"taskId": "roleObjects"}]},
        "error": {"type": "ValueError", "message": "fixture failure"}})
    report = renderTrainingReport(runDir, tmp_path / "failedReport")
    assert "roleObjects | notExecuted" in report.read_text()
    assert "Research evaluation: **not executed**" in report.read_text()
    assert not list(report.parent.rglob("*.png"))


def testMatchPointSuccessTablesRetainToleranceAndReferenceDenominator(reportEvaluation, tmp_path):
    detection = readJson(reportEvaluation / "evidence" / "genericObjects" / "detection.json")
    summary = {"className": "point", "support": 2, "localizedCount": 1, "successRates": {"0.01": 0.5}}
    points = {"perClass": [summary], "errors": [{"recordId": "frame", "nativePoint": [0, 0], "predictedPoint": None}],
              "byMatch": {"matchB": {"perClass": [summary]}}}
    directory = tmp_path / "tables"
    stageTables(detection, points, directory)
    with (directory / "matchPointToleranceSuccess.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [{"matchId": "matchB", "className": "point", "support": "2",
                     "toleranceDiagonal": "0.01", "successRate": "0.5"}]


def testUnsealedOptionalHistoryCannotInfluenceAResearchReport(reportEvaluation):
    (reportEvaluation / "evidence" / "genericObjects" / "results.csv").write_text(
        "epoch,metrics/mAP50-95(B)\n1,1.0\n2,1.0\n")
    with pytest.raises(ValueError, match="membership"):
        verifyEvaluation(reportEvaluation)
