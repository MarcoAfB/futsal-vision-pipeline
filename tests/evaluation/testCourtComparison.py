import sys
from pathlib import Path

import pytest
import yaml

from futsal.datasets.datasetFiles import readJson, writeJson
from futsal.evaluation.configurationRun import evaluateConfiguration
from futsal.evaluation.courtComparison import compareCourtTraining
from futsal.evaluation.evaluationRun import evaluateRun, sealEvaluation
from futsal.reporting.researchReport import renderReport, verifyEvaluation
from futsal.training.trainingRun import runTraining


def testMixedRunSupportsJointValidationSealedComparisonAndPortableReports(
    configurationEvaluation, syntheticSource, tmp_path, monkeypatch
):
    import futsal.training.trainingRun as trainingModule

    _, runs, data, parent, _ = configurationEvaluation
    originalTrain = trainingModule.trainStage

    def train(*args):
        result = originalTrain(*args)
        result["training"] = {
            "completedEpochs": 1,
            "batches": 1,
            "optimizerUpdates": 1,
            "elapsedSeconds": 1.0,
        }
        return result

    monkeypatch.setattr(trainingModule, "trainStage", train)
    config = yaml.safe_load(Path("configs/training/courtPoints.yaml").read_text())
    config.update(device="cpu", evaluationDevice="cpu")
    configPath = tmp_path / "sameCourtConfig.yaml"
    configPath.write_text(yaml.safe_dump(config))
    weights = tmp_path / "initial/yolo26n.pt"
    courtRuns = []
    for name, source in (("real", None), ("mixed", syntheticSource)):
        runDir = tmp_path / name
        run = runTraining(
            data, configPath, weights, runDir, syntheticDir=source, progress=lambda message: None
        )
        assert run["status"] == "completed", run.get("error")
        courtRuns.append(runDir)
    mixedJoint = evaluateConfiguration(
        runs[0],
        courtRuns[1],
        data,
        parent,
        "configs/evaluation/researchValidation.yaml",
        tmp_path / "mixedJoint",
        progress=lambda message: None,
    )
    assert mixedJoint["status"] == "completed", mixedJoint.get("error")
    monkeypatch.setattr("futsal.evaluation.evaluationRun.initializeBackend", lambda *args: None)

    def predict(checkpoint, task, references, stage, trainingConfig, config, workDir):
        return [
            {
                "recordId": r["recordId"],
                "dimensions": r["dimensions"],
                "detections": [],
                "nativeOrder": i,
                "correctAtIou": [],
            }
            for i, r in enumerate(references)
        ], {}

    monkeypatch.setattr("futsal.evaluation.evaluationRun.predictValidation", predict)
    evaluations = []
    for index, runDir in enumerate(courtRuns):
        output = tmp_path / f"validation{index}"
        result = evaluateRun(
            runDir,
            data,
            "configs/evaluation/researchValidation.yaml",
            "runValidation",
            output,
            progress=lambda message: None,
            workRoot=tmp_path / "evaluationWork",
        )
        assert result["status"] == "completed", result.get("error")
        evaluations.append(output)
    output = tmp_path / "reports/comparison"
    state = compareCourtTraining(*evaluations, output)
    assert verifyEvaluation(output)[0] == state
    unsupported = [
        r
        for r in readJson(output / "evidence/comparisonMetrics.json")["points"]
        if r["className"] == "lowerCornerSideB" and r["metric"] == "successRate"
    ]
    assert unsupported[0]["baseline"] is None and unsupported[0]["delta"] is None
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    assert "N/A" in renderReport(output).read_text()
    assert "courtTrainingComparison" in (output.parent / "index.md").read_text()
    regenerated = tmp_path / "regenerated"
    assert renderReport(output, regenerated).is_file()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from resultTransfer import packResults, validateResults
    from transferArchive import unpackArchive

    archive, receipt = tmp_path / "results.tar.gz", tmp_path / "results.json"
    packResults(courtRuns[1], [evaluations[1], output], archive, receipt)
    received = tmp_path / "received"
    unpackArchive(archive, receipt, received, "curriculumResultsTransfer", validateResults)
    assert (received / "run/inputs/syntheticContract.json").is_file()
    assert renderReport(received / "reports/report02", tmp_path / "receivedPresentation").is_file()
    baseline = readJson(evaluations[0] / "evaluation.json")
    baseline["training"]["configuration"]["seed"] += 1
    writeJson(evaluations[0] / "evidence/trainingRun.json", baseline["training"])
    from futsal.inventory.catalog import hashFile

    baseline["trainingRunSha256"] = hashFile(evaluations[0] / "evidence/trainingRun.json")
    sealEvaluation(evaluations[0], baseline)
    with pytest.raises(ValueError, match="parameters"):
        compareCourtTraining(*evaluations, tmp_path / "incompatible")
    run = readJson(courtRuns[1] / "run.json")
    run["syntheticIntegrityAfterRun"] = False
    writeJson(courtRuns[1] / "run.json", run)
    with pytest.raises(ValueError, match="integrity"):
        from futsal.evaluation.configurationRun import configurationInputs

        configurationInputs(runs[0], courtRuns[1], data, parent)
