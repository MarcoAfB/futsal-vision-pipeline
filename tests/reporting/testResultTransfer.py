import sys
from pathlib import Path

import pytest

from futsal.datasets.datasetFiles import readJson, writeJson
from futsal.evaluation.evaluationRun import sealEvaluation
from futsal.training.trainingRun import fileEvidence

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import resultTransfer
import transferArchive


@pytest.fixture
def resultBundle(tmp_path):
    runDir, reportDir = tmp_path / "run", tmp_path / "report"
    (runDir / "stages/first").mkdir(parents=True)
    checkpoint = runDir / "stages/first/best.pt"
    checkpoint.write_bytes(b"isolated software fixture checkpoint")
    (runDir / "runtimeData").mkdir()
    (runDir / "runtimeData/omitted.jpg").write_bytes(b"runtime bytes")
    writeJson(runDir / "runtimeEvidence.json", [fileEvidence(runDir / "runtimeData/omitted.jpg", runDir)])
    writeJson(runDir / "run.json", {"status": "completed", "purpose": "syntheticFixture",
        "artifacts": [fileEvidence(checkpoint, runDir)]})
    (reportDir / "evidence").mkdir(parents=True)
    writeJson(reportDir / "evidence/metrics.json", {"fixture": True})
    sealEvaluation(reportDir, {"status": "completed", "trainingRunSha256": transferArchive.hashFile(runDir / "run.json")})
    return runDir, reportDir


def testResultsRoundTripChecksSealsAndOmitsRuntimeData(resultBundle, tmp_path):
    runDir, reportDir = resultBundle
    archive, receiptPath = tmp_path / "results.tar", tmp_path / "results.json"
    receipt = resultTransfer.packResults(runDir, [reportDir], archive, receiptPath, compression="none")
    assert receipt["metadata"]["excludedRuntimeBytesVerifiedAtReceiver"] is False
    assert not any(item["path"].startswith("run/runtimeData/") for item in receipt["files"])
    destination = tmp_path / "received"
    transferArchive.unpackArchive(archive, receiptPath, destination, "curriculumResultsTransfer", resultTransfer.validateResults)
    assert (destination / "run/runtimeEvidence.json").is_file()
    assert not (destination / "run/runtimeData").exists()
    assert (destination / "run/run.json").read_bytes() == (runDir / "run.json").read_bytes()
    resultTransfer.verifyReport(destination / "reports/report01")


@pytest.mark.parametrize("mutation", ["seal", "checkpoint", "otherRun", "running"])
def testResultsRejectChangedEvidenceAndMismatchedReports(resultBundle, tmp_path, mutation):
    runDir, reportDir = resultBundle
    if mutation == "seal":
        (reportDir / "evidence/metrics.json").write_text("changed")
    elif mutation == "checkpoint":
        (runDir / "stages/first/best.pt").write_bytes(b"changed")
    elif mutation == "otherRun":
        run = readJson(runDir / "run.json")
        run["differentAttempt"] = True
        writeJson(runDir / "run.json", run)
    else:
        run = readJson(runDir / "run.json")
        run["status"] = "running"
        writeJson(runDir / "run.json", run)
    with pytest.raises(ValueError):
        resultTransfer.packResults(runDir, [reportDir], tmp_path / "results.tar.gz", tmp_path / "results.json")
