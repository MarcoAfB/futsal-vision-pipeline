import sys
from pathlib import Path

import pytest

from futsal.datasets.datasetFiles import readJson, writeJson
from futsal.evaluation.configurationComparison import compareConfigurations
from futsal.inventory.catalog import hashFile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import resultTransfer


def testConfigurationResultsRoundTripIncludesTwoRunsAndSealedComparison(comparisonEvaluations, tmp_path):
    baseline, candidate = comparisonEvaluations
    comparison = tmp_path / "comparison"
    compareConfigurations(baseline, candidate, comparison)
    original = hashFile(candidate / "evaluation.json")
    archive, receipt = tmp_path / "results.tar.gz", tmp_path / "results.receipt.json"
    assert resultTransfer.main(["pack", "--configurationDir", str(candidate), "--reportDir", str(comparison),
                                "--archive", str(archive), "--receipt", str(receipt)]) == 0
    metadata = readJson(receipt)
    assert metadata["kind"] == "configurationResultsTransfer"
    assert metadata["metadata"]["excludedRuntimeBytesVerifiedAtReceiver"] is False
    assert not any("/runtimeData/" in item["path"] or "/backend/" in item["path"] for item in metadata["files"])
    received = tmp_path / "received"
    assert resultTransfer.main(["unpack", "--archive", str(archive), "--receipt", str(receipt), "--outputDir", str(received)]) == 0
    for taskId in ("detailedObjects", "courtPoints"):
        resultTransfer.verifyRun(received / "runs" / taskId)
    resultTransfer.verifyReport(received / "configuration")
    resultTransfer.verifyReport(received / "reports/report01")
    assert hashFile(candidate / "evaluation.json") == original


def testConfigurationTransportRefusesComponentMetadataDrift(configurationEvaluation, tmp_path):
    candidate, runs, _, _, _ = configurationEvaluation
    run = readJson(runs[0] / "run.json")
    run["differentAttempt"] = True
    writeJson(runs[0] / "run.json", run)
    with pytest.raises(ValueError, match="metadata differ"):
        resultTransfer.packConfigurationResults(candidate, [], tmp_path / "results.tar.gz", tmp_path / "receipt.json")
