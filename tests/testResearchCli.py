import pytest

from futsal.catalog import hashFile
from futsal.cli import main
from futsal.datasetFiles import readJson, writeJson
from futsal.researchCli import generateRunReport


def testResearchCliNeverAcceptsTestSelection(tmp_path):
    output = tmp_path / "mustNotExist"
    with pytest.raises(SystemExit) as error:
        main(["evaluateRun", "--runDir", "missing", "--datasetDir", "missing", "--selection", "test",
              "--outputDir", str(output)])
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("status", ["failed", "completed"])
def testAutomaticReportKeepsTrainingStatusWhenResearchCannotRun(tmp_path, monkeypatch, status):
    runDir = tmp_path / "run"
    runDir.mkdir()
    writeJson(runDir / "run.json", {"status": status, "purpose": "technicalSmoke", "stages": [],
                                    "configuration": {"stages": [{"taskId": "genericObjects"}]}})
    before = hashFile(runDir / "run.json")
    output = tmp_path / "reports" / "run" / "automatic"
    calls = []

    def failEvaluation(*args, **kwargs):
        calls.append(True)
        raise ValueError("fixture evaluation failure")

    monkeypatch.setattr("futsal.evaluationRun.newEvaluationPath", lambda *args: output)
    monkeypatch.setattr("futsal.evaluationRun.evaluateRun", failEvaluation)
    complete, path = generateRunReport(runDir, tmp_path / "dataset", "unusedConfig")
    assert not complete and path.is_file()
    assert hashFile(runDir / "run.json") == before
    report = readJson(output / "reportState.json")
    assert report["trainingStatus"] == status
    if status == "failed":
        assert calls == [] and report["evaluationStatus"] == "notExecuted"
        assert "notExecuted" in (output / "report.md").read_text()
    else:
        assert calls == [True] and report["status"] == "failed"
        assert "fixture evaluation failure" in (output / "pipelineFailure.txt").read_text()
    assert "automatic/report" in (tmp_path / "reports" / "index.md").read_text()
