import pytest

from futsal.cli import main


def testVersion(capsys):
    with pytest.raises(SystemExit) as result:
        main(["--version"])
    assert result.value.code == 0
    assert capsys.readouterr().out.strip() == "futsal 0.1.0"


@pytest.mark.parametrize("status,exitCode", [("completed", 0), ("partial", 1)])
def testBallTrackingCommandForwardsDeviceAndResume(monkeypatch, tmp_path, status, exitCode):
    calls = []
    monkeypatch.setattr("futsal.evaluation.ballTrackingRun.compareBallTrackers",
                        lambda *args, **kwargs: calls.append((args, kwargs)) or {"status": status})
    monkeypatch.setattr("futsal.reporting.researchReport.renderReport", lambda *args: "report.md")
    assert main(["compareBallTrackers", "--runsConfig", "localSettings.ballTracking.yaml",
                 "--datasetDir", "view", "--mediaStoreDir", "store", "--device", "0",
                 "--resumeFrom", "previous", "--outputDir", str(tmp_path / "comparison")]) == exitCode
    assert calls[0][1]["device"] == "0" and str(calls[0][1]["resumeFrom"]) == "previous"


def testTrainForwardsModelSizeOverride(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("futsal.training.trainingRun.runTraining",
                        lambda *args, **kwargs: calls.append(kwargs) or {"status": "completed"})
    monkeypatch.setattr("futsal.reporting.researchCli.generateRunReport", lambda *args, **kwargs: (True, "report.md"))
    assert main(["train", "--datasetDir", str(tmp_path / "data"), "--weights", "yolo26s.pt",
                 "--modelSize", "s", "--outputDir", str(tmp_path / "run")]) == 0
    assert calls[0]["modelSize"] == "s"


@pytest.mark.parametrize("command", ["compareConfigurations", "compareCourtTraining"])
@pytest.mark.parametrize("mode", [None, "modelVariants"])
def testComparisonCommandsForwardMode(monkeypatch, tmp_path, command, mode):
    calls = []
    module = "configurationComparison" if command == "compareConfigurations" else "courtComparison"
    monkeypatch.setattr(f"futsal.evaluation.{module}.{command}",
                        lambda *args, **kwargs: calls.append(kwargs) or {"status": "completed"})
    monkeypatch.setattr("futsal.reporting.researchReport.renderReport", lambda *args: "report.md")
    arguments = [command, "--baselineEvaluationDir", "before", "--candidateEvaluationDir", "after",
                 "--outputDir", str(tmp_path / "comparison")]
    if mode is not None:
        arguments += ["--comparisonMode", mode]
    assert main(arguments) == 0
    assert calls[0]["comparisonMode"] == (mode or "controlled")
