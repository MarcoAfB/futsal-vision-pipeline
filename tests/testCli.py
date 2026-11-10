import pytest

from futsal.cli import main


def testVersion(capsys):
    with pytest.raises(SystemExit) as result:
        main(["--version"])
    assert result.value.code == 0
    assert capsys.readouterr().out.strip() == "futsal 0.1.0"


def testTrainForwardsModelSizeOverride(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("futsal.training.trainingRun.runTraining",
                        lambda *args, **kwargs: calls.append(kwargs) or {"status": "completed"})
    monkeypatch.setattr("futsal.reporting.researchCli.generateRunReport", lambda *args, **kwargs: (True, "report.md"))
    assert main(["train", "--datasetDir", str(tmp_path / "data"), "--weights", "yolo26s.pt",
                 "--modelSize", "s", "--outputDir", str(tmp_path / "run")]) == 0
    assert calls[0]["modelSize"] == "s"
