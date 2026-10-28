import pytest

from futsal.trainingEnvironment import inspectEnvironment, writeEnvironmentReport


def testMissingTrainingPackagesStillProduceAUsefulReport(tmp_path, monkeypatch):
    monkeypatch.setattr("futsal.trainingEnvironment.packageVersions", lambda: {"torch": None})
    monkeypatch.setattr("futsal.trainingEnvironment.shutil.which", lambda name: None)
    report = inspectEnvironment()
    assert report["ready"] is False
    assert "training extra" in report["findings"][0]
    output = tmp_path / "environment"
    writeEnvironmentReport(report, output)
    assert (output / "environment.json").is_file()
    assert (output / "report.md").is_file()


def testDeviceSelectionDoesNotAcceptImplicitFallbacks():
    with pytest.raises(ValueError, match="device"):
        inspectEnvironment("auto")
