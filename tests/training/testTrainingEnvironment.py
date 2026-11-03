from types import SimpleNamespace

import pytest

from futsal.training.trainingEnvironment import inspectEnvironment, writeEnvironmentReport


def testMissingTrainingPackagesStillProduceAUsefulReport(tmp_path, monkeypatch):
    monkeypatch.setattr("futsal.training.trainingEnvironment.packageVersions", lambda: {"torch": None})
    monkeypatch.setattr("futsal.training.trainingEnvironment.shutil.which", lambda name: None)
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


def testDriverLibraryMismatchIsRetainedAsAGpuReadinessFailure(monkeypatch):
    monkeypatch.setattr("futsal.training.trainingEnvironment.packageVersions", lambda: {"torch": None})
    monkeypatch.setattr("futsal.training.trainingEnvironment.shutil.which", lambda name: "nvidia-smi")
    monkeypatch.setattr("futsal.training.trainingEnvironment.subprocess.run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=255, stdout="", stderr="NVML: Driver/library version mismatch"))
    report = inspectEnvironment("0")
    assert not report["ready"]
    assert "mismatch" in report["nvidiaSmi"]["error"]
    assert any("driver" in finding for finding in report["findings"])
