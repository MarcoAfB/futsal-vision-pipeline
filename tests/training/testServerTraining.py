import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from futsal.datasets.datasetFiles import readJson, writeJson
from futsal.training.modelWeights import prepareWeights
from futsal.training.trainingConfig import loadTrainingConfig
from futsal.training.trainingData import selectTrainingData

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import runServerTraining


@pytest.mark.parametrize("failure", ["environment", "training", "none"])
@pytest.mark.parametrize("modelSize", ["n", "s"])
def testOfflineLauncherUsesCurrentInterpreterAndKeepsExitStatus(trainingBundle, tmp_path, monkeypatch, failure, modelSize):
    bundle, configPath, _ = trainingBundle
    config = yaml.safe_load(Path("configs/training/yolo26CurriculumPilot.yaml").read_text(encoding="utf-8"))
    configPath.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr("futsal.training.modelWeights.downloadWeights", lambda url, path: path.write_bytes(b"fixture"))
    prepareWeights(modelSize, tmp_path / "weights")
    commit = "1" * 40
    monkeypatch.setattr(runServerTraining, "gitOutput", lambda root, *args: commit if args[0] == "rev-parse" else "")
    calls = []

    def command(arguments, **kwargs):
        calls.append(arguments)
        assert kwargs["env"]["UV_OFFLINE"] == "1"
        assert arguments[0] == sys.executable
        return SimpleNamespace(returncode=3 if failure == "environment" or (failure == "training" and "train" in arguments) else 0)

    monkeypatch.setattr(runServerTraining.subprocess, "run", command)
    arguments = ["--datasetDir", str(bundle), "--config", str(configPath), "--weights", str(tmp_path / "weights" / f"yolo26{modelSize}.pt"),
                 "--modelSize", modelSize,
                 "--outputDir", str(tmp_path / "attempt"), "--reportRoot", str(tmp_path / "reports"),
                 "--workRoot", str(tmp_path / "work"), "--logDir", str(tmp_path / "launch"), "--expectedCommit", commit]
    code = runServerTraining.main(arguments)
    state = readJson(tmp_path / "launch/launch.json")
    assert code == (0 if failure == "none" else 3)
    if failure == "environment":
        assert len(calls) == 1 and state["status"] == "environmentFailed"
    else:
        assert len(calls) == 2 and "--reportRoot" in calls[-1]
        assert state["trainingExitCode"] == code
    assert state["pythonExecutable"] == sys.executable
    assert state["modelSize"] == modelSize
    if failure != "environment":
        assert calls[-1][calls[-1].index("--modelSize") + 1] == modelSize


def testScientificLaunchRequiresMatchingCompletedPilot(trainingBundle, tmp_path):
    bundle, _, _ = trainingBundle
    recipe = readJson(bundle / "plan.json")["recipe"]
    config = loadTrainingConfig("configs/training/yolo26Curriculum.yaml", recipe)
    pilotConfig = loadTrainingConfig("configs/training/yolo26CurriculumPilot.yaml", recipe)
    selection = selectTrainingData(bundle, config)
    root = tmp_path / "pilot"
    root.mkdir()
    pilot = {"status": "completed", "purpose": "scientificPilot", "contractId": "fixtureContract",
             "initialWeights": {"sha256": "fixtureWeights"}, "configuration": pilotConfig,
             "code": {"commit": "fixtureCommit", "dirty": False}, "runtimeIntegrityAfterRun": True,
             "bundleIntegrityAfterRun": {"hashesVerified": True},
             "stages": [{"taskId": stage["taskId"], "status": "completed"} for stage in pilotConfig["stages"]]}
    writeJson(root / "run.json", pilot)
    writeJson(root / "selection.json", selection)
    assert runServerTraining.validatePilot(root, config, selection, pilot["initialWeights"],
                                           "fixtureContract", "fixtureCommit")["runSha256"]
    config["parameters"]["batch"] = 4
    with pytest.raises(ValueError, match="changed after the pilot"):
        runServerTraining.validatePilot(root, config, selection, pilot["initialWeights"], "fixtureContract", "fixtureCommit")
    config["parameters"]["batch"] = 2
    config["modelSize"] = "s"
    with pytest.raises(ValueError, match="scale"):
        runServerTraining.validatePilot(root, config, selection, pilot["initialWeights"], "fixtureContract", "fixtureCommit")
    config["modelSize"] = "n"
    pilot["status"] = "failed"
    writeJson(root / "run.json", pilot)
    with pytest.raises(ValueError, match="completed, verified pilot"):
        runServerTraining.validatePilot(root, config, selection, pilot["initialWeights"], "fixtureContract", "fixtureCommit")
