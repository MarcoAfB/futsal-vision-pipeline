import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def playerLauncher():
    spec = importlib.util.spec_from_file_location("playerLauncher", Path("scripts/runPlayerTraining.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def testPlayerLauncherSelectsMixedDefaultsAndAllSizes(playerLauncher, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(playerLauncher, "trainCampaign", lambda *a, **kw: calls.append(kw) or 0)
    assert playerLauncher.main(["train", "--workDir", str(tmp_path), "--datasetDir", str(tmp_path / "view"),
                              "--mediaStoreDir", str(tmp_path / "store")]) == 0
    assert calls[0]["curriculum"] == "playerMixed"
    assert calls[0]["configPath"].name == "playerMixed.yaml"
    assert calls[0]["campaignDir"].name.startswith("playerMixedAllSizesSeed42_")
    assert calls[0]["modelSizes"] == ["n", "s", "m", "l", "x"]
    assert calls[0]["allowUncommitted"] is False


def testPlayerLauncherForwardsExplicitUncommittedCheckout(playerLauncher, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(playerLauncher, "trainCampaign", lambda *a, **kw: calls.append(kw) or 0)
    assert playerLauncher.main(["train", "--workDir", str(tmp_path), "--datasetDir", str(tmp_path / "view"),
                              "--mediaStoreDir", str(tmp_path / "store"), "--allowUncommittedCheckout"]) == 0
    assert calls[0]["allowUncommitted"] is True


def testPlayerPreparationUsesMixedRecipeAndRawBinding(playerLauncher, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(playerLauncher, "prepareObjectCurriculum", lambda *a, **kw:
                        calls.append((a, kw)) or {"contractId": "fixture"})
    assert playerLauncher.main(["prepare", "--sourceDatasetDir", str(tmp_path / "source"), "--mediaStoreDir", str(tmp_path / "store"),
        "--planDir", str(tmp_path / "plan"), "--datasetDir", str(tmp_path / "view"), "--dataRoot", str(tmp_path / "raw")]) == 0
    assert calls[0][0][1].name == "playerMixed.yaml"
    assert calls[0][1] == {"curriculum": "playerMixed", "dataRoot": tmp_path / "raw"}


def testPlayerResumeUsesExplicitCampaignWithoutStartingNewOne(playerLauncher, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(playerLauncher, "trainCampaign", lambda *a, **kw: calls.append(kw) or 0)
    arguments = ["train", "--workDir", str(tmp_path), "--datasetDir", str(tmp_path / "view"),
                 "--mediaStoreDir", str(tmp_path / "store"), "--resume"]
    assert playerLauncher.main(arguments) == 2 and calls == []
    assert playerLauncher.main([*arguments, "--campaignDir", str(tmp_path / "campaign"), "--modelSizes", "x"]) == 0
    assert calls[0]["resume"] and calls[0]["modelSizes"] == ["x"]
