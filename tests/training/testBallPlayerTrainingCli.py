import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def ballPlayerLauncher():
    spec = importlib.util.spec_from_file_location("ballPlayerLauncher", Path("scripts/runBallPlayerTraining.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def testBallPlayerLauncherSelectsMixedDefaultsAndAllSizes(ballPlayerLauncher, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(ballPlayerLauncher, "trainCampaign", lambda *a, **kw: calls.append(kw) or 0)
    assert ballPlayerLauncher.main(["train", "--workDir", str(tmp_path), "--datasetDir", str(tmp_path / "view"),
                                    "--mediaStoreDir", str(tmp_path / "store")]) == 0
    assert calls[0]["curriculum"] == "ballPlayerMixed"
    assert calls[0]["configPath"].name == "ballPlayerMixed.yaml"
    assert calls[0]["campaignDir"].name.startswith("ballPlayerMixedAllSizesSeed42_")
    assert calls[0]["modelSizes"] == ["n", "s", "m", "l", "x"]
    assert calls[0]["allowUncommitted"] is False


def testBallPlayerLauncherForwardsExplicitUncommittedCheckout(ballPlayerLauncher, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(ballPlayerLauncher, "trainCampaign", lambda *a, **kw: calls.append(kw) or 0)
    assert ballPlayerLauncher.main(["train", "--workDir", str(tmp_path), "--datasetDir", str(tmp_path / "view"),
                                    "--mediaStoreDir", str(tmp_path / "store"), "--allowUncommittedCheckout"]) == 0
    assert calls[0]["allowUncommitted"] is True


def testBallPlayerPreparationUsesMixedRecipeAndRawBinding(ballPlayerLauncher, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(ballPlayerLauncher, "prepareObjectCurriculum", lambda *a, **kw:
                        calls.append((a, kw)) or {"contractId": "fixture"})
    assert ballPlayerLauncher.main(["prepare", "--sourceDatasetDir", str(tmp_path / "source"),
        "--mediaStoreDir", str(tmp_path / "store"), "--planDir", str(tmp_path / "plan"),
        "--datasetDir", str(tmp_path / "view"), "--dataRoot", str(tmp_path / "raw")]) == 0
    assert calls[0][0][1].name == "ballPlayerMixed.yaml"
    assert calls[0][1] == {"curriculum": "ballPlayerMixed", "dataRoot": tmp_path / "raw"}


def testBallPlayerResumeUsesExplicitCampaignWithoutStartingNewOne(ballPlayerLauncher, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(ballPlayerLauncher, "trainCampaign", lambda *a, **kw: calls.append(kw) or 0)
    arguments = ["train", "--workDir", str(tmp_path), "--datasetDir", str(tmp_path / "view"),
                 "--mediaStoreDir", str(tmp_path / "store"), "--resume"]
    assert ballPlayerLauncher.main(arguments) == 2 and calls == []
    assert ballPlayerLauncher.main([*arguments, "--campaignDir", str(tmp_path / "campaign"), "--modelSizes", "m"]) == 0
    assert calls[0]["resume"] and calls[0]["modelSizes"] == ["m"]
