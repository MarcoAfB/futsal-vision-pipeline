import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def ballLauncher():
    spec = importlib.util.spec_from_file_location("ballLauncher", Path("scripts/runBallTraining.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def testBallLauncherSelectsMixedDefaultsAndAllSizes(ballLauncher, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(ballLauncher, "trainCampaign", lambda *a, **kw: calls.append(kw) or 0)
    assert ballLauncher.main(["train", "--workDir", str(tmp_path), "--datasetDir", str(tmp_path / "view"),
                              "--mediaStoreDir", str(tmp_path / "store")]) == 0
    assert calls[0]["curriculum"] == "ballMixed"
    assert calls[0]["configPath"].name == "ballMixed.yaml"
    assert calls[0]["campaignDir"].name.startswith("ballMixedAllSizesSeed42_")
    assert calls[0]["modelSizes"] == ["n", "s", "m", "l", "x"]


def testBallPreparationUsesMixedRecipeAndRawBinding(ballLauncher, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(ballLauncher, "prepareObjectCurriculum", lambda *a, **kw:
                        calls.append((a, kw)) or {"contractId": "fixture"})
    assert ballLauncher.main(["prepare", "--sourceDatasetDir", str(tmp_path / "source"), "--mediaStoreDir", str(tmp_path / "store"),
        "--planDir", str(tmp_path / "plan"), "--datasetDir", str(tmp_path / "view"), "--dataRoot", str(tmp_path / "raw")]) == 0
    assert calls[0][0][1].name == "ballMixed.yaml"
    assert calls[0][1] == {"curriculum": "ballMixed", "dataRoot": tmp_path / "raw"}


def testBallResumeUsesExplicitCampaignWithoutStartingNewOne(ballLauncher, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(ballLauncher, "trainCampaign", lambda *a, **kw: calls.append(kw) or 0)
    arguments = ["train", "--workDir", str(tmp_path), "--datasetDir", str(tmp_path / "view"),
                 "--mediaStoreDir", str(tmp_path / "store"), "--resume"]
    assert ballLauncher.main(arguments) == 2 and calls == []
    assert ballLauncher.main([*arguments, "--campaignDir", str(tmp_path / "campaign"), "--modelSizes", "x"]) == 0
    assert calls[0]["resume"] and calls[0]["modelSizes"] == ["x"]
