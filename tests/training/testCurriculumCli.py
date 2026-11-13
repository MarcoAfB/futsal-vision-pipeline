import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def launcher():
    spec = importlib.util.spec_from_file_location("curriculumLauncher", Path("scripts/runObjectCurriculum.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("variant,stem", [("objects", "object"), ("players", "player")])
def testLauncherChoosesVariantDefaultsAndAllSizes(launcher, monkeypatch, tmp_path, variant, stem):
    calls = []
    monkeypatch.setattr(launcher, "trainCampaign", lambda *a, **kw: calls.append(kw) or 0)
    arguments = ["train", "--workDir", str(tmp_path), "--datasetDir", str(tmp_path / "view"), "--mediaStoreDir", str(tmp_path / "store")]
    if variant == "players":
        arguments += ["--curriculum", variant]
    assert launcher.main(arguments) == 0
    assert calls[0]["curriculum"] == variant
    assert calls[0]["configPath"].name == f"yolo26{stem.capitalize()}Curriculum.yaml"
    assert calls[0]["campaignDir"].name.startswith(f"{stem}CurriculumAllSizesSeed42_")
    assert calls[0]["modelSizes"] == ["n", "s", "m", "l", "x"]
    assert calls[0]["weightsRoot"] == tmp_path / "data/modelWeights"


@pytest.mark.parametrize("variant,stem", [("objects", "object"), ("players", "player")])
def testPreparationPassesVariantRecipeAndRawSource(launcher, monkeypatch, tmp_path, variant, stem):
    calls = []
    monkeypatch.setattr(launcher, "prepareObjectCurriculum", lambda *a, **kw:
                        calls.append((a, kw)) or {"contractId": "fixture"})
    arguments = ["prepare", "--curriculum", variant, "--sourceDatasetDir", str(tmp_path / "source"),
                 "--mediaStoreDir", str(tmp_path / "store"), "--planDir", str(tmp_path / "plan"),
                 "--datasetDir", str(tmp_path / "view"), "--dataRoot", str(tmp_path / "raw")]
    assert launcher.main(arguments) == 0
    assert calls[0][0][1].name == f"{stem}Curriculum.yaml"
    assert calls[0][1] == {"curriculum": variant, "dataRoot": tmp_path / "raw"}


def testResumeRequiresExplicitCampaignDirectory(launcher, monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "trainCampaign", lambda *a, **kw: pytest.fail("Allocated a campaign"))
    assert launcher.main(["train", "--workDir", str(tmp_path), "--datasetDir", str(tmp_path / "view"),
                          "--mediaStoreDir", str(tmp_path / "store"), "--resume", "--curriculum", "players"]) == 2
