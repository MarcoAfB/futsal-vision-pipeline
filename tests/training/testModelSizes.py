from pathlib import Path

import pytest
import yaml

from futsal.training.modelWeights import modelSizes
from futsal.training.trainingConfig import loadTrainingConfig, resolveTrainingConfig


@pytest.mark.parametrize("modelSize", modelSizes)
@pytest.mark.parametrize("profile", sorted(Path("configs/training").glob("*.yaml")), ids=lambda p: p.stem)
def testEveryTrainingProfileAcceptsEveryModelSize(tmp_path, profile, modelSize):
    config = yaml.safe_load(profile.read_text())
    recipePath = "configs/curriculum.yaml" if profile.stem.startswith("yolo26") else "configs/sharedDetection.yaml"
    if profile.stem == "yolo26ObjectCurriculum":
        recipePath = "configs/objectCurriculum.yaml"
    recipe = yaml.safe_load(Path(recipePath).read_text())
    config["modelSize"] = modelSize
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(config))
    assert loadTrainingConfig(path, recipe)["modelSize"] == modelSize


@pytest.mark.parametrize("modelSize", ["unknown", "S", "", 1, True, ["s"], {"size": "s"}, None])
def testTrainingProfileRejectsInvalidModelSizes(tmp_path, modelSize):
    config = yaml.safe_load(Path("configs/training/yolo26Smoke.yaml").read_text())
    config["modelSize"] = modelSize
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(config))
    recipe = yaml.safe_load(Path("configs/curriculum.yaml").read_text())
    with pytest.raises(ValueError, match="modelSize"):
        loadTrainingConfig(path, recipe)


def testSizeAndSeedResolutionPreservesProfile():
    source = {"modelSize": "m", "seed": 42}
    assert resolveTrainingConfig(source) == source
    assert resolveTrainingConfig(source, modelSize="s", seed=43) == {"modelSize": "s", "seed": 43}
    assert source == {"modelSize": "m", "seed": 42}
    with pytest.raises(ValueError, match="modelSize"):
        resolveTrainingConfig(source, modelSize="invalid")
    with pytest.raises(ValueError, match="Seed"):
        resolveTrainingConfig(source, seed=True)
