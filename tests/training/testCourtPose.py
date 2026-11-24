from pathlib import Path

import pytest
import yaml

from futsal.training.trainingConfig import courtRepresentation, loadTrainingConfig

poseProfile = Path("configs/training/courtPoseMAugmented.yaml")
boxProfile = Path("configs/training/courtPointsMAugmented.yaml")
recipe = yaml.safe_load(Path("configs/separatedDetection.yaml").read_text(encoding="utf-8"))


def loadModified(tmp_path, change):
    config = yaml.safe_load(poseProfile.read_text(encoding="utf-8"))
    change(config)
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return loadTrainingConfig(path, recipe)


def testPoseProfileChangesOnlyTheRepresentation():
    pose, box = loadTrainingConfig(poseProfile, recipe), loadTrainingConfig(boxProfile, recipe)
    assert courtRepresentation(pose) == "courtPose" and courtRepresentation(box) == "pointBoxes"
    assert {key: value for key, value in pose.items() if key != "representation"} == box


@pytest.mark.parametrize("change,message", [
    (lambda c: c.update(representation="heatmap"), "only declared alternative"),
    (lambda c: c.pop("checkpointSelection"), "point-based checkpoint selection"),
])
def testPoseRequiresDeclaredRepresentationAndPointSelection(tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        loadModified(tmp_path, change)

