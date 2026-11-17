from pathlib import Path

import pytest
import yaml

from futsal.geometry.courtModel import landmarkCoordinates, landmarkNames
from futsal.training.augmentationPolicy import mirroredClassIndices
from futsal.training.trainingConfig import checkpointSelectionCriterion, loadTrainingConfig

augmentedProfile = Path("configs/training/courtPointsMAugmented.yaml")
recipe = yaml.safe_load(Path("configs/separatedDetection.yaml").read_text(encoding="utf-8"))


def loadModified(tmp_path, change, profile=augmentedProfile):
    config = yaml.safe_load(profile.read_text(encoding="utf-8"))
    change(config)
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return loadTrainingConfig(path, recipe)


def testAugmentedProfileDeclaresPolicyAndPointSelection():
    config = loadTrainingConfig(augmentedProfile, recipe)
    assert config["augmentation"] == {"policy": "courtSideSwapLight", "exemptOrigins": ["synthetic"]}
    assert config["parameters"]["fliplr"] == 0.5 and config["parameters"]["mosaic"] == 0.0
    assert checkpointSelectionCriterion(config) == "nativeValidationPointSuccessMacro"
    assert checkpointSelectionCriterion(loadTrainingConfig("configs/training/courtPoints.yaml", recipe)) == \
        "nativeValidationMap50To95"


@pytest.mark.parametrize("change,message", [
    (lambda c: c["parameters"].update(mosaic=1.0), "mosaic is not enabled"),
    (lambda c: c["parameters"].update(flipud=0.5), "flipud is not enabled"),
    (lambda c: c["parameters"].update(degrees=45.0), "degrees must be within"),
    (lambda c: c["parameters"].update(hsv_s=-0.1), "hsv_s must be within"),
    (lambda c: c["augmentation"].update(policy="heavy"), "Unknown augmentation policy"),
    (lambda c: c["augmentation"].update(exemptOrigins=["own"]), "exemptOrigins"),
    (lambda c: c.pop("augmentation"), "must be disabled without an augmentation policy"),
    (lambda c: c["checkpointSelection"].update(metric="pointSuccessMicro"), "pointSuccessMacro"),
    (lambda c: c["checkpointSelection"].update(confidence=1.0), "confidence must be in"),
])
def testInvalidAugmentationOrSelectionIsRejected(tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        loadModified(tmp_path, change)


def testPolicyAndPointSelectionAreRestrictedToCourtPoints(tmp_path):
    def objectStage(config):
        config["stages"] = [{"taskId": "detailedObjects", "epochs": 100}]

    with pytest.raises(ValueError, match="only for independent courtPoints"):
        loadModified(tmp_path, objectStage)
    with pytest.raises(ValueError, match="Point-based checkpoint selection is supported only"):
        loadModified(tmp_path, objectStage, Path("configs/training/courtPointsMPointSelection.yaml"))


def testMirrorMapMatchesTheGeometricMirrorOfTheCourt():
    length, width = 40.0, 20.0
    coordinates = landmarkCoordinates(length, width, {"goalWidth": 3.0, "penaltyDistance": 6.0,
                                                      "secondPenaltyDistance": 10.0, "centerCircleRadius": 3.0,
                                                      "substitutionZoneOffset": 5.0, "substitutionZoneLength": 5.0})
    names = list(landmarkNames)
    mirrored = mirroredClassIndices(names)
    for index, name in enumerate(names):
        x, y = coordinates[name]
        assert coordinates[names[mirrored[index]]] == pytest.approx((length - x, y))
    assert [mirrored[target] for target in mirrored] == list(range(len(names)))
    assert names[mirrored[names.index("lowerGoalpostSideA")]] == "lowerGoalpostSideB"
    assert names[mirrored[names.index("substitutionPoint2")]] == "substitutionPoint3"


def testMirrorMapKeepsObjectsAndRejectsIncompleteOrUnknownLandmarks():
    assert mirroredClassIndices(["ball", "outfieldPlayerTeamA", "goalkeeperTeamB"]) == [0, 1, 2]
    with pytest.raises(ValueError, match="missing from the task"):
        mirroredClassIndices(["upperCornerSideA"])
    with pytest.raises(ValueError, match="Unknown court landmark"):
        mirroredClassIndices(["middleCornerSideA"])
