import copy

import pytest

from futsal.detectionGeometry import convertTargets, renderYoloLabels, validNormalizedBox


def geometrySample(shape, coordinates):
    return {
        "recordId": "own:0", "coordinateSystem": "imagePixels", "dimensions": [1920, 1080],
        "annotations": [{"shapeType": shape, "coordinates": coordinates}],
        "tasks": {"task": {"eligible": True, "targets": [{"annotationIndex": 0, "classId": 7}]}},
    }


policy = {"widthFraction": 1 / 64, "heightFraction": 1 / 36}
task = {"taskId": "task"}


def testPointBoxesUseTheDeclaredSizeAndKeepOriginalCoordinates():
    sample = geometrySample("points", [960, 540])
    original = copy.deepcopy(sample)
    target = convertTargets(sample, task, policy)[0]
    assert target["coordinates"] == pytest.approx([0.5, 0.5, 30 / 1920, 30 / 1080])
    assert target["transformation"] == {"kind": "pointToBox", "truncated": False}
    assert sample == original


def testArtificialBoxesAtImageEdgesAreIntersectedAndDeclared():
    sample = geometrySample("points", [0, 1080])
    target = convertTargets(sample, task, policy)[0]
    assert target["coordinates"] == pytest.approx([7.5 / 1920, 1072.5 / 1080, 15 / 1920, 15 / 1080])
    assert target["transformation"]["truncated"]
    assert validNormalizedBox([float(value) for value in renderYoloLabels([target]).split()[1:]])
    assert sample["annotations"][0]["coordinates"] == [0, 1080]


def testEllipseUsesItsEnclosingBoxWithoutClipping():
    sample = geometrySample("ellipse", [960, 540, 10, 15])
    target = convertTargets(sample, task, policy)[0]
    assert target["coordinates"] == pytest.approx([0.5, 0.5, 20 / 1920, 30 / 1080])
    assert target["transformation"]["kind"] == "ellipseToBox"
    sample["annotations"][0]["coordinates"] = [0, 0, 10, 15]
    with pytest.raises(ValueError, match="Invalid derived"):
        convertTargets(sample, task, policy)


def testQuarantinedSamplesCannotBeConverted():
    sample = geometrySample("box", [0, 0, 100, 100])
    sample["tasks"]["task"]["eligible"] = False
    with pytest.raises(ValueError, match="quarantined"):
        convertTargets(sample, task, policy)


def testPublicBoxesRetainTheirNormalizedCoordinates():
    sample = geometrySample("box", [0.5, 0.5, 0.2, 0.3])
    sample["coordinateSystem"] = "normalizedCenterBoxes"
    target = convertTargets(sample, task, policy)[0]
    assert target["coordinates"] == [0.5, 0.5, 0.2, 0.3]
    assert renderYoloLabels([target]).startswith("7 0.500000000000")
