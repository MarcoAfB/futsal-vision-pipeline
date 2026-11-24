from pathlib import Path

import pytest
import yaml

from futsal.geometry.courtModel import landmarkNames
from futsal.inventory.catalog import hashFile
from futsal.training.augmentationPolicy import mirroredClassIndices
from futsal.training.courtPose import nativePoints, poseDatasetEntries, poseLabels
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


def courtRow(recordId, targets, origin="own", dimensions=(1920, 1080)):
    return {"recordId": recordId, "origin": origin, "dimensions": list(dimensions), "targets": targets}


def boxTarget(classId, annotationIndex, center, truncated=False, dimensions=(1920, 1080)):
    width, height = dimensions
    return {"classId": classId, "annotationIndex": annotationIndex,
            "coordinates": [center[0] / width, center[1] / height, 1 / 64, 1 / 36],
            "transformation": {"kind": "pointToBox", "truncated": truncated}}


def testTruncatedTargetsKeepTheirNativePointAndSyntheticUsesExactCentres():
    # The truncated box centre was moved inward by clipping; the label must keep the annotated point.
    real = courtRow("video01:0", [boxTarget(0, 0, (8.0, 5.0), truncated=True), boxTarget(14, 1, (900.0, 400.0))])
    synthetic = courtRow("synthetic:0", [boxTarget(6, 0, (960.0, 300.0))], origin="synthetic")
    plan = {"samples": [{"recordId": "video01:0", "annotations": [
        {"shapeType": "points", "coordinates": [3.0, 2.0]},
        {"shapeType": "points", "coordinates": [900.0, 400.0]}]}]}
    points = nativePoints([real, synthetic], plan)
    assert points["video01:0"] == [{"classId": 0, "point": [3.0, 2.0]}, {"classId": 14, "point": [900.0, 400.0]}]
    assert points["synthetic:0"] == [{"classId": 6, "point": pytest.approx([960.0, 300.0])}]
    with pytest.raises(ValueError, match="Synthetic court target was truncated"):
        nativePoints([courtRow("synthetic:1", [boxTarget(6, 0, (1.0, 300.0), truncated=True)], "synthetic")], plan)


def testPoseLabelsFollowTheCourtModelOrderAndLeaveMissingLandmarksInvisible():
    names = list(landmarkNames)
    rows = [courtRow("a", []), courtRow("b", [])]
    labels = poseLabels(rows, {"a": [{"classId": 14, "point": [960.0, 540.0]}], "b": []}, names)
    assert labels["a"][14] == [0.5, 0.5, 2.0]
    assert sum(item[2] > 0 for item in labels["a"]) == 1 and all(item == [0.0, 0.0, 0.0] for item in labels["b"])
    with pytest.raises(ValueError, match="outside its image"):
        poseLabels([courtRow("c", [])], {"c": [{"classId": 0, "point": [1921.0, 5.0]}]}, names)
    with pytest.raises(ValueError, match="court-model order"):
        poseLabels(rows, {"a": [], "b": []}, names[::-1])


def testPoseDatasetFlipSwapsMirroredLandmarks():
    entries = poseDatasetEntries(list(landmarkNames))
    assert entries["names"] == {0: "court"} and entries["kpt_shape"] == [20, 3]
    flip = entries["flip_idx"]
    assert flip == mirroredClassIndices(list(landmarkNames)) and [flip[index] for index in flip] == list(range(20))


@pytest.mark.trainingIntegration
def testSharedPoseDatasetBuildsOneFullFrameInstanceAndFlipsKeypoints(tmp_path):
    import cv2
    import numpy as np

    from futsal.training.trainingBackend import initializeBackend

    initializeBackend(tmp_path / "backend", 2)
    from ultralytics.cfg import get_cfg

    from futsal.training.sharedDataset import SharedPoseDataset

    path = tmp_path / "frame.png"
    assert cv2.imwrite(str(path), np.full((32, 64, 3), 100, dtype=np.uint8))
    rows = [{"recordId": recordId, "imagePath": "media/frame", "imageSha256": hashFile(path), "resolvedImage": str(path),
             "dimensions": [64, 32], "targets": []} for recordId in ("visible", "empty")]
    keypoints = [[0.0, 0.0, 0.0] for _ in landmarkNames]
    keypoints[0] = [0.25, 0.25, 2.0]
    labels = {"landmarks": list(landmarkNames), "records": {"visible": keypoints, "empty": [[0.0, 0.0, 0.0]] * 20}}
    data = {**poseDatasetEntries(list(landmarkNames)), "nc": 1, "channels": 3}
    samples = {}
    for fliplr in (0.0, 1.0):
        hyp = get_cfg(overrides={"fliplr": fliplr, "mosaic": 0.0, "scale": 0.0, "translate": 0.0, "hsv_h": 0.0,
                                 "hsv_s": 0.0, "hsv_v": 0.0, "cache": False})
        dataset = SharedPoseDataset(rows, labels, data=data, hyp=hyp, imgsz=64, augment=True, batch_size=1,
                                    task="pose")
        byRecord = {label["recordId"]: index for index, label in enumerate(dataset.labels)}
        assert dataset.labels[byRecord["empty"]]["keypoints"].shape == (0, 20, 3)
        assert dataset.labels[byRecord["visible"]]["bboxes"].tolist() == [[0.5, 0.5, 1.0, 1.0]]
        samples[fliplr] = dataset[byRecord["visible"]]["keypoints"][0]
    plain, flipped = samples[0.0], samples[1.0]
    mirrored = landmarkNames.index("upperCornerSideB")
    assert float(flipped[mirrored, 0]) == pytest.approx(1 - float(plain[0, 0]))
    assert float(flipped[mirrored, 1]) == pytest.approx(float(plain[0, 1]))
    assert float(flipped[0, 2]) == 0 and float(flipped[mirrored, 2]) > 0
