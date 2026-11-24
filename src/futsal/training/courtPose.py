from futsal.geometry.courtModel import landmarkNames
from futsal.training.augmentationPolicy import mirroredClassIndices
from futsal.training.trainingConfig import courtRepresentation

poseClass = "court"
fullFrameBox = [0.5, 0.5, 1.0, 1.0]


def requireLandmarkOrder(names):
    if list(names) != list(landmarkNames):
        raise ValueError("courtPose requires the twenty court landmarks in court-model order.")


def poseDatasetEntries(names):
    requireLandmarkOrder(names)
    return {"names": {0: poseClass}, "kpt_shape": [len(names), 3],
            "flip_idx": mirroredClassIndices(names), "kpt_names": {0: list(names)}}


def nativePoints(rows, plan):
    """Pixel landmark points per record: native annotations, or the exact centre of untruncated synthetic boxes."""
    samples = {sample["recordId"]: sample for sample in plan["samples"]}
    points = {}
    for row in rows:
        width, height = row["dimensions"]
        items = []
        for target in row["targets"]:
            if row["origin"] == "synthetic":
                if target["transformation"]["truncated"]:
                    raise ValueError(f"Synthetic court target was truncated: {row['recordId']}.")
                point = [target["coordinates"][0] * width, target["coordinates"][1] * height]
            else:
                annotation = samples[row["recordId"]]["annotations"][target["annotationIndex"]]
                if annotation["shapeType"] != "points":
                    raise ValueError(f"Court landmark is not a native point: {row['recordId']}.")
                point = list(annotation["coordinates"])
            items.append({"classId": target["classId"], "point": point})
        points[row["recordId"]] = items
    return points


def poseLabels(rows, points, names):
    """Normalized [x, y, visibility] per landmark; unannotated landmarks are (0, 0, 0)."""
    requireLandmarkOrder(names)
    labels = {}
    for row in rows:
        width, height = row["dimensions"]
        keypoints = [[0.0, 0.0, 0.0] for _ in names]
        for item in points[row["recordId"]]:
            x, y = item["point"]
            if not (0 <= x <= width and 0 <= y <= height):
                raise ValueError(f"Court landmark lies outside its image: {row['recordId']}.")
            keypoints[item["classId"]] = [x / width, y / height, 2.0]
        labels[row["recordId"]] = keypoints
    return labels


def courtPoseLabels(config, selection, plan):
    if courtRepresentation(config) != "courtPose":
        return None
    task = selection["tasks"]["courtPoints"]
    rows = [*task["training"], *task["validation"]]
    return {"courtPoints": poseLabels(rows, nativePoints(rows, plan), task["classes"])}


def referencePoints(references):
    return {row["recordId"]: [{"classId": target["classId"], "point": target["point"]} for target in row["targets"]]
            for row in references}
