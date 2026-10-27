import math


def validNormalizedBox(coordinates):
    if len(coordinates) != 4 or not all(math.isfinite(value) for value in coordinates):
        return False
    x, y, width, height = coordinates
    tolerance = 1e-10
    return width > 0 and height > 0 and (
        x - width / 2 >= -tolerance and x + width / 2 <= 1 + tolerance
        and y - height / 2 >= -tolerance and y + height / 2 <= 1 + tolerance
    )


def annotationBox(annotation, sample, pointBoxes):
    coordinates = annotation["coordinates"]
    if sample["coordinateSystem"] == "normalizedCenterBoxes":
        return list(coordinates), {"kind": "preservedNormalizedBox", "truncated": False}
    width, height = sample["dimensions"]
    shape = annotation["shapeType"]
    truncated = False
    if shape == "points":
        centerX, centerY = coordinates
        halfWidth = width * pointBoxes["widthFraction"] / 2
        halfHeight = height * pointBoxes["heightFraction"] / 2
        raw = [centerX - halfWidth, centerY - halfHeight, centerX + halfWidth, centerY + halfHeight]
        left, top, right, bottom = max(0, raw[0]), max(0, raw[1]), min(width, raw[2]), min(height, raw[3])
        truncated = [left, top, right, bottom] != raw
        kind = "pointToBox"
    elif shape == "ellipse":
        centerX, centerY, radiusX, radiusY = coordinates
        left, top, right, bottom = centerX - radiusX, centerY - radiusY, centerX + radiusX, centerY + radiusY
        kind = "ellipseToBox"
    elif shape == "box":
        left, top, right, bottom = coordinates
        kind = "preservedPixelBox"
    else:
        raise ValueError(f"Unsupported detection shape: {shape}.")
    normalized = [(left + right) / (2 * width), (top + bottom) / (2 * height),
                  (right - left) / width, (bottom - top) / height]
    return normalized, {"kind": kind, "truncated": truncated}


def convertTargets(sample, task, pointBoxes):
    status = sample["tasks"][task["taskId"]]
    if not status["eligible"]:
        raise ValueError(f"Cannot convert quarantined sample {sample['recordId']}.")
    converted = []
    for target in status["targets"]:
        annotation = sample["annotations"][target["annotationIndex"]]
        coordinates, transformation = annotationBox(annotation, sample, pointBoxes)
        if not validNormalizedBox(coordinates):
            raise ValueError(f"Invalid derived box in {sample['recordId']}.")
        converted.append({**target, "coordinates": coordinates, "transformation": transformation})
    return converted


def renderYoloLabels(targets):
    lines = []
    for target in targets:
        coordinates = [f"{value:.12f}" for value in target["coordinates"]]
        if not validNormalizedBox([float(value) for value in coordinates]):
            raise ValueError("Serialized detection box falls outside normalized image bounds.")
        lines.append(" ".join([str(target["classId"]), *coordinates]))
    return "\n".join(lines) + ("\n" if lines else "")
