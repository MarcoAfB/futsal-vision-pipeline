def keypointBox(x, y, dimensions, pointBoxes):
    """Reference-sized box centred exactly on the keypoint, shrunk symmetrically at the image border."""
    width, height = dimensions
    halfWidth = min(width * pointBoxes["widthFraction"] / 2, x, width - x)
    halfHeight = min(height * pointBoxes["heightFraction"] / 2, y, height - y)
    return [x - halfWidth, y - halfHeight, x + halfWidth, y + halfHeight]


def bestInstance(instances):
    return max(instances, key=lambda item: item["confidence"]) if instances else None


def landmarkDetections(instances, dimensions, names, pointBoxes, confidenceFloor):
    """Native landmark detections from the most confident court instance.

    Instances hold pixel keypoints [x, y, visibility] in landmark order. A landmark's confidence is
    the instance confidence times its keypoint visibility; keypoints outside the image are not detections.
    """
    instance = bestInstance(instances)
    if instance is None:
        return []
    width, height = dimensions
    detections = []
    for classId, (x, y, visibility) in enumerate(instance["keypoints"]):
        confidence = instance["confidence"] * visibility
        if confidence >= confidenceFloor and 0 <= x <= width and 0 <= y <= height:
            detections.append({"classId": classId, "className": names[classId], "confidence": confidence,
                               "box": keypointBox(x, y, dimensions, pointBoxes)})
    return sorted(detections, key=lambda item: -item["confidence"])


def poseInstances(scaled):
    return [{"confidence": float(confidence), "keypoints": keypoints}
            for confidence, keypoints in zip(scaled["conf"].tolist(), scaled["keypoints"].tolist(), strict=True)]


def landmarkMatches(validator, detections, reference):
    """Ultralytics class-aware matches at its ten IoU thresholds, against the reference landmark boxes."""
    import torch
    from ultralytics.utils.metrics import box_iou

    if not detections or not reference["targets"]:
        return [[False] * len(validator.iouv) for _ in detections]
    iou = box_iou(torch.tensor([target["box"] for target in reference["targets"]], dtype=torch.float64),
                  torch.tensor([item["box"] for item in detections], dtype=torch.float64))
    classes = torch.tensor([item["classId"] for item in detections])
    truth = torch.tensor([target["classId"] for target in reference["targets"]])
    return validator.match_predictions(classes, truth, iou).tolist()
