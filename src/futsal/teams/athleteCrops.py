import numpy as np

from futsal.evaluation.evaluationMetrics import boxOverlaps


def torsoBox(box, config, dimensions):
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    left = max(0, round(x0 + config["sideMargin"] * width))
    right = min(dimensions[0], round(x1 - config["sideMargin"] * width))
    top = max(0, round(y0 + config["torsoTop"] * height))
    bottom = min(dimensions[1], round(y0 + config["torsoBottom"] * height))
    if min(right - left, bottom - top) < config["minCropPixels"]:
        return None
    return [left, top, right, bottom]


def detectionCrops(prediction, config):
    kept = [(index, item) for index, item in enumerate(prediction["detections"])
            if item["className"] == "player" and item["confidence"] >= config["detectionConfidence"]]
    overlaps = boxOverlaps([item for _, item in kept], [item for _, item in kept])
    np.fill_diagonal(overlaps, 0.0)
    rows = []
    for position, (index, item) in enumerate(kept):
        region = torsoBox(item["box"], config, prediction["dimensions"])
        overlap = float(overlaps[position].max())
        rows.append({"detectionIndex": index, "box": item["box"], "confidence": item["confidence"],
                     "torsoBox": region, "maxOverlap": overlap,
                     "clean": region is not None and overlap <= config["overlapIouMax"]})
    return rows


def cropImage(image, region):
    left, top, right, bottom = region
    return image[top:bottom, left:right]
