import math
from itertools import combinations

import cv2
import numpy as np

from futsal.geometry.courtProjection import jacobians, projectPoints


def triangleArea(first, second, third):
    return abs((second[0] - first[0]) * (third[1] - first[1]) -
               (second[1] - first[1]) * (third[0] - first[0])) / 2


def generalPosition(points, minArea):
    return all(triangleArea(*points[list(indices)]) >= minArea
               for indices in combinations(range(len(points)), 3))


def hasGeneralSubset(points, minArea):
    return any(generalPosition(points[list(subset)], minArea)
               for subset in combinations(range(len(points)), 4))


def orientedHomography(court, image):
    """Least-squares court-to-image matrix with unit norm and positive scale at its points."""
    matrix, _ = cv2.findHomography(np.asarray(court, dtype=np.float64),
                                   np.asarray(image, dtype=np.float64), 0)
    if matrix is None or not np.isfinite(matrix).all():
        return None
    matrix = matrix / np.linalg.norm(matrix)
    _, scale = projectPoints(matrix, court)
    return -matrix if np.median(scale) < 0 else matrix


def reprojectionErrors(matrix, court, image):
    projected, scale = projectPoints(matrix, court)
    errors = np.linalg.norm(projected - image, axis=1)
    errors[(scale <= 0) | ~np.isfinite(errors)] = np.inf
    return errors


def searchInliers(court, image, threshold, minArea):
    # At most twenty landmarks give at most 4845 minimal subsets, so the search
    # is exhaustive and deterministic instead of randomly sampled.
    best, bestKey = None, None
    for subset in combinations(range(len(court)), 4):
        indices = list(subset)
        if not generalPosition(court[indices], minArea):
            continue
        matrix = orientedHomography(court[indices], image[indices])
        if matrix is None:
            continue
        errors = reprojectionErrors(matrix, court, image)
        inliers = errors <= threshold
        # A subset whose own points project behind the camera is not a valid model.
        if inliers.sum() < 4:
            continue
        key = (int(inliers.sum()), -float((np.minimum(errors, threshold) ** 2).sum()))
        if bestKey is None or key > bestKey:
            best, bestKey = inliers, key
    return best


def hullFraction(points, dimensions):
    hull = cv2.convexHull(np.asarray(points, dtype=np.float32))
    return float(cv2.contourArea(hull)) / (dimensions[0] * dimensions[1])


def gateReasons(matrix, court, image, inliers, dimensions, config):
    errors = reprojectionErrors(matrix, court, image)
    rmse = float(np.sqrt(np.mean(errors[inliers] ** 2)))
    reasons = []
    if not math.isfinite(rmse) or rmse > config["maxInlierRmseDiagonal"] * math.hypot(*dimensions):
        reasons.append("inlierRmseHigh")
    if hullFraction(image[inliers], dimensions) < config["minImageHullFraction"]:
        reasons.append("insufficientImageSpread")
    _, scale = projectPoints(matrix, court[inliers])
    if np.any(scale <= 0):
        reasons.append("pointsBehindCamera")
    local = jacobians(matrix, court[inliers])
    # Court x points toward Side B and y toward the camera; the image keeps
    # that handedness, so a mirrored Side A/B or upper/lower labelling flips it.
    if np.any(np.linalg.det(local) <= 0):
        reasons.append("orientationReversed")
    if np.any(local[:, 1, 1] <= 0):
        reasons.append("farNearInverted")
    return reasons, errors, rmse


def estimateHomography(court, image, dimensions, config, *, robust):
    """Estimate court-to-image from correspondences in metres and original pixels.

    robust=True searches the inlier set among detected landmarks; robust=False
    fits every annotated reference point. Status is verified (at least five
    inliers), minimal (exactly four: zero residual, no internal evidence) or
    rejected with reason codes.
    """
    court = np.asarray(court, dtype=float).reshape(-1, 2)
    image = np.asarray(image, dtype=float).reshape(-1, 2)
    result = {"pointCount": len(court), "status": "rejected", "reasons": [], "courtToImage": None,
              "inliers": [False] * len(court), "residualsPixels": [None] * len(court),
              "inlierCount": 0, "inlierRmsePixels": None}
    minArea = config["minTriangleAreaSquareMeters"]
    if len(court) < 4:
        return {**result, "reasons": ["insufficientPoints"]}
    if not hasGeneralSubset(court, minArea):
        return {**result, "reasons": ["degenerateCourtConfiguration"]}
    inliers = np.ones(len(court), dtype=bool)
    if robust:
        threshold = config["inlierThresholdDiagonal"] * math.hypot(*dimensions)
        inliers = searchInliers(court, image, threshold, minArea)
        if inliers is None:
            return {**result, "reasons": ["noConsistentSubset"]}
    matrix = orientedHomography(court[inliers], image[inliers])
    if matrix is None:
        return {**result, "reasons": ["estimationFailed"]}
    reasons, errors, rmse = gateReasons(matrix, court, image, inliers, dimensions, config)
    count = int(inliers.sum())
    return {**result, "status": "rejected" if reasons else "verified" if count >= 5 else "minimal",
            "reasons": reasons, "courtToImage": matrix.tolist(), "inliers": inliers.tolist(),
            "residualsPixels": [float(value) if math.isfinite(value) else None for value in errors],
            "inlierCount": count, "inlierRmsePixels": rmse if math.isfinite(rmse) else None}
