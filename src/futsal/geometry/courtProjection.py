import numpy as np


def projectPoints(matrix, points):
    """Return projected Cartesian points and their homogeneous scale.

    A nonpositive scale means the point lies behind the camera for a
    court-to-image matrix whose sign was fixed by orientedHomography, or outside
    the visible half-plane for its inverse. Such points are nonfinite or
    meaningless and callers must exclude them.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    homogeneous = np.column_stack([points, np.ones(len(points))]) @ np.asarray(matrix, dtype=float).T
    scale = homogeneous[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        projected = homogeneous[:, :2] / scale[:, None]
    return projected, scale


def imageToCourt(courtToImage, points):
    return projectPoints(np.linalg.inv(courtToImage), points)


def groundPoints(boxes):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]])


def jacobians(matrix, points):
    matrix = np.asarray(matrix, dtype=float)
    projected, scale = projectPoints(matrix, points)
    rows = []
    for (u, v), w in zip(projected, scale, strict=True):
        rows.append([[(matrix[0, 0] - u * matrix[2, 0]) / w, (matrix[0, 1] - u * matrix[2, 1]) / w],
                     [(matrix[1, 0] - v * matrix[2, 0]) / w, (matrix[1, 1] - v * matrix[2, 1]) / w]])
    return np.asarray(rows, dtype=float).reshape(-1, 2, 2)


def metresPerPixel(courtToImage, imagePoints):
    """Local isotropic scale of the image-to-court mapping at image points."""
    determinants = np.linalg.det(jacobians(np.linalg.inv(courtToImage), imagePoints))
    return np.sqrt(np.abs(determinants))


def insideCourt(points, court, margin):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    return ((points[:, 0] >= -margin) & (points[:, 0] <= court["length"] + margin) &
            (points[:, 1] >= -margin) & (points[:, 1] <= court["width"] + margin))
