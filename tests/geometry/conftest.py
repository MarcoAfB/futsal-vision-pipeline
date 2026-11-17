import numpy as np
import pytest

from futsal.geometry.courtModel import landmarkCoordinates, loadCourtGeometry
from futsal.geometry.courtProjection import projectPoints


def cameraHomography(position, target, focal=1400.0, size=(1920, 1080)):
    # World z points down, so the court frame (x toward Side B, y toward the
    # camera) and the camera frame (x right, y down, z forward) share handedness.
    position, target = np.asarray(position, dtype=float), np.asarray(target, dtype=float)
    forward = (target - position) / np.linalg.norm(target - position)
    right = np.array([1.0, 0.0, 0.0]) - forward * forward[0]
    right /= np.linalg.norm(right)
    rotation = np.stack([right, np.cross(forward, right), forward])
    translation = -rotation @ position
    intrinsic = np.array([[focal, 0, size[0] / 2], [0, focal, size[1] / 2], [0, 0, 1]])
    return intrinsic @ np.column_stack([rotation[:, 0], rotation[:, 1], translation])


@pytest.fixture
def broadcastView():
    geometry = loadCourtGeometry("configs/courtGeometry.yaml")
    landmarks = landmarkCoordinates(40.0, 20.0, geometry["markings"])
    matrix = cameraHomography((20.0, 34.0, -9.0), (20.0, 9.0, 0.0))
    names = list(landmarks)
    image, scale = projectPoints(matrix, [landmarks[name] for name in names])
    visible = [index for index, (u, v) in enumerate(image)
               if scale[index] > 0 and 0 <= u < 1920 and 0 <= v < 1080]
    return {"matrix": matrix, "landmarks": landmarks, "names": [names[index] for index in visible],
            "court": np.array([landmarks[names[index]] for index in visible]), "image": image[visible],
            "dimensions": [1920, 1080], "geometry": geometry}
