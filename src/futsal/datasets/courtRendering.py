import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

referenceWidth = 1920


referenceHeight = 1080


pointBoxWidth = 1 / 64


pointBoxHeight = 1 / 36


@dataclass(frozen=True)
class RenderConfig:
    width: int = 1280
    height: int = 720
    textureScale: int = 48
    minJpegQuality: int = 84
    maxJpegQuality: int = 96
    maxCameraAttempts: int = 300
    maxSampleAttempts: int = 80


@dataclass(frozen=True)
class LabeledReference:
    classId: int
    className: str
    referenceName: str
    coordinate: tuple[float, float]


@dataclass(frozen=True)
class SynthesisModel:
    name: str
    length: float
    width: float
    goalWidth: float
    penaltyDistance: float
    secondPenaltyDistance: float
    centerCircleRadius: float
    references: tuple[LabeledReference, ...]
    classNames: Mapping[int, str]


@dataclass(frozen=True)
class VirtualCamera:
    position: np.ndarray
    target: np.ndarray
    horizontalFovDegrees: float
    rollDegrees: float
    intrinsicMatrix: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    projectionMatrix: np.ndarray
    undistortedHomography: np.ndarray
    distortion: tuple[float, float, float, float]


@dataclass(frozen=True)
class ProjectedLabel:
    reference: LabeledReference
    x: float
    y: float


@dataclass(frozen=True)
class SyntheticSample:
    image: np.ndarray
    debugImage: np.ndarray
    labels: tuple[ProjectedLabel, ...]
    metadata: Mapping[str, Any]
    jpegQuality: int


def sampleRng(masterSeed: int, index: int) -> tuple[np.random.Generator, int]:
    sequence = np.random.SeedSequence([int(masterSeed), int(index)])
    sampleSeed = int(sequence.generate_state(1, dtype=np.uint64)[0])
    return (np.random.default_rng(sequence), sampleSeed)


def clampColor(color: Sequence[float]) -> tuple[int, int, int]:
    values = np.clip(np.rint(color), 0, 255).astype(np.uint8)
    return (int(values[0]), int(values[1]), int(values[2]))


def pinholeCamera(
    position: np.ndarray,
    target: np.ndarray,
    horizontalFovDegrees: float,
    rollDegrees: float,
    width: int,
    height: int,
    rng: np.random.Generator,
) -> VirtualCamera:
    axisZ = target - position
    axisZ = axisZ / np.linalg.norm(axisZ)
    worldVertical = np.asarray([0.0, 0.0, 1.0])
    axisX = np.cross(axisZ, worldVertical)
    norm = np.linalg.norm(axisX)
    if norm < 1e-08:
        raise ValueError("Degenerate camera: view direction is parallel to the vertical axis.")
    axisX /= norm
    axisY = np.cross(axisZ, axisX)
    axisY /= np.linalg.norm(axisY)
    roll = math.radians(rollDegrees)
    rotatedAxisX = math.cos(roll) * axisX + math.sin(roll) * axisY
    rotatedAxisY = -math.sin(roll) * axisX + math.cos(roll) * axisY
    rotation = np.stack([rotatedAxisX, rotatedAxisY, axisZ])
    translation = -rotation @ position
    focal = width / 2 / math.tan(math.radians(horizontalFovDegrees) / 2)
    centerX = width * (0.5 + float(rng.normal(0.0, 0.004)))
    centerY = height * (0.5 + float(rng.normal(0.0, 0.004)))
    intrinsic = np.asarray(
        [[focal, 0.0, centerX], [0.0, focal, centerY], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    extrinsic = np.column_stack([rotation, translation])
    projection = intrinsic @ extrinsic
    homography = intrinsic @ np.column_stack([rotation[:, 0], rotation[:, 1], translation])
    distortion = (
        float(rng.uniform(-0.014, 0.01)),
        float(rng.uniform(-0.003, 0.004)),
        float(rng.uniform(-0.00035, 0.00035)),
        float(rng.uniform(-0.00035, 0.00035)),
    )
    return VirtualCamera(
        position=position,
        target=target,
        horizontalFovDegrees=horizontalFovDegrees,
        rollDegrees=rollDegrees,
        intrinsicMatrix=intrinsic,
        rotation=rotation,
        translation=translation,
        projectionMatrix=projection,
        undistortedHomography=homography,
        distortion=distortion,
    )


def project3d(
    points: np.ndarray, camera: VirtualCamera, distort: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cameraCoords = (camera.rotation @ points.T).T + camera.translation
    depths = cameraCoords[:, 2]
    normalized = cameraCoords[:, :2] / depths[:, None]
    x = normalized[:, 0]
    y = normalized[:, 1]
    if distort:
        k1, k2, p1, p2 = camera.distortion
        radius2 = x * x + y * y
        radial = 1.0 + k1 * radius2 + k2 * radius2 * radius2
        originalX = x.copy()
        originalY = y.copy()
        x = (
            originalX * radial
            + 2.0 * p1 * originalX * originalY
            + p2 * (radius2 + 2.0 * originalX * originalX)
        )
        y = (
            originalY * radial
            + p1 * (radius2 + 2.0 * originalY * originalY)
            + 2.0 * p2 * originalX * originalY
        )
    fx = camera.intrinsicMatrix[0, 0]
    fy = camera.intrinsicMatrix[1, 1]
    cx = camera.intrinsicMatrix[0, 2]
    cy = camera.intrinsicMatrix[1, 2]
    pixels = np.column_stack([fx * x + cx, fy * y + cy])
    return (pixels, depths)


def fullBoxInImage(x: float, y: float, width: int, height: int) -> bool:
    halfWidth = width * pointBoxWidth / 2
    halfHeight = height * pointBoxHeight / 2
    return halfWidth <= x <= width - halfWidth and halfHeight <= y <= height - halfHeight
