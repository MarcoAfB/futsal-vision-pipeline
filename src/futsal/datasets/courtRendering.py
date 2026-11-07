import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
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


def sampleCamera(
    model: SynthesisModel,
    targetReference: LabeledReference,
    rng: np.random.Generator,
    width: int,
    height: int,
    maxAttempts: int,
) -> tuple[VirtualCamera, np.ndarray]:
    courtPoints = np.asarray(
        [[*item.coordinate, 0.0] for item in model.references], dtype=np.float64
    )
    center = np.asarray([[model.length / 2, model.width / 2, 0.0]], dtype=np.float64)
    neighbors = np.asarray(
        [
            [model.length / 2 + 1, model.width / 2, 0.0],
            [model.length / 2, model.width / 2 + 1, 0.0],
        ],
        dtype=np.float64,
    )
    corners = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [model.length, 0.0, 0.0],
            [model.length, model.width, 0.0],
            [0.0, model.width, 0.0],
        ],
        dtype=np.float64,
    )
    for _ in range(maxAttempts):
        targetX = float(
            np.clip(rng.normal(targetReference.coordinate[0], 5.0), -7.0, model.length + 7.0)
        )
        position = np.asarray(
            [
                model.length / 2 + rng.normal(0.0, 1.2),
                rng.uniform(model.width + 15.0, model.width + 30.0),
                rng.uniform(10.0, 17.0),
            ],
            dtype=np.float64,
        )
        target = np.asarray(
            [targetX, rng.uniform(9.0, 11.0), rng.uniform(-2.0, -0.5)], dtype=np.float64
        )
        camera = pinholeCamera(
            position,
            target,
            float(rng.uniform(20.0, 40.0)),
            float(np.clip(rng.normal(0.0, 1.5), -4.0, 4.0)),
            width,
            height,
            rng,
        )
        projected, depths = project3d(courtPoints, camera)
        if np.any(depths <= 0) or not np.isfinite(projected).all():
            continue
        visible = np.asarray(
            [fullBoxInImage(x, y, width, height) for x, y in projected], dtype=bool
        )
        visibleCount = int(visible.sum())
        if not 4 <= visibleCount <= 12:
            continue
        if not visible[targetReference.classId]:
            continue
        projectedCenter, _ = project3d(center, camera)
        projectedNeighbors, _ = project3d(neighbors, camera)
        scaleX = float(np.linalg.norm(projectedNeighbors[0] - projectedCenter[0]))
        scaleY = float(np.linalg.norm(projectedNeighbors[1] - projectedCenter[0]))
        normalizedCenter = projectedCenter[0] / np.asarray([width, height])
        projectedCorners, _ = project3d(corners, camera)
        projectedArea = abs(float(cv2.contourArea(projectedCorners.astype(np.float32))))
        normalizedArea = projectedArea / (width * height)
        if not 70.0 <= scaleX <= 155.0:
            continue
        if not 20.0 <= scaleY <= 60.0:
            continue
        if not -0.65 <= normalizedCenter[0] <= 1.65:
            continue
        if not 0.25 <= normalizedCenter[1] <= 0.95:
            continue
        if not 0.65 <= normalizedArea <= 5.5:
            continue
        return (camera, projected)
    raise RuntimeError(
        "Cannot sample a valid camera. Increase maxCameraAttempts or review the court geometry."
    )


class MetricPlane:
    def __init__(self, model: SynthesisModel, scale: int) -> None:
        self.model = model
        self.scale = scale
        self.xMin = -6.0
        self.xMax = model.length + 6.0
        self.yMin = -5.0
        self.yMax = model.width + 5.0
        self.widthPixels = int(round((self.xMax - self.xMin) * scale)) + 1
        self.heightPixels = int(round((self.yMax - self.yMin) * scale)) + 1

    def point(self, x: float, y: float) -> tuple[int, int]:
        return (int(round((x - self.xMin) * self.scale)), int(round((y - self.yMin) * self.scale)))

    @property
    def worldToPixel(self) -> np.ndarray:
        return np.asarray(
            [
                [self.scale, 0.0, -self.xMin * self.scale],
                [0.0, self.scale, -self.yMin * self.scale],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


def lowFrequencyNoise(
    height: int, width: int, rng: np.random.Generator, scale: int = 48
) -> np.ndarray:
    smallHeight = max(3, math.ceil(height / scale))
    smallWidth = max(3, math.ceil(width / scale))
    small = rng.normal(0.0, 1.0, (smallHeight, smallWidth)).astype(np.float32)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)


def createWoodFloor(plane: MetricPlane, rng: np.random.Generator) -> np.ndarray:
    palettes = np.asarray(
        [(43, 83, 131), (43, 98, 144), (108, 167, 190), (138, 208, 215), (155, 228, 246)],
        dtype=np.float32,
    )
    base = palettes[int(rng.integers(0, len(palettes)))]
    base += rng.normal(0.0, 7.0, 3)
    image = np.empty((plane.heightPixels, plane.widthPixels, 3), np.float32)
    image[:] = base
    noise = lowFrequencyNoise(plane.heightPixels, plane.widthPixels, rng, scale=72)
    image += noise[..., None] * rng.uniform(5.0, 13.0)
    image = np.clip(image, 0, 255).astype(np.uint8)
    boardHeightMeters = float(rng.uniform(0.13, 0.22))
    boardHeightPixels = max(4, int(round(boardHeightMeters * plane.scale)))
    jointColor = clampColor(base * rng.uniform(0.84, 0.94))
    for y in range(0, plane.heightPixels, boardHeightPixels):
        cv2.line(image, (0, y), (plane.widthPixels - 1, y), jointColor, 1)
        offset = int(rng.integers(0, max(1, int(2.5 * plane.scale))))
        boardLength = int(rng.uniform(1.6, 3.2) * plane.scale)
        for x in range(offset, plane.widthPixels, boardLength):
            cv2.line(
                image,
                (x, y),
                (x, min(plane.heightPixels - 1, y + boardHeightPixels)),
                jointColor,
                1,
            )
    for _ in range(180):
        y = int(rng.integers(0, plane.heightPixels))
        x1 = int(rng.integers(0, plane.widthPixels))
        x2 = min(plane.widthPixels - 1, x1 + int(rng.uniform(30, 240)))
        color = clampColor(base * rng.uniform(0.76, 1.05))
        cv2.line(image, (x1, y), (x2, y), color, 1, cv2.LINE_AA)
    return image


def createPaintedFloor(plane: MetricPlane, rng: np.random.Generator) -> np.ndarray:
    palettes = np.asarray(
        [(230, 196, 160), (176, 128, 96), (182, 94, 76), (213, 114, 90), (190, 118, 80)],
        dtype=np.float32,
    )
    base = palettes[int(rng.integers(0, len(palettes)))]
    base += rng.normal(0.0, 8.0, 3)
    image = np.empty((plane.heightPixels, plane.widthPixels, 3), np.float32)
    image[:] = base
    smooth = lowFrequencyNoise(plane.heightPixels, plane.widthPixels, rng, scale=56)
    thin = rng.normal(0.0, 1.3, image.shape[:2]).astype(np.float32)
    image += (smooth * rng.uniform(4.0, 10.0) + thin)[..., None]
    return np.clip(image, 0, 255).astype(np.uint8)


def penaltyAreaPolygon(model: SynthesisModel, rightSide: bool, arcCount: int = 32) -> np.ndarray:
    radius = model.penaltyDistance
    centerY = model.width / 2
    halfAreaSegment = (model.goalWidth + 0.16) / 2
    upperY = centerY - halfAreaSegment
    lowerY = centerY + halfAreaSegment
    if not rightSide:
        upperArc = [
            (radius * math.cos(angle), upperY + radius * math.sin(angle))
            for angle in np.linspace(-math.pi / 2, 0.0, arcCount)
        ]
        lowerArc = [
            (radius * math.cos(angle), lowerY + radius * math.sin(angle))
            for angle in np.linspace(0.0, math.pi / 2, arcCount)
        ]
        return np.asarray(
            [(0.0, upperY - radius), *upperArc, *lowerArc, (0.0, lowerY + radius)], dtype=np.float64
        )
    return np.asarray(
        [(model.length - x, y) for x, y in penaltyAreaPolygon(model, False, arcCount)[::-1]],
        dtype=np.float64,
    )


def drawLogosAndDistractors(
    image: np.ndarray, plane: MetricPlane, rng: np.random.Generator
) -> None:
    model = plane.model
    for _ in range(int(rng.integers(1, 4))):
        x = float(rng.choice([rng.uniform(8, 16), rng.uniform(17, 23), rng.uniform(24, 32)]))
        y = float(rng.uniform(5.5, 14.5))
        radiusX = float(rng.uniform(1.8, 4.2))
        radiusY = float(rng.uniform(0.9, 2.6))
        center = plane.point(x, y)
        axes = (int(radiusX * plane.scale), int(radiusY * plane.scale))
        color = clampColor(rng.integers(35, 225, 3))
        overlay = image.copy()
        cv2.ellipse(overlay, center, axes, 0, 0, 360, color, -1, cv2.LINE_AA)
        alpha = float(rng.uniform(0.28, 0.52))
        cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0, image)
        for offset in (-0.45, 0.0, 0.45):
            p1 = plane.point(x - radiusX * 0.55, y + offset)
            p2 = plane.point(x + radiusX * 0.55, y + offset)
            cv2.line(image, p1, p2, (225, 225, 225), max(1, plane.scale // 24), cv2.LINE_AA)
    colors = [(35, 185, 225), (50, 70, 205), (200, 90, 55), (70, 180, 85)]
    for _ in range(int(rng.integers(2, 7))):
        color = colors[int(rng.integers(0, len(colors)))]
        thickness = max(1, int(round(rng.uniform(0.035, 0.06) * plane.scale)))
        if rng.random() < 0.65:
            y = float(rng.uniform(1.0, model.width - 1.0))
            cv2.line(
                image,
                plane.point(0.0, y),
                plane.point(model.length, y + rng.uniform(-1.0, 1.0)),
                color,
                thickness,
                cv2.LINE_AA,
            )
        else:
            x = float(rng.uniform(3.0, model.length - 3.0))
            radius = int(rng.uniform(1.5, 4.5) * plane.scale)
            cv2.circle(
                image, plane.point(x, model.width / 2), radius, color, thickness, cv2.LINE_AA
            )


def drawOfficialMarkings(image: np.ndarray, plane: MetricPlane, rng: np.random.Generator) -> None:
    model = plane.model
    s = plane.scale
    line = max(2, int(round(0.08 * s)))
    color = clampColor(np.asarray([238, 238, 236]) + rng.normal(0, 5, 3))
    centerX = model.length / 2
    centerY = model.width / 2
    halfAreaSegment = (model.goalWidth + 0.16) / 2
    upperY = centerY - halfAreaSegment
    lowerY = centerY + halfAreaSegment
    areaRadiusPixels = int(round(model.penaltyDistance * s))
    cv2.rectangle(
        image,
        plane.point(0.0, 0.0),
        plane.point(model.length, model.width),
        color,
        line,
        cv2.LINE_AA,
    )
    cv2.line(
        image,
        plane.point(centerX, 0.0),
        plane.point(centerX, model.width),
        color,
        line,
        cv2.LINE_AA,
    )
    cv2.circle(
        image,
        plane.point(centerX, centerY),
        int(round(model.centerCircleRadius * s)),
        color,
        line,
        cv2.LINE_AA,
    )
    cv2.circle(
        image, plane.point(centerX, centerY), max(2, int(round(0.06 * s))), color, -1, cv2.LINE_AA
    )
    cv2.ellipse(
        image,
        plane.point(0.0, upperY),
        (areaRadiusPixels, areaRadiusPixels),
        0,
        270,
        360,
        color,
        line,
        cv2.LINE_AA,
    )
    cv2.ellipse(
        image,
        plane.point(0.0, lowerY),
        (areaRadiusPixels, areaRadiusPixels),
        0,
        0,
        90,
        color,
        line,
        cv2.LINE_AA,
    )
    cv2.line(
        image,
        plane.point(model.penaltyDistance, upperY),
        plane.point(model.penaltyDistance, lowerY),
        color,
        line,
        cv2.LINE_AA,
    )
    cv2.ellipse(
        image,
        plane.point(model.length, upperY),
        (areaRadiusPixels, areaRadiusPixels),
        0,
        180,
        270,
        color,
        line,
        cv2.LINE_AA,
    )
    cv2.ellipse(
        image,
        plane.point(model.length, lowerY),
        (areaRadiusPixels, areaRadiusPixels),
        0,
        90,
        180,
        color,
        line,
        cv2.LINE_AA,
    )
    cv2.line(
        image,
        plane.point(model.length - model.penaltyDistance, upperY),
        plane.point(model.length - model.penaltyDistance, lowerY),
        color,
        line,
        cv2.LINE_AA,
    )
    markRadius = max(2, int(round(0.06 * s)))
    for x in (
        model.penaltyDistance,
        model.secondPenaltyDistance,
        model.length - model.penaltyDistance,
        model.length - model.secondPenaltyDistance,
    ):
        cv2.circle(image, plane.point(x, centerY), markRadius, color, -1, cv2.LINE_AA)
    for x in (reference.coordinate[0] for reference in model.references[10:14]):
        cv2.line(
            image,
            plane.point(x, model.width - 0.4),
            plane.point(x, model.width + 0.4),
            color,
            line,
            cv2.LINE_AA,
        )
    cornerRadius = max(2, int(round(0.25 * s)))
    for x, y, start, end in (
        (0.0, 0.0, 0, 90),
        (0.0, model.width, 270, 360),
        (model.length, 0.0, 90, 180),
        (model.length, model.width, 180, 270),
    ):
        cv2.ellipse(
            image,
            plane.point(x, y),
            (cornerRadius, cornerRadius),
            0,
            start,
            end,
            color,
            line,
            cv2.LINE_AA,
        )


def applyFloorReflections(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    height, width = image.shape[:2]
    mapImage = np.ones((max(8, height // 16), max(8, width // 16)), np.float32)
    yy, xx = np.mgrid[0 : mapImage.shape[0], 0 : mapImage.shape[1]]
    for _ in range(int(rng.integers(2, 7))):
        cx = rng.uniform(0, mapImage.shape[1])
        cy = rng.uniform(0, mapImage.shape[0])
        sx = rng.uniform(mapImage.shape[1] * 0.025, mapImage.shape[1] * 0.13)
        sy = rng.uniform(mapImage.shape[0] * 0.08, mapImage.shape[0] * 0.35)
        intensity = rng.uniform(0.04, 0.24)
        mapImage += intensity * np.exp(-0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
    mapImage = cv2.resize(mapImage, (width, height), interpolation=cv2.INTER_CUBIC)
    resultado = image.astype(np.float32) * mapImage[..., None]
    return np.clip(resultado, 0, 255).astype(np.uint8)


def createCourtPlane(
    model: SynthesisModel, scale: int, rng: np.random.Generator
) -> tuple[np.ndarray, MetricPlane, str]:
    plane = MetricPlane(model, scale)
    family = "wood" if rng.random() < 0.48 else "painted"
    if family == "wood":
        image = createWoodFloor(plane, rng)
    else:
        image = createPaintedFloor(plane, rng)
    courtMask = np.zeros(image.shape[:2], np.uint8)
    cv2.rectangle(courtMask, plane.point(0.0, 0.0), plane.point(model.length, model.width), 255, -1)
    surroundColor = np.asarray(
        rng.choice(
            np.asarray([(45, 55, 70), (70, 55, 48), (95, 55, 45), (38, 75, 105)], dtype=np.float32)
        ),
        dtype=np.float32,
    )
    outside = courtMask == 0
    image[outside] = np.clip(
        image[outside].astype(np.float32) * rng.uniform(0.25, 0.55)
        + surroundColor * rng.uniform(0.45, 0.75),
        0,
        255,
    ).astype(np.uint8)
    if rng.random() < 0.48:
        areaColor = clampColor(
            rng.choice(np.asarray([(45, 70, 175), (185, 90, 65), (75, 120, 190), (55, 145, 110)]))
        )
        overlay = image.copy()
        for right in (False, True):
            polygon = penaltyAreaPolygon(model, right)
            pixels = np.asarray([plane.point(x, y) for x, y in polygon], np.int32)
            cv2.fillPoly(overlay, [pixels], areaColor, cv2.LINE_AA)
        alpha = float(rng.uniform(0.45, 0.82))
        cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0, image)
    drawLogosAndDistractors(image, plane, rng)
    drawOfficialMarkings(image, plane, rng)
    image = applyFloorReflections(image, rng)
    return (image, plane, family)


def createArenaBackground(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    base = np.asarray(
        rng.choice(np.asarray([(22, 24, 28), (35, 31, 28), (25, 32, 40), (45, 42, 37)])),
        dtype=np.float32,
    )
    y = np.linspace(0.72, 1.18, height, dtype=np.float32)[:, None, None]
    background = np.broadcast_to(base[None, None, :] * y, (height, width, 3)).copy()
    smallW = max(80, width // 8)
    smallH = max(45, height // 8)
    audience = np.zeros((smallH, smallW, 3), np.uint8)
    audience[:] = clampColor(base * 0.7)
    colors = np.asarray(
        [(35, 45, 60), (65, 75, 95), (40, 65, 125), (85, 55, 45), (110, 105, 85)], dtype=np.uint8
    )
    for _ in range(int(smallW * smallH * 0.035)):
        cx = int(rng.integers(0, smallW))
        cy = int(rng.integers(0, max(1, int(smallH * 0.68))))
        color = tuple(int(v) for v in colors[int(rng.integers(0, len(colors)))])
        cv2.circle(audience, (cx, cy), int(rng.integers(1, 3)), color, -1)
    audience = cv2.GaussianBlur(audience, (5, 5), 0)
    audience = cv2.resize(audience, (width, height), interpolation=cv2.INTER_LINEAR)
    verticalMask = np.linspace(1.0, 0.0, height, dtype=np.float32)[:, None, None]
    background = background * (1.0 - 0.55 * verticalMask) + audience * (0.55 * verticalMask)
    return np.clip(background, 0, 255).astype(np.uint8)


def composePlane(
    background: np.ndarray,
    texture: np.ndarray,
    plane: MetricPlane,
    camera: VirtualCamera,
    width: int,
    height: int,
) -> np.ndarray:
    planeToWorld = np.linalg.inv(plane.worldToPixel)
    planeToImage = camera.undistortedHomography @ planeToWorld
    projected = cv2.warpPerspective(
        texture,
        planeToImage,
        (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )
    horizonLine = np.linalg.inv(camera.undistortedHomography).T @ np.asarray([0.0, 0.0, 1.0])
    center, _ = project3d(
        np.asarray([[plane.model.length / 2, plane.model.width / 2, 0.0]]), camera, False
    )
    centerSign = float(horizonLine @ np.asarray([center[0, 0], center[0, 1], 1.0]))
    sign = 1.0 if centerSign >= 0 else -1.0
    yy, xx = np.mgrid[0:height, 0:width]
    values = horizonLine[0] * xx + horizonLine[1] * yy + horizonLine[2]
    mask = values * sign >= 0
    resultado = background.copy()
    resultado[mask] = projected[mask]
    return resultado


def projectSegment(
    start: Sequence[float], end: Sequence[float], camera: VirtualCamera
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    points, depths = project3d(np.asarray([start, end]), camera, False)
    if np.any(depths <= 0) or not np.isfinite(points).all():
        return None
    return (tuple(np.rint(points[0]).astype(int)), tuple(np.rint(points[1]).astype(int)))


def drawBoards(
    image: np.ndarray, camera: VirtualCamera, model: SynthesisModel, rng: np.random.Generator
) -> None:
    overlay = image.copy()
    panelWidth = float(rng.uniform(3.0, 5.5))
    x = -5.0
    colors = [(225, 225, 220), (45, 120, 185), (35, 55, 125), (185, 65, 45), (55, 150, 90)]
    while x < model.length + 5.0:
        x2 = min(model.length + 5.0, x + panelWidth)
        vertices = np.asarray(
            [[x, -0.35, 0.0], [x2, -0.35, 0.0], [x2, -0.35, 1.05], [x, -0.35, 1.05]],
            dtype=np.float64,
        )
        pixels, depths = project3d(vertices, camera, False)
        if np.all(depths > 0) and np.isfinite(pixels).all():
            polygon = np.rint(pixels).astype(np.int32)
            color = colors[int(rng.integers(0, len(colors)))]
            cv2.fillConvexPoly(overlay, polygon, color, cv2.LINE_AA)
            cv2.polylines(overlay, [polygon], True, (215, 215, 210), 2, cv2.LINE_AA)
            middle1 = tuple(np.rint(pixels[0] * 0.65 + pixels[3] * 0.35).astype(int))
            middle2 = tuple(np.rint(pixels[1] * 0.65 + pixels[2] * 0.35).astype(int))
            cv2.line(overlay, middle1, middle2, (235, 235, 230), 3, cv2.LINE_AA)
        x = x2
        panelWidth = float(rng.uniform(3.0, 5.5))
    for fixedX in (-0.35, model.length + 0.35):
        y = -0.5
        while y < model.width + 0.5:
            y2 = min(model.width + 0.5, y + float(rng.uniform(2.6, 4.5)))
            vertices = np.asarray(
                [[fixedX, y, 0.0], [fixedX, y2, 0.0], [fixedX, y2, 1.05], [fixedX, y, 1.05]],
                dtype=np.float64,
            )
            pixels, depths = project3d(vertices, camera, False)
            if np.all(depths > 0) and np.isfinite(pixels).all():
                polygon = np.rint(pixels).astype(np.int32)
                color = colors[int(rng.integers(0, len(colors)))]
                cv2.fillConvexPoly(overlay, polygon, color, cv2.LINE_AA)
                cv2.polylines(overlay, [polygon], True, (215, 215, 210), 2, cv2.LINE_AA)
            y = y2
    cv2.addWeighted(overlay, 0.92, image, 0.08, 0, image)


def drawGoals(
    image: np.ndarray, camera: VirtualCamera, model: SynthesisModel, rng: np.random.Generator
) -> None:
    centerY = model.width / 2
    y1 = centerY - model.goalWidth / 2
    y2 = centerY + model.goalWidth / 2
    stripColor = tuple(
        int(v) for v in rng.choice(np.asarray([(35, 35, 185), (185, 55, 35), (45, 90, 190)]))
    )
    for frontX, direction in ((0.0, -1.0), (model.length, 1.0)):
        backX = frontX + direction * 1.0
        netOverlay = image.copy()
        netLines: list[tuple[Sequence[float], Sequence[float]]] = []
        for y in np.linspace(y1, y2, 7):
            netLines.append(((frontX, y, 0.0), (backX, y, 0.0)))
            netLines.append(((backX, y, 0.0), (backX, y, 2.0)))
        for z in np.linspace(0.0, 2.0, 6):
            netLines.append(((backX, y1, z), (backX, y2, z)))
            netLines.append(((frontX, y1, z), (backX, y1, z)))
            netLines.append(((frontX, y2, z), (backX, y2, z)))
        for start, end in netLines:
            segment = projectSegment(start, end, camera)
            if segment is not None:
                cv2.line(netOverlay, *segment, (195, 195, 190), 1, cv2.LINE_AA)
        cv2.addWeighted(netOverlay, 0.48, image, 0.52, 0, image)
        structure = [
            ((frontX, y1, 0.0), (frontX, y1, 2.0)),
            ((frontX, y2, 0.0), (frontX, y2, 2.0)),
            ((frontX, y1, 2.0), (frontX, y2, 2.0)),
        ]
        for start, end in structure:
            segment = projectSegment(start, end, camera)
            if segment is None:
                continue
            p1, p2 = segment
            reference, _ = project3d(
                np.asarray([start, (start[0], start[1] + 0.08, start[2])]), camera, False
            )
            thickness = int(np.clip(np.linalg.norm(reference[1] - reference[0]), 3, 10))
            cv2.line(image, p1, p2, (238, 238, 234), thickness, cv2.LINE_AA)
            for fraction in (0.18, 0.48, 0.78):
                a = tuple(
                    np.rint(np.asarray(p1) * (1 - fraction) + np.asarray(p2) * fraction).astype(int)
                )
                bFraction = min(1.0, fraction + 0.1)
                b = tuple(
                    np.rint(np.asarray(p1) * (1 - bFraction) + np.asarray(p2) * bFraction).astype(
                        int
                    )
                )
                cv2.line(image, a, b, stripColor, thickness, cv2.LINE_AA)


def uniformColors(rng: np.random.Generator) -> tuple[tuple[int, int, int], ...]:
    pairs = (
        ((25, 205, 235), (45, 45, 45)),
        ((45, 55, 205), (225, 225, 220)),
        ((220, 105, 45), (35, 35, 35)),
        ((60, 165, 65), (225, 220, 210)),
        ((35, 130, 225), (150, 55, 45)),
    )
    teamA, teamB = pairs[int(rng.integers(0, len(pairs)))]
    goalkeeperA = tuple(
        int(v) for v in rng.choice(np.asarray([(40, 190, 210), (200, 70, 170), (55, 180, 75)]))
    )
    goalkeeperB = tuple(
        int(v) for v in rng.choice(np.asarray([(185, 85, 45), (55, 200, 200), (160, 55, 180)]))
    )
    return (teamA, teamB, goalkeeperA, goalkeeperB)


def drawPerson(
    image: np.ndarray,
    occlusionMask: np.ndarray,
    base: np.ndarray,
    top: np.ndarray,
    shirtColor: tuple[int, int, int],
    rng: np.random.Generator,
) -> None:
    axis = top - base
    height = float(np.linalg.norm(axis))
    if height < 18:
        return
    unidade = axis / height
    perpendicular = np.asarray([-unidade[1], unidade[0]])

    def point(fraction: float, lateral: float = 0.0) -> tuple[int, int]:
        value = base + axis * fraction + perpendicular * lateral * height
        return tuple(np.rint(value).astype(int))

    skin = tuple(
        int(v)
        for v in rng.choice(
            np.asarray([(80, 135, 180), (60, 100, 145), (105, 165, 205), (45, 75, 110)])
        )
    )
    shortsColor = clampColor(np.asarray(shirtColor) * rng.uniform(0.28, 0.58))
    shadowColor = clampColor(np.asarray(shirtColor) * rng.uniform(0.7, 0.9))
    stepSide = -1.0 if rng.random() < 0.5 else 1.0
    shadowCenter = point(0.02, 0.08 * stepSide)
    shadow = image.copy()
    cv2.ellipse(
        shadow,
        shadowCenter,
        (max(5, int(height * 0.16)), max(2, int(height * 0.035))),
        float(rng.uniform(-20, 20)),
        0,
        360,
        (18, 20, 22),
        -1,
        cv2.LINE_AA,
    )
    cv2.addWeighted(shadow, 0.28, image, 0.72, 0, image)
    legThickness = max(2, int(round(height * 0.055)))
    armThickness = max(2, int(round(height * 0.045)))
    leftHip = point(0.43, -0.075)
    rightHip = point(0.43, 0.075)
    leftFoot = point(0.0, -0.07 + 0.055 * stepSide)
    rightFoot = point(0.0, 0.07 + 0.08 * stepSide)
    cv2.line(image, leftHip, leftFoot, skin, legThickness, cv2.LINE_AA)
    cv2.line(image, rightHip, rightFoot, skin, legThickness, cv2.LINE_AA)
    cv2.line(occlusionMask, leftHip, leftFoot, 255, legThickness + 2, cv2.LINE_AA)
    cv2.line(occlusionMask, rightHip, rightFoot, 255, legThickness + 2, cv2.LINE_AA)
    short = np.asarray(
        [point(0.46, -0.105), point(0.46, 0.105), point(0.33, 0.075), point(0.33, -0.075)], np.int32
    )
    torso = np.asarray(
        [point(0.78, -0.145), point(0.78, 0.145), point(0.45, 0.105), point(0.45, -0.105)], np.int32
    )
    cv2.fillConvexPoly(image, short, shortsColor, cv2.LINE_AA)
    cv2.fillConvexPoly(image, torso, shirtColor, cv2.LINE_AA)
    cv2.polylines(image, [torso], True, shadowColor, max(1, armThickness // 2), cv2.LINE_AA)
    cv2.fillConvexPoly(occlusionMask, short, 255, cv2.LINE_AA)
    cv2.fillConvexPoly(occlusionMask, torso, 255, cv2.LINE_AA)
    leftShoulder = point(0.73, -0.13)
    rightShoulder = point(0.73, 0.13)
    leftHand = point(0.48, -0.2 - 0.04 * stepSide)
    rightHand = point(0.5, 0.19 - 0.04 * stepSide)
    cv2.line(image, leftShoulder, leftHand, skin, armThickness, cv2.LINE_AA)
    cv2.line(image, rightShoulder, rightHand, skin, armThickness, cv2.LINE_AA)
    cv2.line(occlusionMask, leftShoulder, leftHand, 255, armThickness + 2, cv2.LINE_AA)
    cv2.line(occlusionMask, rightShoulder, rightHand, 255, armThickness + 2, cv2.LINE_AA)
    headCenter = point(0.89)
    headRadius = max(3, int(round(height * 0.068)))
    cv2.circle(image, headCenter, headRadius, skin, -1, cv2.LINE_AA)
    cv2.circle(image, point(0.94), max(2, int(headRadius * 0.75)), (35, 40, 45), -1, cv2.LINE_AA)
    cv2.circle(occlusionMask, headCenter, headRadius + 1, 255, -1, cv2.LINE_AA)


def drawHumanOccluders(
    image: np.ndarray, camera: VirtualCamera, model: SynthesisModel, rng: np.random.Generator
) -> tuple[np.ndarray, int]:
    mask = np.zeros(image.shape[:2], np.uint8)
    uniforms = uniformColors(rng)
    targetX = float(camera.target[0])
    candidatos: list[tuple[float, np.ndarray, np.ndarray, tuple[int, int, int]]] = []
    outfieldCount = int(rng.integers(7, 11))
    positions: list[tuple[float, float, int]] = []
    for index in range(outfieldCount):
        for _ in range(30):
            x = float(np.clip(rng.normal(targetX, 7.5), 0.8, model.length - 0.8))
            y = float(rng.uniform(1.0, model.width - 1.0))
            if all((math.hypot(x - px, y - py) >= 0.85 for px, py, _ in positions)):
                positions.append((x, y, index % 2))
                break
    if rng.random() < 0.86:
        positions.append((float(rng.uniform(0.4, 2.0)), float(rng.uniform(7.7, 12.3)), 2))
    if rng.random() < 0.86:
        positions.append(
            (
                float(rng.uniform(model.length - 2.0, model.length - 0.4)),
                float(rng.uniform(7.7, 12.3)),
                3,
            )
        )
    for x, y, team in positions:
        heightMeters = float(rng.uniform(1.62, 1.91))
        points, depths = project3d(np.asarray([[x, y, 0.0], [x, y, heightMeters]]), camera, False)
        if np.any(depths <= 0) or not np.isfinite(points).all():
            continue
        base, top = points
        heightPixels = float(np.linalg.norm(top - base))
        if heightPixels < 25:
            continue
        margin = heightPixels * 0.25
        if not (
            -margin <= base[0] <= image.shape[1] + margin
            and -margin <= base[1] <= image.shape[0] + margin
        ):
            continue
        candidatos.append((float(depths[0]), base, top, uniforms[team]))
    for _, base, top, uniform in sorted(candidatos, reverse=True, key=lambda item: item[0]):
        drawPerson(image, mask, base, top, uniform, rng)
    if rng.random() < 0.72:
        for _ in range(int(rng.integers(2, 6))):
            x = float(np.clip(rng.normal(targetX, 5.5), 0.5, model.length - 0.5))
            y = model.width + float(rng.uniform(0.7, 1.8))
            points, depths = project3d(
                np.asarray([[x, y, 0.0], [x, y, rng.uniform(1.55, 1.85)]]), camera, False
            )
            if np.all(depths > 0) and np.isfinite(points).all():
                drawPerson(image, mask, points[0], points[1], (45, 55, 65), rng)
    return (mask, len(candidatos))


def distortionMaps(camera: VirtualCamera, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    fx = float(camera.intrinsicMatrix[0, 0])
    fy = float(camera.intrinsicMatrix[1, 1])
    cx = float(camera.intrinsicMatrix[0, 2])
    cy = float(camera.intrinsicMatrix[1, 2])
    xd = (xx - cx) / fx
    yd = (yy - cy) / fy
    xu = xd.copy()
    yu = yd.copy()
    k1, k2, p1, p2 = camera.distortion
    for _ in range(6):
        radius2 = xu * xu + yu * yu
        radial = 1.0 + k1 * radius2 + k2 * radius2 * radius2
        estimatedX = xu * radial + 2 * p1 * xu * yu + p2 * (radius2 + 2 * xu * xu)
        estimatedY = yu * radial + p1 * (radius2 + 2 * yu * yu) + 2 * p2 * xu * yu
        xu += xd - estimatedX
        yu += yd - estimatedY
    return ((fx * xu + cx).astype(np.float32), (fy * yu + cy).astype(np.float32))


def applyDistortion(
    image: np.ndarray, mask: np.ndarray, camera: VirtualCamera
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    mapX, mapY = distortionMaps(camera, width, height)
    distortedImage = cv2.remap(
        image,
        mapX,
        mapY,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(20, 22, 24),
    )
    distortedMask = cv2.remap(
        mask, mapX, mapY, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    return (distortedImage, distortedMask)


def motionKernel(size: int, angleDegrees: float) -> np.ndarray:
    kernel = np.zeros((size, size), np.float32)
    center = (size - 1) / 2
    length = (size - 1) / 2
    angle = math.radians(angleDegrees)
    dx = math.cos(angle) * length
    dy = math.sin(angle) * length
    cv2.line(
        kernel,
        (int(round(center - dx)), int(round(center - dy))),
        (int(round(center + dx)), int(round(center + dy))),
        1.0,
        1,
        cv2.LINE_AA,
    )
    total = float(kernel.sum())
    return kernel / total if total > 0 else np.eye(size, dtype=np.float32)


def applyPhotometryAndVideo(
    image: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, Mapping[str, Any]]:
    resultado = image.astype(np.float32) / 255.0
    gamma = float(rng.uniform(0.78, 1.28))
    exposure = float(rng.uniform(0.78, 1.28))
    contrast = float(rng.uniform(0.86, 1.15))
    balance = rng.uniform(0.9, 1.11, 3).astype(np.float32)
    resultado = np.power(np.clip(resultado, 0, 1), gamma)
    resultado = (resultado - 0.5) * contrast + 0.5
    resultado *= exposure * balance[None, None, :]
    height, width = resultado.shape[:2]
    yy, xx = np.mgrid[-1 : 1 : complex(0, height), -1 : 1 : complex(0, width)]
    vignetteStrength = float(rng.uniform(0.0, 0.18))
    vignette = 1.0 - vignetteStrength * np.clip(xx * xx + yy * yy, 0, 1)
    resultado *= vignette[..., None]
    resultado = np.clip(resultado * 255.0, 0, 255).astype(np.uint8)
    motionBlur = 0
    motionAngle = 0.0
    if rng.random() < 0.32:
        motionBlur = int(rng.choice(np.asarray([3, 5, 7])))
        motionAngle = float(rng.uniform(-22.0, 22.0))
        resultado = cv2.filter2D(resultado, -1, motionKernel(motionBlur, motionAngle))
    blurSigma = float(rng.uniform(0.35, 1.15))
    resultado = cv2.GaussianBlur(resultado, (0, 0), blurSigma)
    ycrcb = cv2.cvtColor(resultado, cv2.COLOR_BGR2YCrCb)
    for channel in (1, 2):
        reduced = cv2.resize(
            ycrcb[..., channel],
            (max(1, width // 2), max(1, height // 2)),
            interpolation=cv2.INTER_AREA,
        )
        ycrcb[..., channel] = cv2.resize(reduced, (width, height), interpolation=cv2.INTER_LINEAR)
    resultado = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    noiseSigma = float(rng.uniform(0.3, 2.4))
    noise = rng.normal(0.0, noiseSigma, resultado.shape).astype(np.float32)
    resultado = np.clip(resultado.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return (
        resultado,
        {
            "gamma": gamma,
            "exposure": exposure,
            "contrast": contrast,
            "bgrBalance": [float(v) for v in balance],
            "vignette": vignetteStrength,
            "motionBlurKernel": motionBlur,
            "motionBlurAngle": motionAngle,
            "defocusSigma": blurSigma,
            "noiseSigma": noiseSigma,
            "chroma420Aproximado": True,
        },
    )
