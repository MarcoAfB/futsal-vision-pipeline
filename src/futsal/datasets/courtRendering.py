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
