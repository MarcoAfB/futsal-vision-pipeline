import math
from pathlib import Path

import numpy as np

from futsal.inventory.catalog import readYaml

landmarkNames = (
    "upperCornerSideA", "lowerCornerSideA", "upperGoalpostSideA", "lowerGoalpostSideA",
    "penaltyMarkSideA", "secondPenaltyMarkSideA", "upperCenterCourtPoint", "lowerCenterCourtPoint",
    "upperHalfwayLinePoint", "lowerHalfwayLinePoint", "substitutionPoint1", "substitutionPoint2",
    "substitutionPoint3", "substitutionPoint4", "upperCornerSideB", "lowerCornerSideB",
    "upperGoalpostSideB", "lowerGoalpostSideB", "penaltyMarkSideB", "secondPenaltyMarkSideB",
)
markingNames = {"goalWidth", "penaltyDistance", "secondPenaltyDistance", "centerCircleRadius",
                "substitutionZoneOffset", "substitutionZoneLength"}
# Futsal Laws of the Game: touchline 25-42 m, goal line 16-25 m.
lengthRange = (25.0, 42.0)
widthRange = (16.0, 25.0)


def positiveNumber(value, context):
    if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{context} must be a positive finite number.")
    return float(value)


def courtDimensions(value, context):
    if not isinstance(value, dict) or set(value) != {"length", "width"}:
        raise ValueError(f"{context} must define exactly length and width.")
    length = positiveNumber(value["length"], f"{context} length")
    width = positiveNumber(value["width"], f"{context} width")
    if not lengthRange[0] <= length <= lengthRange[1] or not widthRange[0] <= width <= widthRange[1]:
        raise ValueError(f"{context} is outside the futsal ranges: length 25-42 m, width 16-25 m.")
    return {"length": length, "width": width}


def loadCourtGeometry(path, length=None, width=None):
    config, evidence = readYaml(Path(path))
    if set(config) != {"schemaVersion", "defaultDimensions", "matchDimensions", "markings"}:
        raise ValueError("Unexpected or missing court geometry fields.")
    if config["schemaVersion"] != 1:
        raise ValueError("Only court geometry schemaVersion 1 is supported.")
    if not isinstance(config["markings"], dict) or set(config["markings"]) != markingNames:
        raise ValueError(f"Court markings must define exactly: {sorted(markingNames)}.")
    if not isinstance(config["matchDimensions"], dict):
        raise ValueError("matchDimensions must map match IDs to dimensions.")
    if (length is None) != (width is None):
        raise ValueError("Supply both court length and width, or neither.")
    supplied = length is not None
    default = courtDimensions({"length": length, "width": width} if supplied
                              else config["defaultDimensions"], "Default court dimensions")
    return {"evidence": evidence,
            "markings": {name: positiveNumber(value, name) for name, value in config["markings"].items()},
            "defaultDimensions": default,
            "defaultSource": "commandLine" if supplied else "nominalDefault",
            "matchDimensions": {str(matchId): courtDimensions(value, f"Match {matchId} dimensions")
                                for matchId, value in config["matchDimensions"].items()}}


def landmarkCoordinates(length, width, markings):
    half, middle = length / 2, width / 2
    goal = markings["goalWidth"] / 2
    penalty, second = markings["penaltyDistance"], markings["secondPenaltyDistance"]
    radius = markings["centerCircleRadius"]
    near = markings["substitutionZoneOffset"]
    far = near + markings["substitutionZoneLength"]
    return {
        "upperCornerSideA": (0.0, 0.0), "lowerCornerSideA": (0.0, width),
        "upperGoalpostSideA": (0.0, middle - goal), "lowerGoalpostSideA": (0.0, middle + goal),
        "penaltyMarkSideA": (penalty, middle), "secondPenaltyMarkSideA": (second, middle),
        "upperCenterCourtPoint": (half, 0.0), "lowerCenterCourtPoint": (half, width),
        "upperHalfwayLinePoint": (half, middle - radius), "lowerHalfwayLinePoint": (half, middle + radius),
        "substitutionPoint1": (half - far, width), "substitutionPoint2": (half - near, width),
        "substitutionPoint3": (half + near, width), "substitutionPoint4": (half + far, width),
        "upperCornerSideB": (length, 0.0), "lowerCornerSideB": (length, width),
        "upperGoalpostSideB": (length, middle - goal), "lowerGoalpostSideB": (length, middle + goal),
        "penaltyMarkSideB": (length - penalty, middle), "secondPenaltyMarkSideB": (length - second, middle),
    }


def matchCourt(geometry, matchId):
    measured = geometry["matchDimensions"].get(matchId)
    dimensions = measured or geometry["defaultDimensions"]
    return {**dimensions, "dimensionSource": "matchConfig" if measured else geometry["defaultSource"],
            "landmarks": landmarkCoordinates(dimensions["length"], dimensions["width"], geometry["markings"])}


def arc(center, radius, startDegrees, endDegrees, count=24):
    angles = np.radians(np.linspace(startDegrees, endDegrees, count))
    return np.column_stack([center[0] + radius * np.cos(angles), center[1] + radius * np.sin(angles)])


def penaltyArea(goalLineX, direction, width, markings):
    middle, goal = width / 2, markings["goalWidth"] / 2
    radius = markings["penaltyDistance"]
    upper = arc((0.0, middle - goal), radius, -90, 0)
    lower = arc((0.0, middle + goal), radius, 0, 90)
    area = np.vstack([upper, lower])
    return np.column_stack([goalLineX + direction * area[:, 0], area[:, 1]])


def courtPolylines(court, markings):
    length, width = court["length"], court["width"]
    outline = np.array([[0, 0], [length, 0], [length, width], [0, width], [0, 0]], dtype=float)
    halfway = np.array([[length / 2, 0], [length / 2, width]], dtype=float)
    circle = arc((length / 2, width / 2), markings["centerCircleRadius"], 0, 360, 64)
    return [outline, halfway, circle, penaltyArea(0.0, 1.0, width, markings),
            penaltyArea(length, -1.0, width, markings)]
