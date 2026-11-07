import math
from pathlib import Path

import cv2

from futsal.annotations.detectionGeometry import annotationBox
from futsal.datasets.courtRendering import LabeledReference, SynthesisModel
from futsal.inventory.catalog import hashFile, readYaml
from futsal.inventory.yoloInventory import scanFiles

projectRoot = Path(__file__).resolve().parents[3]


defaultConfig = projectRoot / "configs/syntheticCourt.yaml"


pointBoxes = {"widthFraction": 1 / 64, "heightFraction": 1 / 36}


def courtClassNames():
    labels, _ = readYaml(projectRoot / "configs/labelMap.yaml")
    return [item["name"] for item in labels["canonicalClasses"]][5:]


def loadSynthesisConfig(configPath=defaultConfig, **overrides):
    config, _ = readYaml(Path(configPath))
    config.update({key: value for key, value in overrides.items() if value is not None})
    fields = {
        "schemaVersion",
        "count",
        "seed",
        "width",
        "height",
        "textureScale",
        "debugCount",
        "minJpegQuality",
        "maxJpegQuality",
        "maxCameraAttempts",
        "maxSampleAttempts",
        "geometry",
    }
    if set(config) != fields or config["schemaVersion"] != 1:
        raise ValueError("Unsupported synthesis configuration fields or schema.")
    for key in fields - {"schemaVersion", "geometry"}:
        if type(config[key]) is not int or config[key] < (
            0 if key in {"seed", "debugCount"} else 1
        ):
            raise ValueError(f"Invalid synthesis integer: {key}.")
    if (
        config["width"] < 320
        or config["height"] < 180
        or not math.isclose(config["width"] / config["height"], 16 / 9, abs_tol=0.02)
    ):
        raise ValueError("Synthesis requires a 16:9 image of at least 320 x 180 pixels.")
    if (
        not 24 <= config["textureScale"] <= 128
        or not 50 <= config["minJpegQuality"] <= config["maxJpegQuality"] <= 100
    ):
        raise ValueError("Invalid synthesis texture scale or JPEG quality interval.")
    synthesisModel(config)
    return config


def synthesisModel(config):
    geometry = config["geometry"]
    names = courtClassNames()
    if (
        len(names) != 20
        or set(geometry)
        != {
            "length",
            "width",
            "goalWidth",
            "penaltyDistance",
            "secondPenaltyDistance",
            "centerCircleRadius",
            "references",
        }
        or set(geometry["references"]) != set(names)
    ):
        raise ValueError(
            "Synthesis geometry must define exactly the twenty canonical court references."
        )
    if any(
        type(value) not in {float, int} or not math.isfinite(value) or value <= 0
        for key, value in geometry.items()
        if key != "references"
    ):
        raise ValueError("Court dimensions must be finite and positive.")
    length, width = geometry["length"], geometry["width"]
    if (
        geometry["goalWidth"] >= width
        or geometry["penaltyDistance"] >= geometry["secondPenaltyDistance"]
        or geometry["secondPenaltyDistance"] >= length / 2
        or geometry["centerCircleRadius"] >= width / 2
    ):
        raise ValueError("Court geometry has incompatible goal, penalty or circle dimensions.")
    references = []
    for index, name in enumerate(names):
        coordinates = geometry["references"][name]
        if (
            not isinstance(coordinates, list)
            or len(coordinates) != 2
            or any(
                type(v) not in {int, float} or not math.isfinite(v) or not 0 <= v <= bound
                for v, bound in zip(coordinates, (length, width), strict=True)
            )
        ):
            raise ValueError(f"Invalid metric court reference: {name}.")
        references.append(LabeledReference(index, name, name, tuple(coordinates)))
    if len({r.coordinate for r in references}) != 20:
        raise ValueError("Court references must have distinct coordinates.")
    centerX, centerY = length / 2, width / 2
    upperPost, lowerPost = centerY - geometry["goalWidth"] / 2, centerY + geometry["goalWidth"] / 2
    penalty, second, radius = (
        geometry["penaltyDistance"],
        geometry["secondPenaltyDistance"],
        geometry["centerCircleRadius"],
    )
    expected = [
        (0, 0),
        (0, width),
        (0, upperPost),
        (0, lowerPost),
        (penalty, centerY),
        (second, centerY),
        (centerX, 0),
        (centerX, width),
        (centerX, centerY - radius),
        (centerX, centerY + radius),
        (centerX - 10, width),
        (centerX - 5, width),
        (centerX + 5, width),
        (centerX + 10, width),
        (length, 0),
        (length, width),
        (length, upperPost),
        (length, lowerPost),
        (length - penalty, centerY),
        (length - second, centerY),
    ]
    if [r.coordinate for r in references] != expected:
        raise ValueError(
            "Metric references must coincide with the rendered court markings in canonical order."
        )
    return SynthesisModel(
        "futsalCourt",
        **{k: v for k, v in geometry.items() if k != "references"},
        references=tuple(references),
        classNames=dict(enumerate(names)),
    )


def derivedTargets(points, dimensions):
    targets = []
    for index, point in enumerate(points):
        coordinates, transformation = annotationBox(
            {"shapeType": "points", "coordinates": point["imagePixels"]},
            {"coordinateSystem": "pixelPoints", "dimensions": dimensions},
            pointBoxes,
        )
        targets.append(
            {
                "classId": point["classId"],
                "annotationIndex": index,
                "coordinates": coordinates,
                "transformation": transformation,
            }
        )
    return targets


def evidenceFiles(root):
    return [
        {
            "path": p.relative_to(root).as_posix(),
            "byteSize": p.stat().st_size,
            "sha256": hashFile(p),
        }
        for p in scanFiles(root)
    ]


def encodeJpeg(image, quality):
    success, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise ValueError("Cannot encode a synthetic image.")
    return data.tobytes()
