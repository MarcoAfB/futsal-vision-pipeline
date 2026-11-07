import hashlib
import json
import math
import platform
from collections import Counter
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np

from futsal.annotations.detectionGeometry import annotationBox, renderYoloLabels
from futsal.datasets.courtRendering import (
    LabeledReference,
    RenderConfig,
    SynthesisModel,
    generateSample,
    sampleRng,
)
from futsal.datasets.datasetFiles import (
    hashJson,
    readJson,
    stagedDirectory,
    verifyEvidence,
    writeJson,
)
from futsal.inventory.catalog import hashFile, readYaml, resolveSourcePath
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


def generateSyntheticCourt(outputDir, configPath=defaultConfig, progress=print, **overrides):
    config = loadSynthesisConfig(configPath, **overrides)
    from futsal.evaluation.evaluationData import validateResearchOutput

    output = validateResearchOutput(
        outputDir,
        [
            projectRoot / "data/raw",
            projectRoot / "data/prepared",
            projectRoot / "data/projections",
            projectRoot / "data/modelWeights",
            projectRoot / ".git",
        ],
        [configPath],
    )
    model = synthesisModel(config)
    renderConfig = RenderConfig(**{key: config[key] for key in asdict(RenderConfig())})
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "pyyaml": version("PyYAML"),
    }
    generator = {
        "implementation": "nativeCourtRenderingV1",
        "versions": versions,
        "sourceHashes": {
            name: hashFile(Path(__file__).with_name(name))
            for name in ("courtRendering.py", "syntheticCourt.py")
        },
        "parameterProvenance": "portedRenderingDefaultsNoCurrentValidationOrTestCalibration",
    }
    generationId = hashJson(
        {
            "configuration": config,
            "generator": generator,
            "classes": list(model.classNames.values()),
            "pointBoxes": pointBoxes,
        }
    )
    counts = Counter()
    cv2.setNumThreads(1)
    with stagedDirectory(
        output, projectRoot / "data/raw", [configPath, projectRoot / "src", projectRoot / "configs"]
    ) as staging:
        for directory in ("images/train", "labels/train", "debug"):
            (staging / directory).mkdir(parents=True)
        writeJson(staging / "configuration.json", config)
        with (staging / "manifest.jsonl").open("w", encoding="utf-8", newline="\n") as manifest:
            for index in range(config["count"]):
                least = min(counts.get(r.classId, 0) for r in model.references)
                candidates = [r for r in model.references if counts.get(r.classId, 0) == least]
                rng, sampleSeed = sampleRng(config["seed"], index)
                sample = generateSample(
                    model,
                    renderConfig,
                    rng,
                    sampleSeed,
                    candidates[int(rng.integers(len(candidates)))],
                )
                stem = f"syntheticCourt{index:06d}"
                imagePath, labelPath = f"images/train/{stem}.jpg", f"labels/train/{stem}.txt"
                (staging / imagePath).write_bytes(encodeJpeg(sample.image, sample.jpegQuality))
                points = sample.metadata["visiblePoints"]
                targets = derivedTargets(points, [config["width"], config["height"]])
                (staging / labelPath).write_text(
                    renderYoloLabels(targets), encoding="utf-8", newline="\n"
                )
                row = {
                    "recordId": f"syntheticCourt:{generationId}:{index:06d}",
                    "taskId": "courtPoints",
                    "origin": "synthetic",
                    "sourceId": "syntheticCourt",
                    "split": "training",
                    "dimensions": [config["width"], config["height"]],
                    "imagePath": imagePath,
                    "labelPath": labelPath,
                    "imageSha256": hashFile(staging / imagePath),
                    "labelSha256": hashFile(staging / labelPath),
                    "targets": targets,
                    "metadata": sample.metadata,
                }
                manifest.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
                )
                counts.update(p["classId"] for p in points)
                if index < config["debugCount"]:
                    (staging / f"debug/{stem}.jpg").write_bytes(encodeJpeg(sample.debugImage, 94))
                if index == 0 or (index + 1) % 25 == 0 or index + 1 == config["count"]:
                    progress(f"Generated {index + 1}/{config['count']} court images.")
        writeJson(
            staging / "summary.json",
            {
                "imageCount": config["count"],
                "trainingOnly": True,
                "support": {name: counts[index] for index, name in model.classNames.items()},
            },
        )
        contract = {
            "schemaVersion": 1,
            "kind": "syntheticCourtDataset",
            "trainingOnly": True,
            "generationId": generationId,
            "configuration": config,
            "generator": generator,
            "classes": list(model.classNames.values()),
            "pointBoxes": pointBoxes,
            "artifactEvidence": evidenceFiles(staging),
        }
        contract["contractId"] = hashJson(contract)
        writeJson(staging / "contract.json", contract)
        verifySyntheticCourt(staging)
    return contract


def validateSyntheticSnapshot(contract, rows, classes):
    if (
        contract.get("kind") != "syntheticCourtDataset"
        or contract.get("schemaVersion") != 1
        or contract.get("trainingOnly") is not True
        or contract.get("pointBoxes") != pointBoxes
        or contract.get("classes") != classes
    ):
        raise ValueError("Synthetic source must be a training-only canonical court dataset.")
    if contract["contractId"] != hashJson({k: v for k, v in contract.items() if k != "contractId"}):
        raise ValueError("Synthetic contract checksum mismatch.")
    config = contract["configuration"]
    model = synthesisModel(config)
    if len(rows) != config["count"] or not rows:
        raise ValueError("Synthetic manifest count differs from its contract.")
    identity = {
        "configuration": config,
        "generator": contract["generator"],
        "classes": classes,
        "pointBoxes": pointBoxes,
    }
    if contract["generationId"] != hashJson(identity):
        raise ValueError("Synthetic generation identity mismatch.")
    counts = Counter()
    for index, row in enumerate(rows):
        stem = f"syntheticCourt{index:06d}"
        if row["recordId"] != f"syntheticCourt:{contract['generationId']}:{index:06d}" or any(
            row.get(key) != value
            for key, value in {
                "origin": "synthetic",
                "sourceId": "syntheticCourt",
                "taskId": "courtPoints",
                "split": "training",
                "dimensions": [config["width"], config["height"]],
                "imagePath": f"images/train/{stem}.jpg",
                "labelPath": f"labels/train/{stem}.txt",
            }.items()
        ):
            raise ValueError("Synthetic record identity, paths or development membership changed.")
        points = row["metadata"]["visiblePoints"]
        ids = [p["classId"] for p in points]
        if (
            not 4 <= len(ids) <= 11
            or len(set(ids)) != len(ids)
            or any(
                type(i) is not int
                or not 0 <= i < 20
                or p["className"] != classes[i]
                or p["courtMeters"] != list(model.references[i].coordinate)
                or len(p["imagePixels"]) != 2
                or any(
                    not math.isfinite(v) or not 0 <= v < bound
                    for v, bound in zip(p["imagePixels"], row["dimensions"], strict=True)
                )
                for i, p in zip(ids, points, strict=True)
            )
        ):
            raise ValueError("Invalid synthetic native point annotations.")
        if row["targets"] != derivedTargets(points, row["dimensions"]):
            raise ValueError("Synthetic detection boxes differ from their native points.")
        if (
            row["labelSha256"]
            != hashlib.sha256(renderYoloLabels(row["targets"]).encode()).hexdigest()
        ):
            raise ValueError("Synthetic labels differ from native-derived boxes.")
        counts.update(ids)
    if len(rows) >= 20 and set(counts) != set(range(20)):
        raise ValueError("Synthetic generation lost court class coverage.")
    return counts


def readSyntheticManifest(root):
    return [
        json.loads(line)
        for line in (Path(root) / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def verifySyntheticCourt(datasetDir):
    root = Path(datasetDir).resolve()
    contract, rows = readJson(root / "contract.json"), readSyntheticManifest(root)
    counts = validateSyntheticSnapshot(contract, rows, courtClassNames())
    expected = {item["path"] for item in contract["artifactEvidence"]} | {"contract.json"}
    if len(expected) != len(contract["artifactEvidence"]) + 1:
        raise ValueError("Synthetic evidence contains duplicate or self-referential paths.")
    if {p.relative_to(root).as_posix() for p in scanFiles(root)} != expected:
        raise ValueError("Synthetic dataset has missing or unexpected files.")
    verifyEvidence(root, contract["artifactEvidence"])
    allowed = {"contract.json", "manifest.jsonl", "summary.json", "configuration.json"}
    for index, row in enumerate(rows):
        image = cv2.imread(str(resolveSourcePath(root, row["imagePath"])))
        if image is None or [image.shape[1], image.shape[0]] != row["dimensions"]:
            raise ValueError("Synthetic image cannot be decoded at its declared dimensions.")
        for kind in ("image", "label"):
            if hashFile(root / row[f"{kind}Path"]) != row[f"{kind}Sha256"]:
                raise ValueError("Synthetic manifest file hash mismatch.")
            allowed.add(row[f"{kind}Path"])
        if index < contract["configuration"]["debugCount"]:
            allowed.add(f"debug/syntheticCourt{index:06d}.jpg")
    if expected != allowed or readJson(root / "configuration.json") != contract["configuration"]:
        raise ValueError("Synthetic file layout or configuration differs from its contract.")
    summary = {
        "imageCount": len(rows),
        "trainingOnly": True,
        "support": {name: counts[i] for i, name in enumerate(contract["classes"])},
    }
    if readJson(root / "summary.json") != summary:
        raise ValueError("Synthetic support summary differs from its native annotations.")
    return contract
