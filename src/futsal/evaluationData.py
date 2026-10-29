import json
import math
from pathlib import Path

from futsal.catalog import hashFile, readYaml
from futsal.datasetFiles import readJson
from futsal.recipes import activeTasks


def loadEvaluationConfig(path):
    config, _ = readYaml(Path(path))
    fractions = {"confidenceFloor", "operationalConfidence", "operationalIou", "primaryPointTolerance"}
    integers = {"maxDetections", "seed", "fixedExampleCount", "hardExampleCount"}
    if set(config) != fractions | integers | {"schemaVersion", "apIouThresholds", "pointTolerances"}:
        raise ValueError("Unexpected or missing evaluation configuration fields.")
    if config["schemaVersion"] != 1:
        raise ValueError("Only evaluation schemaVersion 1 is supported.")
    for key in fractions:
        value = config[key]
        if type(value) not in {int, float} or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{key} must be finite and in (0, 1].")
    for key in integers:
        if type(config[key]) is not int or config[key] < (0 if key == "seed" else 1):
            raise ValueError(f"{key} must be an integer in its supported range.")
    if config["confidenceFloor"] > config["operationalConfidence"]:
        raise ValueError("The confidence floor must not exceed the operational threshold.")
    if config["apIouThresholds"] != [round(0.5 + index * 0.05, 2) for index in range(10)]:
        raise ValueError("AP requires the fixed IoU thresholds 0.50:0.95.")
    values = config["pointTolerances"]
    if not isinstance(values, list) or not values or any(
        type(value) not in {int, float} or not math.isfinite(value) or not 0 < value <= 1
        for value in values
    ) or values != sorted(set(values)) or config["primaryPointTolerance"] not in values:
        raise ValueError("Point tolerances must increase uniquely and include the primary tolerance.")
    return config


def writeJsonLines(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def readJsonLines(path):
    def rejectConstant(value):
        raise ValueError(f"Nonfinite JSON value: {value}.")

    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line, parse_constant=rejectConstant) for line in stream if line.strip()]


def evaluationTasks(datasetDir, runDir, run, selection):
    if selection not in {"runValidation", "fullValidation"}:
        raise ValueError("Evaluation is restricted to runValidation or fullValidation.")
    root = Path(datasetDir)
    plan, manifest = readJson(root / "plan.json"), readJson(root / "manifest.json")
    saved = readJson(Path(runDir) / "selection.json")
    if hashFile(Path(runDir) / "selection.json") != run["selectionSha256"]:
        raise ValueError("Training selection changed since the run.")
    tasks = {task["taskId"]: task for task in activeTasks(plan["recipe"])}
    result = {}
    for stage in run["configuration"]["stages"]:
        taskId = stage["taskId"]
        task = tasks[taskId]
        rows = [row for row in manifest["samples"]
                if row["taskId"] == taskId and row["split"] == "validation"]
        byId = {row["recordId"]: row for row in rows}
        if selection == "runValidation":
            requested = saved["tasks"][taskId]["validation"]
            if any(byId.get(row["recordId"]) != row for row in requested):
                raise ValueError("Saved validation selection differs from the prepared manifest.")
            rows = requested
        if not rows or any(row["origin"] != "own" for row in rows):
            raise ValueError("Evaluation requires eligible own validation records.")
        result[taskId] = {**task, "rows": sorted(rows, key=lambda row: row["recordId"])}
    return plan, result


def buildReferences(plan, task):
    samples = {sample["recordId"]: sample for sample in plan["samples"]}
    references = []
    for row in task["rows"]:
        width, height = row["dimensions"]
        targets = []
        for item in row["targets"]:
            native = samples[row["recordId"]]["annotations"][item["annotationIndex"]]
            x, y, boxWidth, boxHeight = [float(f"{value:.12f}") for value in item["coordinates"]]
            box = [(x - boxWidth / 2) * width, (y - boxHeight / 2) * height,
                   (x + boxWidth / 2) * width, (y + boxHeight / 2) * height]
            targets.append({"annotationIndex": item["annotationIndex"], "classId": item["classId"],
                            "className": task["classes"][item["classId"]], "box": box,
                            "point": native["coordinates"] if native["shapeType"] == "points" else None,
                            "truncated": item["transformation"]["truncated"]})
        references.append({key: row[key] for key in ("recordId", "taskId", "matchId", "sourceId",
                           "frameIndex", "dimensions", "imagePath", "imageSha256", "labelSha256")}
                          | {"targets": targets})
    return references


def validatePredictions(references, predictions, names, config):
    byId = {row["recordId"]: row for row in predictions}
    if len(byId) != len(predictions) or set(byId) != {row["recordId"] for row in references}:
        raise ValueError("Prediction membership is missing, duplicated or unexpected.")
    for reference in references:
        row = byId[reference["recordId"]]
        if row["dimensions"] != reference["dimensions"]:
            raise ValueError("Prediction dimensions differ from the original image.")
        detections = row["detections"]
        if not isinstance(detections, list) or len(detections) > config["maxDetections"]:
            raise ValueError("Prediction count exceeds the declared detection limit.")
        if "correctAtIou" in row and (len(row["correctAtIou"]) != len(detections) or any(
            len(values) != 10 or any(type(value) is not bool for value in values)
            for values in row["correctAtIou"]
        )):
            raise ValueError("Native AP evidence must contain ten boolean matches per detection.")
        for detection in detections:
            classId, score, box = detection["classId"], detection["confidence"], detection["box"]
            if type(classId) is not int or not 0 <= classId < len(names) or (
                detection["className"] != names[classId]
            ):
                raise ValueError("Prediction class differs from the stage ontology.")
            if type(score) not in {int, float} or not math.isfinite(score) or not config["confidenceFloor"] <= score <= 1:
                raise ValueError("Prediction confidence is outside the recorded protocol.")
            if len(box) != 4 or not all(type(value) in {int, float} and math.isfinite(value) for value in box):
                raise ValueError("Prediction boxes require four finite pixel coordinates.")
            width, height = reference["dimensions"]
            if not (-1e-4 <= box[0] <= box[2] <= width + 1e-4 and
                    -1e-4 <= box[1] <= box[3] <= height + 1e-4):
                raise ValueError("Prediction boxes must stay in original-image coordinates.")
    return [byId[row["recordId"]] for row in references]
