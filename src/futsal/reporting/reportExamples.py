import hashlib

import cv2
import numpy as np


def classColour(name):
    digest = hashlib.sha256(name.encode()).digest()
    return tuple(70 + value % 160 for value in digest[:3])


def drawReference(image, reference):
    for target in reference["targets"]:
        colour = classColour(target["className"])
        label = str(target["classId"])
        if target["point"] is not None:
            x, y = [round(value) for value in target["point"]]
            cv2.drawMarker(image, (x, y), colour, cv2.MARKER_CROSS, 16, 2)
        else:
            left, top, right, bottom = [round(value) for value in target["box"]]
            cv2.rectangle(image, (left, top), (right, bottom), colour, 2)
            x, y = left, top
        cv2.putText(image, label, (max(0, x + 4), max(16, y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, colour, 2, cv2.LINE_AA)


def drawPredictions(image, prediction, names, config, pointClasses, pointErrors):
    byClass = {row["classId"]: row for row in pointErrors if row["nativePoint"] is not None}
    for item in prediction["detections"]:
        if item["confidence"] < config["operationalConfidence"]:
            continue
        name, box = names[item["classId"]], item["box"]
        colour = classColour(name)
        left, top, right, bottom = [round(value) for value in box]
        if name in pointClasses:
            x, y = round((left + right) / 2), round((top + bottom) / 2)
            cv2.circle(image, (x, y), 7, colour, 2)
            error = byClass.get(item["classId"])
            if error is not None and abs(item["confidence"] - error["confidence"]) < 1e-8:
                truth = tuple(round(value) for value in error["nativePoint"])
                cv2.line(image, truth, (x, y), colour, 2)
        else:
            cv2.rectangle(image, (left, top), (right, bottom), colour, 2)
            x, y = left, top
        cv2.putText(image, f"{item['classId']} {item['confidence']:.2f}", (max(0, x + 4), max(16, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2, cv2.LINE_AA)


def examplePanel(image, reference, prediction, names, config, pointClasses, pointErrors):
    displayScale = min(1, 1000 / image.shape[1])
    if displayScale < 1:
        image = cv2.resize(image, (round(image.shape[1] * displayScale), round(image.shape[0] * displayScale)),
                            interpolation=cv2.INTER_AREA)
    scaledReference = {**reference, "targets": [{**item, "box": [value * displayScale for value in item["box"]],
                        "point": [value * displayScale for value in item["point"]] if item["point"] is not None else None}
                       for item in reference["targets"]]}
    scaledPrediction = {**prediction, "detections": [{**item, "box": [value * displayScale for value in item["box"]]}
                                                    for item in prediction["detections"]]}
    scaledErrors = [{**row, "nativePoint": [value * displayScale for value in row["nativePoint"]]
                    if row["nativePoint"] is not None else None} for row in pointErrors]
    truth, predicted = image.copy(), image.copy()
    drawReference(truth, scaledReference)
    drawPredictions(predicted, scaledPrediction, names, config, pointClasses, scaledErrors)
    panels = np.hstack((truth, predicted))
    height, width = image.shape[:2]
    balls = [target for target in scaledReference["targets"] if target["className"] == "ball"]
    zooms = []
    if balls:
        box = balls[0]["box"]
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        radius = max(32, 2 * max(box[2] - box[0], box[3] - box[1]))
        x1, y1, x2, y2 = max(0, int(cx - radius)), max(0, int(cy - radius)), min(width, int(cx + radius)), min(height, int(cy + radius))
        zoomSize = min(240, width // 3)
        for canvas in (truth, predicted):
            crop = cv2.resize(canvas[y1:y2, x1:x2], (zoomSize, zoomSize), interpolation=cv2.INTER_NEAREST)
            zooms.append(crop)
        strip = np.full((zoomSize + 32, panels.shape[1], 3), 255, dtype=np.uint8)
        for index, crop in enumerate(zooms):
            strip[32:, index * width:index * width + zoomSize] = crop
            cv2.putText(strip, "Ball detail at annotated location", (index * width + 12, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
        panels = np.vstack((panels, strip))
    scale = min(2000, max(1200, panels.shape[1])) / panels.shape[1]
    panels = cv2.resize(panels, (round(panels.shape[1] * scale), round(panels.shape[0] * scale)),
                        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST)
    header = np.full((64, panels.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(header, "Reference (crosses = native points)", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
    cv2.putText(header, f"Prediction (confidence >= {config['operationalConfidence']:g})", (panels.shape[1] // 2 + 12, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
    cv2.putText(header, f"{reference['recordId']} | {reference['matchId']} | labels = class IDs | ball inset when annotated",
                (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1)
    return np.vstack((header, panels))


def renderExamples(root, stage, references, predictions, points, names, config, pointClasses, directory):
    directory.mkdir(parents=True)
    byId = {row["recordId"]: row for row in references}
    predicted = {row["recordId"]: row for row in predictions}
    result = []
    for index, selected in enumerate(stage["examples"], start=1):
        recordId = selected["recordId"]
        image = cv2.imread(str(root / selected["image"]))
        if image is None:
            raise ValueError(f"Archived example image is unreadable: {recordId}.")
        rows = [row for row in points["errors"] if row["recordId"] == recordId]
        panel = examplePanel(image, byId[recordId], predicted[recordId], names, config, pointClasses, rows)
        filename = f"example{index:02d}.png"
        if not cv2.imwrite(str(directory / filename), panel):
            raise OSError("Cannot write qualitative example.")
        result.append({**selected, "figure": filename})
    return result
