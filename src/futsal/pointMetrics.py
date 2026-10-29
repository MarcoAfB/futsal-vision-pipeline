import math

import numpy as np


def pointErrors(references, predictions, pointNames, config):
    rows = []
    for reference, prediction in zip(references, predictions, strict=True):
        width, height = reference["dimensions"]
        diagonal = math.hypot(width, height)
        for classId, name in enumerate(pointNames):
            if name is None:
                continue
            truth = [item for item in reference["targets"] if item["classId"] == classId]
            if len(truth) > 1:
                raise ValueError("Eligible point references must be unique per class and image.")
            detections = [item for item in prediction["detections"] if item["classId"] == classId
                          and item["confidence"] >= config["operationalConfidence"]]
            selected = max(detections, key=lambda item: item["confidence"]) if detections else None
            native = truth[0]["point"] if truth else None
            predicted = ([(selected["box"][0] + selected["box"][2]) / 2,
                          (selected["box"][1] + selected["box"][3]) / 2] if selected else None)
            distance = math.dist(native, predicted) if native is not None and predicted is not None else None
            box = truth[0]["box"] if truth else None
            proxyOffset = math.dist(native, [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]) if box else None
            rows.append({"recordId": reference["recordId"], "matchId": reference["matchId"],
                         "classId": classId, "className": name, "annotated": bool(truth),
                         "nativePoint": native, "predictedPoint": predicted,
                         "confidence": selected["confidence"] if selected else None,
                         "predictionCount": len(detections), "distancePixels": distance,
                         "distanceDiagonal": distance / diagonal if distance is not None else None,
                         "targetCenterOffsetPixels": proxyOffset,
                         "targetCenterOffsetDiagonal": proxyOffset / diagonal if proxyOffset is not None else None,
                         "truncated": truth[0]["truncated"] if truth else False})
    return rows


def summarizePoints(rows, names, config):
    summaries = []
    for name in names:
        subset = [row for row in rows if row["className"] == name]
        annotated = [row for row in subset if row["annotated"]]
        distances = [row["distanceDiagonal"] for row in annotated if row["distanceDiagonal"] is not None]
        correct = sum(value <= config["primaryPointTolerance"] for value in distances)
        predictionCount = sum(row["predictionCount"] for row in subset)
        summaries.append({"className": name, "support": len(annotated), "localizedCount": len(distances),
                          "missingCount": len(annotated) - len(distances), "predictions": predictionCount,
                          "extraPredictions": sum(max(0, row["predictionCount"] - int(row["annotated"])) for row in subset),
                          "truncatedTargets": sum(row["truncated"] for row in annotated),
                          "medianDistanceDiagonal": float(np.median(distances)) if distances else None,
                          "p90DistanceDiagonal": float(np.quantile(distances, 0.9)) if distances else None,
                          "medianTargetCenterOffsetDiagonal": float(np.median([
                              row["targetCenterOffsetDiagonal"] for row in annotated])) if annotated else None,
                          "localizationPrecision": correct / predictionCount if annotated and predictionCount else
                          (0.0 if annotated else None),
                          "successRates": {str(tolerance): sum(value <= tolerance for value in distances) / len(annotated)
                                           if annotated else None for tolerance in config["pointTolerances"]}})
    support = sum(row["support"] for row in summaries)
    rates = {str(t): sum(row["distanceDiagonal"] is not None and row["distanceDiagonal"] <= t for row in rows
                        if row["annotated"]) / support if support else None for t in config["pointTolerances"]}
    return {"perClass": summaries, "support": support, "successRates": rates}


def scorePoints(references, predictions, names, pointClasses, config):
    indexed = [name if name in pointClasses else None for name in names]
    rows = pointErrors(references, predictions, indexed, config)
    pointNames = [name for name in names if name in pointClasses]
    return {**summarizePoints(rows, pointNames, config), "errors": rows,
            "byMatch": {matchId: summarizePoints([row for row in rows if row["matchId"] == matchId],
                                                  pointNames, config)
                        for matchId in sorted({row["matchId"] for row in references})}}
