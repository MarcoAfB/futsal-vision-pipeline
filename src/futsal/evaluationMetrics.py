from collections import Counter

import numpy as np


def boxOverlaps(targets, detections):
    truth = np.array([target["box"] for target in targets], dtype=float).reshape(-1, 4)
    predicted = np.array([item["box"] for item in detections], dtype=float).reshape(-1, 4)
    intersection = np.maximum(0, np.minimum(truth[:, None, 2:], predicted[None, :, 2:]) -
                              np.maximum(truth[:, None, :2], predicted[None, :, :2])).prod(2)
    area = np.maximum(0, truth[:, 2:] - truth[:, :2]).prod(1)
    predictedArea = np.maximum(0, predicted[:, 2:] - predicted[:, :2]).prod(1)
    return intersection / np.maximum(area[:, None] + predictedArea[None, :] - intersection, 1e-16)


def matchBoxes(targets, detections, threshold, sameClass=False):
    overlap = boxOverlaps(targets, detections)
    available = set(range(len(targets)))
    matches = []
    order = sorted(range(len(detections)), key=lambda i: (-detections[i]["confidence"], i))
    for predictionIndex in order:
        candidates = [index for index in sorted(available) if not sameClass or
                      targets[index]["classId"] == detections[predictionIndex]["classId"]]
        if not candidates:
            continue
        referenceIndex = max(candidates, key=lambda index: overlap[index, predictionIndex])
        value = float(overlap[referenceIndex, predictionIndex])
        if value >= threshold:
            matches.append((referenceIndex, predictionIndex, value))
            available.remove(referenceIndex)
    return matches


def operationalMetrics(references, predictions, names, config):
    count = len(names)
    matrix = np.zeros((count + 1, count + 1), dtype=int)
    evidence, imageErrors = [], []
    for reference, prediction in zip(references, predictions, strict=True):
        targets = reference["targets"]
        detections = [{**item, "rawIndex": index} for index, item in enumerate(prediction["detections"])
                      if item["confidence"] >= config["operationalConfidence"]]
        matches = matchBoxes(targets, detections, config["operationalIou"])
        usedTruth, usedPredicted = set(), set()
        errors = {"recordId": reference["recordId"], "matchId": reference["matchId"],
                  "falsePositives": 0, "falseNegatives": 0, "classConfusions": 0, "ballMisses": 0,
                  "highestFalsePositiveConfidence": 0.0}
        for truthIndex, predictionIndex, overlap in matches:
            truthClass, predictedClass = targets[truthIndex]["classId"], detections[predictionIndex]["classId"]
            matrix[truthClass, predictedClass] += 1
            usedTruth.add(truthIndex)
            usedPredicted.add(predictionIndex)
            correct = truthClass == predictedClass
            evidence.append({"recordId": reference["recordId"], "referenceIndex": truthIndex,
                             "predictionIndex": detections[predictionIndex]["rawIndex"], "iou": overlap,
                             "kind": "truePositive" if correct else "classConfusion"})
            if not correct:
                errors["classConfusions"] += 1
                errors["falsePositives"] += 1
                errors["falseNegatives"] += 1
                errors["ballMisses"] += names[truthClass] == "ball"
                errors["highestFalsePositiveConfidence"] = max(
                    errors["highestFalsePositiveConfidence"], detections[predictionIndex]["confidence"])
        for index, item in enumerate(targets):
            if index not in usedTruth:
                matrix[item["classId"], count] += 1
                errors["falseNegatives"] += 1
                errors["ballMisses"] += names[item["classId"]] == "ball"
                evidence.append({"recordId": reference["recordId"], "referenceIndex": index,
                                 "predictionIndex": None, "iou": None, "kind": "falseNegative"})
        for index, item in enumerate(detections):
            if index not in usedPredicted:
                matrix[count, item["classId"]] += 1
                errors["falsePositives"] += 1
                errors["highestFalsePositiveConfidence"] = max(errors["highestFalsePositiveConfidence"],
                                                               item["confidence"])
                evidence.append({"recordId": reference["recordId"], "referenceIndex": None,
                                 "predictionIndex": item["rawIndex"], "iou": None, "kind": "falsePositive"})
        imageErrors.append(errors)
    perClass = []
    for classId, name in enumerate(names):
        support, predicted = int(matrix[classId].sum()), int(matrix[:, classId].sum())
        truePositives = int(matrix[classId, classId])
        precision = truePositives / predicted if predicted else 0.0
        recall = truePositives / support if support else 0.0
        perClass.append({"classId": classId, "className": name, "support": support,
                         "imageSupport": sum(any(t["classId"] == classId for t in row["targets"])
                                             for row in references), "predictions": predicted,
                         "truePositives": truePositives, "falsePositives": predicted - truePositives,
                         "falseNegatives": support - truePositives,
                         "precision": precision if support else None, "recall": recall if support else None,
                         "f1": (2 * precision * recall / (precision + recall) if precision + recall else 0.0)
                         if support else None})
    return {"perClass": perClass, "confusionLabels": [*names, "background"],
            "confusionMatrix": matrix.tolist(), "matches": evidence, "imageErrors": imageErrors}


def nativeAveragePrecision(references, predictions, names, config):
    from ultralytics.utils.metrics import ap_per_class

    correct, confidence, predictedClasses, targetClasses, evidence = [], [], [], [], []
    for reference, prediction in zip(references, predictions, strict=True):
        detections = prediction["detections"]
        if "correctAtIou" in prediction:
            mask = np.array(prediction["correctAtIou"], dtype=bool).reshape(len(detections), 10)
        else:
            mask = np.zeros((len(detections), 10), dtype=bool)
            for column, threshold in enumerate(config["apIouThresholds"]):
                for _, index, _ in matchBoxes(reference["targets"], detections, threshold, sameClass=True):
                    mask[index, column] = True
        correct.extend(mask)
        confidence.extend(item["confidence"] for item in detections)
        predictedClasses.extend(item["classId"] for item in detections)
        targetClasses.extend(item["classId"] for item in reference["targets"])
        evidence.append({"recordId": reference["recordId"], "correctAtIou": mask.tolist()})
    values = ap_per_class(np.array(correct, dtype=bool).reshape(-1, 10), np.array(confidence),
                          np.array(predictedClasses), np.array(targetClasses), names=dict(enumerate(names)),
                          plot=False)
    byClass = {int(classId): index for index, classId in enumerate(values[6])}
    perClass, curves = {}, {}
    for classId, name in enumerate(names):
        index = byClass.get(classId)
        perClass[name] = {"ap50": float(values[5][index, 0]) if index is not None else None,
                          "ap50To95": float(values[5][index].mean()) if index is not None else None}
        if index is not None:
            curves[name] = {"recall": values[10].tolist(), "precision": values[11][index].tolist()}
    return {"perClass": perClass, "prCurves": curves, "matches": evidence}


def aggregateGroups(perClass, pointClasses):
    groups = {"all": perClass, "ball": [row for row in perClass if row["className"] == "ball"],
              "athletes": [row for row in perClass if row["className"] != "ball" and
                           row["className"] not in pointClasses],
              "courtLandmarks": [row for row in perClass if row["className"] in pointClasses]}
    result = {}
    for name, rows in groups.items():
        if not rows:
            continue
        supported = [row for row in rows if row["support"]]
        result[name] = {"classCount": len(rows), "supportedClassCount": len(supported),
                        "support": sum(row["support"] for row in rows),
                        **{key: float(np.mean([row[key] for row in supported])) if supported else None
                           for key in ("ap50", "ap50To95", "precision", "recall", "f1")}}
    return result


def scoreDetectionSubset(references, predictions, names, pointClasses, config):
    operational = operationalMetrics(references, predictions, names, config)
    averagePrecision = nativeAveragePrecision(references, predictions, names, config)
    for row in operational["perClass"]:
        row.update(averagePrecision["perClass"][row["className"]])
    return {**operational, "groups": aggregateGroups(operational["perClass"], pointClasses),
            "prCurves": averagePrecision["prCurves"], "apMatches": averagePrecision["matches"],
            "imageCount": len(references),
            "matchCounts": dict(Counter(row["matchId"] for row in references))}


def scoreDetections(references, predictions, names, pointClasses, config):
    result = scoreDetectionSubset(references, predictions, names, pointClasses, config)
    result["byMatch"] = {}
    for matchId in sorted(result["matchCounts"]):
        pairs = [(r, p) for r, p in zip(references, predictions, strict=True) if r["matchId"] == matchId]
        subset = scoreDetectionSubset([r for r, _ in pairs], [p for _, p in pairs], names, pointClasses, config)
        result["byMatch"][matchId] = {key: subset[key] for key in ("imageCount", "perClass", "groups")}
    return result
