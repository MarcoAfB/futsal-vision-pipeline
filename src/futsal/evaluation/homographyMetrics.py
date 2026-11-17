import math
from collections import Counter

import numpy as np

from futsal.evaluation.evaluationMetrics import matchBoxes
from futsal.geometry.courtModel import landmarkNames
from futsal.geometry.courtProjection import (
    groundPoints,
    imageToCourt,
    insideCourt,
    metresPerPixel,
    projectPoints,
)
from futsal.geometry.homographyEstimation import (
    estimateHomography,
    hasGeneralSubset,
    orientedHomography,
)

athleteClasses = {"player", "outfieldPlayerTeamA", "outfieldPlayerTeamB", "goalkeeperTeamA",
                  "goalkeeperTeamB"}
acceptedStatuses = ("verified", "minimal")


def finiteOrNone(value):
    return float(value) if value is not None and math.isfinite(value) else None


def nearestRank(values, fraction):
    ordered = np.sort(np.asarray(values, dtype=float))
    return float(ordered[max(0, math.ceil(fraction * len(ordered)) - 1)])


def distribution(values):
    values = np.asarray([value for value in values if value is not None], dtype=float)
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "p90": None, "max": None}
    return {"count": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9)), "max": float(values.max())}


def acceptedMatrix(estimate):
    return np.asarray(estimate["courtToImage"]) if estimate["status"] in acceptedStatuses else None


def referenceCorrespondences(reference, court):
    targets = [target for target in reference["targets"]
               if target["className"] in court["landmarks"] and target["point"] is not None]
    return ([target["className"] for target in targets],
            [court["landmarks"][target["className"]] for target in targets],
            [target["point"] for target in targets])


def leaveOneOut(court, image, config):
    court, image = np.asarray(court, dtype=float), np.asarray(image, dtype=float)
    errors = []
    for index in range(len(court)):
        keep = np.arange(len(court)) != index
        if keep.sum() < 4 or not hasGeneralSubset(court[keep], config["minTriangleAreaSquareMeters"]):
            continue
        matrix = orientedHomography(court[keep], image[keep])
        if matrix is None:
            continue
        projected, scale = projectPoints(matrix, court[index])
        if scale[0] > 0:
            errors.append(float(np.linalg.norm(projected[0] - image[index])))
    return errors


def referenceHomography(reference, court, config):
    names, courtPoints, imagePoints = referenceCorrespondences(reference, court)
    estimate = estimateHomography(courtPoints, imagePoints, reference["dimensions"], config, robust=False)
    accepted = estimate["status"] in acceptedStatuses
    return {**estimate, "classNames": names, "imagePoints": imagePoints,
            "leaveOneOutPixels": leaveOneOut(courtPoints, imagePoints, config) if accepted else []}


def touchesBorder(box, dimensions):
    return box[0] <= 0.5 or box[1] <= 0.5 or box[2] >= dimensions[0] - 0.5 or box[3] >= dimensions[1] - 0.5


def predictedCandidates(prediction, court, confidence, dimensions):
    best = {}
    for item in prediction["detections"]:
        name = item["className"]
        if name in court["landmarks"] and item["confidence"] >= confidence and (
                name not in best or item["confidence"] > best[name]["confidence"]):
            best[name] = item
    return [{"className": name, "confidence": item["confidence"],
             "image": [(item["box"][0] + item["box"][2]) / 2, (item["box"][1] + item["box"][3]) / 2],
             "court": list(court["landmarks"][name]), "edgeTruncated": touchesBorder(item["box"], dimensions)}
            for name, item in sorted(best.items(), key=lambda pair: landmarkNames.index(pair[0]))]


def predictedHomography(prediction, court, dimensions, config, confidence):
    candidates = predictedCandidates(prediction, court, confidence, dimensions)
    estimate = estimateHomography([item["court"] for item in candidates], [item["image"] for item in candidates],
                                  dimensions, config, robust=True)
    inliers, residuals = estimate.pop("inliers"), estimate.pop("residualsPixels")
    for item, inlier, residual in zip(candidates, inliers, residuals, strict=True):
        item.update(inlier=inlier, residualPixels=residual)
    return {**estimate, "candidates": candidates}


def courtGrid(court, step):
    xs = np.arange(0.0, court["length"] + step / 2, step)
    ys = np.arange(0.0, court["width"] + step / 2, step)
    return np.array([(x, y) for y in ys for x in xs])


def gridErrors(referenceMatrix, predictedMatrix, court, dimensions, step):
    grid = courtGrid(court, step)
    image, scale = projectPoints(referenceMatrix, grid)
    visible = ((scale > 0) & np.isfinite(image).all(1) & (image[:, 0] >= 0) & (image[:, 1] >= 0) &
               (image[:, 0] < dimensions[0]) & (image[:, 1] < dimensions[1]))
    if not visible.any():
        return None
    mapped, mappedScale = imageToCourt(predictedMatrix, image[visible])
    errors = np.linalg.norm(mapped - grid[visible], axis=1)
    errors[(mappedScale <= 0) | ~np.isfinite(errors)] = np.inf
    return errors


def gridSummary(errors):
    if errors is None:
        return None
    finite = errors[np.isfinite(errors)]
    return {"pointCount": len(errors), "behindCount": int(len(errors) - len(finite)),
            "meanFiniteMeters": float(finite.mean()) if len(finite) else None,
            "medianMeters": finiteOrNone(nearestRank(errors, 0.5)),
            "p90Meters": finiteOrNone(nearestRank(errors, 0.9)), "maxMeters": finiteOrNone(errors.max())}


def visibleCourtIou(referenceMatrix, predictedMatrix, court, dimensions, step):
    us = np.arange(step / 2, dimensions[0], step)
    vs = np.arange(step / 2, dimensions[1], step)
    pixels = np.array([(u, v) for v in vs for u in us])
    masks = []
    for matrix in (referenceMatrix, predictedMatrix):
        mapped, scale = imageToCourt(matrix, pixels)
        masks.append((scale > 0) & np.isfinite(mapped).all(1) & insideCourt(mapped, court, 0.0))
    union = int((masks[0] | masks[1]).sum())
    return float((masks[0] & masks[1]).sum() / union) if union else None


def landmarkTransfer(predictedMatrix, reference, court):
    names, courtPoints, imagePoints = referenceCorrespondences(reference, court)
    if not names:
        return []
    projected, scale = projectPoints(predictedMatrix, courtPoints)
    errors = np.linalg.norm(projected - np.asarray(imagePoints, dtype=float), axis=1)
    return [{"className": name, "errorPixels": finiteOrNone(error) if positive > 0 else None}
            for name, error, positive in zip(names, errors, scale, strict=True)]


def courtPosition(matrix, imagePoint):
    mapped, scale = imageToCourt(matrix, imagePoint)
    return mapped[0] if scale[0] > 0 else None


def courtDistance(position, truth):
    return None if position is None else float(np.linalg.norm(position - truth))


def athleteErrors(reference, prediction, referenceMatrix, predictedMatrix, court, config):
    targets = [target for target in reference["targets"] if target["className"] in athleteClasses]
    detections = [item for item in prediction["detections"]
                  if item["className"] in athleteClasses and item["confidence"] >= config["athleteConfidence"]]
    matches = {target: detection for target, detection, _ in
               matchBoxes(targets, detections, config["athleteMatchIou"])}
    feet = groundPoints([target["box"] for target in targets])
    predictedFeet = groundPoints([item["box"] for item in detections])
    truth, truthScale = imageToCourt(referenceMatrix, feet)
    scales = metresPerPixel(referenceMatrix, feet) if len(feet) else []
    rows = []
    for index, target in enumerate(targets):
        row = {"recordId": reference["recordId"], "matchId": reference["matchId"],
               "annotationIndex": target["annotationIndex"], "className": target["className"],
               "truthAvailable": bool(truthScale[index] > 0), "matched": index in matches,
               "truthMeters": None, "insideCourt": None, "metresPerPixel": None,
               "courtOnlyMeters": None, "detectionMeters": None, "endToEndMeters": None,
               "endToEndPositionMeters": None}
        if row["truthAvailable"]:
            row.update(truthMeters=truth[index].tolist(), metresPerPixel=float(scales[index]),
                       insideCourt=bool(insideCourt(truth[index], court, config["courtMarginMeters"])[0]))
            if predictedMatrix is not None:
                row["courtOnlyMeters"] = courtDistance(courtPosition(predictedMatrix, feet[index]), truth[index])
            if index in matches:
                foot = predictedFeet[matches[index]]
                row["detectionMeters"] = courtDistance(courtPosition(referenceMatrix, foot), truth[index])
                if predictedMatrix is not None:
                    position = courtPosition(predictedMatrix, foot)
                    row["endToEndMeters"] = courtDistance(position, truth[index])
                    row["endToEndPositionMeters"] = None if position is None else position.tolist()
        rows.append(row)
    return rows, {"detections": len(detections), "matched": len(matches)}


def geometryOutcome(referenceMatrix, estimate, reference, court, config):
    matrix = None if estimate["courtToImage"] is None else np.asarray(estimate["courtToImage"])
    if referenceMatrix is None or matrix is None:
        return None
    dimensions = reference["dimensions"]
    return {"grid": gridSummary(gridErrors(referenceMatrix, matrix, court, dimensions, config["gridStepMeters"])),
            "visibleIou": visibleCourtIou(referenceMatrix, matrix, court, dimensions, config["iouSampleStepPixels"]),
            "landmarkTransfer": landmarkTransfer(matrix, reference, court)}


def evaluateFrame(reference, prediction, court, config, athletes=None):
    dimensions = reference["dimensions"]
    referenceEstimate = referenceHomography(reference, court, config)
    referenceMatrix = acceptedMatrix(referenceEstimate)
    predicted = predictedHomography(prediction, court, dimensions, config, config["pointConfidence"])
    diagnostic = {}
    for confidence in config["diagnosticConfidences"]:
        estimate = (predicted if confidence == config["pointConfidence"] else
                    predictedHomography(prediction, court, dimensions, config, confidence))
        outcome = geometryOutcome(referenceMatrix, estimate, reference, court, config)
        diagnostic[str(confidence)] = {"status": estimate["status"], "pointCount": estimate["pointCount"],
                                       "p90Meters": outcome["grid"]["p90Meters"]
                                       if outcome and outcome["grid"] else None}
    row = {"recordId": reference["recordId"], "matchId": reference["matchId"],
           "sourceId": reference["sourceId"], "frameIndex": reference["frameIndex"],
           "imageSha256": reference["imageSha256"], "dimensions": dimensions,
           "court": {key: court[key] for key in ("length", "width", "dimensionSource")},
           "reference": referenceEstimate, "prediction": predicted,
           "geometry": geometryOutcome(referenceMatrix, predicted, reference, court, config),
           "diagnostic": diagnostic, "athletes": None}
    athleteRows = []
    if athletes is not None and referenceMatrix is not None:
        athleteRows, counts = athleteErrors(*athletes, referenceMatrix, acceptedMatrix(predicted), court, config)
        for item in athleteRows:
            item["predictionStatus"] = predicted["status"]
        row["athletes"] = {**counts, "references": len(athleteRows)}
    return row, athleteRows


def frameP90(row):
    geometry = row["geometry"]
    return geometry["grid"]["p90Meters"] if geometry and geometry["grid"] else None


def rate(count, total):
    return count / total if total else None


def transferSummary(rows, tolerances):
    errors = [item["errorPixels"] / math.hypot(*row["dimensions"])
              if item["errorPixels"] is not None else None
              for row in rows for item in row["geometry"]["landmarkTransfer"]]
    return {"landmarks": len(errors),
            "successRates": {str(tolerance): rate(sum(error is not None and error <= tolerance for error in errors),
                                                  len(errors)) for tolerance in tolerances},
            "errorDiagonal": distribution(errors)}


def diagnosticSummary(rows, config):
    result = {}
    for confidence in map(str, config["diagnosticConfidences"]):
        outcomes = [row["diagnostic"][confidence] for row in rows]
        accepted = [item for item in outcomes if item["status"] in acceptedStatuses]
        falseValid = [item for item in accepted
                      if item["p90Meters"] is None or item["p90Meters"] > config["falseValidP90Meters"]]
        result[confidence] = {
            "coverage": rate(len(accepted), len(rows)),
            "successRates": {str(threshold): rate(sum(item["p90Meters"] is not None and
                                                      item["p90Meters"] <= threshold for item in accepted), len(rows))
                             for threshold in config["errorThresholdsMeters"]},
            "falseValidRate": rate(len(falseValid), len(accepted))}
    return result


def endToEndRates(rows, thresholds, statuses):
    return {str(threshold): rate(sum(row["predictionStatus"] in statuses and row["endToEndMeters"] is not None
                                     and row["endToEndMeters"] <= threshold for row in rows), len(rows))
            for threshold in thresholds}


def athleteSummary(rows, config):
    available = [row for row in rows if row["truthAvailable"]]
    thresholds = config["errorThresholdsMeters"]
    return {"references": len(rows), "truthAvailable": len(available),
            "insideCourt": sum(bool(row["insideCourt"]) for row in available),
            "matched": sum(row["matched"] for row in available),
            "metresPerPixel": distribution(row["metresPerPixel"] for row in available),
            "courtOnlyMeters": distribution(row["courtOnlyMeters"] for row in available),
            "detectionMeters": distribution(row["detectionMeters"] for row in available),
            "endToEndMeters": distribution(row["endToEndMeters"] for row in available),
            "endToEndVerifiedOnlyMeters": distribution(row["endToEndMeters"] for row in available
                                                       if row["predictionStatus"] == "verified"),
            "endToEndSuccessRates": endToEndRates(available, thresholds, acceptedStatuses),
            "endToEndSuccessRatesVerifiedOnly": endToEndRates(available, thresholds, ("verified",))}


def acceptancePolicy(rows, statuses, config):
    accepted = [row for row in rows if row["prediction"]["status"] in statuses]
    falseValid = [row for row in accepted
                  if frameP90(row) is None or frameP90(row) > config["falseValidP90Meters"]]
    return {"coverage": rate(len(accepted), len(rows)),
            "successRates": {str(threshold): rate(sum(frameP90(row) is not None and frameP90(row) <= threshold
                                                      for row in accepted), len(rows))
                             for threshold in config["errorThresholdsMeters"]},
            "falseValidRate": rate(len(falseValid), len(accepted))}


def statusQuality(rows, config):
    result = {}
    for status in acceptedStatuses:
        subset = [row for row in rows if row["prediction"]["status"] == status]
        result[status] = {"frames": len(subset), "p90Meters": distribution(frameP90(row) for row in subset),
                          "unboundedP90": sum(frameP90(row) is None for row in subset),
                          "falseValid": sum(frameP90(row) is None or frameP90(row) > config["falseValidP90Meters"]
                                            for row in subset)}
    return result


def cohortSummary(rows, athleteRows, config):
    accepted = [row for row in rows if row["prediction"]["status"] in acceptedStatuses]
    falseValid = [row for row in accepted
                  if frameP90(row) is None or frameP90(row) > config["falseValidP90Meters"]]
    rejectedGood = [row for row in rows if row["prediction"]["status"] == "rejected" and
                    frameP90(row) is not None and frameP90(row) <= config["falseValidP90Meters"]]
    records = {row["recordId"] for row in rows}
    return {
        "frames": len(rows),
        "predictionStatus": dict(Counter(row["prediction"]["status"] for row in rows)),
        "coverage": rate(len(accepted), len(rows)),
        "successRates": {str(threshold): rate(sum(frameP90(row) is not None and frameP90(row) <= threshold
                                                  for row in accepted), len(rows))
                         for threshold in config["errorThresholdsMeters"]},
        "acceptedP90Meters": distribution(frameP90(row) for row in accepted),
        "acceptedMedianMeters": distribution(row["geometry"]["grid"]["medianMeters"] for row in accepted
                                             if row["geometry"]["grid"]),
        "falseValid": {"count": len(falseValid), "rate": rate(len(falseValid), len(accepted)),
                       "byStatus": dict(Counter(row["prediction"]["status"] for row in falseValid))},
        "rejectedWithinTolerance": len(rejectedGood),
        "acceptedByStatus": statusQuality(rows, config),
        "verifiedOnly": acceptancePolicy(rows, ("verified",), config),
        "visibleIou": distribution(row["geometry"]["visibleIou"] for row in accepted),
        "landmarkTransfer": transferSummary(accepted, config["transferTolerancesDiagonal"]),
        "referenceLeaveOneOutPixels": distribution(value for row in rows
                                                   for value in row["reference"]["leaveOneOutPixels"]),
        "diagnostic": diagnosticSummary(rows, config),
        "athletes": athleteSummary([row for row in athleteRows if row["recordId"] in records], config),
    }


def groupSummary(rows, athleteRows, config):
    return {"frames": len(rows),
            "referenceStatus": dict(Counter(row["reference"]["status"] for row in rows)),
            "referenceReasons": dict(Counter(reason for row in rows for reason in row["reference"]["reasons"])),
            "predictionStatus": dict(Counter(row["prediction"]["status"] for row in rows)),
            "predictionReasons": dict(Counter(reason for row in rows for reason in row["prediction"]["reasons"])),
            "predictedPoints": distribution(row["prediction"]["pointCount"] for row in rows),
            "edgeTruncatedCandidates": sum(item["edgeTruncated"] for row in rows
                                           for item in row["prediction"]["candidates"]),
            "cohorts": {status: cohortSummary([row for row in rows if row["reference"]["status"] == status],
                                              athleteRows, config) for status in acceptedStatuses}}


def summarizeEvaluation(rows, athleteRows, config):
    matches = sorted({row["matchId"] for row in rows})
    return {"overall": groupSummary(rows, athleteRows, config),
            "byMatch": {matchId: groupSummary([row for row in rows if row["matchId"] == matchId],
                                              athleteRows, config) for matchId in matches}}
