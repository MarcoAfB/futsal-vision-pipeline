from futsal.evaluation.evaluationData import validatePredictions
from futsal.evaluation.evaluationMetrics import matchBoxes


def validateTrackingPredictions(references, predictions, config, tracked):
    predictions = validatePredictions(references, predictions, ["ball"],
                                     {**config, "confidenceFloor": config["detectorConfidence"]})
    for reference, prediction in zip(references, predictions, strict=True):
        if any(prediction.get(key) != reference[key] for key in ("segmentId", "frameIndex")):
            raise ValueError("Tracking predictions changed temporal membership.")
        identities = []
        for detection in prediction["detections"]:
            identity, box = detection.get("trackId"), detection["box"]
            if box[0] >= box[2] or box[1] >= box[3]:
                raise ValueError("Tracking predictions require positive-area boxes.")
            if tracked and (type(identity) is not int or identity < 0):
                raise ValueError("Tracker output requires nonnegative integer track IDs.")
            if not tracked and identity is not None:
                raise ValueError("Detector-only output cannot contain track IDs.")
            identities.append(identity)
        if tracked and len(identities) != len(set(identities)):
            raise ValueError("Track IDs must be distinct within a frame.")
    return predictions


def summarizeTrackingFrames(frames, tracked):
    losses, pending, previous = [], None, None
    changes, comparable = 0, 0
    for frame in frames:
        adjacent = previous is not None and previous["segmentId"] == frame["segmentId"] and previous["frameIndex"] + 1 == frame["frameIndex"]
        if not adjacent or not frame["positive"]:
            if pending is not None:
                losses.append(pending)
                pending = None
        if frame["fn"]:
            if pending is None:
                pending = {"segmentId": frame["segmentId"], "startFrame": frame["frameIndex"],
                           "length": 0, "precededByHit": bool(adjacent and previous["tp"]), "recovered": False}
            pending["length"] += 1
        elif frame["tp"] and pending is not None:
            pending["recovered"] = pending["precededByHit"]
            losses.append(pending)
            pending = None
        if tracked and adjacent and previous["tp"] and frame["tp"]:
            comparable += 1
            changes += previous["matchedTrackId"] != frame["matchedTrackId"]
        previous = frame
    if pending is not None:
        losses.append(pending)
    tp, fp, fn = (sum(row[key] for row in frames) for key in ("tp", "fp", "fn"))
    recovered = [loss["length"] for loss in losses if loss["recovered"]]
    return {"frames": len(frames), "positiveFrames": sum(f["positive"] for f in frames),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "lossCount": len(losses), "maxLossFrames": max((loss["length"] for loss in losses), default=0),
            "recoveredLossCount": len(recovered),
            "meanRecoveryFrames": sum(recovered) / len(recovered) if recovered else None,
            "observedIdChanges": changes if tracked else None,
            "adjacentMatchedPairs": comparable if tracked else None, "lossRuns": losses}


def scoreBallTracking(references, predictions, config, *, tracked):
    predictions = validateTrackingPredictions(references, predictions, config, tracked)
    frames = []
    for reference, prediction in zip(references, predictions, strict=True):
        detections = [d for d in prediction["detections"] if d["confidence"] >= config["operationalConfidence"]]
        matches = matchBoxes(reference["targets"], detections, config["operationalIou"])
        matched = detections[matches[0][1]] if matches else None
        tp = int(matched is not None)
        frames.append({key: reference[key] for key in ("recordId", "matchId", "segmentId", "frameIndex")}
                      | {"positive": bool(reference["targets"]), "tp": tp, "fp": len(detections) - tp,
                         "fn": int(bool(reference["targets"]) and not tp),
                         "matchedTrackId": matched["trackId"] if matched else None})
    return {"overall": summarizeTrackingFrames(frames, tracked),
            "byMatch": {matchId: summarizeTrackingFrames([f for f in frames if f["matchId"] == matchId], tracked)
                        for matchId in sorted({f["matchId"] for f in frames})},
            "frames": frames, "identityInterpretation": "exploratoryAdjacentMatchesOnly" if tracked else "notApplicable"}
