import statistics
from collections import Counter, defaultdict

from futsal.evaluation.evaluationData import validatePredictions
from futsal.evaluation.evaluationMetrics import matchBoxes


def validateTrackedPredictions(references, predictions, classNames, config, tracked):
    predictions = validatePredictions(references, predictions, classNames,
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


def linkReferenceChains(references, iou):
    """Greedy same-class IoU links between adjacent frames of one segment.

    The chains are exploratory pseudo-identities. Annotation track IDs are not verified athlete
    identities, so they are neither used nor trusted here.
    """
    chains, previous, nextChain = [], None, 0
    for reference in references:
        targets = reference["targets"]
        current = [None] * len(targets)
        adjacent = (previous is not None and previous[0]["segmentId"] == reference["segmentId"]
                    and previous[0]["frameIndex"] + 1 == reference["frameIndex"])
        if adjacent and targets and previous[0]["targets"]:
            candidates = [{**target, "confidence": 1.0} for target in targets]
            for before, after, _ in matchBoxes(previous[0]["targets"], candidates, iou, sameClass=True):
                current[after] = previous[1][before]
        for index, chain in enumerate(current):
            if chain is None:
                current[index], nextChain = nextChain, nextChain + 1
        chains.append(current)
        previous = reference, current
    return chains


def scoreFrames(references, predictions, config, chains, tracked):
    frames = []
    for reference, prediction, chain in zip(references, predictions, chains, strict=True):
        targets = reference["targets"]
        detections = [item for item in prediction["detections"] if item["confidence"] >= config["operationalConfidence"]]
        matches = matchBoxes(targets, detections, config["operationalIou"], sameClass=True)
        matched = [None] * len(targets)
        byClass = defaultdict(lambda: [0, 0, 0])
        for targetIndex, detectionIndex, _ in matches:
            matched[targetIndex] = detections[detectionIndex]["trackId"] if tracked else -1
            byClass[targets[targetIndex]["className"]][0] += 1
        hit = {detectionIndex for _, detectionIndex, _ in matches}
        hitTargets = {targetIndex for targetIndex, _, _ in matches}
        for index, item in enumerate(detections):
            if index not in hit:
                byClass[item["className"]][1] += 1
        for index, item in enumerate(targets):
            if index not in hitTargets:
                byClass[item["className"]][2] += 1
        frames.append({key: reference[key] for key in ("recordId", "matchId", "segmentId", "frameIndex")}
                      | {"targets": len(targets), "detections": len(detections), "tp": len(matches),
                         "fp": len(detections) - len(matches), "fn": len(targets) - len(matches),
                         "byClass": {name: dict(zip(("tp", "fp", "fn"), counts, strict=True))
                                     for name, counts in sorted(byClass.items())},
                         "chains": chain, "matchedTrackIds": matched if tracked else None,
                         "trackIds": [item["trackId"] for item in detections] if tracked else None})
    return frames


def rates(tp, fp, fn):
    return {"precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}


def stabilitySummary(frames):
    lengths = Counter((frame["segmentId"], identity) for frame in frames for identity in frame["trackIds"])
    sequences = defaultdict(list)
    for frame in frames:
        for chain, identity in zip(frame["chains"], frame["matchedTrackIds"], strict=True):
            sequences[chain].append((frame["segmentId"], frame["frameIndex"], identity))
    chainCount = adjacent = adjacentChanges = recoveries = recoveryChanges = multiple = 0
    identityCounts = []
    for sequence in sequences.values():
        if len(sequence) < 2:
            continue
        chainCount += 1
        for before, after in zip(sequence, sequence[1:], strict=False):
            if before[2] is not None and after[2] is not None:
                adjacent += 1
                adjacentChanges += before[2] != after[2]
        matched = [item for item in sequence if item[2] is not None]
        for before, after in zip(matched, matched[1:], strict=False):
            if after[1] - before[1] > 1:
                recoveries += 1
                recoveryChanges += before[2] != after[2]
        if matched:
            distinct = len({item[2] for item in matched})
            identityCounts.append(distinct)
            multiple += distinct > 1
    values = list(lengths.values())
    return {"trackCount": len(values), "meanTrackLength": statistics.fmean(values) if values else None,
            "medianTrackLength": statistics.median(values) if values else None,
            "singleFrameTracks": sum(value == 1 for value in values),
            "referenceChains": chainCount, "matchedChains": len(identityCounts), "chainsWithMultipleIds": multiple,
            "meanIdsPerMatchedChain": statistics.fmean(identityCounts) if identityCounts else None,
            "adjacentMatchedPairs": adjacent, "adjacentIdChanges": adjacentChanges,
            "gapRecoveries": recoveries, "gapIdChanges": recoveryChanges}


def summarizeTracking(frames, classNames, tracked):
    tp, fp, fn = (sum(frame[key] for frame in frames) for key in ("tp", "fp", "fn"))
    perClass = {}
    for name in classNames:
        counts = {key: sum(frame["byClass"].get(name, {}).get(key, 0) for frame in frames) for key in ("tp", "fp", "fn")}
        perClass[name] = {**counts, **rates(counts["tp"], counts["fp"], counts["fn"])}
    return {"frames": len(frames), "targets": sum(frame["targets"] for frame in frames),
            "positiveFrames": sum(frame["targets"] > 0 for frame in frames), "tp": tp, "fp": fp, "fn": fn,
            **rates(tp, fp, fn), "perClass": perClass, "stability": stabilitySummary(frames) if tracked else None}


def scoreTracking(references, predictions, classNames, config, *, tracked):
    predictions = validateTrackedPredictions(references, predictions, classNames, config, tracked)
    frames = scoreFrames(references, predictions, config, linkReferenceChains(references, config["operationalIou"]), tracked)
    return {"overall": summarizeTracking(frames, classNames, tracked),
            "byMatch": {matchId: summarizeTracking([f for f in frames if f["matchId"] == matchId], classNames, tracked)
                        for matchId in sorted({f["matchId"] for f in frames})},
            "frames": frames,
            "identityInterpretation": "exploratoryIouLinkedReferenceChains" if tracked else "notApplicable"}
