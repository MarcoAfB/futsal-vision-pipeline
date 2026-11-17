from collections import defaultdict

from futsal.evaluation.evaluationMetrics import matchBoxes, operationalMetrics

athleteClassNames = ("outfieldPlayerTeamA", "outfieldPlayerTeamB", "goalkeeperTeamA", "goalkeeperTeamB")
swappedName = {"outfieldPlayerTeamA": "outfieldPlayerTeamB", "outfieldPlayerTeamB": "outfieldPlayerTeamA",
               "goalkeeperTeamA": "goalkeeperTeamB", "goalkeeperTeamB": "goalkeeperTeamA"}


def athleteRecord(record, key):
    items = [{**item, "classId": athleteClassNames.index(item["className"])}
             for item in record[key] if item["className"] in athleteClassNames]
    return {**record, key: items}


def swapRecord(prediction):
    return {**prediction, "detections": [
        {**item, "className": swappedName[item["className"]],
         "classId": athleteClassNames.index(swappedName[item["className"]])} for item in prediction["detections"]]}


def matchedPairs(reference, prediction, config):
    detections = [item for item in prediction["detections"] if item["confidence"] >= config["operationalConfidence"]]
    return [(reference["targets"][truth]["className"], detections[predicted]["className"])
            for truth, predicted, _ in matchBoxes(reference["targets"], detections, config["operationalIou"])]


def pairScores(pairs):
    count = len(pairs)

    def rate(correct):
        return sum(correct) / count if count else None

    return {"matched": count,
            "roleAccuracy": rate(truth[:4] == predicted[:4] for truth, predicted in pairs),
            "teamAccuracy": rate(truth[-1] == predicted[-1] for truth, predicted in pairs),
            "classAccuracy": rate(truth == predicted for truth, predicted in pairs)}


def bestSwaps(references, predictions, config):
    """Choose one joint A/B exchange per match, applied to outfield players and goalkeepers together."""
    correct = defaultdict(lambda: [0, 0])
    for reference, prediction in zip(references, predictions, strict=True):
        for truth, predicted in matchedPairs(reference, prediction, config):
            correct[reference["matchId"]][0] += truth == predicted
            correct[reference["matchId"]][1] += truth == swappedName[predicted]
    return {matchId: named < swapped for matchId, (named, swapped) in sorted(correct.items())}


def scoreVariant(references, predictions, config):
    pairs = defaultdict(list)
    for reference, prediction in zip(references, predictions, strict=True):
        pairs[reference["matchId"]].extend(matchedPairs(reference, prediction, config))
    operational = operationalMetrics(references, predictions, list(athleteClassNames), config)
    supported = [row["f1"] for row in operational["perClass"] if row["support"]]
    return {"assignment": pairScores([pair for values in pairs.values() for pair in values]),
            "byMatch": {matchId: pairScores(values) for matchId, values in sorted(pairs.items())},
            "perClass": operational["perClass"], "macroF1": sum(supported) / len(supported) if supported else None,
            "confusionLabels": operational["confusionLabels"], "confusionMatrix": operational["confusionMatrix"]}


def scoreAssignment(references, predictions, config):
    """Score four-class predictions as named and after the best joint swap per match.

    Inputs are athlete-only records whose classId follows athleteClassNames.
    """
    swaps = bestSwaps(references, predictions, config)
    adjusted = [swapRecord(prediction) if swaps.get(reference["matchId"]) else prediction
                for reference, prediction in zip(references, predictions, strict=True)]
    return {"named": scoreVariant(references, predictions, config),
            "bestSwap": {**scoreVariant(references, adjusted, config),
                         "swappedMatches": [matchId for matchId, swapped in swaps.items() if swapped]}}
