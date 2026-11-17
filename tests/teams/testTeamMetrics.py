from futsal.evaluation.teamMetrics import athleteRecord, scoreAssignment, swappedName

config = {"operationalConfidence": 0.25, "operationalIou": 0.5}
names = ["outfieldPlayerTeamA", "outfieldPlayerTeamB", "goalkeeperTeamA", "outfieldPlayerTeamA"]


def boxes():
    return [[100.0 * index, 0.0, 100.0 * index + 40.0, 100.0] for index in range(len(names))]


def pair(matchId, predictedNames):
    reference = athleteRecord({"recordId": f"{matchId}:0", "matchId": matchId, "targets": [
        {"className": name, "box": box} for name, box in zip(names, boxes(), strict=True)]}, "targets")
    prediction = athleteRecord({"recordId": reference["recordId"], "detections": [
        {"className": name, "box": box, "confidence": 0.9} for name, box in zip(predictedNames, boxes(), strict=True)]},
        "detections")
    return reference, prediction


def testOneJointSwapPerMatchRestoresInvertedNamesWithoutChangingRoles():
    inverted = pair("match01", [swappedName[name] for name in names])
    # match11 has one role error that no team exchange can repair.
    correct = pair("match11", ["outfieldPlayerTeamA", "outfieldPlayerTeamB", "outfieldPlayerTeamA",
                               "outfieldPlayerTeamA"])
    result = scoreAssignment([inverted[0], correct[0]], [inverted[1], correct[1]], config)
    named, swapped = result["named"], result["bestSwap"]

    assert swapped["swappedMatches"] == ["match01"]
    assert named["byMatch"]["match01"]["classAccuracy"] == 0.0 and swapped["byMatch"]["match01"]["classAccuracy"] == 1.0
    assert named["byMatch"]["match11"] == swapped["byMatch"]["match11"]
    assert named["assignment"]["roleAccuracy"] == swapped["assignment"]["roleAccuracy"] == 7 / 8
    assert swapped["assignment"]["teamAccuracy"] == 1.0 and swapped["assignment"]["classAccuracy"] == 7 / 8


def testUnmatchedPredictionsAndAthleteReferencesStayInOperationalScores():
    reference, prediction = pair("match01", names)
    prediction["detections"] = prediction["detections"][:3] + [
        {**prediction["detections"][0], "box": [900.0, 500.0, 940.0, 600.0]}]
    perClass = {row["className"]: row for row in scoreAssignment([reference], [prediction], config)["named"]["perClass"]}

    assert perClass["outfieldPlayerTeamA"]["support"] == 2 and perClass["outfieldPlayerTeamA"]["truePositives"] == 1
    assert perClass["outfieldPlayerTeamA"]["falsePositives"] == 1 and perClass["goalkeeperTeamB"]["support"] == 0
