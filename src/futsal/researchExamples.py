import hashlib
from collections import defaultdict


def sampleOrder(recordId, seed):
    return hashlib.sha256(f"{seed}:researchExamples:{recordId}".encode()).hexdigest()


def fixedExamples(tasks, config):
    shared = set.intersection(*({row["recordId"] for row in task["rows"]} for task in tasks.values()))
    first = next(iter(tasks.values()))
    matches = defaultdict(list)
    for row in first["rows"]:
        if row["recordId"] in shared:
            matches[row["matchId"]].append(row["recordId"])
    def ordered(values):
        return sorted(values, key=lambda recordId: sampleOrder(recordId, config["seed"]))
    chosen = [ordered(matches[matchId])[0] for matchId in sorted(matches)]
    chosen += ordered(shared - set(chosen))
    return chosen[:config["fixedExampleCount"]]


def selectExamples(fixedIds, detection, points, config):
    selected = [{"recordId": recordId, "reason": "fixedSharedExample"} for recordId in fixedIds]
    errors = {row["recordId"]: dict(row) for row in detection["imageErrors"]}
    for row in points["errors"]:
        error = errors[row["recordId"]]
        error["pointDistance"] = max(error.get("pointDistance", 0), row["distanceDiagonal"] or 0)
        error["missingPoints"] = error.get("missingPoints", 0) + (row["annotated"] and row["distanceDiagonal"] is None)
    criteria = [("ballMisses", "ballMiss"), ("classConfusions", "classConfusion")]
    criteria += ([("pointDistance", "pointLocalizationError"), ("missingPoints", "missingLandmarks")]
                 if points["perClass"] else [("highestFalsePositiveConfidence", "confidentFalsePositive")])
    criteria += [("totalErrors", "detectionErrors")]
    for error in errors.values():
        error["totalErrors"] = error["falsePositives"] + error["falseNegatives"]
    used = set(fixedIds)
    for field, reason in criteria:
        candidates = [row for row in errors.values() if row["recordId"] not in used and row.get(field, 0) > 0]
        if candidates:
            row = min(candidates, key=lambda item: (-item[field], sampleOrder(item["recordId"], config["seed"])))
            selected.append({"recordId": row["recordId"], "reason": reason})
            used.add(row["recordId"])
        if len(selected) >= len(fixedIds) + config["hardExampleCount"]:
            break
    return selected
