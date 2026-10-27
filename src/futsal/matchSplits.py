import hashlib
import json
import math
from fractions import Fraction

from futsal.datasetSamples import eligibleClassCounts


def ownEligibilityDigest(samples, taskId):
    records = [
        {"recordId": sample["recordId"], "matchId": sample["matchId"],
         "status": sample["tasks"][taskId]}
        for sample in samples if sample["origin"] == "own"
    ]
    records.sort(key=lambda item: item["recordId"])
    content = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def coverageTask(recipe):
    return next(task for task in recipe.data["tasks"] if task["taskId"] == recipe.data["coverageTask"])


def summarizeSplits(samples, assignments, task):
    own = [sample for sample in samples if sample["origin"] == "own"]
    total = len(own)
    eligibleTotal = sum(sample["tasks"][task["taskId"]]["eligible"] for sample in own)
    summary = {}
    for split in ("training", "validation", "test"):
        selected = [sample for sample in own if assignments[sample["matchId"]] == split]
        eligible = sum(sample["tasks"][task["taskId"]]["eligible"] for sample in selected)
        counts = eligibleClassCounts(selected, task["taskId"])
        summary[split] = {
            "matchIds": sorted({sample["matchId"] for sample in selected}),
            "reviewedFrameCount": len(selected),
            "reviewedFraction": len(selected) / total if total else 0,
            "eligibleFrameCount": eligible,
            "eligibleFraction": eligible / eligibleTotal if eligibleTotal else 0,
            "quarantinedFrameCount": len(selected) - eligible,
            "classAnnotationCounts": {name: counts[index] for index, name in enumerate(task["classes"])},
            "missingClasses": [name for index, name in enumerate(task["classes"]) if not counts[index]],
        }
    return summary


def validateAssignments(samples, assignments, recipe):
    own = [sample for sample in samples if sample["origin"] == "own"]
    matchIds = {sample["matchId"] for sample in own}
    if not isinstance(assignments, dict) or set(assignments) != matchIds or set(
        assignments.values()
    ) != {"training", "validation", "test"}:
        raise ValueError("Splits must assign every original match exactly once to three nonempty splits.")
    task = coverageTask(recipe)
    summary = summarizeSplits(samples, assignments, task)
    for split in ("training", "test"):
        if summary[split]["missingClasses"]:
            raise ValueError(f"{split} lacks eligible annotations: {summary[split]['missingClasses']}.")
    unsupported = set(recipe.data["unsupportedValidationClasses"])
    if set(summary["validation"]["missingClasses"]) != unsupported:
        raise ValueError("Validation coverage differs from the declared unsupported classes.")
    return summary


def subsetStatistics(own, task, rareClassId):
    matchIds = sorted({sample["matchId"] for sample in own})
    grouped = {matchId: [sample for sample in own if sample["matchId"] == matchId]
               for matchId in matchIds}
    matchCounts = [len(grouped[matchId]) for matchId in matchIds]
    classCounts = [eligibleClassCounts(grouped[matchId], task["taskId"]) for matchId in matchIds]
    masks = [sum(1 << classId for classId in counts) for counts in classCounts]
    sizes, coverage, rare = [0], [0], [0]
    for mask in range(1, 1 << len(matchIds)):
        bit = mask & -mask
        index = bit.bit_length() - 1
        previous = mask ^ bit
        sizes.append(sizes[previous] + matchCounts[index])
        coverage.append(coverage[previous] | masks[index])
        rare.append(rare[previous] + classCounts[index][rareClassId])
    return matchIds, sizes, coverage, rare


def selectAssignments(samples, recipe):
    task = coverageTask(recipe)
    own = [sample for sample in samples if sample["origin"] == "own"]
    matchIds, sizes, coverage, rare = subsetStatistics(
        own, task, task["classes"].index(recipe.data["rareClass"])
    )
    complete = (1 << len(task["classes"])) - 1
    unsupported = sum(1 << task["classes"].index(name)
                      for name in recipe.data["unsupportedValidationClasses"])
    fractions = recipe.catalog.data["splitPolicy"]["targetFractions"]
    targets = {name: Fraction(str(value)) for name, value in fractions.items()}
    scale = math.lcm(*(value.denominator for value in targets.values()))
    costs = {name: [abs(size * scale - len(own) * int(target * scale)) for size in sizes]
             for name, target in targets.items()}
    validations = sorted(
        (mask for mask in range(1, len(sizes)) if coverage[mask] == complete ^ unsupported),
        key=lambda mask: costs["validation"][mask],
    )
    tests = sorted((mask for mask in range(1, len(sizes)) if coverage[mask] == complete),
                   key=lambda mask: costs["test"][mask])
    best = None
    allMatches = len(sizes) - 1
    for validation in validations:
        if best and costs["validation"][validation] > best[0]:
            break
        for test in tests:
            partialCost = costs["validation"][validation] + costs["test"][test]
            if best and partialCost > best[0]:
                break
            if validation & test:
                continue
            training = allMatches ^ (validation | test)
            if coverage[training] != complete:
                continue
            score = partialCost + costs["training"][training]
            if best and (score, -rare[training]) > best[:2]:
                continue
            groups = tuple(tuple(name for index, name in enumerate(matchIds) if mask & (1 << index))
                           for mask in (training, validation, test))
            candidate = (score, -rare[training], groups)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        support = {name: sorted({sample["matchId"] for sample in own
                                if index in eligibleClassCounts([sample], task["taskId"])})
                   for index, name in enumerate(task["classes"])}
        raise ValueError(f"No match-level split satisfies eligible class coverage. Support: {support}")
    assignments = {matchId: split for split, group in zip(
        ("training", "validation", "test"), best[2], strict=True
    ) for matchId in group}
    validateAssignments(samples, assignments, recipe)
    return dict(sorted(assignments.items()))


def frozenSplitDocument(samples, assignments, recipe):
    validateAssignments(samples, assignments, recipe)
    return {
        "schemaVersion": 1,
        "countingUnit": "reviewedFrameRecord",
        "reviewedFrameCount": sum(sample["origin"] == "own" for sample in samples),
        "targetFractions": recipe.catalog.data["splitPolicy"]["targetFractions"],
        "selectionMethod": "minimumAbsoluteDeviationThenRareTrainingSupportThenLexicographicGroups",
        "coverageTask": recipe.data["coverageTask"],
        "ownEligibilitySha256": ownEligibilityDigest(samples, recipe.data["coverageTask"]),
        "assignments": assignments,
    }


def validateFrozenSplit(document, samples, recipe):
    expected = frozenSplitDocument(samples, document["assignments"], recipe)
    if document != expected:
        raise ValueError("Frozen split does not match current own eligibility or split policy.")
    return validateAssignments(samples, document["assignments"], recipe)
