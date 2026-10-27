import copy
from types import SimpleNamespace

import pytest

from futsal.matchSplits import (
    frozenSplitDocument,
    selectAssignments,
    validateAssignments,
    validateFrozenSplit,
)


def splitFixture():
    recipe = SimpleNamespace(data={
        "tasks": [{"taskId": "unified", "classes": ["ball", "point", "rare"]}],
        "coverageTask": "unified", "rareClass": "rare", "unsupportedValidationClasses": ["rare"],
    }, catalog=SimpleNamespace(data={"splitPolicy": {
        "targetFractions": {"training": 0.7, "validation": 0.15, "test": 0.15}
    }}))
    samples = []
    for matchId, size in zip("abcde", [7, 5, 3, 2, 1], strict=True):
        for frame in range(size):
            classIds = [0, 1, 2] if matchId in "ab" else [0, 1]
            samples.append({
                "recordId": f"{matchId}:{frame}", "matchId": matchId, "origin": "own",
                "tasks": {"unified": {"eligible": True, "reasons": [], "targets": [
                    {"annotationIndex": index, "classId": index} for index in classIds
                ]}},
            })
    return samples, recipe


def testCoverageOverridesProportionsAndSelectionIsDeterministic():
    samples, recipe = splitFixture()
    assignments = selectAssignments(samples, recipe)
    summary = validateAssignments(samples, assignments, recipe)
    assert not summary["test"]["missingClasses"]
    assert not summary["training"]["missingClasses"]
    assert summary["validation"]["missingClasses"] == ["rare"]
    assert summary["test"]["reviewedFraction"] > 0.15
    assert assignments["a"] != assignments["b"]
    assert assignments == selectAssignments(list(reversed(samples)), recipe)


def testQuarantinedRareAnnotationsCannotSatisfyTestCoverage():
    samples, recipe = splitFixture()
    for sample in samples:
        if sample["matchId"] == "b":
            sample["tasks"]["unified"]["eligible"] = False
    with pytest.raises(ValueError, match="No match-level split"):
        selectAssignments(samples, recipe)


def testFrozenSplitDetectsEligibilityChangesAndReportsBothDenominators():
    samples, recipe = splitFixture()
    assignments = selectAssignments(samples, recipe)
    frozen = frozenSplitDocument(samples, assignments, recipe)
    sample = next(sample for sample in samples if sample["matchId"] in "cde")
    sample["tasks"]["unified"]["eligible"] = False
    summary = validateAssignments(samples, assignments, recipe)
    assert sum(item["reviewedFrameCount"] for item in summary.values()) == 18
    assert sum(item["eligibleFrameCount"] for item in summary.values()) == 17
    with pytest.raises(ValueError, match="Frozen split"):
        validateFrozenSplit(frozen, samples, recipe)


def testMissingMatchesAndInvalidAssignmentValuesAreRejected():
    samples, recipe = splitFixture()
    assignments = selectAssignments(samples, recipe)
    broken = copy.deepcopy(assignments)
    del broken["a"]
    with pytest.raises(ValueError, match="every original match"):
        validateAssignments(samples, broken, recipe)
    broken = {name: "training" for name in assignments}
    with pytest.raises(ValueError):
        validateAssignments(samples, broken, recipe)
