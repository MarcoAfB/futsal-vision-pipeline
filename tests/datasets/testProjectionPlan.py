import copy
from pathlib import Path

from futsal.datasets.datasetFiles import readJson
from futsal.datasets.datasetPlan import planRecipe
from futsal.datasets.datasetSamples import classifySample
from futsal.datasets.projectionPlan import buildProjectionPlan
from futsal.inventory.catalog import readYaml


def projectionRecipe(parent):
    recipe = planRecipe(parent)
    recipe.data = readYaml(Path("configs/separatedDetection.yaml"))[0]
    recipe.data.pop("splitPath")
    recipe.data["tasks"][0]["classes"] = [item["name"] for item in recipe.catalog.labels["canonicalClasses"]]
    recipe.data["tasks"][0]["classMap"] = {name: name for name in recipe.data["tasks"][0]["classes"]}
    return recipe


def testProjectionRetainsFrozenSplitNativeReferencesAndUnsupportedPoint(trainingBundle):
    bundle, _, _ = trainingBundle
    parent = readJson(bundle / "plan.json")
    original = copy.deepcopy(parent)
    plan = buildProjectionPlan(parent, projectionRecipe(parent))
    assert parent == original
    assert plan["frozenSplit"] == parent["frozenSplit"]
    assert plan["parentPlanId"] == parent["planId"]
    assert [len(task["classes"]) for task in plan["recipe"]["tasks"]] == [25, 5, 20]
    assert plan["ownSplitSummaries"]["courtPoints"]["validation"]["missingClasses"] == ["lowerCornerSideB"]


def testTaskQuarantineDoesNotPropagateAcrossIndependentTargets(trainingBundle):
    bundle, _, _ = trainingBundle
    parent = readJson(bundle / "plan.json")
    recipe = projectionRecipe(parent)
    sample = copy.deepcopy(next(sample for sample in parent["samples"] if sample["origin"] == "own"))
    objects, court = recipe.data["tasks"][1:]
    point = next(annotation for annotation in sample["annotations"] if annotation["shapeType"] == "points")
    points = set(court["classes"])
    point["geometryFindings"] = ["coordinatesOutOfBounds"]
    assert classifySample(sample, objects, points)["eligible"]
    assert not classifySample(sample, court, points)["eligible"]
    point["geometryFindings"] = []
    ball = next(annotation for annotation in sample["annotations"] if annotation.get("canonicalClass") == "ball")
    ball["geometryFindings"] = ["geometryOutOfBounds"]
    assert not classifySample(sample, objects, points)["eligible"]
    assert classifySample(sample, court, points)["eligible"]
