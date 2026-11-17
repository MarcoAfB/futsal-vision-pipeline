import copy

import pytest

from futsal.datasets.datasetSamples import classifySample
from futsal.datasets.recipes import loadRecipe, validateRecipe


@pytest.mark.parametrize("ignored", [["unknown"], ["ball"], ["player", "player"], "player", [None]])
def testIgnoredClassesMustBeKnownUniqueAndUnmapped(ignored):
    recipe = loadRecipe("configs/ballMixed.yaml")
    recipe.data["tasks"][-1]["ignoredSourceClasses"] = ignored
    with pytest.raises(ValueError):
        validateRecipe(recipe.data, recipe.catalog)


def testIgnoringAthletesIsExplicitAndDoesNotChangeSourceIndices():
    sample = {"origin": "public", "dimensions": [96, 64], "emptyLabel": False, "annotations": [
        {"mappingAction": "preserve", "semanticClass": "player", "geometryFindings": ["degenerateBox"], "shapeType": "box"},
        {"mappingAction": "map", "canonicalClass": "ball", "geometryFindings": [], "shapeType": "box"}]}
    original = copy.deepcopy(sample)
    task = {"classes": ["ball"], "classMap": {"ball": "ball"}}
    assert classifySample(sample, task, set())["reasons"] == ["unresolvedAthleteSemantics"]
    task["ignoredSourceClasses"] = ["player"]
    assert classifySample(sample, task, set()) == {"eligible": True, "reasons": [], "targets": [{"annotationIndex": 1, "classId": 0}]}
    assert sample == original
    sample["annotations"][0]["mappingAction"] = "unmapped"
    assert classifySample(sample, task, set())["reasons"] == ["unmappedLabel"]
    sample["annotations"][0]["mappingAction"] = "preserve"
    sample["annotations"][1]["geometryFindings"] = ["geometryOutOfBounds"]
    assert classifySample(sample, task, set())["reasons"] == ["geometryOutOfBounds"]
    sample["emptyLabel"] = True
    assert "emptyPublicLabel" in classifySample(sample, task, set())["reasons"]
    sample["readFindings"] = ["missingOrAmbiguousLabel"]
    assert "missingOrAmbiguousLabel" in classifySample(sample, task, set())["reasons"]


@pytest.mark.parametrize("name", ["curriculum", "objectCurriculum", "playerCurriculum", "sharedDetection", "playerMixed"])
def testOptionalIgnorePolicyDoesNotRewriteOlderRecipes(name):
    recipe = loadRecipe(f"configs/{name}.yaml")
    original = copy.deepcopy(recipe.data)
    validateRecipe(recipe.data, recipe.catalog)
    assert recipe.data == original
    assert not any("ignoredSourceClasses" in task for task in recipe.data["tasks"])
