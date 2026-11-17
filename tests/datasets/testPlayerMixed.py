import copy

import pytest

from futsal.datasets.datasetFiles import hashJson, readJson
from futsal.datasets.datasetPlan import loadDatasetPlan
from futsal.datasets.datasetSamples import classifySample
from futsal.datasets.playerMixed import buildPlayerMixedPlan, ownAthletes
from futsal.datasets.recipes import loadRecipe
from futsal.datasets.sharedViews import verifyView
from futsal.training.trainingConfig import loadTrainingConfig, resolveTrainingConfig
from futsal.training.trainingData import selectTrainingData


def testPlayerMixUsesPublicOwnAndValidNegativesWithUnchangedValidation(playerMixedView, tmp_path):
    view, store, configPath = playerMixedView
    parent = loadDatasetPlan(tmp_path / "playerParent/plan.json")
    plan = loadDatasetPlan(view / "plan.json")
    assert plan["parentPlanId"] == parent["planId"]
    for key in ("frozenSplit", "sourceEvidence", "catalog"):
        assert plan[key] == parent[key]
    for before, after in zip(parent["samples"], plan["samples"], strict=True):
        assert {k: v for k, v in before.items() if k != "tasks"} == {k: v for k, v in after.items() if k != "tasks"}
        assert all(after["tasks"][key] == value for key, value in before["tasks"].items())
    assert plan["recipe"]["curriculum"] == ["playerOnly"]
    assert verifyView(view, store)["contractId"]
    assert not list(view.rglob("*.png")) and not list(view.rglob("*.jpg"))
    for size in ("n", "s", "m", "l", "x"):
        config = resolveTrainingConfig(loadTrainingConfig(configPath, plan["recipe"]), modelSize=size)
        task = selectTrainingData(view, config)["tasks"]["playerOnly"]
        assert task["classes"] == ["player"]
        assert len(task["training"]) == 5 and len(task["validation"]) == 2
        assert sum(not row["targets"] for row in task["training"]) == 2
        assert sum(not row["targets"] for row in task["validation"]) == 1
        assert sum(len(row["targets"]) for row in task["training"]) == 7
        assert {row["origin"] for row in task["training"]} == {"own", "public"}
        assert {row["origin"] for row in task["validation"]} == {"own"}
        assert all(target["classId"] == 0 for split in ("training", "validation") for row in task[split] for target in row["targets"])
    # Own membership equals the own-only playerOnly task, which has no generic player semantic.
    ownOnly = {"classes": ["player"], "classMap": {name: "player" for name in ownAthletes}}
    assert all(sample["tasks"]["playerOnly"] == classifySample(sample, ownOnly, set())
               for sample in plan["samples"] if sample["origin"] == "own")
    assert not any(row["split"] == "test" for row in readJson(view / "manifest.json")["samples"])


@pytest.mark.parametrize("change", ["sources", "semantics", "ignore", "ontology", "coverage"])
def testMixedPlayerDerivationRejectsIncompatibleRecipe(playerMixedView, preparedSources, tmp_path, change):
    parent = loadDatasetPlan(tmp_path / "playerParent/plan.json")
    original = copy.deepcopy(parent)
    recipe = loadRecipe(preparedSources[1].with_name("playerMixed.yaml"))
    task = recipe.data["tasks"][-1]
    if change == "sources":
        task["trainingSources"] = ["own"]
    elif change == "semantics":
        task["classMap"].pop("goalkeeper")
    elif change == "ignore":
        task["ignoredSourceClasses"] = ["ball"]
    elif change == "ontology":
        task["classes"].append("ball")
        task["classMap"]["ball"] = "ball"
    else:
        recipe.data["pointBoxes"]["widthFraction"] *= 2
    with pytest.raises(ValueError):
        buildPlayerMixedPlan(parent, recipe)
    assert parent == original


def testBadBallGeometryDoesNotQuarantinePlayersButBadPlayerStillDoes(playerMixedView, preparedSources, tmp_path):
    parent = loadDatasetPlan(tmp_path / "playerParent/plan.json")
    recipe = loadRecipe(preparedSources[1].with_name("playerMixed.yaml"))
    sample = next(sample for sample in parent["samples"] if sample["sourceId"] == "futsal4400" and not sample["mediaPath"].endswith("negative.jpg"))
    ball = next(annotation for annotation in sample["annotations"] if annotation.get("canonicalClass") == "ball")
    ball["geometryFindings"] = ["geometryOutOfBounds"]
    parent["planId"] = hashJson({key: value for key, value in parent.items() if key != "planId"})
    derived = buildPlayerMixedPlan(parent, recipe)
    assert next(row for row in derived["samples"] if row["recordId"] == sample["recordId"])["tasks"]["playerOnly"]["eligible"]
    player = next(annotation for annotation in sample["annotations"] if annotation.get("semanticClass") == "player")
    player["geometryFindings"] = ["degenerateBox"]
    parent["planId"] = hashJson({key: value for key, value in parent.items() if key != "planId"})
    derived = buildPlayerMixedPlan(parent, recipe)
    assert not next(row for row in derived["samples"] if row["recordId"] == sample["recordId"])["tasks"]["playerOnly"]["eligible"]


def testMixedPlayerDerivationRefusesAnOwnOnlyNativeProjection(playerMixedView, preparedSources, tmp_path):
    parent = loadDatasetPlan(tmp_path / "playerParent/plan.json")
    parent["samples"] = [sample for sample in parent["samples"] if sample["origin"] == "own"]
    parent["planId"] = hashJson({key: value for key, value in parent.items() if key != "planId"})
    recipe = loadRecipe(preparedSources[1].with_name("playerMixed.yaml"))
    with pytest.raises(ValueError, match="native public records"):
        buildPlayerMixedPlan(parent, recipe)
