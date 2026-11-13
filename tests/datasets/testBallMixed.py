import copy

import pytest

from futsal.datasets.ballMixed import buildBallMixedPlan
from futsal.datasets.datasetFiles import hashJson, readJson
from futsal.datasets.datasetPlan import loadDatasetPlan
from futsal.datasets.recipes import loadRecipe
from futsal.datasets.sharedViews import verifyView
from futsal.training.trainingConfig import loadTrainingConfig, resolveTrainingConfig
from futsal.training.trainingData import selectTrainingData


def testBallMixUsesPublicOwnAndValidNegativesWithUnchangedValidation(ballMixedView, tmp_path):
    view, store, configPath = ballMixedView
    parent = loadDatasetPlan(tmp_path / "ballParent/plan.json")
    plan = loadDatasetPlan(view / "plan.json")
    assert plan["parentPlanId"] == parent["planId"]
    for key in ("frozenSplit", "sourceEvidence", "catalog"):
        assert plan[key] == parent[key]
    for before, after in zip(parent["samples"], plan["samples"], strict=True):
        assert {k: v for k, v in before.items() if k != "tasks"} == {k: v for k, v in after.items() if k != "tasks"}
        assert all(after["tasks"][key] == value for key, value in before["tasks"].items())
    assert plan["recipe"]["curriculum"] == ["ballOnly"]
    assert verifyView(view, store)["contractId"]
    assert not list(view.rglob("*.png")) and not list(view.rglob("*.jpg"))
    for size in ("n", "s", "m", "l", "x"):
        config = resolveTrainingConfig(loadTrainingConfig(configPath, plan["recipe"]), modelSize=size)
        task = selectTrainingData(view, config)["tasks"]["ballOnly"]
        assert task["classes"] == ["ball"]
        assert len(task["training"]) == 5 and len(task["validation"]) == 2
        assert sum(not row["targets"] for row in task["training"]) == 2
        assert sum(not row["targets"] for row in task["validation"]) == 1
        assert {row["origin"] for row in task["training"]} == {"own", "public"}
        assert {row["origin"] for row in task["validation"]} == {"own"}
        assert all(target["classId"] == 0 for split in ("training", "validation") for row in task[split] for target in row["targets"])
    # Reproduce the own-only projection's status without public ignore rules.
    from futsal.datasets.datasetSamples import classifySample

    oldTask = {"classes": ["ball"], "classMap": {"ball": "ball"}}
    assert all(sample["tasks"]["ballOnly"] == classifySample(sample, oldTask, set()) for sample in plan["samples"] if sample["origin"] == "own")
    assert not any(row["split"] == "test" for row in readJson(view / "manifest.json")["samples"])


@pytest.mark.parametrize("change", ["sources", "ignore", "ontology", "coverage"])
def testMixedDerivationRejectsIncompatibleRecipe(ballMixedView, preparedSources, tmp_path, change):
    parent = loadDatasetPlan(tmp_path / "ballParent/plan.json")
    original = copy.deepcopy(parent)
    recipe = loadRecipe(preparedSources[1].with_name("ballMixed.yaml"))
    if change == "sources":
        recipe.data["tasks"][-1]["trainingSources"] = ["own"]
    elif change == "ignore":
        recipe.data["tasks"][-1]["ignoredSourceClasses"] = []
    elif change == "ontology":
        recipe.data["tasks"][-1]["classes"].append("player")
        recipe.data["tasks"][-1]["classMap"]["player"] = "player"
    else:
        recipe.data["pointBoxes"]["widthFraction"] *= 2
    with pytest.raises(ValueError):
        buildBallMixedPlan(parent, recipe)
    assert parent == original


def testBadPlayerGeometryDoesNotQuarantineBallButBadBallStillDoes(ballMixedView, preparedSources, tmp_path):
    parent = loadDatasetPlan(tmp_path / "ballParent/plan.json")
    recipe = loadRecipe(preparedSources[1].with_name("ballMixed.yaml"))
    sample = next(sample for sample in parent["samples"] if sample["sourceId"] == "futsal4400" and not sample["mediaPath"].endswith("negative.jpg"))
    player = next(annotation for annotation in sample["annotations"] if annotation.get("semanticClass") == "player")
    player["geometryFindings"] = ["degenerateBox"]
    parent["planId"] = hashJson({key: value for key, value in parent.items() if key != "planId"})
    derived = buildBallMixedPlan(parent, recipe)
    assert next(row for row in derived["samples"] if row["recordId"] == sample["recordId"])["tasks"]["ballOnly"]["eligible"]
    ball = next(annotation for annotation in sample["annotations"] if annotation.get("canonicalClass") == "ball")
    ball["geometryFindings"] = ["geometryOutOfBounds"]
    parent["planId"] = hashJson({key: value for key, value in parent.items() if key != "planId"})
    derived = buildBallMixedPlan(parent, recipe)
    assert not next(row for row in derived["samples"] if row["recordId"] == sample["recordId"])["tasks"]["ballOnly"]["eligible"]


def testMixedDerivationRefusesAnOwnOnlyNativeProjection(ballMixedView, preparedSources, tmp_path):
    parent = loadDatasetPlan(tmp_path / "ballParent/plan.json")
    parent["samples"] = [sample for sample in parent["samples"] if sample["origin"] == "own"]
    parent["planId"] = hashJson({key: value for key, value in parent.items() if key != "planId"})
    recipe = loadRecipe(preparedSources[1].with_name("ballMixed.yaml"))
    with pytest.raises(ValueError, match="native public records"):
        buildBallMixedPlan(parent, recipe)
