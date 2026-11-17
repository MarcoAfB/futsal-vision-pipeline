import copy

import pytest

from futsal.datasets.ballPlayerMixed import buildBallPlayerMixedPlan
from futsal.datasets.datasetFiles import hashJson, readJson
from futsal.datasets.datasetPlan import loadDatasetPlan
from futsal.datasets.datasetSamples import classifySample
from futsal.datasets.playerMixed import ownAthletes
from futsal.datasets.recipes import loadRecipe
from futsal.datasets.sharedViews import verifyView
from futsal.training.trainingConfig import loadTrainingConfig, resolveTrainingConfig
from futsal.training.trainingData import selectTrainingData


def testBallPlayerMixUsesPublicOwnAndBallFreeImagesWithUnchangedValidation(ballPlayerMixedView, tmp_path):
    view, store, configPath = ballPlayerMixedView
    parent = loadDatasetPlan(tmp_path / "ballPlayerParent/plan.json")
    plan = loadDatasetPlan(view / "plan.json")
    assert plan["parentPlanId"] == parent["planId"]
    for key in ("frozenSplit", "sourceEvidence", "catalog"):
        assert plan[key] == parent[key]
    for before, after in zip(parent["samples"], plan["samples"], strict=True):
        assert {k: v for k, v in before.items() if k != "tasks"} == {k: v for k, v in after.items() if k != "tasks"}
        assert all(after["tasks"][key] == value for key, value in before["tasks"].items())
    assert plan["recipe"]["curriculum"] == ["ballPlayer"]
    assert verifyView(view, store)["contractId"]
    assert not list(view.rglob("*.png")) and not list(view.rglob("*.jpg"))
    for size in ("n", "s", "m", "l", "x"):
        config = resolveTrainingConfig(loadTrainingConfig(configPath, plan["recipe"]), modelSize=size)
        task = selectTrainingData(view, config)["tasks"]["ballPlayer"]
        assert task["classes"] == ["ball", "player"]
        assert len(task["training"]) == 5 and len(task["validation"]) == 2
        classIds = {split: [target["classId"] for row in task[split] for target in row["targets"]]
                    for split in ("training", "validation")}
        assert classIds["training"].count(0) == 3 and classIds["training"].count(1) == 12
        assert classIds["validation"].count(0) == 1 and classIds["validation"].count(1) == 8
        assert all(row["targets"] for split in ("training", "validation") for row in task[split])
        assert sum(all(t["classId"] == 1 for t in row["targets"]) for row in task["training"]) == 2
        assert sum(all(t["classId"] == 1 for t in row["targets"]) for row in task["validation"]) == 1
        assert {row["origin"] for row in task["training"]} == {"own", "public"}
        assert {row["origin"] for row in task["validation"]} == {"own"}
    # Own membership equals the own-only generic objects task (sharedDetection genericObjects).
    ownOnly = {"classes": ["ball", "player"], "classMap": {"ball": "ball", **{name: "player" for name in ownAthletes}}}
    assert all(sample["tasks"]["ballPlayer"] == classifySample(sample, ownOnly, set())
               for sample in plan["samples"] if sample["origin"] == "own")
    assert not any(row["split"] == "test" for row in readJson(view / "manifest.json")["samples"])


@pytest.mark.parametrize("change", ["sources", "semantics", "ignore", "noBall", "order", "coverage"])
def testMixedBallPlayerDerivationRejectsIncompatibleRecipe(ballPlayerMixedView, preparedSources, tmp_path, change):
    parent = loadDatasetPlan(tmp_path / "ballPlayerParent/plan.json")
    original = copy.deepcopy(parent)
    recipe = loadRecipe(preparedSources[1].with_name("ballPlayerMixed.yaml"))
    task = recipe.data["tasks"][-1]
    if change == "sources":
        task["trainingSources"] = ["own"]
    elif change == "semantics":
        task["classMap"].pop("goalkeeper")
    elif change == "ignore":
        task["ignoredSourceClasses"] = ["upperCornerSideA"]
    elif change == "noBall":
        task["classes"].remove("ball")
        task["classMap"].pop("ball")
    elif change == "order":
        task["classes"].reverse()
    else:
        recipe.data["pointBoxes"]["widthFraction"] *= 2
    with pytest.raises(ValueError):
        buildBallPlayerMixedPlan(parent, recipe)
    assert parent == original


def testBadBallOrPlayerGeometryQuarantinesTheImage(ballPlayerMixedView, preparedSources, tmp_path):
    parent = loadDatasetPlan(tmp_path / "ballPlayerParent/plan.json")
    recipe = loadRecipe(preparedSources[1].with_name("ballPlayerMixed.yaml"))
    sample = next(sample for sample in parent["samples"]
                  if sample["sourceId"] == "futsal4400" and not sample["mediaPath"].endswith("playerOnly.jpg"))
    record = sample["recordId"]
    assert next(row for row in buildBallPlayerMixedPlan(parent, recipe)["samples"]
                if row["recordId"] == record)["tasks"]["ballPlayer"]["eligible"]
    for semantic, finding in (("ball", "geometryOutOfBounds"), ("player", "degenerateBox")):
        broken = copy.deepcopy(parent)
        target = next(row for row in broken["samples"] if row["recordId"] == record)
        annotation = next(item for item in target["annotations"]
                          if (item.get("canonicalClass") or item.get("semanticClass")) == semantic)
        annotation["geometryFindings"] = [finding]
        broken["planId"] = hashJson({key: value for key, value in broken.items() if key != "planId"})
        status = next(row for row in buildBallPlayerMixedPlan(broken, recipe)["samples"]
                      if row["recordId"] == record)["tasks"]["ballPlayer"]
        assert not status["eligible"] and finding in status["reasons"]


def testMixedBallPlayerDerivationRefusesAnOwnOnlyNativeProjection(ballPlayerMixedView, preparedSources, tmp_path):
    parent = loadDatasetPlan(tmp_path / "ballPlayerParent/plan.json")
    parent["samples"] = [sample for sample in parent["samples"] if sample["origin"] == "own"]
    parent["planId"] = hashJson({key: value for key, value in parent.items() if key != "planId"})
    recipe = loadRecipe(preparedSources[1].with_name("ballPlayerMixed.yaml"))
    with pytest.raises(ValueError, match="native public records"):
        buildBallPlayerMixedPlan(parent, recipe)
