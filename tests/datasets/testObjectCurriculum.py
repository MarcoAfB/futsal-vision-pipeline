import copy
from pathlib import Path

import pytest

from futsal.datasets.datasetFiles import hashJson, readJson
from futsal.datasets.datasetPlan import loadDatasetPlan
from futsal.datasets.objectCurriculum import buildObjectCurriculumPlan
from futsal.datasets.projectionPlan import buildProjectionPlan
from futsal.datasets.recipes import activeTasks, loadRecipe
from futsal.datasets.sharedViews import verifyView
from futsal.training.trainingConfig import (
    loadTrainingConfig,
    resolveTrainingConfig,
    stageParameters,
)


def testObjectViewPreservesPublicSamplesNativeSplitAndSources(objectCurriculumView, trainingBundle, preparedSources):
    view, store, configPath = objectCurriculumView
    before = loadDatasetPlan(trainingBundle[0] / "plan.json")
    plan = loadDatasetPlan(view / "plan.json")
    assert plan["parentPlanId"] == before["planId"]
    assert plan["frozenSplit"] == before["frozenSplit"]
    assert plan["sourceEvidence"] == before["sourceEvidence"]
    assert plan["catalog"] == before["catalog"]
    assert [sample for sample in plan["samples"] if sample["origin"] == "public"] == [
        sample for sample in before["samples"] if sample["origin"] == "public"]
    for old, new in zip(before["samples"], plan["samples"], strict=True):
        for key, value in old.items():
            assert value == ({k: v for k, v in new[key].items() if k != "detailedObjects"} if key == "tasks" else new[key])
    rows = readJson(view / "manifest.json")["samples"]
    tasks = ["genericObjects", "roleObjects", "detailedObjects"]
    assert {row["taskId"] for row in rows} == set(tasks)
    assert {row["origin"] for row in rows if row["taskId"] != "detailedObjects" and row["split"] == "training"} == {"public"}
    assert {row["origin"] for row in rows if row["taskId"] == "detailedObjects"} == {"own"}
    validations = [{row["recordId"] for row in rows if row["taskId"] == task and row["split"] == "validation"} for task in tasks]
    assert validations[0] == validations[1] == validations[2]
    assert not any(row["split"] == "test" for row in rows)
    assert not any(path.suffix.lower() in {".png", ".jpg", ".mp4", ".avi"} for path in view.rglob("*"))
    assert verifyView(view, store)["contractId"]
    recipe = loadRecipe(preparedSources[1].with_name("objectCurriculum.yaml"))
    with pytest.raises(ValueError, match="own-data"):
        buildProjectionPlan(before, recipe)
    for size in ("n", "s", "m", "l", "x"):
        config = resolveTrainingConfig(loadTrainingConfig(configPath, plan["recipe"]), modelSize=size)
        assert [stageParameters(config, stage)["batch"] for stage in config["stages"]] == [8, 8, 8]
        assert [len(task["classes"]) for task in activeTasks(plan["recipe"])] == [2, 3, 5]


@pytest.mark.parametrize("change", ["sources", "labels", "points", "classes", "coverage"])
def testObjectDerivationRejectsChangedScientificContracts(objectCurriculumView, trainingBundle, preparedSources, change):
    parent = loadDatasetPlan(trainingBundle[0] / "plan.json")
    original = copy.deepcopy(parent)
    recipe = loadRecipe(preparedSources[1].with_name("objectCurriculum.yaml"))
    if change == "sources":
        recipe.data["tasks"][0]["trainingSources"] = ["own"]
    elif change == "labels":
        recipe.catalog.labels["canonicalClasses"][0]["name"] = "changed"
    elif change == "points":
        recipe.data["pointBoxes"]["widthFraction"] *= 2
    elif change == "classes":
        recipe.data["tasks"][-1]["classes"].reverse()
    else:
        recipe.data["coverageTask"] = "detailedObjects"
    with pytest.raises(ValueError):
        buildObjectCurriculumPlan(parent, recipe)
    assert parent == original


def testTrackedObjectProfilesHaveInactiveCanonicalCoverage():
    recipe = loadRecipe(Path("configs/objectCurriculum.yaml"))
    assert recipe.data["coverageTask"] == "unified25"
    assert "unified25" not in recipe.data["curriculum"]
    assert [task["trainingSources"] for task in activeTasks(recipe.data)] == [
        ["futsal4400", "smartFutsal"], ["futsalMot", "futsalAnaliza", "futsalDetect"], ["own"]]


def testDetailedEligibilityIsIndependentOfCourtQuarantine(objectCurriculumView, trainingBundle, preparedSources):
    from futsal.datasets.datasetSamples import classifySample

    parent = loadDatasetPlan(trainingBundle[0] / "plan.json")
    recipe = loadRecipe(preparedSources[1].with_name("objectCurriculum.yaml"))
    # A quarantined court annotation is carried unchanged, while object eligibility is derived separately.
    sample = next(sample for sample in parent["samples"] if sample["origin"] == "own")
    point = next(item for item in sample["annotations"] if item.get("canonicalClass") == "lowerCornerSideA")
    point["geometryFindings"] = ["geometryOutOfBounds"]
    coverage = next(task for task in parent["recipe"]["tasks"] if task["taskId"] == "unified25")
    sample["tasks"]["unified25"] = classifySample(sample, coverage, {point["canonicalClass"]})
    # Canonical split evidence must be rebuilt by the native planner, never patched by the derivative.
    from futsal.datasets.datasetPlan import planRecipe
    from futsal.datasets.matchSplits import frozenSplitDocument

    parent["frozenSplit"] = frozenSplitDocument(parent["samples"], parent["frozenSplit"]["assignments"], planRecipe(parent))
    parent["planId"] = hashJson({key: value for key, value in parent.items() if key != "planId"})
    derived = buildObjectCurriculumPlan(parent, recipe)
    assert derived["samples"][0]["tasks"]["detailedObjects"]["eligible"]
    assert not derived["samples"][0]["tasks"]["unified25"]["eligible"]
