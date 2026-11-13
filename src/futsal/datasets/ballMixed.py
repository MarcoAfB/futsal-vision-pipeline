import copy

from futsal.datasets.datasetSamples import classifySample
from futsal.datasets.objectCurriculum import finalizeCurriculumPlan, validateCurriculumParent
from futsal.datasets.recipes import activeTasks


def validateBallMixedRecipe(recipe, catalog):
    tasks = activeTasks(recipe)
    sources = [source["sourceId"] for source in catalog["publicSources"]]
    if len(tasks) != 1 or tasks[0] != {
        "taskId": "ballOnly", "classes": ["ball"], "trainingSources": ["own", *sources],
        "classMap": {"ball": "ball"}, "ignoredSourceClasses": ["player", "outfieldPlayer", "goalkeeper"]
    }:
        raise ValueError("Mixed ball training requires ballOnly with own/all registered public sources and explicit ignored athlete classes.")


def buildBallMixedPlan(parent, recipe):
    validateCurriculumParent(parent, recipe)
    validateBallMixedRecipe(recipe.data, parent["catalog"]["data"])
    publicSources = {source["sourceId"] for source in parent["catalog"]["data"]["publicSources"]}
    if not publicSources <= {sample["sourceId"] for sample in parent["samples"] if sample["origin"] == "public"}:
        raise ValueError("Mixed ball derivation requires native public records; use the original curriculum plan, not an own-only projection.")
    plan = copy.deepcopy(parent)
    plan.pop("planId")
    plan.update(parentPlanId=parent["planId"], recipe=copy.deepcopy(recipe.data), configEvidence=recipe.evidence)
    task = activeTasks(recipe.data)[0]
    videos = {video["videoId"]: video for video in parent["catalog"]["data"]["ownVideos"]}
    for sample in plan["samples"]:
        status = classifySample(sample, task, set())
        if sample["origin"] == "own":
            video = videos[sample["sourceId"]]
            if video["annotationScope"] != "completeForDeclaredClasses" or "ball" not in video["classesAnnotated"]:
                status["eligible"] = False
                status["reasons"].append("annotationScopeUnverified")
        sample["tasks"]["ballOnly"] = status
    return finalizeCurriculumPlan(plan, recipe, ["ballOnly"])
