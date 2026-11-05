import copy

from futsal.datasets.datasetFiles import hashJson
from futsal.datasets.datasetPlan import validatePlan
from futsal.datasets.datasetSamples import classifySample, summarizeEligibility
from futsal.datasets.matchSplits import summarizeSplits
from futsal.datasets.recipes import activeTasks, validateRecipe


def buildProjectionPlan(parent, recipe):
    validatePlan(parent)
    validateRecipe(recipe.data, recipe.catalog)
    if {"data": recipe.catalog.data, "labels": recipe.catalog.labels} != parent["catalog"]:
        raise ValueError("Projection requires the identical registered source/ontology snapshot.")
    if recipe.data["coverageTask"] != parent["recipe"]["coverageTask"] or recipe.data["pointBoxes"] != parent["recipe"]["pointBoxes"]:
        raise ValueError("Projection must preserve canonical coverage and native point representation.")
    if any(task["trainingSources"] != ["own"] for task in activeTasks(recipe.data)):
        raise ValueError("Projection supports real own-data tasks only.")
    plan = copy.deepcopy(parent)
    plan.pop("planId")
    plan["parentPlanId"] = parent["planId"]
    plan["recipe"] = recipe.data
    plan["configEvidence"] = recipe.evidence
    plan["samples"] = [sample for sample in plan["samples"] if sample["origin"] == "own"]
    points = {item["name"] for item in recipe.catalog.labels["canonicalClasses"] if item["geometry"] == "point"}
    videos = {video["videoId"]: video for video in recipe.catalog.data["ownVideos"]}
    for sample in plan["samples"]:
        previous = sample["tasks"][recipe.data["coverageTask"]]
        sample["tasks"] = {}
        for task in recipe.data["tasks"]:
            status = classifySample(sample, task, points)
            video = videos[sample["sourceId"]]
            if video["annotationScope"] != "completeForDeclaredClasses" or not set(task["classMap"]) <= set(video["classesAnnotated"]):
                status["eligible"] = False
                status["reasons"].append("annotationScopeUnverified")
            sample["tasks"][task["taskId"]] = status
        if sample["tasks"][recipe.data["coverageTask"]] != previous:
            raise ValueError("Projection changed the frozen canonical eligibility basis.")
    plan["eligibilitySummary"] = summarizeEligibility(plan["samples"], recipe.data["tasks"])
    plan["ownSplitSummaries"] = {task["taskId"]: summarizeSplits(plan["samples"], plan["frozenSplit"]["assignments"], task)
                                  for task in recipe.data["tasks"]}
    for task in activeTasks(recipe.data):
        if any(plan["ownSplitSummaries"][task["taskId"]][split]["missingClasses"] for split in ("training", "test")):
            raise ValueError("Each projected task must retain all training/test classes.")
    plan["planId"] = hashJson(plan)
    validatePlan(plan)
    return plan
