import copy

from futsal.datasets.datasetSamples import classifySample
from futsal.datasets.objectCurriculum import finalizeCurriculumPlan, validateCurriculumParent
from futsal.datasets.recipes import activeTasks


def deriveMixedPlan(parent, recipe, validateRecipe, ownClasses):
    validateCurriculumParent(parent, recipe)
    validateRecipe(recipe.data, parent["catalog"]["data"])
    publicSources = {source["sourceId"] for source in parent["catalog"]["data"]["publicSources"]}
    if not publicSources <= {sample["sourceId"] for sample in parent["samples"] if sample["origin"] == "public"}:
        raise ValueError("Mixed-source derivation requires native public records; "
                         "use the original curriculum plan, not an own-only projection.")
    plan = copy.deepcopy(parent)
    plan.pop("planId")
    plan.update(parentPlanId=parent["planId"], recipe=copy.deepcopy(recipe.data), configEvidence=recipe.evidence)
    task = activeTasks(recipe.data)[0]
    videos = {video["videoId"]: video for video in parent["catalog"]["data"]["ownVideos"]}
    for sample in plan["samples"]:
        status = classifySample(sample, task, set())
        if sample["origin"] == "own":
            video = videos[sample["sourceId"]]
            if video["annotationScope"] != "completeForDeclaredClasses" or not set(ownClasses) <= set(video["classesAnnotated"]):
                status["eligible"] = False
                status["reasons"].append("annotationScopeUnverified")
        sample["tasks"][task["taskId"]] = status
    return finalizeCurriculumPlan(plan, recipe, [task["taskId"]])
