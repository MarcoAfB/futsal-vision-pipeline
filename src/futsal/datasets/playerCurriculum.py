import copy

from futsal.datasets.curriculumVariants import curriculumVariant
from futsal.datasets.datasetSamples import classifySample
from futsal.datasets.objectCurriculum import finalizeCurriculumPlan, validateCurriculumParent
from futsal.datasets.recipes import activeTasks


def buildPlayerCurriculumPlan(parent, recipe):
    validateCurriculumParent(parent, recipe)
    expected = curriculumVariant("players")["tasks"]
    tasks = activeTasks(recipe.data)
    if {task["taskId"]: task["classes"] for task in tasks} != expected or (
        [task["taskId"] for task in tasks] != list(expected)
    ):
        raise ValueError("Player curriculum requires ordered genericPlayers, rolePlayers, detailedPlayers with 1/2/4 classes.")
    oldTasks = {task["taskId"]: task for task in parent["recipe"]["tasks"]}
    for task, parentId in zip(tasks[:2], ("genericObjects", "roleObjects"), strict=True):
        previous = oldTasks.get(parentId)
        if previous is None or "ball" not in previous["classes"] or task["classes"] != [
            name for name in previous["classes"] if name != "ball"
        ] or task["trainingSources"] != previous["trainingSources"] or task["classMap"] != {
            name: target for name, target in previous["classMap"].items() if name != "ball"
        }:
            raise ValueError("Player curriculum must preserve public sources and athlete semantics from the native object tasks.")
    detailed = tasks[-1]
    if detailed["trainingSources"] != ["own"] or detailed["classMap"] != {
        name: name for name in detailed["classes"]
    }:
        raise ValueError("Detailed players requires the four native athlete classes from own data.")
    plan = copy.deepcopy(parent)
    plan.pop("planId")
    plan.update(parentPlanId=parent["planId"], recipe=copy.deepcopy(recipe.data), configEvidence=recipe.evidence)
    videos = {video["videoId"]: video for video in parent["catalog"]["data"]["ownVideos"]}
    canonical = {item["name"] for item in recipe.catalog.labels["canonicalClasses"]}
    points = {item["name"] for item in recipe.catalog.labels["canonicalClasses"] if item["geometry"] == "point"}
    for sample in plan["samples"]:
        for task in tasks:
            if sample["origin"] == "public" and sample["sourceId"] not in task["trainingSources"]:
                continue
            status = classifySample(sample, task, points)
            if sample["origin"] == "own":
                video = videos[sample["sourceId"]]
                required = set(task["classMap"]) & canonical
                if video["annotationScope"] != "completeForDeclaredClasses" or not required <= set(video["classesAnnotated"]):
                    status["eligible"] = False
                    status["reasons"].append("annotationScopeUnverified")
            sample["tasks"][task["taskId"]] = status
    return finalizeCurriculumPlan(plan, recipe, list(expected))
