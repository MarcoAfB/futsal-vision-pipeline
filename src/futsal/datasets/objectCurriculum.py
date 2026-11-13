import copy
from pathlib import Path

from futsal.datasets.datasetFiles import hashJson, validateNewOutput
from futsal.datasets.datasetPlan import loadDatasetPlan, validatePlan, writeDatasetPlan
from futsal.datasets.datasetSamples import classifySample, summarizeEligibility
from futsal.datasets.matchSplits import summarizeSplits
from futsal.datasets.recipes import activeTasks, loadRecipe, validateRecipe
from futsal.datasets.sourceTransition import scientificCatalog


def buildObjectCurriculumPlan(parent, recipe):
    validatePlan(parent)
    validateRecipe(recipe.data, recipe.catalog)
    if recipe.catalog.labels != parent["catalog"]["labels"] or (
        scientificCatalog(recipe.catalog.data) != scientificCatalog(parent["catalog"]["data"])
    ):
        raise ValueError("Object curriculum requires the same scientific source catalogue and labels.")
    expected = ["genericObjects", "roleObjects", "detailedObjects"]
    if [task["taskId"] for task in activeTasks(recipe.data)] != expected:
        raise ValueError("Object curriculum requires genericObjects, roleObjects, detailedObjects in order.")
    oldTasks = {task["taskId"]: task for task in parent["recipe"]["tasks"]}
    newTasks = {task["taskId"]: task for task in recipe.data["tasks"]}
    coverage = parent["recipe"]["coverageTask"]
    if recipe.data["coverageTask"] != coverage or any(
        recipe.data[key] != parent["recipe"][key]
        for key in ("pointBoxes", "publicValidation", "unsupportedValidationClasses", "rareClass")
    ) or any(newTasks.get(key) != oldTasks.get(key) for key in (*expected[:2], coverage)):
        raise ValueError("Object curriculum must preserve public tasks, review and canonical coverage.")
    names = ["ball", "outfieldPlayerTeamA", "outfieldPlayerTeamB", "goalkeeperTeamA", "goalkeeperTeamB"]
    detailed = newTasks["detailedObjects"]
    if detailed != {"taskId": "detailedObjects", "classes": names,
                    "trainingSources": ["own"], "classMap": dict(zip(names, names, strict=True))}:
        raise ValueError("Detailed objects requires ball and the four native athlete classes from own data.")
    plan = copy.deepcopy(parent)
    plan.pop("planId")
    plan.update(parentPlanId=parent["planId"], recipe=copy.deepcopy(recipe.data), configEvidence=recipe.evidence)
    videos = {video["videoId"]: video for video in parent["catalog"]["data"]["ownVideos"]}
    points = {item["name"] for item in recipe.catalog.labels["canonicalClasses"] if item["geometry"] == "point"}
    for sample in plan["samples"]:
        if sample["origin"] != "own":
            continue
        status = classifySample(sample, detailed, points)
        video = videos[sample["sourceId"]]
        if video["annotationScope"] != "completeForDeclaredClasses" or not set(names) <= set(video["classesAnnotated"]):
            status["eligible"] = False
            status["reasons"].append("annotationScopeUnverified")
        sample["tasks"]["detailedObjects"] = status
    plan["eligibilitySummary"] = summarizeEligibility(plan["samples"], recipe.data["tasks"])
    plan["ownSplitSummaries"] = {task["taskId"]: summarizeSplits(plan["samples"], plan["frozenSplit"]["assignments"], task)
                                 for task in recipe.data["tasks"]}
    if any(plan["ownSplitSummaries"]["detailedObjects"][split]["missingClasses"] for split in ("training", "test")):
        raise ValueError("Detailed objects must retain all own training/test classes.")
    plan["planId"] = hashJson(plan)
    validatePlan(plan)
    return plan


def prepareObjectCurriculum(sourceDatasetDir, configPath, mediaStoreDir, planDir, datasetDir, *, progress=print):
    from futsal.datasets.mediaRegistration import registerMedia
    from futsal.datasets.sharedViews import prepareView

    source, configPath, store, planDir, output = (Path(path).resolve() for path in
        (sourceDatasetDir, configPath, mediaStoreDir, planDir, datasetDir))
    project = Path(__file__).resolve().parents[3]
    protected = [source, configPath, store, project / "src", project / "configs", project / "data/raw"]
    for target, other in ((planDir, output), (output, planDir)):
        for path in [*protected, other]:
            validateNewOutput(target, path, [path])
    plan = buildObjectCurriculumPlan(loadDatasetPlan(source / "plan.json"), loadRecipe(configPath))
    writeDatasetPlan(plan, planDir, source, protectedPaths=protected)
    registration = registerMedia(store, planDir, progress=progress)
    return prepareView(store, configPath, output, collectionId=registration["collectionId"], progress=progress)
