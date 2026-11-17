import copy
from pathlib import Path

from futsal.datasets.curriculumVariants import curriculumVariant, trainingVariant
from futsal.datasets.datasetFiles import hashJson, validateNewOutput
from futsal.datasets.datasetPlan import loadDatasetPlan, validatePlan, writeDatasetPlan
from futsal.datasets.datasetSamples import classifySample, summarizeEligibility
from futsal.datasets.matchSplits import summarizeSplits
from futsal.datasets.recipes import activeTasks, loadRecipe, validateRecipe
from futsal.datasets.sourceTransition import scientificCatalog


def validateCurriculumParent(parent, recipe):
    validatePlan(parent)
    validateRecipe(recipe.data, recipe.catalog)
    if recipe.catalog.labels != parent["catalog"]["labels"] or (
        scientificCatalog(recipe.catalog.data) != scientificCatalog(parent["catalog"]["data"])
    ):
        raise ValueError("Training derivation requires the same scientific source catalogue and labels.")
    oldTasks = {task["taskId"]: task for task in parent["recipe"]["tasks"]}
    newTasks = {task["taskId"]: task for task in recipe.data["tasks"]}
    coverage = parent["recipe"]["coverageTask"]
    if recipe.data["coverageTask"] != coverage or any(
        recipe.data[key] != parent["recipe"][key]
        for key in ("pointBoxes", "publicValidation", "unsupportedValidationClasses", "rareClass")
    ) or newTasks.get(coverage) != oldTasks.get(coverage):
        raise ValueError("Curriculum must preserve public review and canonical coverage.")


def finalizeCurriculumPlan(plan, recipe, taskIds):
    plan["eligibilitySummary"] = summarizeEligibility(plan["samples"], recipe.data["tasks"])
    plan["ownSplitSummaries"] = {task["taskId"]: summarizeSplits(plan["samples"], plan["frozenSplit"]["assignments"], task)
                                 for task in recipe.data["tasks"]}
    if any(plan["ownSplitSummaries"][taskId][split]["missingClasses"] for taskId in taskIds for split in ("training", "test")):
        raise ValueError("Curriculum tasks must retain all own training/test classes.")
    plan["planId"] = hashJson(plan)
    validatePlan(plan)
    return plan


def buildObjectCurriculumPlan(parent, recipe):
    validateCurriculumParent(parent, recipe)
    expected = list(curriculumVariant("objects")["tasks"])
    if [task["taskId"] for task in activeTasks(recipe.data)] != expected:
        raise ValueError("Object curriculum requires genericObjects, roleObjects, detailedObjects in order.")
    oldTasks = {task["taskId"]: task for task in parent["recipe"]["tasks"]}
    newTasks = {task["taskId"]: task for task in recipe.data["tasks"]}
    if any(newTasks.get(key) != oldTasks.get(key) for key in expected[:2]):
        raise ValueError("Object curriculum must preserve public tasks.")
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
    return finalizeCurriculumPlan(plan, recipe, ["detailedObjects"])


def prepareObjectCurriculum(sourceDatasetDir, configPath, mediaStoreDir, planDir, datasetDir, *,
                            progress=print, curriculum="objects", dataRoot=None):
    from futsal.datasets.mediaRegistration import registerMedia
    from futsal.datasets.sharedViews import prepareView

    trainingVariant(curriculum)
    source, configPath, store, planDir, output = (Path(path).resolve() for path in
        (sourceDatasetDir, configPath, mediaStoreDir, planDir, datasetDir))
    project = Path(__file__).resolve().parents[3]
    protected = [source, configPath, store, project / "src", project / "configs", project / "data/raw"]
    if dataRoot is not None:
        protected.append(Path(dataRoot).resolve())
    for target, other in ((planDir, output), (output, planDir)):
        for path in [*protected, other]:
            validateNewOutput(target, path, [path])
    builder = buildObjectCurriculumPlan
    if curriculum == "players":
        from futsal.datasets.playerCurriculum import buildPlayerCurriculumPlan

        builder = buildPlayerCurriculumPlan
    elif curriculum == "ballMixed":
        from futsal.datasets.ballMixed import buildBallMixedPlan

        builder = buildBallMixedPlan
    elif curriculum == "playerMixed":
        from futsal.datasets.playerMixed import buildPlayerMixedPlan

        builder = buildPlayerMixedPlan
    plan = builder(loadDatasetPlan(source / "plan.json"), loadRecipe(configPath))
    writeDatasetPlan(plan, planDir, source, protectedPaths=protected)
    registration = registerMedia(store, planDir, progress=progress)
    return prepareView(store, configPath, output, collectionId=registration["collectionId"],
                       dataRoot=dataRoot, progress=progress)
