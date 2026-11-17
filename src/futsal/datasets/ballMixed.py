from futsal.datasets.mixedSources import deriveMixedPlan
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
    return deriveMixedPlan(parent, recipe, validateBallMixedRecipe, ["ball"])
