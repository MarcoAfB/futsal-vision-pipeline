from futsal.datasets.mixedSources import deriveMixedPlan
from futsal.datasets.playerMixed import ownAthletes
from futsal.datasets.recipes import activeTasks


def validateBallPlayerMixedRecipe(recipe, catalog):
    tasks = activeTasks(recipe)
    sources = [source["sourceId"] for source in catalog["publicSources"]]
    athletes = {name: "player" for name in ["player", "outfieldPlayer", "goalkeeper", *ownAthletes]}
    if len(tasks) != 1 or tasks[0] != {
        "taskId": "ballPlayer", "classes": ["ball", "player"], "trainingSources": ["own", *sources],
        "classMap": {"ball": "ball", **athletes}
    }:
        raise ValueError("Mixed ball and player training requires ballPlayer with own/all registered public sources, "
                         "ball mapped to ball and every athlete semantic mapped to player.")


def buildBallPlayerMixedPlan(parent, recipe):
    return deriveMixedPlan(parent, recipe, validateBallPlayerMixedRecipe, ["ball", *ownAthletes])
