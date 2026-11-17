from futsal.datasets.mixedSources import deriveMixedPlan
from futsal.datasets.recipes import activeTasks

ownAthletes = ["outfieldPlayerTeamA", "outfieldPlayerTeamB", "goalkeeperTeamA", "goalkeeperTeamB"]


def validatePlayerMixedRecipe(recipe, catalog):
    tasks = activeTasks(recipe)
    sources = [source["sourceId"] for source in catalog["publicSources"]]
    classMap = {name: "player" for name in ["player", "outfieldPlayer", "goalkeeper", *ownAthletes]}
    if len(tasks) != 1 or tasks[0] != {
        "taskId": "playerOnly", "classes": ["player"], "trainingSources": ["own", *sources], "classMap": classMap
    }:
        raise ValueError("Mixed player training requires playerOnly with own/all registered public sources "
                         "and every athlete semantic mapped to player.")


def buildPlayerMixedPlan(parent, recipe):
    return deriveMixedPlan(parent, recipe, validatePlayerMixedRecipe, ownAthletes)
