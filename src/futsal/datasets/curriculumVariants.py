def trainingVariant(name):
    athletes = ["outfieldPlayerTeamA", "outfieldPlayerTeamB", "goalkeeperTeamA", "goalkeeperTeamB"]
    if name == "objects":
        return {"stem": "object", "tasks": {
            "genericObjects": ["ball", "player"],
            "roleObjects": ["ball", "outfieldPlayer", "goalkeeper"],
            "detailedObjects": ["ball", *athletes]}}
    if name == "players":
        return {"stem": "player", "tasks": {
            "genericPlayers": ["player"],
            "rolePlayers": ["outfieldPlayer", "goalkeeper"],
            "detailedPlayers": athletes}}
    if name == "ballMixed":
        return {"stem": "ballMixed", "trainingMode": "naturalMixedSources", "tasks": {"ballOnly": ["ball"]}}
    if name == "playerMixed":
        return {"stem": "playerMixed", "trainingMode": "naturalMixedSources", "tasks": {"playerOnly": ["player"]}}
    if name == "ballPlayerMixed":
        return {"stem": "ballPlayerMixed", "trainingMode": "naturalMixedSources",
                "tasks": {"ballPlayer": ["ball", "player"]}}
    raise ValueError("training variant must be objects, players, ballMixed, playerMixed or ballPlayerMixed.")


def curriculumVariant(name):
    if name in {"objects", "players"}:
        return trainingVariant(name)
    raise ValueError("curriculum must be objects or players.")
