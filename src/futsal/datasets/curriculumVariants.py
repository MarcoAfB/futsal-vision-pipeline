def curriculumVariant(name):
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
    raise ValueError("curriculum must be objects or players.")
