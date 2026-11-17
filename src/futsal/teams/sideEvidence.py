import numpy as np

from futsal.geometry.courtProjection import groundPoints, imageToCourt


def frameEvidence(row, statuses):
    """Summarize one sealed homography frame for team assignment.

    Side votes add the confidences of predicted side-specific landmarks; they say
    which goal area the camera shows, not where a particular athlete stands.
    """
    prediction = row["prediction"]
    votes = {"A": 0.0, "B": 0.0}
    for candidate in prediction["candidates"]:
        for side in votes:
            if candidate["className"].endswith(f"Side{side}"):
                votes[side] += candidate["confidence"]
    return {"status": prediction["status"], "sideVotes": votes,
            "sideVote": None if votes["A"] == votes["B"] else max(votes, key=votes.get),
            "courtToImage": prediction["courtToImage"] if prediction["status"] in statuses else None}


def courtPositions(boxes, courtToImage):
    positions = np.full((len(boxes), 2), np.nan)
    if courtToImage is None or not boxes:
        return positions
    points, scale = imageToCourt(np.asarray(courtToImage, dtype=float), groundPoints(boxes))
    valid = (scale > 0) & np.isfinite(points).all(1)
    positions[valid] = points[valid]
    return positions


def courtSide(x, court):
    return "A" if x < court["length"] / 2 else "B"


def goalDistances(positions, court):
    goals = np.array([[0.0, court["width"] / 2], [court["length"], court["width"] / 2]])
    positions = np.asarray(positions, dtype=float).reshape(-1, 2)
    return np.linalg.norm(positions[:, None, :] - goals[None], axis=2).min(1)


def outfieldNames(teams, teamOf, positions, minEvidence):
    """Name the outfield team with the lower mean court x as team A.

    Team A is the team defending Side A in the analysed clip. Without enough
    projected members the names are arbitrary and the source is "none".
    """
    meanX = {}
    for team in teams:
        values = positions[teamOf == team, 0]
        values = values[np.isfinite(values)]
        meanX[team] = float(values.mean()) if len(values) >= minEvidence else None
    first, second = teams
    if None not in meanX.values() and meanX[first] != meanX[second]:
        teamA, source = min(teams, key=meanX.get), "meanX"
    else:
        teamA, source = first, "none"
    return {team: "A" if team == teamA else "B" for team in teams}, source, meanX
