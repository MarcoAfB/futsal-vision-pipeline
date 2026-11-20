from collections import Counter, defaultdict

import numpy as np

from futsal.teams.sideEvidence import courtSide, goalDistances, outfieldNames


def majority(values):
    counts = Counter(values)
    return min(counts, key=lambda value: (-counts[value], value))


def trackGroups(rows, votes):
    """Group detections by (segmentId, trackId); tracks without any embedded member stay unassigned."""
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(row["segmentId"], row["trackId"])].append(index)
    return {key: members for key, members in groups.items() if any(votes[index] for index in members)}


def trackCandidate(members, votes, positions, court, config):
    """Goal-zone geometry of the track decides when enough members are projected; member roles otherwise."""
    projected = positions[members][np.isfinite(positions[members]).all(1)]
    if len(projected) >= config["minEvidence"]:
        nearGoal = projected[goalDistances(projected, court) <= config["goalRadiusMeters"]]
        if len(nearGoal) / len(projected) < config["goalkeeperZoneMin"]:
            return None
        side = majority(courtSide(x, court) for x in nearGoal[:, 0])
        return {"side": side, "source": "trackGoalZone", "strength": [1, len(nearGoal)]}
    embedded = [votes[index] for index in members if votes[index]]
    keepers = [vote["side"] for vote in embedded if vote["role"] == "goalkeeper"]
    if 2 * len(keepers) > len(embedded):
        return {"side": majority(keepers), "source": "trackRoleMajority", "strength": [0, len(embedded)]}
    return None


def trackTeam(members, votes, rows):
    counts, confidence = Counter(), Counter()
    for index in members:
        if votes[index]:
            counts[votes[index]["teamVote"]] += 1
            confidence[votes[index]["teamVote"]] += rows[index]["confidence"]
    return min(counts, key=lambda team: (-counts[team], -confidence[team], team))


def acceptGoalkeepers(tracks, candidates, rows):
    """Keep at most one goalkeeper track per side in any frame, strongest first; demote the others whole."""
    order = sorted(candidates, key=lambda key: (-candidates[key]["strength"][0], -candidates[key]["strength"][1],
                                                -len(tracks[key]), key))
    keepers, demoted, occupied = {}, [], set()
    for key in order:
        frames = {(rows[index]["recordId"], candidates[key]["side"]) for index in tracks[key]}
        if frames & occupied:
            demoted.append(key)
        else:
            occupied |= frames
            keepers[key] = candidates[key]
    return keepers, demoted


def trackRoles(tracks, votes, rows, keepers, demoted):
    roles = {}
    for key, members in tracks.items():
        if key in keepers:
            roles[key] = {"role": "goalkeeper", "side": keepers[key]["side"], "source": keepers[key]["source"]}
        else:
            roles[key] = {"role": "outfield", "team": trackTeam(members, votes, rows),
                          "source": "trackGoalkeeperLimit" if key in demoted else "trackTeamVote"}
    return roles


def assignTracks(rows, votes, positions, court, teams, config):
    """Aggregate per-detection team votes and goalkeeper evidence over each track of one match.

    rows are every crop row of the match, votes the per-detection assignment of rows with an
    embedding (None otherwise) and positions their court positions (NaN when not projected).
    Rows of a track without an embedding inherit the track's label.
    """
    tracks = trackGroups(rows, votes)
    candidates = {key: candidate for key, members in tracks.items()
                  if (candidate := trackCandidate(members, votes, positions, court, config))}
    keepers, demoted = acceptGoalkeepers(tracks, candidates, rows)
    roles = trackRoles(tracks, votes, rows, keepers, demoted)
    assigned = [roles.get((row["segmentId"], row["trackId"])) for row in rows]
    teamOf = np.array([item["team"] if item and item["role"] == "outfield" else -1 for item in assigned])
    names, namingSource, meanX = outfieldNames(teams, teamOf, positions, config["minEvidence"])
    classNames = [None if item is None else f"outfieldPlayerTeam{names[item['team']]}"
                  if item["role"] == "outfield" else f"goalkeeperTeam{item['side']}" for item in assigned]
    decision = {"tracks": len(tracks), "teams": {str(team): names[team] for team in teams},
                "outfieldNaming": {"source": namingSource, "meanX": {str(team): value for team, value in meanX.items()}},
                "goalkeeperTracks": [{"segmentId": key[0], "trackId": key[1], "members": len(tracks[key]),
                                      **keepers[key]} for key in sorted(keepers)],
                "demotedGoalkeeperTracks": len(demoted),
                "inheritedDetections": sum(item is not None and not vote for item, vote in
                                           zip(assigned, votes, strict=True))}
    return assigned, classNames, decision
