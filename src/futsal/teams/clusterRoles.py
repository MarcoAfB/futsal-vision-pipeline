from collections import Counter

import numpy as np

from futsal.teams.sideEvidence import courtSide, goalDistances, outfieldNames
from futsal.teams.teamClustering import clusterFeatures


def clusterEvidence(labels, positions, sideVotes, court, config):
    projectedMask = np.isfinite(positions).all(1)
    rows = {}
    for cluster in range(config["clusterCount"]):
        members = labels == cluster
        projected = positions[members & projectedMask]
        nearGoal = projected[goalDistances(projected, court) <= config["goalRadiusMeters"]]
        zoneSides = Counter(courtSide(x, court) for x in nearGoal[:, 0])
        votes = Counter(vote for vote, member in zip(sideVotes, members, strict=True) if member and vote)
        voteCount = votes["A"] + votes["B"]
        enoughProjected, enoughVotes = len(projected) >= config["minEvidence"], voteCount >= config["minEvidence"]
        rows[cluster] = {
            "size": int(members.sum()), "share": float(members.mean()) if len(labels) else 0.0,
            "projected": len(projected), "nearGoal": len(nearGoal),
            "goalZoneFraction": len(nearGoal) / len(projected) if enoughProjected else None,
            "goalZoneSide": zoneSides.most_common(1)[0][0] if zoneSides else None,
            "sideVotes": {"A": votes["A"], "B": votes["B"]},
            "sideConsistency": max(votes["A"], votes["B"]) / voteCount if enoughVotes else None,
            "voteSide": votes.most_common(1)[0][0] if voteCount else None}
    return rows


def goalkeeperCandidate(evidence, config):
    """Goal-zone evidence decides when enough members are projected; frame side votes otherwise."""
    if evidence["share"] > config["goalkeeperMaxShare"]:
        return None
    if evidence["goalZoneFraction"] is not None:
        if evidence["goalZoneFraction"] >= config["goalkeeperZoneMin"] and evidence["goalZoneSide"]:
            return {"side": evidence["goalZoneSide"], "source": "goalZone", "strength": evidence["goalZoneFraction"]}
        return None
    if evidence["sideConsistency"] is not None and evidence["sideConsistency"] >= config["goalkeeperSideMin"]:
        return {"side": evidence["voteSide"], "source": "sideVote", "strength": evidence["sideConsistency"]}
    return None


def clusterRoles(centers, evidence, config):
    order = sorted(evidence, key=lambda cluster: (-evidence[cluster]["size"], cluster))
    teams = order[:2]
    roles = {cluster: {"role": "outfield", "team": cluster, "source": "largestCluster"} for cluster in teams}
    candidates = [(cluster, candidate) for cluster in order[2:]
                  if (candidate := goalkeeperCandidate(evidence[cluster], config))]
    for cluster, candidate in sorted(candidates, key=lambda item: (-item[1]["strength"], item[0])):
        if all(role.get("side") != candidate["side"] for role in roles.values()):
            roles[cluster] = {"role": "goalkeeper", "side": candidate["side"], "source": candidate["source"]}
    for cluster in order[2:]:
        if cluster not in roles:
            roles[cluster] = {"role": "outfield", "team": nearestTeam(centers[cluster], centers, teams),
                              "source": "mergedNearestTeam"}
    return roles, teams


def nearestTeam(point, centers, teams):
    return min(teams, key=lambda team: (float(np.linalg.norm(point - centers[team])), team))


def detectionRoles(labels, points, centers, recordIds, roles, teams):
    """Keep at most one goalkeeper per side in a frame; the others join the nearest team."""
    assigned = [dict(roles[label]) for label in labels]
    keeperCenters = {role["side"]: centers[cluster] for cluster, role in roles.items() if role["role"] == "goalkeeper"}
    byFrame = {}
    for index, item in enumerate(assigned):
        if item["role"] == "goalkeeper":
            byFrame.setdefault((recordIds[index], item["side"]), []).append(index)
    for (_, side), indices in byFrame.items():
        keep = min(indices, key=lambda index: (float(np.linalg.norm(points[index] - keeperCenters[side])), index))
        for index in indices:
            if index != keep:
                assigned[index] = {"role": "outfield", "team": nearestTeam(points[index], centers, teams),
                                   "source": "frameGoalkeeperLimit"}
    return assigned


def assignMatch(features, clean, positions, sideVotes, recordIds, court, config):
    labels, centers, points = clusterFeatures(features, clean, config)
    evidence = clusterEvidence(labels, positions, sideVotes, court, config)
    roles, teams = clusterRoles(centers, evidence, config)
    assigned = detectionRoles(labels, points, centers, recordIds, roles, teams)
    for item, point in zip(assigned, points, strict=True):
        # The outfield team a detection would join; a track-level vote reads it for goalkeepers too.
        item["teamVote"] = item["team"] if item["role"] == "outfield" else nearestTeam(point, centers, teams)
    teamOf = np.array([item.get("team", -1) for item in assigned])
    names, namingSource, meanX = outfieldNames(teams, teamOf, positions, config["minEvidence"])
    classNames = [f"outfieldPlayerTeam{names[item['team']]}" if item["role"] == "outfield"
                  else f"goalkeeperTeam{item['side']}" for item in assigned]
    decision = {"clusters": {str(cluster): {**evidence[cluster], **roles[cluster]} for cluster in sorted(evidence)},
                "teams": {str(team): names[team] for team in teams},
                "outfieldNaming": {"source": namingSource, "meanX": {str(team): value for team, value in meanX.items()}},
                "goalkeeperSides": sorted(role["side"] for role in roles.values() if role["role"] == "goalkeeper"),
                "reassignedGoalkeepers": sum(item["source"] == "frameGoalkeeperLimit" for item in assigned)}
    return labels, assigned, classNames, decision, teams
