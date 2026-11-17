from pathlib import Path

import numpy as np

from futsal.inventory.catalog import readYaml
from futsal.teams.clusterRoles import clusterEvidence, clusterRoles, detectionRoles

config = readYaml(Path("configs/evaluation/teamAssignment.yaml"))[0]
court = {"length": 40.0, "width": 20.0}
centers = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 8.0], [9.0, 1.0]])
labels = np.array([0] * 10 + [1] * 9 + [2] * 4 + [3] * 2)


def positions(keeper=(1.0, 10.0)):
    return np.array([(15.0, 5.0)] * 10 + [(25.0, 15.0)] * 9 + [keeper] * 4 + [(20.0, 10.0)] * 2)


def testGoalZoneMakesASmallClusterAGoalkeeperAndMergesTheRest():
    evidence = clusterEvidence(labels, positions(), [None] * len(labels), court, config)
    roles, teams = clusterRoles(centers, evidence, config)

    assert teams == [0, 1]
    assert roles[2] == {"role": "goalkeeper", "side": "A", "source": "goalZone"}
    assert roles[3] == {"role": "outfield", "team": 1, "source": "mergedNearestTeam"}
    assert evidence[3]["goalZoneFraction"] is None


def testSideVotesDecideWithoutProjectionsAndFarClustersStayOutfield():
    unprojected = np.full((len(labels), 2), np.nan)
    votes = ["A"] * 19 + ["B"] * 4 + [None] * 2
    roles, _ = clusterRoles(centers, clusterEvidence(labels, unprojected, votes, court, config), config)
    assert roles[2] == {"role": "goalkeeper", "side": "B", "source": "sideVote"}

    midfield = clusterEvidence(labels, positions(keeper=(20.0, 10.0)), [None] * len(labels), court, config)
    assert clusterRoles(centers, midfield, config)[0][2]["role"] == "outfield"


def testOnlyOneGoalkeeperPerSideIsKeptInAFrame():
    roles = {0: {"role": "outfield", "team": 0, "source": "largestCluster"},
             1: {"role": "outfield", "team": 1, "source": "largestCluster"},
             2: {"role": "goalkeeper", "side": "A", "source": "goalZone"}}
    points = np.array([[5.0, 8.0], [9.0, 1.0], [0.0, 0.0]])
    assigned = detectionRoles(np.array([2, 2, 0]), points, centers, ["f0", "f0", "f0"], roles, [0, 1])

    assert assigned[0]["role"] == "goalkeeper"
    assert assigned[1] == {"role": "outfield", "team": 1, "source": "frameGoalkeeperLimit"}
