import numpy as np

from futsal.teams.trackRoles import assignTracks

court = {"length": 40.0, "width": 20.0}
config = {"minEvidence": 3, "goalRadiusMeters": 4.0, "goalkeeperZoneMin": 0.5}
unprojected = [np.nan, np.nan]


def row(frame, track, confidence=0.9):
    return {"recordId": f"video01:{frame}", "segmentId": "video01:interval:0", "trackId": track,
            "confidence": confidence}


def outfield(team):
    return {"role": "outfield", "team": team, "teamVote": team, "source": "largestCluster"}


def keeper(side, teamVote=0):
    return {"role": "goalkeeper", "side": side, "teamVote": teamVote, "source": "goalZone"}


def assign(rows, votes, positions):
    return assignTracks(rows, votes, np.array(positions, dtype=float), court, [0, 1], config)


def testMajorityTeamVoteRelabelsAMinorityDetection():
    rows = [row(0, 1), row(1, 1), row(2, 1), row(0, 2)]
    _, classes, decision = assign(rows, [outfield(0), outfield(0), outfield(1), outfield(1)], [unprojected] * 4)

    assert classes == ["outfieldPlayerTeamA"] * 3 + ["outfieldPlayerTeamB"]
    assert decision["tracks"] == 2 and decision["outfieldNaming"]["source"] == "none"


def testGoalZoneTrackTakesItsOwnSideOverItsCluster():
    # Both keepers share one cluster named side B; the track near goal A becomes goalkeeper A.
    rows = [row(frame, 1) for frame in range(3)] + [row(frame, 2) for frame in range(3)] + \
        [row(frame, 3) for frame in range(3)]
    votes = [keeper("B")] * 3 + [outfield(0)] * 3 + [outfield(1)] * 3
    positions = [[1.5, 10]] * 3 + [[15, 8]] * 3 + [[25, 12]] * 3
    assigned, classes, decision = assign(rows, votes, positions)

    assert classes[:3] == ["goalkeeperTeamA"] * 3 and assigned[0]["source"] == "trackGoalZone"
    assert classes[3:] == ["outfieldPlayerTeamA"] * 3 + ["outfieldPlayerTeamB"] * 3
    assert decision["outfieldNaming"]["source"] == "meanX"
    assert decision["goalkeeperTracks"] == [{"segmentId": "video01:interval:0", "trackId": 1, "members": 3,
                                             "side": "A", "source": "trackGoalZone", "strength": [1, 3]}]


def testProjectedTrackAwayFromGoalsStaysOutfield():
    rows = [row(frame, 1) for frame in range(3)]
    assigned, classes, _ = assign(rows, [keeper("B", teamVote=1)] * 3, [[20, 10]] * 3)

    assert classes == ["outfieldPlayerTeamB"] * 3 and assigned[0]["source"] == "trackTeamVote"


def testWeakerGoalkeeperTrackSharingAFrameAndSideIsDemotedWhole():
    rows = [row(frame, 1) for frame in range(4)] + [row(frame, 2) for frame in range(3, 6)]
    votes = [keeper("A")] * 4 + [keeper("A", teamVote=1)] * 3
    assigned, classes, decision = assign(rows, votes, [[1.5, 10]] * 4 + [[2.5, 9]] * 3)

    assert classes[:4] == ["goalkeeperTeamA"] * 4
    assert classes[4:] == ["outfieldPlayerTeamB"] * 3
    assert {item["source"] for item in assigned[4:]} == {"trackGoalkeeperLimit"}
    assert decision["demotedGoalkeeperTracks"] == 1


def testUnprojectedTrackFollowsTheMajorityOfItsDetectionRoles():
    rows = [row(frame, 1) for frame in range(3)]
    assigned, classes, _ = assign(rows, [keeper("B"), keeper("B"), outfield(0)], [unprojected] * 3)

    assert classes == ["goalkeeperTeamB"] * 3 and assigned[0]["source"] == "trackRoleMajority"


def testDetectionsWithoutCropsInheritTheirTrackOnly():
    rows = [row(0, 1), row(1, 1), row(2, 1), row(0, 9)]
    assigned, classes, decision = assign(rows, [outfield(1), None, outfield(1), None], [unprojected] * 4)

    assert classes == ["outfieldPlayerTeamB"] * 3 + [None]
    assert assigned[3] is None and decision["inheritedDetections"] == 1 and decision["tracks"] == 1


def testSingleDetectionTrackKeepsItsDetectionLabel():
    rows = [row(0, 1), row(0, 2)]
    _, classes, _ = assign(rows, [outfield(1), keeper("A")], [[20, 10], [2, 10]])

    assert classes == ["outfieldPlayerTeamB", "goalkeeperTeamA"]
