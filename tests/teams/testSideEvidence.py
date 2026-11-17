import cv2
import numpy as np

from futsal.geometry.courtProjection import projectPoints
from futsal.teams.sideEvidence import courtPositions, frameEvidence, outfieldNames

courtToImage = cv2.getPerspectiveTransform(
    np.float32([[0, 0], [40, 0], [40, 20], [0, 20]]),
    np.float32([[300, 150], [1620, 150], [1900, 1000], [20, 1000]])).astype(float)


def athleteBox(foot):
    u, v = projectPoints(courtToImage, foot)[0][0]
    return [u - 20.0, v - 120.0, u + 20.0, v]


def frame(status, candidates):
    return {"prediction": {"status": status, "courtToImage": courtToImage.tolist(), "candidates": candidates}}


def testSideVotesAddLandmarkConfidencesAndMinimalFitsCanBeExcluded():
    candidates = [{"className": "penaltyMarkSideB", "confidence": 0.6},
                  {"className": "upperGoalpostSideA", "confidence": 0.4},
                  {"className": "upperHalfwayLinePoint", "confidence": 0.9}]
    minimal = frameEvidence(frame("minimal", candidates), ["verified"])
    tied = frameEvidence(frame("verified", []), ["verified"])

    assert minimal["sideVote"] == "B" and minimal["courtToImage"] is None
    assert tied["sideVote"] is None and tied["courtToImage"] is not None


def testCourtPositionsUseTheBoxBottomCentre():
    positions = courtPositions([athleteBox((12.0, 7.0))], courtToImage)
    assert np.allclose(positions, [[12.0, 7.0]], atol=1e-6)
    assert np.isnan(courtPositions([athleteBox((12.0, 7.0))], None)).all()


def testLowerMeanCourtXNamesTeamAAndMissingEvidenceIsArbitrary():
    teamOf = np.array([0, 0, 0, 1, 1, 1])
    positions = np.array([[30.0, 5.0]] * 3 + [[12.0, 5.0]] * 3)
    names, source, meanX = outfieldNames([0, 1], teamOf, positions, 3)
    assert names == {0: "B", 1: "A"} and source == "meanX" and meanX == {0: 30.0, 1: 12.0}

    names, source, _ = outfieldNames([0, 1], teamOf, np.full((6, 2), np.nan), 3)
    assert names == {0: "A", 1: "B"} and source == "none"
