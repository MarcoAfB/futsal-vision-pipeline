from pathlib import Path

import numpy as np
import pytest
import yaml

from futsal.geometry.courtProjection import imageToCourt, projectPoints
from futsal.geometry.homographyEstimation import estimateHomography


@pytest.fixture
def config():
    return yaml.safe_load(Path("configs/evaluation/homography.yaml").read_text())


def testExactBroadcastProjectionIsVerifiedAndInvertible(broadcastView, config):
    view = broadcastView
    result = estimateHomography(view["court"], view["image"], view["dimensions"], config, robust=True)
    mapped, scale = imageToCourt(np.asarray(result["courtToImage"]), view["image"])

    assert len(view["court"]) >= 8
    assert result["status"] == "verified" and result["reasons"] == []
    assert result["inlierCount"] == len(view["court"])
    assert np.all(scale > 0) and np.allclose(mapped, view["court"], atol=1e-6)


def testGrossOutliersAreExcludedFromTheRobustFit(broadcastView, config):
    image = broadcastView["image"].copy()
    image[[0, 3]] += [[180.0, -140.0], [-220.0, 90.0]]
    result = estimateHomography(broadcastView["court"], image, broadcastView["dimensions"], config, robust=True)

    assert result["status"] == "verified"
    assert [index for index, inlier in enumerate(result["inliers"]) if not inlier] == [0, 3]
    assert result["inlierRmsePixels"] < 1e-3


@pytest.mark.parametrize("mirror,reason", [
    (lambda court: np.column_stack([40 - court[:, 0], court[:, 1]]), "orientationReversed"),
    (lambda court: np.column_stack([court[:, 0], 20 - court[:, 1]]), "orientationReversed"),
    (lambda court: np.column_stack([40 - court[:, 0], 20 - court[:, 1]]), "farNearInverted"),
])
def testSwappedSideOrTouchlineLabelsAreRejectedDespiteLowResidual(broadcastView, config, mirror, reason):
    court = mirror(broadcastView["court"])
    result = estimateHomography(court, broadcastView["image"], broadcastView["dimensions"], config, robust=True)

    assert result["status"] == "rejected" and reason in result["reasons"]
    assert result["inlierRmsePixels"] < 1e-3


def testThreeCollinearHalfwayPointsCannotDefineAHomography(broadcastView, config):
    names = ["lowerHalfwayLinePoint", "upperCenterCourtPoint", "upperHalfwayLinePoint", "secondPenaltyMarkSideA"]
    court = np.array([broadcastView["landmarks"][name] for name in names])
    image = np.array([[900.0, 700.0], [920.0, 180.0], [910.0, 420.0], [430.0, 520.0]])
    for robust in (True, False):
        result = estimateHomography(court, image, broadcastView["dimensions"], config, robust=robust)
        assert result["reasons"] == ["degenerateCourtConfiguration"]


def testFitsThatPutTheirOwnPointsBehindTheCameraAreNotModels(broadcastView, config):
    # The fourth plane point lies behind the camera; its finite pixel is a mirrored
    # projection, so the only exact fit has mixed homogeneous signs.
    court = np.array([[0.0, 0.0], [40.0, 0.0], [20.0, 10.0], [20.0, 60.0]])
    image, scale = projectPoints(broadcastView["matrix"], court)
    result = estimateHomography(court, image, broadcastView["dimensions"], config, robust=True)

    assert list(scale > 0) == [True, True, True, False]
    assert result["reasons"] == ["noConsistentSubset"]


def testFourOrFewerLandmarksAreMinimalOrInsufficient(broadcastView, config):
    view = broadcastView
    four = estimateHomography(view["court"][:4], view["image"][:4], view["dimensions"], config, robust=True)
    three = estimateHomography(view["court"][:3], view["image"][:3], view["dimensions"], config, robust=True)

    assert four["status"] == "minimal" and four["inlierCount"] == 4
    assert three["status"] == "rejected" and three["reasons"] == ["insufficientPoints"]


def testReferenceFitRejectsInconsistentAnnotationsInsteadOfDroppingThem(broadcastView, config):
    image = broadcastView["image"].copy()
    image[1] += [60.0, 40.0]
    result = estimateHomography(broadcastView["court"], image, broadcastView["dimensions"], config, robust=False)

    assert result["status"] == "rejected" and "inlierRmseHigh" in result["reasons"]
    assert all(result["inliers"])
