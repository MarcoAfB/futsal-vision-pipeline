from futsal.evaluation.teamFeatureRun import loadTeamFeatureConfig
from futsal.teams.athleteCrops import detectionCrops, torsoBox

config = loadTeamFeatureConfig("configs/evaluation/teamFeatures.yaml")[0]


def detection(box, confidence=0.9, className="player"):
    return {"box": box, "confidence": confidence, "className": className, "classId": 0}


def testTorsoWindowFollowsBoxFractionsAndImageBorder():
    assert torsoBox([100, 100, 200, 300], config, [1920, 1080]) == [120, 130, 180, 210]
    assert torsoBox([-50, 1000, 50, 1200], config, [1920, 1080]) == [0, 1030, 30, 1080]
    assert torsoBox([10, 10, 20, 30], config, [1920, 1080]) is None


def testOverlappingAndLowConfidenceDetectionsAreHandledSeparately():
    prediction = {"dimensions": [1920, 1080], "detections": [
        detection([100, 100, 200, 300]), detection([110, 100, 210, 300]), detection([600, 100, 700, 300]),
        detection([900, 100, 1000, 300], confidence=0.1), detection([1200, 100, 1300, 300], className="ball")]}
    rows = detectionCrops(prediction, config)

    assert [row["detectionIndex"] for row in rows] == [0, 1, 2]
    assert [row["clean"] for row in rows] == [False, False, True]
    assert rows[0]["maxOverlap"] > config["overlapIouMax"] and rows[2]["maxOverlap"] == 0.0
