import json

import yaml

from futsal.catalog import loadCatalog
from futsal.inventory import buildInventory
from futsal.ownInventory import fileError


def testReviewedLimitsAndSharedMatch(smallDataset):
    dataRoot, configPath, outputDir = smallDataset
    inventory = buildInventory(dataRoot, loadCatalog(configPath))
    assert inventory["summary"]["reviewedFrames"] == 4
    assert inventory["summary"]["matchCount"] == 1
    assert inventory["summary"]["errorCount"] == 0
    assert [frame["frameIndex"] for frame in inventory["ownVideos"][0]["frames"]] == [0, 1]
    assert inventory["ownVideos"][0]["activeShapeCountOutsideReviewedFrames"] == 2
    assert inventory["ownVideos"][0]["outsideShapeCountInFile"] == 1
    assert inventory["ownVideos"][0]["frames"][1]["annotations"] == []
    assert inventory["matches"][0]["reviewedFrameRecordCount"] == 4
    assert inventory["matches"][0]["videoIds"] == ["montage", "video01"]
    assert inventory["splitPolicy"]["selectionStatus"] == "unassigned"
    assert not outputDir.exists()


def testAmbiguitiesAndSourceSemanticsArePreserved(smallDataset):
    dataRoot, configPath, outputDir = smallDataset
    inventory = buildInventory(dataRoot, loadCatalog(configPath))
    codes = inventory["summary"]["issueCounts"]
    assert codes["duplicateLandmark"] == 1
    assert codes["multipleBallAnnotations"] == 1
    frames = inventory["ownVideos"][1]["frames"]
    assert len(frames[0]["annotations"]) == 3
    assert len(frames[1]["annotations"]) == 3
    referee = frames[0]["annotations"][2]
    genericPlayer = frames[1]["annotations"][2]
    assert referee["mappingAction"] == "ignore" and referee["canonicalClass"] is None
    assert genericPlayer["mappingAction"] == "preserve" and genericPlayer["canonicalClass"] is None
    assert genericPlayer["sourceLabel"] == "Jogador" and genericPlayer["trackId"] == "15"
    coverage = next(
        item for item in inventory["classCoverage"] if item["className"] == "lowerCornerSideB"
    )
    assert coverage["annotationCount"] == 2
    assert coverage["frameRecordCount"] == 1
    assert coverage["matchCount"] == 1


def testRegistryMismatchExcludesUnverifiedVideo(smallDataset):
    dataRoot, configPath, outputDir = smallDataset
    registry = yaml.safe_load((dataRoot / "reviewed.yaml").read_text())
    registry["partidas"]["Jogo1"]["maximoFrames"] = 1
    (dataRoot / "reviewed.yaml").write_text(yaml.safe_dump(registry), encoding="utf-8")
    inventory = buildInventory(dataRoot, loadCatalog(configPath))
    assert inventory["ownVideos"][0]["availability"] == "reviewLimitsUnverified"
    assert inventory["ownVideos"][0]["frames"] == []
    assert inventory["summary"]["reviewedFrames"] == 2
    assert inventory["summary"]["issueCounts"]["reviewRegistryMismatch"] == 1
    assert inventory["summary"]["issueCounts"]["expectedCountMismatch"] >= 1


def testNonfiniteGeometryIsReportedAndJsonRemainsStrict(smallDataset):
    dataRoot, configPath, outputDir = smallDataset
    annotationPath = dataRoot / "old.xml"
    annotationPath.write_text(
        annotationPath.read_text().replace('cx="10"', 'cx="nan"'), encoding="utf-8"
    )
    inventory = buildInventory(dataRoot, loadCatalog(configPath))
    assert inventory["summary"]["issueCounts"]["nonfiniteGeometry"] == 1
    shape = inventory["ownVideos"][0]["frames"][0]["annotations"][0]
    assert shape["coordinates"][0] == "nan"
    assert shape["canonicalClass"] == "ball"
    json.dumps(inventory, allow_nan=False)


def testImageIdsAreNotAssumedToBeVideoFrames(smallDataset):
    dataRoot, configPath, outputDir = smallDataset
    (dataRoot / "montage.xml").write_text(
        '<annotations><image id="0" name="frame1000.jpg" width="100" height="100"/></annotations>',
        encoding="utf-8",
    )
    inventory = buildInventory(dataRoot, loadCatalog(configPath))
    assert inventory["ownVideos"][1]["availability"] == "frameMappingRequired"
    assert inventory["ownVideos"][1]["frames"] == []
    assert inventory["summary"]["issueCounts"]["videoFrameMappingRequired"] == 1


def testUnreadableMediaIsNotReportedAsAvailable(smallDataset):
    dataRoot, configPath, outputDir = smallDataset
    (dataRoot / "old.mp4").unlink()
    inventory = buildInventory(dataRoot, loadCatalog(configPath))
    assert inventory["ownVideos"][0]["availability"] == "mediaUnavailable"
    assert inventory["summary"]["readableVideoCount"] == 1
    assert inventory["summary"]["errorCount"] == 1
    assert "useful reason" in fileError(OSError("useful reason"))


def testDuplicateAuthoritativeLimitsAreRejected(smallDataset):
    dataRoot, configPath, outputDir = smallDataset
    (dataRoot / "reviewed.yaml").write_text(
        "partidas:\n  Jogo1:\n    maximoFrames: 2\n    maximoFrames: 1000\n", encoding="utf-8"
    )
    result = buildInventory(dataRoot, loadCatalog(configPath))
    assert result["summary"]["issueCounts"]["reviewRegistryUnreadable"] == 1
    assert result["ownVideos"][0]["frames"] == []
