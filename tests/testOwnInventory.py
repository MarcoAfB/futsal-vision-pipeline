from futsal.catalog import hashFile, loadCatalog
from futsal.ownInventory import inventoryOwnVideo, readReviewRegistry


def testOwnVideoUsesOnlyAuthoritativeReviewedFrames(smallDataset):
    dataRoot, configPath, outputDir = smallDataset
    catalog = loadCatalog(configPath)
    issues = []
    registry, evidence = readReviewRegistry(catalog, dataRoot, issues)
    report = inventoryOwnVideo(catalog.data["ownVideos"][0], catalog, dataRoot, registry, issues)
    assert not issues
    assert evidence["sha256"] == hashFile(dataRoot / "reviewed.yaml")
    assert [frame["frameIndex"] for frame in report["frames"]] == [0, 1]
    assert report["frames"][0]["annotations"][0]["shapeType"] == "ellipse"
    assert report["frames"][1]["annotations"] == []
    assert report["activeShapeCountOutsideReviewedFrames"] == 2
    assert report["outsideShapeCountInFile"] == 1
    assert not outputDir.exists()
