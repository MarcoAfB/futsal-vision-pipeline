import copy
from pathlib import Path

from futsal.catalog import loadCatalog
from futsal.datasetSamples import classifySample, collectOwnSamples, collectPublicSamples
from futsal.inventory import buildInventory
from futsal.recipes import loadRecipe


def annotation(name, shape="box", findings=(), action="map"):
    return {
        "mappingAction": action, "canonicalClass": name,
        "semanticClass": None, "shapeType": shape,
        "coordinates": [10, 10] if shape == "points" else [1, 2, 3, 4],
        "geometryFindings": list(findings),
    }


def testQuarantineIsTaskSpecificAndKeepsAllSourceObservations():
    recipe = loadRecipe(Path("configs/curriculum.yaml"))
    generic, _, unified = recipe.data["tasks"]
    sample = {"origin": "own", "dimensions": [100, 100], "annotations": [
        annotation("ball"), annotation("penaltyMarkSideA", "points"),
        annotation("penaltyMarkSideA", "points"),
        annotation(None, findings=["geometryOutOfBounds"], action="ignore"),
    ]}
    before = copy.deepcopy(sample)
    assert classifySample(sample, generic, {"penaltyMarkSideA"})["eligible"]
    result = classifySample(sample, unified, {"penaltyMarkSideA"})
    assert result["reasons"] == ["duplicateLandmark:penaltyMarkSideA"]
    assert len(result["targets"]) == 3
    assert sample == before


def testInvalidBallQuarantinesTheImageRatherThanDroppingOneLabel():
    task = loadRecipe(Path("configs/curriculum.yaml")).data["tasks"][0]
    sample = {"origin": "own", "dimensions": [100, 100], "annotations": [
        annotation("ball", "ellipse", ["geometryOutOfBounds"]),
        annotation("outfieldPlayerTeamA"),
    ]}
    status = classifySample(sample, task, set())
    assert not status["eligible"]
    assert len(status["targets"]) == 2
    assert status["reasons"] == ["geometryOutOfBounds"]


def testLandmarkBoxesAreNotReinterpretedAsPoints():
    task = loadRecipe(Path("configs/curriculum.yaml")).data["tasks"][-1]
    sample = {"origin": "own", "dimensions": [100, 100],
              "annotations": [annotation("penaltyMarkSideA")]}
    assert classifySample(sample, task, {"penaltyMarkSideA"})["reasons"] == [
        "landmarkShapeMismatch"
    ]


def testReviewedOwnRecordsAndDetailedLabelsArePreserved(smallDataset):
    dataRoot, configPath, _ = smallDataset
    recipe = loadRecipe(Path("configs/curriculum.yaml"))
    catalog = loadCatalog(configPath)
    inventory = buildInventory(dataRoot, catalog)
    samples = collectOwnSamples(inventory, recipe)
    assert len(samples) == 4
    assert {sample["frameIndex"] for sample in samples} == {0, 1}
    assert len({sample["matchId"] for sample in samples}) == 1
    assert samples[0]["annotations"][0]["shapeType"] == "ellipse"
    assert samples[2]["tasks"]["genericObjects"]["eligible"]
    assert not samples[2]["tasks"]["unified25"]["eligible"]
    assert not samples[3]["tasks"]["genericObjects"]["eligible"]


def testPublicInvalidRefereeDoesNotExcludeACompatibleSample(tmp_path):
    recipe = loadRecipe(Path("configs/curriculum.yaml"))
    label = "0 0.5 0.5 0.1 0.1\n3 1.0 0.9 0.2 0.3\n"
    (tmp_path / "label.txt").write_text(label, encoding="utf-8")
    from futsal.catalog import hashFile

    report = {"sourceId": "futsalDetect", "samples": [{
        "recordId": "public:one", "imagePath": "image.jpg", "labelPath": "label.txt",
        "labelSha256": hashFile(tmp_path / "label.txt"), "media": {"width": 100, "height": 100},
    }]}
    samples = collectPublicSamples({"externalSources": [report]}, recipe, tmp_path)
    assert samples[0]["tasks"]["roleObjects"]["eligible"]
    assert len(samples[0]["tasks"]["roleObjects"]["targets"]) == 1
    (tmp_path / "label.txt").write_text("", encoding="utf-8")
    report["samples"][0]["labelSha256"] = hashFile(tmp_path / "label.txt")
    samples = collectPublicSamples({"externalSources": [report]}, recipe, tmp_path)
    assert samples[0]["tasks"]["roleObjects"]["reasons"] == ["emptyPublicLabel"]
