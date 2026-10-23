from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from futsal.catalog import hashFile, loadCatalog, resolveSourcePath


def fixtureDocuments():
    configRoot = Path(__file__).resolve().parents[1] / "configs"
    data = yaml.safe_load((configRoot / "dataSources.yaml").read_text(encoding="utf-8"))
    labels = yaml.safe_load((configRoot / "labelMap.yaml").read_text(encoding="utf-8"))
    return data, labels


def writeCatalog(tmpPath, data, labels):
    configPath = tmpPath / "dataSources.yaml"
    configPath.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    labelPath = tmpPath / data["labelMapPath"]
    labelPath.parent.mkdir(parents=True, exist_ok=True)
    labelPath.write_text(yaml.safe_dump(labels, sort_keys=False), encoding="utf-8")
    return configPath


def testLoadsDeclarationsAndPreservesSharedMatchWithoutOpeningSources(tmp_path):
    data, labels = fixtureDocuments()
    data["labelMapPath"] = "mappings/labels.yaml"
    catalog = loadCatalog(writeCatalog(tmp_path, data, labels))
    assert catalog.data == data
    assert catalog.labels == labels
    assert catalog.configHashes[1]["path"] == "mappings/labels.yaml"
    for entry in catalog.configHashes:
        assert entry["sha256"] == hashFile(tmp_path / entry["path"])
    shared = catalog.data["ownVideos"][-1]["reviewedIntervals"][3]
    assert shared["matchId"] == "match07"
    assert catalog.data["splitPolicy"]["selectionStatus"] == "unassigned"


@pytest.mark.parametrize("start,end", [(-1, 2), (3, 2), (0.5, 2), (True, 2)])
def testRejectsInvalidReviewedRanges(tmp_path, start, end):
    data, labels = fixtureDocuments()
    data["ownVideos"][0]["reviewedIntervals"][0].update(startFrame=start, endFrame=end)
    with pytest.raises(ValueError, match="integer startFrame"):
        loadCatalog(writeCatalog(tmp_path, data, labels))


def testRejectsOverlappingInclusiveIntervals(tmp_path):
    data, labels = fixtureDocuments()
    intervals = data["ownVideos"][0]["reviewedIntervals"]
    overlap = deepcopy(intervals[0])
    overlap.update(intervalId="overlap", startFrame=199, endFrame=200)
    intervals.append(overlap)
    with pytest.raises(ValueError, match="overlap"):
        loadCatalog(writeCatalog(tmp_path, data, labels))


@pytest.mark.parametrize("problem", ["duplicateLabel", "unknownTarget", "genericTeam", "wrongMap"])
def testRejectsAmbiguousSourceMappings(tmp_path, problem):
    data, labels = fixtureDocuments()
    if problem == "duplicateLabel":
        entry = deepcopy(labels["labelMaps"]["ownCvat"]["entries"][0])
        entry["sourceLabel"] = " BOLA "
        labels["labelMaps"]["ownCvat"]["entries"].append(entry)
    elif problem == "unknownTarget":
        labels["labelMaps"]["ownCvat"]["entries"][0]["targetClass"] = "unknown"
    elif problem == "genericTeam":
        labels["labelMaps"]["publicBallPlayer"]["entries"][1]["targetClass"] = "outfieldPlayerTeamA"
    else:
        data["ownVideos"][0]["labelMapId"] = "publicBallPlayer"
    with pytest.raises(ValueError):
        loadCatalog(writeCatalog(tmp_path, data, labels))


def testRequiresTrainingOnlyPublicSourcesAndConsistentMatchReferences(tmp_path):
    data, labels = fixtureDocuments()
    data["publicSources"][0]["allowedUses"] = ["test"]
    with pytest.raises(ValueError, match="training-only"):
        loadCatalog(writeCatalog(tmp_path, data, labels))
    data, labels = fixtureDocuments()
    data["ownVideos"][-1]["reviewedIntervals"][3]["matchId"] = "differentMatch"
    with pytest.raises(ValueError, match="same original match"):
        loadCatalog(writeCatalog(tmp_path, data, labels))


@pytest.mark.parametrize("relativePath", ["../outside.txt", "/outside.txt", "C:\\outside.txt"])
def testRejectsPathsOutsideSourceRoot(tmp_path, relativePath):
    with pytest.raises(ValueError, match="relative"):
        resolveSourcePath(tmp_path, relativePath)


@pytest.mark.parametrize("content", ["schemaVersion: [", "schemaVersion: 1\nschemaVersion: 2"])
def testRejectsMalformedAndDuplicateYamlKeys(tmp_path, content):
    configPath = tmp_path / "invalid.yaml"
    configPath.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        loadCatalog(configPath)
