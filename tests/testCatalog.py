"""Exercise declaration validation and root-contained source resolution."""

import hashlib
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from futsal.catalog import hashFile, loadCatalog, resolveSourcePath


def fixtureDocuments() -> tuple[dict, dict]:
    classNames = [f"class{index}" for index in range(25)]
    labels = {
        "schemaVersion": 1,
        "canonicalClasses": [
            {"classId": index, "name": name, "geometry": "boundingBox"}
            for index, name in enumerate(classNames)
        ],
        "labelMaps": {
            "own": {
                "inputType": "labelName",
                "normalization": "stripCasefold",
                "entries": [{"sourceLabel": "Ball", "targetClass": "class0", "action": "map"}],
            },
            "public": {
                "inputType": "classId",
                "entries": [
                    {"sourceClassId": 0, "sourceLabel": "ball", "targetClass": "class0",
                     "action": "map"},
                    {"sourceClassId": 1, "sourceLabel": "player", "targetClass": None,
                     "semanticClass": "player", "action": "preserve", "reason": "Team unknown."},
                    {"sourceClassId": 2, "sourceLabel": "referee", "targetClass": None,
                     "semanticClass": "referee", "action": "ignore", "reason": "Outside target."},
                ],
            },
        },
    }
    data = {
        "schemaVersion": 1,
        "labelMapPath": "labelMap.yaml",
        "legacyReviewPath": "legacy/review.yaml",
        "sourceAccess": "readOnly",
        "expected": {"reviewedFrames": 20, "matchCount": 1, "videoCount": 2,
                     "reviewedIntervalCount": 2},
        "splitPolicy": {
            "selectionStatus": "unassigned", "groupBy": "matchId",
            "countingUnit": "reviewedFrameRecord",
            "targetFractions": {"training": 0.7, "validation": 0.15, "test": 0.15},
            "rareClassRequirements": [{"className": "class0", "requiredSplits": ["training"],
                                       "unsupportedSplits": ["validation"], "reason": "Rare."}],
        },
        "ownVideos": [
            {
                "videoId": f"video{index}", "videoPath": f"videos/video{index}.mp4",
                "annotationPath": f"annotations/video{index}.xml", "labelMapId": "own",
                "legacyMatchId": None, "annotationScope": "completeForDeclaredClasses",
                "classesAnnotated": classNames, "annotationScopeEvidence": "researcherDeclaration",
                "reviewedIntervals": [{
                    "intervalId": f"interval{index}", "matchId": "sameMatch",
                    "startFrame": 0, "endFrame": 9, "internalCutsStatus": "pending",
                    "provenance": {"matchIdentityBasis": "researcherDeclaration",
                                   "reviewedRangeBasis": "researcherDeclaration"},
                }],
            }
            for index in range(2)
        ],
        "publicSources": [{
            "sourceId": "publicSource", "sourceName": "Public source", "format": "yolo",
            "required": True, "imageRoot": "public", "labelRoot": "public",
            "labelMapId": "public", "annotationScope": "sourceOntologyOnly",
            "allowedUses": ["training"], "imageExtensions": [".jpg", ".png"],
            "annotationIgnorePatterns": ["README*.txt"], "metadataFiles": ["public/README.txt"],
            "labelMapEvidence": "legacy/labels.yaml",
        }],
        "syntheticSources": [],
    }
    return data, labels


def writeCatalog(tmpPath: Path, data: dict, labels: dict) -> Path:
    configPath = tmpPath / "dataSources.yaml"
    configPath.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    (tmpPath / "labelMap.yaml").write_text(yaml.safe_dump(labels, sort_keys=False), encoding="utf-8")
    return configPath


def replaceValue(document: dict, path: tuple, value: object) -> None:
    current = document
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value


def testLoadsDeclarationsWithoutSourceFilesAndPreservesSharedMatch(tmp_path):
    data, labels = fixtureDocuments()
    configPath = writeCatalog(tmp_path, data, labels)
    catalog = loadCatalog(configPath)
    assert catalog.data == data
    assert catalog.labels == labels
    assert catalog.data["splitPolicy"]["selectionStatus"] == "unassigned"
    assert {video["reviewedIntervals"][0]["matchId"] for video in data["ownVideos"]} == {"sameMatch"}
    for fingerprint in catalog.configHashes:
        content = (tmp_path / fingerprint["path"]).read_bytes()
        assert fingerprint["sha256"] == hashlib.sha256(content).hexdigest()
        assert fingerprint["byteSize"] == len(content)
        assert Path(fingerprint["path"]).name == fingerprint["path"]


@pytest.mark.parametrize(("path", "value"), [
    (("schemaVersion",), True),
    (("schemaVersion",), 2),
    (("expected", "reviewedFrames"), False),
    (("sourceAccess",), "readWrite"),
    (("ownVideos", 0, "videoId"), "video1"),
    (("ownVideos", 0, "videoPath"), "../outside.mp4"),
    (("ownVideos", 0, "reviewedIntervals", 0, "intervalId"), "interval1"),
    (("ownVideos", 0, "reviewedIntervals", 0, "startFrame"), True),
    (("ownVideos", 0, "reviewedIntervals", 0, "startFrame"), -1),
    (("ownVideos", 0, "reviewedIntervals", 0, "startFrame"), 1.5),
    (("ownVideos", 0, "reviewedIntervals", 0, "startFrame"), 10),
    (("ownVideos", 0, "reviewedIntervals", 0, "endFrame"), -1),
    (("ownVideos", 0, "reviewedIntervals", 0, "endFrame"), "9"),
    (("ownVideos", 0, "labelMapId"), "public"),
    (("ownVideos", 0, "classesAnnotated"), ["missingClass"]),
    (("publicSources", 0, "labelMapId"), "own"),
    (("publicSources", 0, "required"), "true"),
    (("publicSources", 0, "allowedUses"), ["test"]),
    (("publicSources", 0, "imageExtensions"), ["jpg"]),
    (("publicSources", 0, "metadataFiles"), ["C:\\secret.txt"]),
    (("publicSources", 0, "labelMapEvidence"), "../../other.yaml"),
    (("splitPolicy", "targetFractions", "test"), float("nan")),
    (("splitPolicy", "targetFractions", "test"), True),
    (("splitPolicy", "targetFractions", "test"), 0.5),
    (("splitPolicy", "rareClassRequirements", 0, "unsupportedSplits"), ["training"]),
])
def testRejectsInvalidSourceDeclarations(tmp_path, path, value):
    data, labels = fixtureDocuments()
    replaceValue(data, path, value)
    with pytest.raises(ValueError):
        loadCatalog(writeCatalog(tmp_path, data, labels))


@pytest.mark.parametrize(("path", "value"), [
    (("canonicalClasses", 0, "classId"), True),
    (("canonicalClasses", 0, "classId"), 25),
    (("canonicalClasses", 0, "classId"), 1),
    (("canonicalClasses", 0, "name"), " CLASS1 "),
    (("canonicalClasses", 0, "geometry"), "polygon"),
    (("labelMaps", "own", "normalization"), "none"),
    (("labelMaps", "own", "entries", 0, "action"), "discard"),
    (("labelMaps", "own", "entries", 0, "targetClass"), "unknown"),
    (("labelMaps", "public", "entries", 1, "sourceClassId"), True),
    (("labelMaps", "public", "entries", 1, "sourceClassId"), 0),
    (("labelMaps", "public", "entries", 1, "sourceClassId"), -1),
    (("labelMaps", "public", "entries", 1, "sourceLabel"), " BALL "),
    (("labelMaps", "public", "entries", 1, "targetClass"), "class0"),
    (("labelMaps", "public", "entries", 2, "semanticClass"), ""),
])
def testRejectsAmbiguousLabelMappings(tmp_path, path, value):
    data, labels = fixtureDocuments()
    replaceValue(labels, path, value)
    with pytest.raises(ValueError):
        loadCatalog(writeCatalog(tmp_path, data, labels))


def testRejectsNormalizedOwnLabels(tmp_path):
    data, labels = fixtureDocuments()
    duplicate = deepcopy(labels["labelMaps"]["own"]["entries"][0])
    duplicate["sourceLabel"] = " ball "
    labels["labelMaps"]["own"]["entries"].append(duplicate)
    with pytest.raises(ValueError, match="normalized label"):
        loadCatalog(writeCatalog(tmp_path, data, labels))


@pytest.mark.parametrize("startFrame", [0, 9])
def testRejectsOverlappingInclusiveIntervals(tmp_path, startFrame):
    data, labels = fixtureDocuments()
    interval = deepcopy(data["ownVideos"][0]["reviewedIntervals"][0])
    interval.update(intervalId="additional", startFrame=startFrame, endFrame=15)
    data["ownVideos"][0]["reviewedIntervals"].append(interval)
    with pytest.raises(ValueError, match="overlap"):
        loadCatalog(writeCatalog(tmp_path, data, labels))


def testAcceptsAdjacentRangesAndConfirmedSharedMatch(tmp_path):
    data, labels = fixtureDocuments()
    interval = deepcopy(data["ownVideos"][0]["reviewedIntervals"][0])
    interval.update(intervalId="additional", startFrame=10, endFrame=10)
    interval["provenance"]["sameMatchAsVideoId"] = "video1"
    data["ownVideos"][0]["reviewedIntervals"].insert(0, interval)
    catalog = loadCatalog(writeCatalog(tmp_path, data, labels))
    assert catalog.data["ownVideos"][0]["reviewedIntervals"][0]["startFrame"] == 10


def testRejectsIncorrectSharedMatchReference(tmp_path):
    data, labels = fixtureDocuments()
    data["ownVideos"][0]["reviewedIntervals"][0]["provenance"]["sameMatchAsVideoId"] = "unknown"
    with pytest.raises(ValueError, match="sameMatchAsVideoId"):
        loadCatalog(writeCatalog(tmp_path, data, labels))


@pytest.mark.parametrize("content", [
    "schemaVersion: 1\nschemaVersion: 1\n",
    "schemaVersion: 1\ntrue: value\n",
    "schemaVersion: [\n",
    "[]",
    "!!python/object/apply:os.system ['echo unsafe']",
])
def testRejectsMalformedOrUnsafeYaml(tmp_path, content):
    configPath = tmp_path / "invalid.yaml"
    configPath.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        loadCatalog(configPath)


def testRejectsUnknownAndMissingFields(tmp_path):
    data, labels = fixtureDocuments()
    data["ownVideos"][0]["videoPaths"] = data["ownVideos"][0].pop("videoPath")
    with pytest.raises(ValueError, match="unknown keys"):
        loadCatalog(writeCatalog(tmp_path, data, labels))


def testRejectsLabelMapOutsideConfigurationDirectory(tmp_path):
    data, labels = fixtureDocuments()
    data["labelMapPath"] = "../labelMap.yaml"
    with pytest.raises(ValueError, match="escapes"):
        loadCatalog(writeCatalog(tmp_path, data, labels))


@pytest.mark.parametrize("relativePath", [
    "/outside", "../outside", "nested/../../outside", "C:/outside", "C:\\outside",
    "C:outside", "\\\\server\\share\\file", "\\outside", "file.txt:stream", "", "\x00",
])
def testRejectsCrossPlatformPathEscapes(tmp_path, relativePath):
    with pytest.raises(ValueError):
        resolveSourcePath(tmp_path, relativePath)


def testResolvesMissingRelativeSourceWithoutCreatingIt(tmp_path):
    result = resolveSourcePath(tmp_path, "nested\\video.mp4")
    assert result == tmp_path / "nested" / "video.mp4"
    assert not result.exists()


def testRejectsSymlinkEscape(tmp_path):
    rootPath = tmp_path / "root"
    outsidePath = tmp_path / "outside"
    rootPath.mkdir()
    outsidePath.mkdir()
    try:
        (rootPath / "linked").symlink_to(outsidePath, target_is_directory=True)
    except OSError:
        pytest.skip("Creating symbolic links requires an OS permission unavailable here.")
    with pytest.raises(ValueError, match="symbolic link"):
        resolveSourcePath(rootPath, "linked/notCreated.txt")


def testHashesMultipleChunksWithoutChangingFile(tmp_path):
    path = tmp_path / "content.bin"
    content = bytes(range(256)) * 9000
    path.write_bytes(content)
    initialTime = path.stat().st_mtime_ns
    assert hashFile(path) == hashlib.sha256(content).hexdigest()
    assert path.stat().st_mtime_ns == initialTime
    assert path.read_bytes() == content
