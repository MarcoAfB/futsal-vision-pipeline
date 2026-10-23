import hashlib

import cv2
import numpy as np
import pytest

from futsal import yoloInventory
from futsal.yoloInventory import inventoryYoloSource


def prepareSource(dataRoot, sourceFormat="yolo"):
    (dataRoot / "images").mkdir()
    (dataRoot / "labels").mkdir()
    (dataRoot / "evidence.yaml").write_text(
        "sourceLabels: [ball, player, referee]", encoding="utf-8"
    )
    source = {
        "sourceId": "example",
        "sourceName": "Example",
        "format": sourceFormat,
        "required": True,
        "imageRoot": "images",
        "labelRoot": "labels",
        "labelMapId": "exampleLabels",
        "annotationScope": "sourceOntologyOnly",
        "allowedUses": ["training"],
        "imageExtensions": [".jpg", ".png"],
        "annotationIgnorePatterns": ["README*.txt"],
        "metadataFiles": [],
        "labelMapEvidence": "evidence.yaml",
    }
    labelMap = {
        "inputType": "classId",
        "entries": [
            {"sourceClassId": 0, "sourceLabel": "ball", "targetClass": "ball", "action": "map"},
            {
                "sourceClassId": 1,
                "sourceLabel": "player",
                "targetClass": None,
                "semanticClass": "player",
                "action": "preserve",
            },
            {
                "sourceClassId": 2,
                "sourceLabel": "referee",
                "targetClass": None,
                "semanticClass": "referee",
                "action": "ignore",
            },
        ],
    }
    return source, labelMap


def writeImage(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix, np.zeros((12, 20, 3), dtype=np.uint8))
    assert success
    path.write_bytes(encoded.tobytes())


def issueCodes(issues):
    return {issue["code"] for issue in issues}


def testPreservesOriginalClassesWithoutInventingTeamsOrRewritingSources(tmp_path):
    source, labelMap = prepareSource(tmp_path)
    writeImage(tmp_path / "images" / "one.png")
    (tmp_path / "labels" / "one.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n1 0.2 0.2 0.1 0.1\n2 0.8 0.8 0.1 0.1\n", encoding="utf-8"
    )
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    report, issues = inventoryYoloSource(source, labelMap, tmp_path)

    assert not issues
    assert report["availability"] == "available"
    assert report["sourceClassCounts"] == {"0": 1, "1": 1, "2": 1}
    assert report["canonicalClassCounts"] == {"ball": 1}
    assert report["samples"][0]["recordId"] == "example:images/one.png"
    assert report["samples"][0]["media"]["width"] == 20
    assert report["samples"][0]["media"]["height"] == 12
    expectedHash = hashlib.sha256((tmp_path / "labels" / "one.txt").read_bytes()).hexdigest()
    assert report["samples"][0]["labelSha256"] == expectedHash
    assert report["annotationFiles"][0]["sha256"] == expectedHash
    assert (
        report["labelMapEvidence"]["sha256"]
        == hashlib.sha256((tmp_path / "evidence.yaml").read_bytes()).hexdigest()
    )
    assert before == {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }


def testOrphansMissingLabelsEmptyLabelsAndCollisionsRetainProvenance(tmp_path):
    source, labelMap = prepareSource(tmp_path)
    for name in ("empty.png", "missing.png", "duplicate.png", "duplicate.jpg"):
        writeImage(tmp_path / "images" / name)
    (tmp_path / "labels" / "empty.txt").write_text(" \n", encoding="utf-8")
    for name in ("orphan.txt", "duplicate.txt"):
        (tmp_path / "labels" / name).write_text("0 0.5 0.5 0.2 0.2", encoding="utf-8")
    (tmp_path / "labels" / "README.dataset.txt").write_text("Not a label", encoding="utf-8")
    (tmp_path / "images" / "Dataset.zip").write_bytes(b"not an image or label")

    report, issues = inventoryYoloSource(source, labelMap, tmp_path)

    assert issueCodes(issues) == {
        "orphanLabel",
        "missingLabel",
        "emptyLabels",
        "ambiguousImageStem",
    }
    assert (report["imageCount"], report["annotationFileCount"], report["emptyLabelCount"]) == (
        4,
        3,
        1,
    )
    assert report["sourceClassCounts"] == {"0": 2}
    assert len(report["annotationFiles"]) == 3
    assert all(label["sha256"] for label in report["annotationFiles"])
    samples = {sample["imagePath"]: sample for sample in report["samples"]}
    assert samples["images/empty.png"]["labelPath"] == "labels/empty.txt"
    assert samples["images/empty.png"]["canonicalClassCounts"] == {}
    assert samples["images/duplicate.png"]["labelPath"] is None
    assert samples["images/duplicate.jpg"]["labelPath"] is None
    assert samples["images/missing.png"]["labelPath"] is None


def testRelativePathsDisambiguateRepeatedBasenamesAndKeepDeterministicRecords(tmp_path):
    source, labelMap = prepareSource(tmp_path)
    for directory in ("b", "a"):
        writeImage(tmp_path / "images" / directory / "same.png")
        labelDirectory = tmp_path / "labels" / directory
        labelDirectory.mkdir()
        (labelDirectory / "same.txt").write_text("0 0.5 0.5 0.2 0.2", encoding="utf-8")

    report, issues = inventoryYoloSource(source, labelMap, tmp_path)

    assert not issues
    assert [sample["recordId"] for sample in report["samples"]] == [
        "example:images/a/same.png",
        "example:images/b/same.png",
    ]
    assert [sample["labelPath"] for sample in report["samples"]] == [
        "labels/a/same.txt",
        "labels/b/same.txt",
    ]


def testMetadataReadmeCountMismatchIsReportedWithoutBecomingAnnotation(tmp_path):
    source, labelMap = prepareSource(tmp_path)
    writeImage(tmp_path / "images" / "sample.png")
    (tmp_path / "labels" / "sample.txt").write_text("0 0.5 0.5 0.2 0.2", encoding="utf-8")
    readmePath = tmp_path / "labels" / "README.dataset.txt"
    readmePath.write_text("The dataset includes 973 images.\n", encoding="utf-8")
    source["metadataFiles"] = ["labels/README.dataset.txt"]

    report, issues = inventoryYoloSource(source, labelMap, tmp_path)

    assert issueCodes(issues) == {"documentedImageCountMismatch"}
    assert issues[0]["documentedImageCount"] == 973
    assert issues[0]["observedImageCount"] == 1
    assert report["annotationFileCount"] == 1
    assert (
        report["metadataFiles"][0]["sha256"] == hashlib.sha256(readmePath.read_bytes()).hexdigest()
    )
    assert report["availability"] == "available"


@pytest.mark.parametrize("required, severity", [(False, "warning"), (True, "error")])
@pytest.mark.parametrize("failure", ["missing", "permission"])
def testUnavailableSourcesAreNeverReportedAsSuccessfulEmptyData(
    tmp_path,
    monkeypatch,
    required,
    severity,
    failure,
):
    source, labelMap = prepareSource(tmp_path)
    source["required"] = required
    if failure == "missing":
        source["imageRoot"] = "missing"
    else:

        def denyScan(path):
            raise PermissionError("synthetic access denial")

        monkeypatch.setattr(yoloInventory, "scanFiles", denyScan)

    report, issues = inventoryYoloSource(source, labelMap, tmp_path)

    assert report["availability"] == "unavailable"
    assert report["samples"] == []
    assert issueCodes(issues) == {"sourceUnavailable"}
    assert issues[0]["severity"] == severity
    assert report["labelMapEvidence"]["sha256"]


def testCorruptImageAndUnreadableAnnotationAreReportedWithoutLosingFileHashes(tmp_path):
    source, labelMap = prepareSource(tmp_path)
    (tmp_path / "images" / "broken.png").write_bytes(b"invalid image")
    (tmp_path / "labels" / "broken.txt").write_bytes(b"\xff\xff")

    report, issues = inventoryYoloSource(source, labelMap, tmp_path)

    assert issueCodes(issues) == {"annotationUnreadable", "imageUnreadable"}
    assert report["samples"][0]["media"] is None
    assert report["samples"][0]["labelSha256"] == hashlib.sha256(b"\xff\xff").hexdigest()
    assert report["emptyLabelCount"] == 0
    assert report["annotationFiles"][0]["empty"] is None


def testFlatSharedImageAndLabelRootIsSupported(tmp_path):
    source, labelMap = prepareSource(tmp_path)
    source["labelRoot"] = "images"
    writeImage(tmp_path / "images" / "sample.png")
    (tmp_path / "images" / "sample.txt").write_text("0 0.5 0.5 0.2 0.2", encoding="utf-8")

    report, issues = inventoryYoloSource(source, labelMap, tmp_path)

    assert not issues
    assert report["samples"][0]["labelPath"] == "images/sample.txt"
    assert report["imageCount"] == report["annotationFileCount"] == 1


def testEmptyRequiredSourceHasExplicitFailure(tmp_path):
    source, labelMap = prepareSource(tmp_path)

    report, issues = inventoryYoloSource(source, labelMap, tmp_path)

    assert report["availability"] == "empty"
    assert issueCodes(issues) == {"noSourceImages"}
    assert issues[0]["severity"] == "error"


def testLandmarkClassesKeepTheirOriginalYoloBoxes(tmp_path):
    source, labelMap = prepareSource(tmp_path)
    labelMap["entries"] = [
        {
            "sourceClassId": 0,
            "sourceLabel": "corner",
            "targetClass": "upperCornerSideA",
            "action": "map",
        }
    ]
    writeImage(tmp_path / "images" / "corner.png")
    (tmp_path / "labels" / "corner.txt").write_text("0 0.5 0.5 0.01 0.02", encoding="utf-8")
    report, issues = inventoryYoloSource(source, labelMap, tmp_path)
    assert not issues
    assert report["canonicalClassCounts"] == {"upperCornerSideA": 1}
    assert report["format"] == "yolo"
