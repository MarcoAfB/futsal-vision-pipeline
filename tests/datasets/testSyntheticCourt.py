import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from futsal.annotations.detectionGeometry import renderYoloLabels
from futsal.datasets.datasetFiles import hashJson, readJson, writeJson
from futsal.datasets.syntheticCourt import (
    derivedTargets,
    generateSyntheticCourt,
    readSyntheticManifest,
    verifySyntheticCourt,
)
from futsal.inventory.catalog import hashFile


def testGeneratedSourcePreservesNativePointsAndExcludesDebug(syntheticSource):
    contract = verifySyntheticCourt(syntheticSource)
    rows = readSyntheticManifest(syntheticSource)
    assert contract["trainingOnly"] is True and len(rows) == 1
    row = rows[0]
    assert 4 <= len(row["targets"]) <= 11
    assert len(row["targets"]) == len({t["classId"] for t in row["targets"]})
    assert len((syntheticSource / row["labelPath"]).read_text().splitlines()[0].split()) == 5
    image = cv2.imread(str(syntheticSource / row["imagePath"]))
    debug = cv2.imread(str(syntheticSource / "debug/syntheticCourt000000.jpg"))
    assert image.shape == (360, 640, 3)
    assert np.mean(cv2.absdiff(image, debug)) > 0.05
    assert all("debug" not in row[key] for key in ("imagePath", "labelPath"))
    assert row["targets"] == derivedTargets(row["metadata"]["visiblePoints"], row["dimensions"])


def testGenerationIsDeterministicAndRefusesOverwrite(syntheticSource, tmp_path):
    generated = tmp_path / "same"
    contract = generateSyntheticCourt(
        generated,
        count=1,
        seed=2026,
        width=640,
        height=360,
        textureScale=24,
        debugCount=1,
        progress=lambda message: None,
    )
    assert contract == readJson(syntheticSource / "contract.json")
    assert (generated / "manifest.jsonl").read_bytes() == (
        syntheticSource / "manifest.jsonl"
    ).read_bytes()
    with pytest.raises(ValueError, match="new directory"):
        generateSyntheticCourt(generated, count=1)


def testSyntheticVerificationRejectsCorruptionAndSemanticTampering(syntheticSource, tmp_path):
    root = tmp_path / "source"
    shutil.copytree(syntheticSource, root)
    row = readSyntheticManifest(root)[0]
    (root / row["imagePath"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="changed"):
        verifySyntheticCourt(root)
    contract = readJson(root / "contract.json")
    contract["trainingOnly"] = False
    contract["contractId"] = hashJson({k: v for k, v in contract.items() if k != "contractId"})
    writeJson(root / "contract.json", contract)
    with pytest.raises(ValueError, match="training-only"):
        verifySyntheticCourt(root)


def testNativePointAtImageEdgeClipsOnlyDerivedBox():
    points = [{"classId": 0, "imagePixels": [0.0, 0.0]}]
    targets = derivedTargets(points, [1280, 720])
    assert points[0]["imagePixels"] == [0.0, 0.0]
    assert targets[0]["transformation"]["truncated"]
    assert len(renderYoloLabels(targets).split()) == 5


def testSyntheticArchiveRoundTrip(syntheticSource, tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from syntheticTransfer import packSynthetic, validateSyntheticTransfer
    from transferArchive import unpackArchive

    archive, receipt = tmp_path / "source.tar.gz", tmp_path / "receipt.json"
    before = hashFile(syntheticSource / "contract.json")
    packed = packSynthetic(syntheticSource, archive, receipt)
    received = tmp_path / "received"
    unpackArchive(archive, receipt, received, "syntheticCourtTransfer", validateSyntheticTransfer)
    assert verifySyntheticCourt(received)["contractId"] == packed["metadata"]["contractId"]
    assert hashFile(syntheticSource / "contract.json") == before
