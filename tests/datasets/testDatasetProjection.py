import sys
from pathlib import Path

import pytest

from futsal.datasets.datasetFiles import readJson, writeJson
from futsal.datasets.datasetProjection import materializeProjection, verifyProjection
from futsal.datasets.datasetVerify import verifyDataset
from futsal.inventory.catalog import hashFile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import projectionTransfer


def testProjectionRecoversOnlyMissingDevelopmentRecordsAndMaterializes(projectionBundle, tmp_path):
    parent, projectionDir, _, _ = projectionBundle
    original = hashFile(parent / "contract.json")
    projection = verifyProjection(projectionDir, parent)
    supplement = [recordId for recordId, image in projection["imageSources"].items() if image["provider"] == "supplement"]
    assert supplement == ["video0:0"]
    assert not any(recordId.startswith("video2:") for recordId in projection["imageSources"])
    derived = tmp_path / "materialized"
    contract = materializeProjection(parent, projectionDir, derived, progress=lambda message: None)
    assert verifyDataset(derived, progress=lambda message: None)["contractId"] == contract["contractId"]
    assert contract["parentContractId"] == readJson(parent / "contract.json")["contractId"]
    assert contract["summary"]["detailedObjects"]["training"]["imageCount"] == 2
    assert contract["summary"]["courtPoints"]["training"]["imageCount"] == 1
    assert hashFile(parent / "contract.json") == original
    with pytest.raises(ValueError, match="new directory"):
        materializeProjection(parent, projectionDir, derived, progress=lambda message: None)


@pytest.mark.parametrize("mutation", ["corruption", "parent", "extra"])
def testProjectionRejectsChangedOrUnrelatedInputs(projectionBundle, mutation):
    parent, projectionDir, _, _ = projectionBundle
    if mutation == "corruption":
        label = next(projectionDir.glob("*/labels/train/*.txt"))
        label.write_text("changed")
    elif mutation == "extra":
        (projectionDir / "unexpected.txt").write_text("extra")
    else:
        contract = readJson(parent / "contract.json")
        contract["differentParent"] = True
        writeJson(parent / "contract.json", contract)
    with pytest.raises(ValueError):
        verifyProjection(projectionDir, parent)


def testProjectionTransportRoundTripUsesNoParentImageCopies(projectionBundle, tmp_path):
    parent, projectionDir, _, _ = projectionBundle
    projection = readJson(projectionDir / "projection.json")
    archive, receipt = tmp_path / "projection.tar.gz", tmp_path / "projection.receipt.json"
    assert projectionTransfer.main(["pack", "--projectionDir", str(projectionDir), "--projectionId", projection["projectionId"],
                                    "--archive", str(archive), "--receipt", str(receipt)]) == 0
    received = tmp_path / "received"
    assert projectionTransfer.main(["unpack", "--archive", str(archive), "--receipt", str(receipt), "--outputDir", str(received)]) == 0
    verifyProjection(received, parent)
    assert not list(received.glob("*/images/*/*.png"))
