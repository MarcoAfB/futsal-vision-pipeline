import io
import sys
import tarfile
from pathlib import Path

import pytest

from futsal.datasets.datasetVerify import verifyDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import datasetTransfer
import transferArchive


@pytest.fixture
def transferFixture(trainingBundle, tmp_path):
    bundle, _, _ = trainingBundle
    result = verifyDataset(bundle, progress=lambda message: None)
    verification = tmp_path / "verification.json"
    transferArchive.writeJson(verification, {"schemaVersion": 1, "kind": "datasetVerification", "status": "passed",
        "contractSha256": transferArchive.hashFile(bundle / "contract.json"), "verification": result})
    archive, receipt = tmp_path / "dataset.tar.gz", tmp_path / "dataset.receipt.json"
    return bundle, archive, receipt, result["contractId"], verification


def testDatasetRoundTripPreservesExactContractAndRefusesOverwrite(transferFixture, tmp_path):
    bundle, archive, receiptPath, contractId, verification = transferFixture
    receipt = datasetTransfer.packDataset(bundle, archive, receiptPath, contractId, verification)
    assert receipt["metadata"]["sourcePolicy"] == "realOnly"
    destination = tmp_path / "received"
    transferArchive.unpackArchive(archive, receiptPath, destination, "curriculumDatasetTransfer", datasetTransfer.validateDataset)
    assert verifyDataset(destination, progress=lambda message: None)["contractId"] == contractId
    assert (destination / "contract.json").read_bytes() == (bundle / "contract.json").read_bytes()
    with pytest.raises(ValueError, match="must be new"):
        transferArchive.unpackArchive(archive, receiptPath, destination, "curriculumDatasetTransfer", datasetTransfer.validateDataset)
    archive.write_bytes(archive.read_bytes()[:-1])
    with pytest.raises(ValueError, match="mismatch"):
        transferArchive.checkArchive(archive, receipt)


@pytest.mark.parametrize("mutation", ["missing", "extra", "lowSpace", "unverified", "changed"])
def testPackingRejectsInvalidInputsBeforePublication(transferFixture, monkeypatch, mutation):
    bundle, archive, receipt, contractId, verification = transferFixture
    if mutation == "missing":
        (bundle / "report.md").unlink()
    elif mutation == "extra":
        (bundle / "unexpected.txt").write_text("extra")
    elif mutation == "lowSpace":
        monkeypatch.setattr(transferArchive.shutil, "disk_usage", lambda path: type("Disk", (), {"free": 0})())
    elif mutation == "unverified":
        verification.write_text("{}")
    else:
        path = bundle / "report.md"
        content = path.read_bytes()
        path.write_bytes(bytes([content[0] ^ 1]) + content[1:])
    with pytest.raises(ValueError):
        datasetTransfer.packDataset(bundle, archive, receipt, contractId, verification)
    assert not archive.exists() and not receipt.exists()


@pytest.mark.parametrize("name,kind", [("../escape", "file"), ("/absolute", "file"),
                                      ("C:/drive", "file"), ("a\\b", "file"),
                                      ("safe.txt", "symlink"), ("safe.txt", "hardlink"),
                                      ("safe.txt", "duplicate"), ("extra.txt", "file"),
                                      ("safe.txt", "oversize"), ("safe.txt", "missing")])
def testExtractionRejectsUnsafeOrInconsistentMembers(tmp_path, name, kind):
    archive, receiptPath = tmp_path / "unsafe.tar", tmp_path / "receipt.json"
    with tarfile.open(archive, "w") as output:
        if kind != "missing":
            info = tarfile.TarInfo(name)
            info.size = 2 if kind == "oversize" else 1
            if kind in {"symlink", "hardlink"}:
                info.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
                info.linkname = "../escape"
                info.size = 0
            output.addfile(info, io.BytesIO(b"xx"))
            if kind == "duplicate":
                output.addfile(info, io.BytesIO(b"x"))
    receipt = {"schemaVersion": 1, "toolVersion": 1, "kind": "fixture", "archiveBytes": archive.stat().st_size,
        "archiveSha256": transferArchive.hashFile(archive), "fileCount": 1, "uncompressedBytes": 1,
        "files": [{"path": "safe.txt", "byteSize": 1, "sha256": transferArchive.hashlib.sha256(b"x").hexdigest()}]}
    receipt["receiptId"] = transferArchive.hashJson(receipt)
    transferArchive.writeJson(receiptPath, receipt)
    destination = tmp_path / "received"
    with pytest.raises(ValueError):
        transferArchive.unpackArchive(archive, receiptPath, destination, "fixture", lambda *args: None)
    assert not destination.exists() and not (tmp_path / "escape").exists()


def testCaseCollisionsAndFileDirectoryConflictsAreRejected():
    evidence = {"byteSize": 0, "sha256": "0" * 64}
    with pytest.raises(ValueError, match="Case-colliding"):
        transferArchive.validateFiles([{"path": "A/file", **evidence}, {"path": "a/other", **evidence}])
    with pytest.raises(ValueError, match="collide"):
        transferArchive.validateFiles([{"path": "a", **evidence}, {"path": "a/file", **evidence}])


def testDatasetCliProducesTheRequiredSemanticReceipt(trainingBundle, tmp_path):
    from futsal.cli import main

    bundle, _, _ = trainingBundle
    receipt = tmp_path / "verification.json"
    assert main(["verifyDataset", "--datasetDir", str(bundle), "--receipt", str(receipt)]) == 0
    assert transferArchive.readJson(receipt)["status"] == "passed"
    with pytest.raises(SystemExit):
        main(["verifyDataset", "--datasetDir", str(bundle), "--receipt", str(receipt)])
