import sys
from pathlib import Path

from futsal.datasets.datasetFiles import readJson, writeJson
from futsal.datasets.mediaRegistration import registerMedia
from futsal.datasets.sharedCli import mediaInventory
from futsal.datasets.sharedViews import prepareView

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import sharedDataTransfer


def testIncrementalTransferSendsOnlyMissingAssetsAndKeepsViewIdentity(projectionBundle, tmp_path):
    parent, supplement, config, _ = projectionBundle
    sourceStore, receiver = tmp_path / "sourceStore", tmp_path / "receiver"
    registerMedia(sourceStore, parent)
    registerMedia(sourceStore, supplement)
    registerMedia(receiver, parent)
    view = tmp_path / "view"
    contract = prepareView(sourceStore, config, view)
    inventory = tmp_path / "inventory.json"
    writeJson(inventory, mediaInventory(receiver))
    archive, receipt = tmp_path / "view.tar.gz", tmp_path / "receipt.json"
    result = sharedDataTransfer.packView(view, sourceStore, inventory, archive, receipt)
    assert len(result["metadata"]["transferred"]) == 1
    received = tmp_path / "received"
    sharedDataTransfer.receiveView(receiver, archive, receipt, received)
    assert readJson(received / "view/contract.json") == contract
    assert len(list((received / "assets").iterdir())) == 1
    assert not list((received / "view").rglob("*.png"))
    secondInventory = tmp_path / "secondInventory.json"
    writeJson(secondInventory, mediaInventory(receiver))
    second = sharedDataTransfer.packView(view, sourceStore, secondInventory,
                                         tmp_path / "second.tar.gz", tmp_path / "second.json")
    assert second["metadata"]["transferred"] == {}


def testTransferRefusesStaleReceiverInventory(projectionBundle, tmp_path):
    import pytest

    parent, supplement, config, _ = projectionBundle
    store = tmp_path / "store"
    registerMedia(store, parent)
    registerMedia(store, supplement)
    view = tmp_path / "view"
    prepareView(store, config, view)
    inventory = tmp_path / "inventory.json"
    writeJson(inventory, mediaInventory(store))
    archive, receipt = tmp_path / "view.tar.gz", tmp_path / "receipt.json"
    sharedDataTransfer.packView(view, store, inventory, archive, receipt)
    with pytest.raises(ValueError, match="Missing media asset binding"):
        sharedDataTransfer.receiveView(tmp_path / "emptyReceiver", archive, receipt, tmp_path / "received")
