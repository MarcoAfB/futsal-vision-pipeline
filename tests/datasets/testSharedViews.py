import copy
from pathlib import Path

import pytest
import yaml

from futsal.datasets.datasetFiles import readJson, verifyEvidence
from futsal.datasets.datasetVerify import verifyDataset
from futsal.datasets.mediaRegistration import registerMedia
from futsal.datasets.mediaStore import readBindings, rebindMedia, resolveAsset, storeCollections
from futsal.datasets.sharedViews import prepareView
from futsal.inventory.catalog import hashFile
from futsal.training.sharedRuntime import prepareSharedRuntime
from futsal.training.trainingData import selectTrainingData
from futsal.training.trainingRun import verifyBundleUnchanged


def testSharedViewsReuseParentAndSupplementWithoutImageCopies(projectionBundle, tmp_path):
    parent, supplement, config, _ = projectionBundle
    store, view = tmp_path / "store", tmp_path / "view"
    before = {path: hashFile(path) for root in (parent, supplement) for path in root.rglob("*") if path.is_file()}
    receipt = registerMedia(store, parent)
    assert registerMedia(store, parent) == receipt
    registerMedia(store, supplement)
    contract = prepareView(store, config, view)
    assert contract["summary"]["detailedObjects"]["training"]["imageCount"] == 2
    assert contract["summary"]["courtPoints"]["training"]["imageCount"] == 1
    assert not list(view.rglob("*.png")) and not list(store.rglob("*.png"))
    assert before == {path: hashFile(path) for path in before}
    verifyDataset(view, mediaStoreDir=store)
    assert len(storeCollections(store)) == 2
    for index in range(2):
        run = tmp_path / f"run{index}"
        run.mkdir()
        config = yaml.safe_load(Path("configs/training/courtPoints.yaml").read_text())
        selection = selectTrainingData(view, config)
        evidence = prepareSharedRuntime(store, run, selection)
        assert not list(run.rglob("*.png")) and not list(run.rglob("*.jpg"))
        verifyEvidence(run, evidence)
        snapshot = readJson(run / "runtimeData/courtPoints/sharedSelection.json")
        assert Path(snapshot["training"][0]["resolvedImage"]).is_file()


def testMissingSupplementCanBeExtractedOnceWithoutTouchingTest(projectionBundle, tmp_path):
    parent, _, config, raw = projectionBundle
    store = tmp_path / "store"
    registerMedia(store, parent)
    with pytest.raises(ValueError, match="Missing 1 development"):
        prepareView(store, config, tmp_path / "missing")
    first = prepareView(store, config, tmp_path / "first", dataRoot=raw)
    assets = list((store / "assets").glob("*.png"))
    assert len(assets) == 1
    second = prepareView(store, config, tmp_path / "second", dataRoot=raw)
    assert list((store / "assets").glob("*.png")) == assets
    assert first["summary"] == second["summary"]


def testAllOwnTaskMappingsUseNativeEligibility(projectionBundle, tmp_path):
    parent, supplement, config, _ = projectionBundle
    store = tmp_path / "store"
    registerMedia(store, parent)
    registerMedia(store, supplement)
    recipe = yaml.safe_load(Path("configs/sharedDetection.yaml").read_text())
    recipe.pop("splitPath")
    path = config.with_name("allTasks.yaml")
    path.write_text(yaml.safe_dump(recipe))
    contract = prepareView(store, path, tmp_path / "allTasks")
    summary = contract["summary"]
    assert set(summary) == {"detailedObjects", "genericObjects", "playerOnly", "ballOnly", "courtPoints"}
    assert summary["playerOnly"]["training"]["classAnnotationCounts"] == {"player": 8}
    assert summary["ballOnly"]["training"]["classAnnotationCounts"] == {"ball": 2}
    assert not list((tmp_path / "allTasks").rglob("*.png"))


def testPinnedViewSurvivesNewCollectionAndDetectsChangedAssets(projectionBundle, syntheticSource, tmp_path):
    parent, supplement, config, _ = projectionBundle
    store, view = tmp_path / "store", tmp_path / "view"
    registerMedia(store, parent)
    registerMedia(store, supplement)
    contract = prepareView(store, config, view)
    digest = hashFile(view / "contract.json")
    registerMedia(store, syntheticSource)
    verifyBundleUnchanged(view, contract, digest, mediaStoreDir=store)
    assetId = next(iter(contract["assets"]))
    original = resolveAsset(store, assetId)
    replacement = tmp_path / "relocated.png"
    replacement.write_bytes(original.read_bytes())
    rebindMedia(store, assetId, tmp_path, replacement.name)
    verifyDataset(view, mediaStoreDir=store)
    assert hashFile(view / "contract.json") == digest
    replacement.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="asset changed"):
        verifyDataset(view, mediaStoreDir=store)


def testLogicalRecordsAreNotCollapsedByIdenticalImageBytes(trainingBundle, preparedSources, tmp_path):
    bundle, _, config = trainingBundle
    store = tmp_path / "store"
    registerMedia(store, bundle)
    prepareView(store, preparedSources[1], tmp_path / "view")
    selection = selectTrainingData(tmp_path / "view", config)
    task = selection["tasks"]["genericObjects"]
    duplicate = copy.deepcopy(task["training"][0])
    duplicate["recordId"] += ":distinctLogicalRecord"
    task["training"].append(duplicate)
    run = tmp_path / "run"
    run.mkdir()
    prepareSharedRuntime(store, run, selection)
    rows = readJson(run / "runtimeData/genericObjects/sharedSelection.json")["training"]
    assert len(rows) == 2 and rows[0]["recordId"] != rows[1]["recordId"]
    assert rows[0]["resolvedImage"] == rows[1]["resolvedImage"]
    assert readBindings(store)["assets"]
