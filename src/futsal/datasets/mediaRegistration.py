from pathlib import Path

from futsal.datasets.datasetFiles import readJson, verifyEvidence
from futsal.datasets.datasetPlan import loadDatasetPlan
from futsal.datasets.mediaStore import (
    bindAsset,
    publishCollection,
    readBindings,
    storeCollections,
    storeLock,
    verifyIdentity,
)
from futsal.inventory.catalog import resolveSourcePath


def registerMedia(storeDir, sourceDir, progress=print):
    storeDir, sourceDir = Path(storeDir).resolve(), Path(sourceDir).resolve()
    if storeDir.is_relative_to(sourceDir) or sourceDir.is_relative_to(storeDir):
        raise ValueError("Media store and registered source must be separate directories.")
    rawRoot = Path(__file__).resolve().parents[3] / "data/raw"
    if storeDir.is_relative_to(rawRoot):
        raise ValueError("Media registry must be outside raw inputs.")
    progress(f"Verifying media provider without copying images: {sourceDir}")
    if (sourceDir / "projection.json").is_file():
        collection, paths = projectionCollection(storeDir, sourceDir)
    elif (sourceDir / "contract.json").is_file():
        contract = readJson(sourceDir / "contract.json")
        if contract.get("kind") == "preparedCurriculumDataset":
            collection, paths = preparedCollection(sourceDir)
        elif contract.get("kind") == "sharedTaskView":
            from futsal.datasets.mediaStore import verifyAssets
            from futsal.datasets.sharedViews import verifyView

            verifyView(sourceDir, storeDir)
            verifyAssets(storeDir, contract["assets"])
            rows = readJson(sourceDir / "manifest.json")["samples"]
            images = {row["recordId"]: {"assetId": row["imageSha256"],
                      "byteSize": contract["assets"][row["imageSha256"]]["byteSize"]} for row in rows}
            collection, paths = realCollection(loadDatasetPlan(sourceDir / "plan.json"), images, contract["contractId"]), {}
        else:
            collection, paths = syntheticCollection(sourceDir)
    elif (sourceDir / "plan.json").is_file():
        plan = loadDatasetPlan(sourceDir / "plan.json")
        collection, paths = realCollection(plan, {}, None), {}
    else:
        raise ValueError("Register a prepared bundle, projection, native plan or completed synthetic source.")
    with storeLock(storeDir) as root:
        bindings = readBindings(root)
        bindings["providers"] = sorted(set(bindings.get("providers", [])) | {str(sourceDir)})
        for assetId, path in paths.items():
            bindAsset(bindings, assetId, path)
        collection = publishCollection(root, collection, bindings)
    return {"collectionId": collection["collectionId"], "assetCount": len(collection["assets"]),
            "recordCount": len(collection["images"]), "copiedImageCount": 0}


def realCollection(plan, images, sourceContractId):
    return {"schemaVersion": 1, "kind": "sharedMediaCollection", "sourcePolicy": "realOnly",
            "purpose": "registeredSource",
            "sourceContractId": sourceContractId, "plan": plan, "images": images,
            "assets": {item["assetId"]: {"byteSize": item["byteSize"]} for item in images.values()}}


def preparedCollection(root):
    from futsal.datasets.datasetVerify import verifyDataset

    verifyDataset(root)
    plan = loadDatasetPlan(root / "plan.json")
    contract = readJson(root / "contract.json")
    rows = readJson(root / "manifest.json")["samples"]
    images, paths = {}, {}
    for row in rows:
        path = resolveSourcePath(root, row["imagePath"])
        item = {"assetId": row["imageSha256"], "byteSize": path.stat().st_size}
        if row["recordId"] in images and images[row["recordId"]] != item:
            raise ValueError("Task exports disagree about the image for a record.")
        images[row["recordId"]] = item
        paths.setdefault(item["assetId"], path)
    return realCollection(plan, images, contract["contractId"]), paths


def projectionCollection(storeDir, root):
    from futsal.datasets.datasetExport import validateExportSummary

    projection = readJson(root / "projection.json")
    verifyIdentity(projection, "projectionId")
    if projection.get("kind") != "preparedTaskProjection" or projection.get("sourcePolicy") != "realOnly":
        raise ValueError("Unsupported real projection.")
    verifyEvidence(root, projection["artifactEvidence"])
    parents = [item for item in storeCollections(storeDir)
               if item["sourceContractId"] == projection["parentContractId"]]
    if not parents:
        raise ValueError("Register the projection's verified parent bundle first.")
    plan = loadDatasetPlan(root / "plan.json")
    if plan["planId"] != projection["planId"] or validateExportSummary(plan, projection["samples"]) != projection["summary"]:
        raise ValueError("Projection plan and membership disagree.")
    images, paths = {}, {}
    for recordId, item in projection["imageSources"].items():
        assetId = item["sha256"]
        if item["provider"] == "parent":
            if not any(parent["images"].get(recordId, {}).get("assetId") == assetId for parent in parents):
                raise ValueError("Projection parent image reference changed.")
        elif item["provider"] == "supplement":
            paths[assetId] = resolveSourcePath(root, item["path"])
        else:
            raise ValueError("Unknown projection image provider.")
        images[recordId] = {"assetId": assetId, "byteSize": item["byteSize"]}
    return realCollection(plan, images, projection["projectionId"]), paths


def syntheticCollection(root):
    from futsal.datasets.syntheticCourt import readSyntheticManifest, verifySyntheticCourt

    contract = verifySyntheticCourt(root)
    rows = readSyntheticManifest(root)
    images, paths = {}, {}
    for row in rows:
        path = resolveSourcePath(root, row["imagePath"])
        images[row["recordId"]] = {"assetId": row["imageSha256"], "byteSize": path.stat().st_size}
        paths[row["imageSha256"]] = path
    return {"schemaVersion": 1, "kind": "sharedMediaCollection", "sourcePolicy": "syntheticCourt",
            "sourceContractId": contract["contractId"], "syntheticContract": contract,
            "samples": rows, "images": images,
            "assets": {item["assetId"]: {"byteSize": item["byteSize"]} for item in images.values()}}, paths
