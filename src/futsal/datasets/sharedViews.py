import copy
from pathlib import Path

from futsal.annotations.detectionGeometry import convertTargets, renderYoloLabels
from futsal.datasets.datasetExport import exportSelections, validateExportSummary
from futsal.datasets.datasetFiles import (
    hashJson,
    readJson,
    stagedDirectory,
    verifyEvidence,
    writeJson,
)
from futsal.datasets.datasetPlan import loadDatasetPlan
from futsal.datasets.mediaStore import (
    protectMediaOutput,
    readCollection,
    resolveAsset,
    storeCollections,
    verifyAssets,
    verifyIdentity,
)
from futsal.datasets.projectionPlan import buildProjectionPlan
from futsal.datasets.recipes import Recipe, loadRecipe
from futsal.datasets.sourceTransition import scientificCatalog
from futsal.inventory.catalog import Catalog, hashFile
from futsal.inventory.yoloInventory import scanFiles


def viewPlan(parent, recipe):
    if recipe.catalog.labels != parent["catalog"]["labels"] or (
        scientificCatalog(recipe.catalog.data) != scientificCatalog(parent["catalog"]["data"])
    ):
        raise ValueError("Shared view requires the same scientific source catalogue.")
    if recipe.data == parent["recipe"]:
        return copy.deepcopy(parent)
    adapted = Recipe(recipe.data, Catalog(parent["catalog"]["data"], recipe.catalog.labels, []), recipe.evidence)
    return buildProjectionPlan(parent, adapted)


def prepareView(storeDir, configPath, outputDir, *, collectionId=None, dataRoot=None, progress=print):
    protectMediaOutput(outputDir, storeDir)
    collections = [item for item in storeCollections(storeDir) if item["sourcePolicy"] == "realOnly"]
    if not collections:
        raise ValueError("Register a real native plan or prepared bundle before preparing views.")
    templates = [item for item in collections if item.get("purpose") != "extractedAssets"]
    parent = readCollection(storeDir, collectionId) if collectionId else max(templates, key=lambda item: (
        "parentPlanId" not in item["plan"], len(item["plan"]["samples"])))
    if parent["sourcePolicy"] != "realOnly":
        raise ValueError("Task views require real native annotations.")
    plan = viewPlan(parent["plan"], loadRecipe(Path(configPath)))
    images = {}
    native = {sample["recordId"]: sample for sample in plan["samples"]}
    for collection in collections:
        samples = {sample["recordId"]: sample for sample in collection["plan"]["samples"]}
        for recordId, item in collection["images"].items():
            if recordId not in native:
                continue
            if any(native[recordId][key] != samples[recordId][key] for key in
                   ("annotations", "annotationSha256", "dimensions", "matchId", "frameIndex", "sourceId")):
                raise ValueError("Registered collections disagree on native sample identity.")
            if recordId in images and images[recordId] != item:
                raise ValueError("Registered collections disagree on image bytes.")
            images[recordId] = item
    needed = {sample["recordId"] for sample in plan["samples"] for task, _ in exportSelections(plan, sample)
              if sample["tasks"][task["taskId"]]["eligible"]}
    missing = needed - images.keys()
    if missing:
        if dataRoot is None:
            raise ValueError(f"Missing {len(missing)} development images; register a supplement or supply --dataRoot.")
        from futsal.datasets.sharedExtraction import extractSharedImages

        images.update(extractSharedImages(storeDir, plan, missing, dataRoot, progress))
    assets = {images[recordId]["assetId"]: {"byteSize": images[recordId]["byteSize"]} for recordId in needed}
    progress(f"Verifying {len(assets)} shared assets; writing task metadata only...")
    verifyAssets(storeDir, assets)
    with stagedDirectory(outputDir, storeDir, [configPath]) as staging:
        rows = []
        for sample in plan["samples"]:
            for task, split in exportSelections(plan, sample):
                if not sample["tasks"][task["taskId"]]["eligible"]:
                    continue
                assetId = images[sample["recordId"]]["assetId"]
                targets = convertTargets(sample, task, plan["recipe"]["pointBoxes"])
                label = renderYoloLabels(targets).encode("utf-8")
                import hashlib

                token = hashlib.sha256(sample["recordId"].encode()).hexdigest()
                labelPath = f"{task['taskId']}/labels/{split}/{token}.txt"
                path = staging / labelPath
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(label)
                row = {key: sample[key] for key in ("recordId", "sourceId", "origin", "matchId", "frameIndex",
                        "mediaPath", "annotationPath", "annotationSha256", "dimensions")}
                rows.append({**row, "taskId": task["taskId"], "split": split, "assetId": assetId,
                             "imagePath": f"media/{assetId}", "imageSha256": assetId,
                             "labelPath": labelPath, "labelSha256": hashlib.sha256(label).hexdigest(), "targets": targets})
        rows.sort(key=lambda row: (row["taskId"], row["split"], row["recordId"]))
        trainAssets = {row["assetId"] for row in rows if row["split"] == "training"}
        validationAssets = {row["assetId"] for row in rows if row["split"] == "validation"}
        if trainAssets & validationAssets:
            raise ValueError("Identical image bytes cross the frozen training/validation split.")
        summary = validateExportSummary(plan, rows)
        writeJson(staging / "plan.json", plan)
        writeJson(staging / "manifest.json", {"schemaVersion": 2, "planId": plan["planId"], "samples": rows, "summary": summary})
        contract = {"schemaVersion": 2, "kind": "sharedTaskView", "planId": plan["planId"],
                    "sourcePreservationVerified": True, "testExported": False,
                    "sourcePolicy": "realOnly", "sourceCollectionId": parent["collectionId"],
                    "parentContractId": parent["sourceContractId"], "assets": assets, "summary": summary,
                    "artifactEvidence": [{"path": path.relative_to(staging).as_posix(),
                        "byteSize": path.stat().st_size, "sha256": hashFile(path)} for path in scanFiles(staging)]}
        contract["contractId"] = hashJson(contract)
        writeJson(staging / "contract.json", contract)
        verifyView(staging, storeDir)
    return contract


def verifyView(datasetDir, storeDir=None):
    root = Path(datasetDir).resolve()
    contract = readJson(root / "contract.json")
    verifyIdentity(contract, "contractId")
    if contract.get("schemaVersion") != 2 or contract.get("kind") != "sharedTaskView" or (
        contract.get("testExported") is not False or contract.get("sourcePolicy") != "realOnly"
        or contract.get("sourcePreservationVerified") is not True
    ):
        raise ValueError("Unsupported shared development view.")
    expected = {item["path"] for item in contract["artifactEvidence"]} | {"contract.json"}
    if {path.relative_to(root).as_posix() for path in scanFiles(root)} != expected:
        raise ValueError("Shared view file membership changed.")
    verifyEvidence(root, contract["artifactEvidence"])
    plan = loadDatasetPlan(root / "plan.json")
    manifest = readJson(root / "manifest.json")
    rows = manifest["samples"]
    if plan["planId"] != contract["planId"] or manifest["planId"] != plan["planId"]:
        raise ValueError("Shared view plan identity mismatch.")
    summary = validateExportSummary(plan, rows)
    if summary != manifest["summary"] or summary != contract["summary"]:
        raise ValueError("Shared view class support changed.")
    samples = {sample["recordId"]: sample for sample in plan["samples"]}
    tasks = {task["taskId"]: task for task in plan["recipe"]["tasks"]}
    assets = set()
    for row in rows:
        sample = samples[row["recordId"]]
        for key in ("sourceId", "origin", "matchId", "frameIndex", "mediaPath", "annotationPath", "annotationSha256", "dimensions"):
            if row[key] != sample[key]:
                raise ValueError("Shared view native provenance changed.")
        targets = convertTargets(sample, tasks[row["taskId"]], plan["recipe"]["pointBoxes"])
        if row["targets"] != targets or row["assetId"] != row["imageSha256"] or row["imagePath"] != f"media/{row['assetId']}":
            raise ValueError("Shared view targets or media identity changed.")
        from futsal.inventory.catalog import resolveSourcePath

        path = resolveSourcePath(root, row["labelPath"])
        if path.read_text(encoding="utf-8") != renderYoloLabels(targets) or hashFile(path) != row["labelSha256"]:
            raise ValueError("Shared view label differs from native targets.")
        assets.add(row["assetId"])
    if assets != contract["assets"].keys():
        raise ValueError("Shared view asset membership mismatch.")
    if storeDir is not None:
        verifyAssets(storeDir, contract["assets"])
    return {"contractId": contract["contractId"], "planId": plan["planId"], "exportedImageCount": len(rows),
            "reservedTestRecordCount": sum(plan["frozenSplit"]["assignments"].get(s["matchId"]) == "test" for s in plan["samples"]),
            "verifiedArtifactCount": len(expected), "summary": summary}


def resolveRowImage(datasetDir, row, storeDir=None):
    if row["imagePath"].startswith("media/"):
        if storeDir is None:
            raise ValueError("Shared task views require --mediaStoreDir.")
        return resolveAsset(storeDir, row["imageSha256"])
    from futsal.inventory.catalog import resolveSourcePath

    return resolveSourcePath(Path(datasetDir), row["imagePath"])
