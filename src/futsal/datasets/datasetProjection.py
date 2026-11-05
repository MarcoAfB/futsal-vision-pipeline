import shutil
from collections import defaultdict
from pathlib import Path

import cv2

from futsal.annotations.detectionGeometry import convertTargets, renderYoloLabels
from futsal.datasets.datasetExport import (
    encodeImage,
    exportSelections,
    initializeTaskDirectories,
    sampleToken,
    validateExportSummary,
)
from futsal.datasets.datasetFiles import (
    hashJson,
    readJson,
    stagedDirectory,
    verifyEvidence,
    writeJson,
)
from futsal.datasets.datasetPlan import loadDatasetPlan, planRecipe, renderPlanReport
from futsal.datasets.datasetVerify import verifyDataset, verifyReviews
from futsal.datasets.frameExtraction import readSelectedFrames
from futsal.datasets.projectionPlan import buildProjectionPlan
from futsal.datasets.recipes import activeTasks, loadRecipe
from futsal.datasets.reviewImages import renderReviewImage
from futsal.evaluation.evaluationData import validateResearchOutput
from futsal.inventory.catalog import hashFile, resolveSourcePath
from futsal.inventory.yoloInventory import scanFiles
from futsal.training.trainingData import requireDiskSpace


def artifactEvidence(root):
    return [{"path": path.relative_to(root).as_posix(), "byteSize": path.stat().st_size,
             "sha256": hashFile(path)} for path in scanFiles(root)]


def projectionRow(sample, task, split, image, pointBoxes):
    directory = "train" if split == "training" else "val"
    stem = f"{task['taskId']}/images/{directory}/{sampleToken(sample)}"
    targets = convertTargets(sample, task, pointBoxes)
    label = renderYoloLabels(targets).encode("utf-8")
    row = {field: sample[field] for field in ("recordId", "sourceId", "origin", "matchId", "frameIndex",
                                             "mediaPath", "annotationPath", "annotationSha256", "dimensions")}
    row.update(taskId=task["taskId"], split=split, imagePath=stem + ".png",
               labelPath=stem.replace("/images/", "/labels/") + ".txt",
               imageSha256=image["sha256"], labelSha256=hashBytes(label), targets=targets)
    return row, label


def hashBytes(value):
    import hashlib

    return hashlib.sha256(value).hexdigest()


def developmentSamples(plan):
    return [sample for sample in plan["samples"] if exportSelections(plan, sample)]


def parentImages(parentDir, manifest, requested):
    images = {}
    for row in manifest["samples"]:
        if row["recordId"] not in requested:
            continue
        previous = images.get(row["recordId"])
        if previous and previous["sha256"] != row["imageSha256"]:
            raise ValueError("Parent task projections disagree on original image bytes.")
        path = resolveSourcePath(parentDir, row["imagePath"])
        images[row["recordId"]] = {"provider": "parent", "path": row["imagePath"],
                                  "sha256": row["imageSha256"], "byteSize": path.stat().st_size}
    return images


def extractSupplement(samples, images, dataRoot, staging, sourceEvidence, progress):
    missing = [sample for sample in samples if sample["recordId"] not in images]
    paths = {sample[field] for sample in missing for field in ("mediaPath", "annotationPath")}
    evidence = [item for item in sourceEvidence if item["path"] in paths]
    if {item["path"] for item in evidence} != paths:
        raise ValueError("Supplement source evidence is incomplete.")
    verifyEvidence(dataRoot, evidence)
    byVideo = defaultdict(dict)
    for sample in missing:
        byVideo[sample["mediaPath"]][sample["frameIndex"]] = sample
    for video, byIndex in sorted(byVideo.items()):
        progress(f"Extracting missing reviewed development records: {video} ({len(byIndex)})")
        for frameIndex, image in readSelectedFrames(resolveSourcePath(dataRoot, video), byIndex):
            sample = byIndex[frameIndex]
            if [image.shape[1], image.shape[0]] != sample["dimensions"]:
                raise ValueError("Supplement image dimensions disagree with native annotations.")
            relative = f"supplement/{sampleToken(sample)}.png"
            target = staging / relative
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(encodeImage(image, ".png", [cv2.IMWRITE_PNG_COMPRESSION, 1]))
            images[sample["recordId"]] = {"provider": "supplement", "path": relative,
                "sha256": hashFile(target), "byteSize": target.stat().st_size}
    verifyEvidence(dataRoot, evidence)
    return evidence


def prepareProjection(sourceDatasetDir, configPath, dataRoot, outputDir, progress=print):
    parentDir, dataRoot = Path(sourceDatasetDir).resolve(), Path(dataRoot).resolve()
    outputDir = validateResearchOutput(outputDir, [parentDir, dataRoot], [Path(configPath)])
    verifyDataset(parentDir, progress=progress)
    parent = loadDatasetPlan(parentDir / "plan.json")
    plan = buildProjectionPlan(parent, loadRecipe(Path(configPath)))
    samples = developmentSamples(plan)
    images = parentImages(parentDir, readJson(parentDir / "manifest.json"), {s["recordId"] for s in samples})
    requireDiskSpace(outputDir, sum(s["dimensions"][0] * s["dimensions"][1] * 3 for s in samples if s["recordId"] not in images))
    with stagedDirectory(outputDir, parentDir, [dataRoot, configPath]) as staging:
        initializeTaskDirectories(staging, activeTasks(plan["recipe"]))
        evidence = extractSupplement(samples, images, dataRoot, staging, parent["sourceEvidence"], progress)
        rows, reviews = [], []
        for sample in samples:
            selected = exportSelections(plan, sample)
            quarantined = {task["taskId"]: sample["tasks"][task["taskId"]]["reasons"] for task, _ in selected
                           if not sample["tasks"][task["taskId"]]["eligible"]}
            if quarantined:
                imageSource = images[sample["recordId"]]
                source = resolveSourcePath(parentDir if imageSource["provider"] == "parent" else staging, imageSource["path"])
                image = cv2.imread(str(source))
                if image is None:
                    raise ValueError("Cannot decode a projected quarantine preview source.")
                relative = f"review/{sampleToken(sample)}.jpg"
                (staging / relative).write_bytes(encodeImage(renderReviewImage(image, sample), ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 90]))
                reviews.append({"recordId": sample["recordId"], "imagePath": relative, "quarantinedTasks": quarantined})
            for task, split in selected:
                if sample["tasks"][task["taskId"]]["eligible"]:
                    row, label = projectionRow(sample, task, split, images[sample["recordId"]], plan["recipe"]["pointBoxes"])
                    (staging / row["labelPath"]).write_bytes(label)
                    rows.append(row)
        rows.sort(key=lambda row: (row["taskId"], row["split"], row["recordId"]))
        writeJson(staging / "plan.json", plan)
        (staging / "report.md").write_text(renderPlanReport(plan), encoding="utf-8")
        (staging / "review.md").write_text("# Projected quarantine\n\nSee projection.json for native reasons and development-only previews.\n", encoding="utf-8")
        projection = {"schemaVersion": 1, "kind": "preparedTaskProjection", "sourcePolicy": "realOnly",
            "parentContractId": readJson(parentDir / "contract.json")["contractId"],
            "parentContractSha256": hashFile(parentDir / "contract.json"), "planId": plan["planId"],
            "imageSources": images, "samples": rows, "reviews": reviews,
            "summary": validateExportSummary(plan, rows), "supplementSourceEvidence": evidence,
            "artifactEvidence": artifactEvidence(staging)}
        projection["projectionId"] = hashJson(projection)
        writeJson(staging / "projection.json", projection)
        verifyProjection(staging, parentDir)
    return projection


def verifyProjection(root, parentDir):
    root, parentDir = Path(root).resolve(), Path(parentDir).resolve()
    projection = readJson(root / "projection.json")
    if projection.get("schemaVersion") != 1 or projection.get("kind") != "preparedTaskProjection" or projection.get("sourcePolicy") != "realOnly":
        raise ValueError("Unsupported real-only projection contract.")
    if projection["projectionId"] != hashJson({k: v for k, v in projection.items() if k != "projectionId"}):
        raise ValueError("Projection checksum mismatch.")
    if projection["parentContractSha256"] != hashFile(parentDir / "contract.json") or projection["parentContractId"] != readJson(parentDir / "contract.json")["contractId"]:
        raise ValueError("Projection belongs to a different parent dataset.")
    if {path.relative_to(root).as_posix() for path in scanFiles(root)} != {item["path"] for item in projection["artifactEvidence"]} | {"projection.json"}:
        raise ValueError("Projection has missing or unexpected files.")
    verifyEvidence(root, projection["artifactEvidence"])
    plan = loadDatasetPlan(root / "plan.json")
    parent = loadDatasetPlan(parentDir / "plan.json")
    if plan != buildProjectionPlan(parent, planRecipe(plan)) or plan["planId"] != projection["planId"]:
        raise ValueError("Projection changed native source data or frozen eligibility.")
    expectedIds = {s["recordId"] for s in developmentSamples(plan)}
    if set(projection["imageSources"]) != expectedIds:
        raise ValueError("Projection image membership includes missing, unrelated or reserved records.")
    available = parentImages(parentDir, readJson(parentDir / "manifest.json"), expectedIds)
    sourcePaths = {s[field] for s in plan["samples"] if s["recordId"] in expectedIds - set(available)
                   for field in ("mediaPath", "annotationPath")}
    if projection["supplementSourceEvidence"] != [item for item in parent["sourceEvidence"] if item["path"] in sourcePaths]:
        raise ValueError("Supplement video/annotation evidence differs from its original snapshot.")
    evidence = []
    for recordId, item in projection["imageSources"].items():
        if recordId in available:
            if item != available[recordId]:
                raise ValueError("Parent image identity differs from the verified record.")
        elif item["provider"] != "supplement" or item["path"] != f"supplement/{sampleToken({'recordId': recordId})}.png":
            raise ValueError("Missing image requires its exact supplemental record.")
        evidence.append({key: item[key] for key in ("path", "byteSize", "sha256")})
        verifyEvidence(parentDir if item["provider"] == "parent" else root, [evidence[-1]])
    samples = {s["recordId"]: s for s in plan["samples"]}
    tasks = {t["taskId"]: t for t in activeTasks(plan["recipe"])}
    for row in projection["samples"]:
        expected, label = projectionRow(samples[row["recordId"]], tasks[row["taskId"]], row["split"], projection["imageSources"][row["recordId"]], plan["recipe"]["pointBoxes"])
        if expected != row or resolveSourcePath(root, row["labelPath"]).read_bytes() != label:
            raise ValueError("Projection labels differ from native task annotations.")
    if validateExportSummary(plan, projection["samples"]) != projection["summary"]:
        raise ValueError("Projection class support changed.")
    allowed = {"plan.json", "report.md", "review.md"}
    allowed.update(f"{taskId}/dataset.yaml" for taskId in tasks)
    allowed.update(row["labelPath"] for row in projection["samples"])
    allowed.update(item["path"] for item in projection["imageSources"].values() if item["provider"] == "supplement")
    allowed.update(verifyReviews(projection, plan, samples))
    if allowed != {item["path"] for item in projection["artifactEvidence"]}:
        raise ValueError("Unrelated files violate the incremental development-only projection.")
    return projection


def materializeProjection(sourceDatasetDir, projectionDir, outputDir, progress=print):
    parentDir, root = Path(sourceDatasetDir).resolve(), Path(projectionDir).resolve()
    outputDir = validateResearchOutput(outputDir, [parentDir, root])
    verifyDataset(parentDir, progress=progress)
    projection = verifyProjection(root, parentDir)
    size = sum(projection["imageSources"][row["recordId"]]["byteSize"] for row in projection["samples"])
    requireDiskSpace(outputDir, size + sum(item["byteSize"] for item in projection["artifactEvidence"]))
    with stagedDirectory(outputDir, parentDir, [root]) as staging:
        for item in projection["artifactEvidence"]:
            if item["path"].startswith("supplement/"):
                continue
            target = resolveSourcePath(staging, item["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(resolveSourcePath(root, item["path"]), target)
        for row in projection["samples"]:
            item = projection["imageSources"][row["recordId"]]
            target = resolveSourcePath(staging, row["imagePath"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(resolveSourcePath(parentDir if item["provider"] == "parent" else root, item["path"]), target)
        manifest = {"schemaVersion": 1, "planId": projection["planId"], **{key: projection[key] for key in ("samples", "reviews", "summary")}}
        writeJson(staging / "manifest.json", manifest)
        contract = {"schemaVersion": 1, "kind": "preparedCurriculumDataset", "planId": projection["planId"],
            "sourcePreservationVerified": True, "testExported": False, "summary": projection["summary"],
            "parentContractId": projection["parentContractId"], "projectionId": projection["projectionId"],
            "artifactEvidence": artifactEvidence(staging)}
        contract["contractId"] = hashJson(contract)
        writeJson(staging / "contract.json", contract)
        verifyDataset(staging, progress=progress)
    return contract
