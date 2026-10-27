import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from futsal.catalog import hashFile, resolveSourcePath
from futsal.datasetFiles import hashJson, stagedDirectory, verifyEvidence, writeJson
from futsal.datasetPlan import loadDatasetPlan, renderPlanReport
from futsal.detectionGeometry import convertTargets, renderYoloLabels
from futsal.frameExtraction import readSelectedFrames
from futsal.reviewImages import renderReviewImage
from futsal.yoloInventory import scanFiles


def taskSplit(sample, task, assignments):
    if sample["origin"] == "public":
        return "training" if sample["sourceId"] in task["trainingSources"] else None
    split = assignments[sample["matchId"]]
    if split == "validation" or (split == "training" and "own" in task["trainingSources"]):
        return split
    return None


def exportSelections(plan, sample):
    return [(task, split) for task in plan["recipe"]["tasks"]
            if (split := taskSplit(sample, task, plan["frozenSplit"]["assignments"])) is not None]


def sampleToken(sample):
    return hashlib.sha256(sample["recordId"].encode("utf-8")).hexdigest()


def writeSample(staging, sample, task, split, imageBytes, suffix, pointBoxes):
    taskId, token = task["taskId"], sampleToken(sample)
    splitDirectory = "train" if split == "training" else "val"
    imagePath = f"{taskId}/images/{splitDirectory}/{token}{suffix}"
    labelPath = f"{taskId}/labels/{splitDirectory}/{token}.txt"
    targets = convertTargets(sample, task, pointBoxes)
    labelBytes = renderYoloLabels(targets).encode("utf-8")
    (staging / imagePath).write_bytes(imageBytes)
    (staging / labelPath).write_bytes(labelBytes)
    return {
        "recordId": sample["recordId"], "taskId": taskId, "split": split,
        "sourceId": sample["sourceId"], "origin": sample["origin"],
        "matchId": sample["matchId"], "frameIndex": sample["frameIndex"],
        "mediaPath": sample["mediaPath"], "annotationPath": sample["annotationPath"],
        "annotationSha256": sample["annotationSha256"], "dimensions": sample["dimensions"],
        "imagePath": imagePath, "labelPath": labelPath,
        "imageSha256": hashlib.sha256(imageBytes).hexdigest(),
        "labelSha256": hashlib.sha256(labelBytes).hexdigest(), "targets": targets,
    }


def encodeImage(image, extension, parameters):
    success, encoded = cv2.imencode(extension, image, parameters)
    if not success:
        raise OSError(f"Image encoding failed for {extension}.")
    return encoded.tobytes()


def exportImage(staging, plan, sample, image, originalBytes=None):
    if [image.shape[1], image.shape[0]] != sample["dimensions"]:
        raise ValueError(f"Decoded dimensions differ from annotations: {sample['recordId']}.")
    selections = exportSelections(plan, sample)
    eligible = [(task, split) for task, split in selections
                if sample["tasks"][task["taskId"]]["eligible"]]
    quarantined = {task["taskId"]: sample["tasks"][task["taskId"]]["reasons"]
                   for task, _ in selections if not sample["tasks"][task["taskId"]]["eligible"]}
    rows = []
    if eligible:
        suffix = Path(sample["mediaPath"]).suffix.lower() if originalBytes is not None else ".png"
        imageBytes = originalBytes if originalBytes is not None else encodeImage(
            image, ".png", [cv2.IMWRITE_PNG_COMPRESSION, 1]
        )
        rows = [writeSample(staging, sample, task, split, imageBytes, suffix, plan["recipe"]["pointBoxes"])
                for task, split in eligible]
    review = None
    if quarantined:
        previewPath = f"review/{sampleToken(sample)}.jpg"
        preview = renderReviewImage(image, sample)
        (staging / previewPath).write_bytes(encodeImage(preview, ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 90]))
        review = {"recordId": sample["recordId"], "quarantinedTasks": quarantined,
                  "imagePath": previewPath}
    return rows, review


def initializeTaskDirectories(staging, tasks):
    (staging / "review").mkdir()
    for task in tasks:
        root = staging / task["taskId"]
        for category in ("images", "labels"):
            for split in ("train", "val"):
                (root / category / split).mkdir(parents=True)
        config = {"train": "images/train", "val": "images/val",
                  "names": dict(enumerate(task["classes"]))}
        (root / "dataset.yaml").write_text(yaml.safe_dump(config, sort_keys=False),
                                           encoding="utf-8", newline="\n")


def exportImages(staging, plan, dataRoot, progress):
    rows, reviews = [], []
    videos = defaultdict(list)
    for sample in plan["samples"]:
        if not exportSelections(plan, sample):
            continue
        if sample["origin"] == "own":
            videos[sample["mediaPath"]].append(sample)
            continue
        path = resolveSourcePath(dataRoot, sample["mediaPath"])
        content = path.read_bytes()
        image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8),
                             cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            if any(sample["tasks"][task["taskId"]]["eligible"]
                   for task, _ in exportSelections(plan, sample)):
                raise OSError(f"Cannot decode eligible image: {sample['mediaPath']}.")
            reviews.append({"recordId": sample["recordId"], "imagePath": None,
                            "quarantinedTasks": {name: status["reasons"]
                                                 for name, status in sample["tasks"].items()}})
            continue
        exported, review = exportImage(staging, plan, sample, image, content)
        rows.extend(exported)
        if review:
            reviews.append(review)
    for videoPath, samples in sorted(videos.items()):
        progress(f"Extracting selected development frames: {videoPath}")
        byIndex = {sample["frameIndex"]: sample for sample in samples}
        for index, image in readSelectedFrames(resolveSourcePath(dataRoot, videoPath), byIndex):
            exported, review = exportImage(staging, plan, byIndex[index], image)
            rows.extend(exported)
            if review:
                reviews.append(review)
    rows.sort(key=lambda row: (row["taskId"], row["split"], row["recordId"]))
    reviews.sort(key=lambda row: row["recordId"])
    return rows, reviews


def validateExportSummary(plan, rows):
    expected = {(task["taskId"], sample["recordId"], split)
                for sample in plan["samples"] for task, split in exportSelections(plan, sample)
                if sample["tasks"][task["taskId"]]["eligible"]}
    observed = {(row["taskId"], row["recordId"], row["split"]) for row in rows}
    if expected != observed or len(observed) != len(rows):
        raise ValueError("Exported sample membership differs from planned eligible development data.")
    summary = {}
    for task in plan["recipe"]["tasks"]:
        summary[task["taskId"]] = {}
        for split in ("training", "validation"):
            selected = [row for row in rows if row["taskId"] == task["taskId"] and row["split"] == split]
            counts = Counter(target["classId"] for row in selected for target in row["targets"])
            allowedMissing = set(plan["recipe"]["unsupportedValidationClasses"]) if (
                split == "validation" and task["taskId"] == plan["recipe"]["coverageTask"]
            ) else set()
            missing = {name for index, name in enumerate(task["classes"]) if not counts[index]}
            if not selected or missing != allowedMissing:
                raise ValueError(f"Exported {task['taskId']} {split} has insufficient class support.")
            summary[task["taskId"]][split] = {
                "imageCount": len(selected),
                "classAnnotationCounts": {name: counts[index] for index, name in enumerate(task["classes"])},
                "truncatedPointBoxCount": sum(target["transformation"]["truncated"]
                                               for row in selected for target in row["targets"]),
            }
    return summary


def prepareDataset(planPath, dataRoot, outputDir, progress=print):
    dataRoot = Path(dataRoot).resolve()
    plan = loadDatasetPlan(planPath)
    with stagedDirectory(outputDir, dataRoot, [planPath]) as staging:
        progress("Checking source hashes before export...")
        verifyEvidence(dataRoot, plan["sourceEvidence"])
        initializeTaskDirectories(staging, plan["recipe"]["tasks"])
        rows, reviews = exportImages(staging, plan, dataRoot, progress)
        summary = validateExportSummary(plan, rows)
        writeJson(staging / "plan.json", plan)
        writeJson(staging / "manifest.json", {"schemaVersion": 1, "planId": plan["planId"],
                                               "samples": rows, "reviews": reviews, "summary": summary})
        lines = ["# Quarantined development samples", "",
                 "Annotation indices in previews refer to the original observations in plan.json.", ""]
        for review in reviews:
            link = f"[preview]({review['imagePath']})" if review["imagePath"] else "Preview unavailable"
            lines.append(f"- `{review['recordId']}`: {link}; {review['quarantinedTasks']}")
        (staging / "review.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        (staging / "report.md").write_text(renderPlanReport(plan), encoding="utf-8", newline="\n")
        progress("Checking source preservation after export...")
        verifyEvidence(dataRoot, plan["sourceEvidence"])
        artifacts = [{"path": path.relative_to(staging).as_posix(), "byteSize": path.stat().st_size,
                      "sha256": hashFile(path)} for path in scanFiles(staging)]
        contract = {"schemaVersion": 1, "kind": "preparedCurriculumDataset", "planId": plan["planId"],
                    "sourcePreservationVerified": True, "artifactEvidence": artifacts,
                    "summary": summary, "testExported": False}
        contract["contractId"] = hashJson(contract)
        writeJson(staging / "contract.json", contract)
    return contract
