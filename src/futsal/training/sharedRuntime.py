from pathlib import Path

import yaml

from futsal.datasets.datasetFiles import readJson, verifyEvidence, writeJson
from futsal.datasets.mediaStore import readBindings, resolveAsset
from futsal.inventory.catalog import hashFile


def prepareSharedRuntime(storeDir, runDir, selection, poseLabels=None):
    root = Path(runDir)
    evidence, assets = [], {}
    bindings = readBindings(storeDir)
    for taskId, task in selection["tasks"].items():
        directory = root / "runtimeData" / taskId
        directory.mkdir(parents=True, exist_ok=True)
        snapshot = {"classes": task["classes"]}
        for split in ("training", "validation"):
            snapshot[split] = []
            for row in task[split]:
                assetId = row["imageSha256"]
                path = resolveAsset(storeDir, assetId, bindings=bindings)
                item = {"kind": "sharedMediaAsset", "assetPath": str(path),
                        "sha256": assetId, "byteSize": path.stat().st_size}
                assets[assetId] = item
                snapshot[split].append({**row, "assetId": assetId, "resolvedImage": str(path)})
        writeJson(directory / "sharedSelection.json", snapshot)
        configuration = {"path": str(directory), "train": "images/train", "val": "images/val",
                         "names": dict(enumerate(task["classes"]))}
        files = [directory / "sharedSelection.json", directory / "dataset.yaml"]
        if poseLabels is not None:
            from futsal.training.courtPose import poseDatasetEntries

            configuration.update(poseDatasetEntries(task["classes"]))
            writeJson(directory / "poseLabels.json", {"landmarks": task["classes"], "records": poseLabels[taskId]})
            files.append(directory / "poseLabels.json")
        (directory / "dataset.yaml").write_text(yaml.safe_dump(configuration, sort_keys=False), encoding="utf-8")
        for path in files:
            evidence.append({"path": path.relative_to(root).as_posix(), "sha256": hashFile(path), "byteSize": path.stat().st_size})
    evidence.extend(assets.values())
    verifyEvidence(root, evidence)
    writeJson(root / "runtimeEvidence.json", evidence)
    writeJson(root / "selection.json", selection)
    return evidence


def sharedRuntimeSelection(runDir, taskId):
    path = Path(runDir) / "runtimeData" / taskId / "sharedSelection.json"
    return readJson(path) if path.is_file() else None


def buildRuntimeDataset(args, runDir, taskId, data, batch, mode, stride):
    snapshot = sharedRuntimeSelection(runDir, taskId)
    if snapshot is None and "kpt_shape" in data:
        raise ValueError("courtPose training and evaluation require a shared task view.")
    if snapshot is None:
        from ultralytics.data import build_yolo_dataset

        return build_yolo_dataset(args, data["train" if mode == "train" else "val"], batch,
                                  data, mode=mode, rect=mode == "val", stride=stride)
    from futsal.training.sharedDataset import SharedDetectionDataset, SharedPoseDataset

    if args.cache:
        raise ValueError("Shared native datasets require cache=false.")
    options = {"imgsz": args.imgsz, "batch_size": batch, "augment": mode == "train", "hyp": args,
               "rect": args.rect or mode == "val", "cache": False, "single_cls": False, "stride": stride,
               "pad": 0.0 if mode == "train" else 0.5, "data": data}
    rows = snapshot["training" if mode == "train" else "validation"]
    if "kpt_shape" in data:
        labels = readJson(Path(runDir) / "runtimeData" / taskId / "poseLabels.json")
        return SharedPoseDataset(rows, labels, task="pose", **options)
    return SharedDetectionDataset(rows, task="detect", **options)
