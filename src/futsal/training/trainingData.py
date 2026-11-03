import hashlib
import shutil
from collections import Counter
from pathlib import Path

import yaml

from futsal.datasets.datasetFiles import readJson, verifyEvidence, writeJson
from futsal.datasets.recipes import activeTasks
from futsal.inventory.catalog import hashFile, resolveSourcePath


def classSupport(rows, names):
    counts = Counter(target["classId"] for row in rows for target in row["targets"])
    return {name: counts[index] for index, name in enumerate(names)}


def selectRows(rows, count, seed, namespace, requiredClasses=None):
    if len(rows) < count:
        raise ValueError(f"{namespace} has {len(rows)} eligible records, fewer than requested {count}.")
    def orderKey(row):
        return hashlib.sha256(f"{seed}:{namespace}:{row['recordId']}".encode()).hexdigest()

    remaining = sorted(rows, key=orderKey)
    uncovered = (set(requiredClasses) if requiredClasses is not None else
                 {target["classId"] for row in rows for target in row["targets"]})
    chosen = []
    while uncovered:
        row = max(remaining, key=lambda row: len(uncovered & {
            target["classId"] for target in row["targets"]}))
        covered = uncovered & {target["classId"] for target in row["targets"]}
        if not covered or len(chosen) >= count:
            raise ValueError(f"{namespace} cannot preserve class coverage within {count} records.")
        chosen.append(row)
        remaining.remove(row)
        uncovered -= covered
    return chosen + remaining[:count - len(chosen)]


def selectTrainingData(datasetDir, config):
    root = Path(datasetDir)
    plan, manifest = readJson(root / "plan.json"), readJson(root / "manifest.json")
    taskIds = [stage["taskId"] for stage in config["stages"]]
    tasks = [task for task in activeTasks(plan["recipe"]) if task["taskId"] in taskIds]
    if config.get("schemaVersion") == 2:
        return selectFullData(plan, manifest, tasks, config)
    rows = {task["taskId"]: {split: [row for row in manifest["samples"]
            if row["taskId"] == task["taskId"] and row["split"] == split]
            for split in ("training", "validation")} for task in tasks}
    sharedIds = set.intersection(*({row["recordId"] for row in rows[taskId]["validation"]}
                                  for taskId in taskIds))
    basis = max(tasks, key=lambda task: len(task["classes"]))
    basisRows = rows[basis["taskId"]]["validation"]
    required = {target["classId"] for row in basisRows for target in row["targets"]}
    selected = selectRows([row for row in basisRows if row["recordId"] in sharedIds],
                          config["validationSampleCount"], config["seed"], "validationShared",
                          required)
    validationIds = [row["recordId"] for row in selected]
    selection = {"schemaVersion": 1, "seed": config["seed"], "validationRecordIds": validationIds,
                 "selectionMethod": "greedyClassCoverageThenSha256", "tasks": {}}
    for task in tasks:
        taskId = task["taskId"]
        byId = {row["recordId"]: row for row in rows[taskId]["validation"]}
        train = selectRows(rows[taskId]["training"], config["trainSampleCount"], config["seed"],
                           f"{taskId}:training")
        if any(value == 0 for value in classSupport(train, task["classes"]).values()):
            raise ValueError(f"Training subset must support every declared class: {taskId}.")
        val = [byId[recordId] for recordId in validationIds]
        fullSupport = classSupport(rows[taskId]["validation"], task["classes"])
        valSupport = classSupport(val, task["classes"])
        if any(count and not valSupport[name] for name, count in fullSupport.items()):
            raise ValueError(f"Shared validation subset lost class support: {taskId}.")
        selection["tasks"][taskId] = {"classes": task["classes"], "training": train,
            "validation": val, "trainingSupport": classSupport(train, task["classes"]),
            "validationSupport": valSupport}
    return selection


def verifyRealSources(plan, manifest, tasks):
    catalog = plan["catalog"]["data"]
    publicIds = {source["sourceId"] for source in catalog["publicSources"]}
    ownIds = {video["videoId"] for video in catalog["ownVideos"]}
    syntheticIds = {source["sourceId"] for source in catalog["syntheticSources"]}
    byRecord = {sample["recordId"]: sample for sample in plan["samples"]}
    byTask = {task["taskId"]: task for task in tasks}
    for task in tasks:
        declared = set(task["trainingSources"])
        if not declared <= publicIds | {"own"} or declared & syntheticIds:
            raise ValueError("realOnly rejects unregistered or synthetic task sources.")
    counts = Counter()
    for row in manifest["samples"]:
        if row["taskId"] not in byTask:
            continue
        native = byRecord.get(row["recordId"], {})
        if any(native.get(key) != row.get(key) for key in ("origin", "sourceId", "mediaPath", "annotationSha256")):
            raise ValueError("realOnly requires matching native source provenance.")
        origin, sourceId = row["origin"], row["sourceId"]
        if sourceId in syntheticIds or not ((origin == "own" and sourceId in ownIds) or
                                            (origin == "public" and sourceId in publicIds)):
            raise ValueError("realOnly rejects synthetic or unregistered sample provenance.")
        allowed = byTask[row["taskId"]]["trainingSources"]
        if row["split"] not in {"training", "validation"} or (row["split"] == "validation" and origin != "own"):
            raise ValueError("Scientific selection permits real development splits only.")
        if row["split"] == "training" and ("own" if origin == "own" else sourceId) not in allowed:
            raise ValueError("Training sample source is not declared by its task.")
        counts[sourceId] += 1
    return {"policy": "realOnly", "provenanceVerified": True, "exportedRowsBySource": dict(counts)}


def selectFullData(plan, manifest, tasks, config):
    policy = verifyRealSources(plan, manifest, tasks)
    selection = {"schemaVersion": 2, "seed": config["seed"], "sourcePolicy": policy,
                 "selectionMethod": "fullPerTask", "tasks": {}}
    for task in tasks:
        taskId, names = task["taskId"], task["classes"]
        splits = {split: sorted((row for row in manifest["samples"] if row["taskId"] == taskId
                                  and row["split"] == split), key=lambda row: row["recordId"])
                  for split in ("training", "validation")}
        if not splits["validation"] or not splits["training"] or not all(classSupport(splits["training"], names).values()):
            raise ValueError(f"Full data require nonempty splits and all training classes: {taskId}.")
        selection["tasks"][taskId] = {"classes": names, **splits,
            "trainingSupport": classSupport(splits["training"], names),
            "validationSupport": classSupport(splits["validation"], names)}
    selection["commonValidationRecordIds"] = sorted(set.intersection(*(
        {row["recordId"] for row in task["validation"]} for task in selection["tasks"].values())))
    return selection


def requireDiskSpace(directory, requiredBytes, reserveBytes=2 * 1024 ** 3):
    directory = Path(directory).resolve()
    while not directory.exists():
        directory = directory.parent
    free = shutil.disk_usage(directory).free
    if free < requiredBytes + reserveBytes:
        raise ValueError(f"Insufficient disk space: need {requiredBytes} bytes plus {reserveBytes} reserve; have {free}.")
    return {"requiredBytes": requiredBytes, "reserveBytes": reserveBytes, "freeBytes": free}


def copyRuntimeData(datasetDir, runDir, selection):
    root, runDir = Path(datasetDir).resolve(), Path(runDir).resolve()
    required = sum(resolveSourcePath(root, row[f"{kind}Path"]).stat().st_size
                   for task in selection["tasks"].values() for split in ("training", "validation")
                   for row in task[split] for kind in ("image", "label"))
    requireDiskSpace(runDir, required)
    evidence = []
    for taskId, task in selection["tasks"].items():
        taskRoot = runDir / "runtimeData" / taskId
        for split in ("training", "validation"):
            for row in task[split]:
                for kind in ("image", "label"):
                    relative = row[f"{kind}Path"]
                    source = resolveSourcePath(root, relative)
                    expected = row[f"{kind}Sha256"]
                    if hashFile(source) != expected:
                        raise ValueError(f"Prepared data changed before runtime copy: {relative}.")
                    destination = runDir / "runtimeData" / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.exists():
                        raise ValueError(f"Runtime destination already exists: {destination}.")
                    shutil.copyfile(source, destination)
                    evidence.append({"path": destination.relative_to(runDir).as_posix(),
                                     "byteSize": source.stat().st_size, "sha256": expected})
        configuration = {"path": str(taskRoot), "train": "images/train", "val": "images/val",
                         "names": dict(enumerate(task["classes"]))}
        (taskRoot / "dataset.yaml").write_text(yaml.safe_dump(configuration, sort_keys=False),
                                                encoding="utf-8")
        path = taskRoot / "dataset.yaml"
        evidence.append({"path": path.relative_to(runDir).as_posix(),
                         "byteSize": path.stat().st_size, "sha256": hashFile(path)})
    verifyEvidence(runDir, evidence)
    writeJson(runDir / "runtimeEvidence.json", evidence)
    writeJson(runDir / "selection.json", selection)
    return evidence
