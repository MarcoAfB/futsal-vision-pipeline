import copy
import shutil
from pathlib import Path

from futsal.datasets.datasetFiles import hashJson, readJson, verifyEvidence
from futsal.inventory.catalog import hashFile, resolveSourcePath
from futsal.training.memoryRecovery import isMemoryFailure
from futsal.training.trainingConfig import stageParameters


def validateMemoryRetry(previous, run, prefixCount):
    before, after = previous["configuration"], run["configuration"]
    if (previous["status"] != "failed" or previous.get("batchRecovery") != "fixed"
        or run.get("batchRecovery") != "fixed" or before.get("schemaVersion") != 2
        or not isMemoryFailure(previous.get("error", {}).get("causes", []))
        or len(previous["stages"]) != prefixCount + 1
        or previous["stages"][-1]["status"] != "failed"):
        raise ValueError("Memory retry requires a verified failed fixed-batch scientific stage with a definite memory error.")
    if before["stages"][prefixCount]["taskId"] != previous["stages"][-1]["taskId"]:
        raise ValueError("Memory retry failed stage does not match the configuration.")
    normalized = copy.deepcopy(after)
    if len(before["stages"]) != len(after["stages"]):
        raise ValueError("Memory retry may change only the failed stage batch.")
    oldStage, newStage = before["stages"][prefixCount], after["stages"][prefixCount]
    oldBatch = stageParameters(before, oldStage)["batch"]
    newBatch = stageParameters(after, newStage)["batch"]
    if oldBatch not in {8, 4, 2} or newBatch != oldBatch // 2:
        raise ValueError("Memory retry requires the next batch in 8 -> 4 -> 2 -> 1.")
    normalized["stages"][prefixCount] = copy.deepcopy(oldStage)
    beforeStage, afterStage = copy.deepcopy(oldStage), copy.deepcopy(newStage)
    for item in (beforeStage, afterStage):
        item.setdefault("parameters", {}).pop("batch", None)
    if normalized != before or beforeStage != afterStage:
        raise ValueError("Memory retry may change only the failed stage batch.")


def continueCompletedStages(previousDir, runDir, run, *, memoryRetry=False):
    previousDir = Path(previousDir).resolve()
    previous = readJson(previousDir / "run.json")
    if previous["status"] not in {"failed", "interrupted"}:
        raise ValueError("Continuation requires a failed or interrupted attempt.")
    keys = ("contractId", "initialWeights", "selectionSha256")
    if not memoryRetry:
        keys = ("configurationId", *keys)
    for key in keys:
        if previous.get(key) != run[key]:
            raise ValueError(f"Continuation requires identical {key}.")
    if previous.get("batchRecovery", "native") != run.get("batchRecovery", "native"):
        raise ValueError("Continuation requires identical batchRecovery.")
    if previous["configurationId"] != hashJson(previous["configuration"]):
        raise ValueError("Prior configuration checksum changed.")
    if (previous["code"]["commit"], previous["code"]["files"]) != (run["code"]["commit"], run["code"]["files"]):
        raise ValueError("Continuation requires identical code commit and source hashes.")
    if previous.get("integrityErrors") or not previous.get("runtimeIntegrityAfterRun") or not previous.get("bundleIntegrityAfterRun", {}).get("hashesVerified"):
        raise ValueError("A prior integrity failure cannot be continued.")
    for key in ("artifacts", "inputEvidence"):
        verifyEvidence(previousDir, previous[key])
    verifyEvidence(previousDir, previous["code"]["files"])
    if hashFile(previousDir / "selection.json") != previous["selectionSha256"]:
        raise ValueError("Prior selection changed before continuation.")
    prefix = []
    for stage in previous["stages"]:
        if stage["status"] != "completed":
            break
        prefix.append(stage)
    if len(prefix) == len(run["configuration"]["stages"]) or (not prefix and not memoryRetry):
        raise ValueError("Continuation requires a completed stage prefix and remaining stages.")
    if memoryRetry:
        validateMemoryRetry(previous, run, len(prefix))
        if not previous.get("failureEvidence"):
            raise ValueError("Memory retry requires sealed failure evidence.")
        verifyEvidence(previousDir, [previous["failureEvidence"]])
    expectedInput = run["initialWeights"]["sha256"]
    for index, stage in enumerate(prefix):
        if stage["taskId"] != run["configuration"]["stages"][index]["taskId"] or stage["inputWeights"]["sha256"] != expectedInput:
            raise ValueError("Completed stage prefix has an inconsistent checkpoint chain.")
        checkpoint = stage["results"]["checkpoints"]["best"]
        verifyEvidence(previousDir, [checkpoint])
        expectedInput = checkpoint["sha256"]
    priorHash = hashFile(previousDir / "run.json")
    for stage in prefix:
        files = [item for item in previous["artifacts"] if item["path"].startswith(stage["directory"] + "/")
                 or item["path"] == f"logs/{stage['taskId']}.log"]
        if stage["results"]["checkpoints"]["best"]["path"] not in {item["path"] for item in files}:
            raise ValueError("Completed checkpoint is missing from prior artifact evidence.")
        for item in files:
            source = resolveSourcePath(previousDir, item["path"])
            target = resolveSourcePath(runDir, item["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        verifyEvidence(runDir, files)
        restored = copy.deepcopy(stage)
        restored["reusedFromRunSha256"] = priorHash
        run["stages"].append(restored)
    run["continuation"] = {"previousDirectory": str(previousDir), "previousRunSha256": priorHash,
                           "completedStageCount": len(prefix), "mode": "restartNextStageFromEpochZero",
                           "reason": "memoryRetry" if memoryRetry else "strictContinuation"}
    if memoryRetry:
        index = len(prefix)
        run["continuation"].update(taskId=run["configuration"]["stages"][index]["taskId"],
            previousBatch=stageParameters(previous["configuration"], previous["configuration"]["stages"][index])["batch"],
            requestedBatch=stageParameters(run["configuration"], run["configuration"]["stages"][index])["batch"])
        run["continuation"]["ancestry"] = [*previous.get("continuation", {}).get("ancestry", []), priorHash]
    shutil.copyfile(previousDir / "run.json", runDir / "inputs" / "previousRun.json")
    if hashFile(runDir / "inputs" / "previousRun.json") != priorHash:
        raise ValueError("Prior run changed during continuation snapshot.")
