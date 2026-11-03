import copy
import shutil
from pathlib import Path

from futsal.datasets.datasetFiles import readJson, verifyEvidence
from futsal.inventory.catalog import hashFile, resolveSourcePath


def continueCompletedStages(previousDir, runDir, run):
    previousDir = Path(previousDir).resolve()
    previous = readJson(previousDir / "run.json")
    if previous["status"] not in {"failed", "interrupted"}:
        raise ValueError("Continuation requires a failed or interrupted attempt.")
    for key in ("configurationId", "contractId", "initialWeights", "selectionSha256"):
        if previous.get(key) != run[key]:
            raise ValueError(f"Continuation requires identical {key}.")
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
    if not prefix or len(prefix) == len(run["configuration"]["stages"]):
        raise ValueError("Continuation requires a completed stage prefix and remaining stages.")
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
                           "completedStageCount": len(prefix), "mode": "restartNextStageFromEpochZero"}
    shutil.copyfile(previousDir / "run.json", runDir / "inputs" / "previousRun.json")
    if hashFile(runDir / "inputs" / "previousRun.json") != priorHash:
        raise ValueError("Prior run changed during continuation snapshot.")
