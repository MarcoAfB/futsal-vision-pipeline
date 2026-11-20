import shutil
import traceback
from pathlib import Path

from futsal.datasets.datasetFiles import hashJson, readJson, writeJson
from futsal.datasets.mediaStore import protectMediaOutput
from futsal.evaluation.ballTrackingRun import campaignIdentity, launchTrackingWorker
from futsal.evaluation.evaluationData import readJsonLines, validateResearchOutput
from futsal.evaluation.evaluationRun import saveEvaluation, sealEvaluation
from futsal.evaluation.trackingMetrics import scoreTracking
from futsal.inventory.catalog import hashFile, resolveSourcePath
from futsal.tracking.trackingBackend import preflightTracking
from futsal.tracking.trackingConfig import loadTrackingConfig, resolveTrackedRuns
from futsal.tracking.trackingData import selectTrackedData
from futsal.tracking.trackingModels import (
    inspectTrackedModels,
    loadTrackedRuns,
    verifyTrackingModels,
)
from futsal.training.trainingRun import snapshotCode, utcNow


def resumeTrackedConditions(source, identity):
    from futsal.reporting.researchReport import verifyEvaluation

    state, contract = verifyEvaluation(source)
    if state.get("kind") != "trackingComparison" or state.get("campaignId") != identity or state.get("integrityError"):
        raise ValueError("Resume requires identical verified tracking campaign inputs and environment.")
    return {item["conditionId"]: item for item in state["conditions"] if item["status"] == "completed"}, contract["contractId"]


def sealedTrackingCondition(trackingDir, conditionId):
    """Read one completed condition of a verified campaign as aligned references and predictions.

    References come from the sealed selection, so they carry matchId, imageSha256 and segmentId.
    """
    from futsal.evaluation.trackingMetrics import validateTrackedPredictions
    from futsal.reporting.researchReport import verifyEvaluation

    root = Path(trackingDir).resolve()
    state, contract = verifyEvaluation(root)
    if state.get("kind") != "trackingComparison" or state["status"] != "completed" or \
            state.get("testEvaluated") is not False:
        raise ValueError("Tracking campaign: a completed validation-only trackingComparison is required.")
    conditions = {item["conditionId"]: item for item in state["conditions"]}
    condition = conditions.get(conditionId)
    if condition is None or condition["status"] != "completed":
        raise ValueError(f"Tracking campaign: condition {conditionId} is not completed; "
                         f"available: {sorted(conditions)}.")
    selection = readJson(root / "evidence" / "selection.json")
    if selection != state["selectionEvidence"]:
        raise ValueError("Tracking campaign: the sealed selection differs from the campaign state.")
    tracked = condition["tracker"] != "detector"
    predictions = validateTrackedPredictions(
        selection["references"], readJsonLines(root / condition["evidenceDirectory"] / "predictions.jsonl"),
        selection["classes"], state["protocol"], tracked)
    model = state["models"][condition["label"]]
    identity = {"directory": str(root), "kind": "trackingComparison", "contractId": contract["contractId"],
                "evaluationSha256": contract["evaluationSha256"], "campaignId": state["campaignId"],
                "conditionId": conditionId, "tracker": condition["tracker"], "tracked": tracked,
                "taskId": selection["taskId"], "classes": selection["classes"],
                "checkpointSha256": model["checkpointSha256"], "recordCount": len(predictions)}
    return identity, selection["references"], predictions


def scoreCondition(state, evidence, tracker):
    selection = state["selectionEvidence"]
    return scoreTracking(selection["references"], readJsonLines(evidence / "predictions.jsonl"),
                         selection["classes"], state["protocol"], tracked=tracker != "detector")


def executeCondition(root, state, condition, model, images):
    work = Path(state["workDirectory"]) / condition["conditionId"]
    work.mkdir(parents=True)
    evidence = root / condition["evidenceDirectory"]
    evidence.mkdir(parents=True)
    tracker = condition["tracker"]
    trackerPath = None if tracker == "detector" else str(root / "evidence" / "trackers" / f"{tracker}.yaml")
    request = {"size": model["size"], "classNames": model["classes"], "config": state["protocol"],
               "device": state["requestedDevice"], "checkpoint": model["checkpointPath"],
               "checkpointSha256": model["checkpointSha256"], "trackerPath": trackerPath,
               "references": state["selectionEvidence"]["references"], "images": images}
    writeJson(work / "request.json", request)
    try:
        result = launchTrackingWorker(work / "request.json")
    finally:
        for name in ("worker.json", "console.log", "predictions.jsonl", "failure.txt", "nativeArguments.json"):
            source = work / name
            if source.is_file():
                shutil.copyfile(source, evidence / name)
    condition.update(result)
    if condition["status"] == "completed":
        if condition["frames"] != state["selectionEvidence"]["frameCount"]:
            raise ValueError("Worker completed with an incomplete frame count.")
        writeJson(evidence / "metrics.json", scoreCondition(state, evidence, tracker))


def verifyTrackedInputs(datasetDir, mediaStoreDir, selection, models, *, full=True):
    if hashFile(datasetDir / "contract.json") != selection["datasetContractSha256"] or any(
        hashFile(datasetDir / item["path"]) != item["sha256"] for item in selection["metadataEvidence"]
    ):
        raise ValueError("Tracking validation metadata changed during the campaign.")
    if full:
        verifyTrackingModels(models)
        current, _ = selectTrackedData(datasetDir, mediaStoreDir, selection["taskId"])
        if current["selectionId"] != selection["selectionId"]:
            raise ValueError("Tracking validation inputs changed during the campaign.")


def reuseCondition(root, state, condition, previous, resumeFrom, tracker):
    old = previous[condition["conditionId"]]
    shutil.copytree(resolveSourcePath(resumeFrom, old["evidenceDirectory"]), root / condition["evidenceDirectory"])
    condition.update(old, reusedFromContractId=state["resumeContractId"])
    evidence = root / condition["evidenceDirectory"]
    if scoreCondition(state, evidence, tracker) != readJson(evidence / "metrics.json") or (
        condition["frames"] != state["selectionEvidence"]["frameCount"]
    ):
        raise ValueError("Reused tracking evidence differs from recomputed metrics or cohort.")


def runConditions(root, state, models, images, previous, resumeFrom, progress, datasetDir, mediaStoreDir):
    for label, model in models.items():
        for tracker in ("detector", *state["protocol"]["trackers"]):
            conditionId = f"{label}_{tracker}"
            condition = {"conditionId": conditionId, "label": label, "size": model["size"], "tracker": tracker,
                         "status": "running", "evidenceDirectory": f"evidence/conditions/{conditionId}"}
            state["conditions"].append(condition)
            saveEvaluation(root, state)
            progress(f"Evaluating {conditionId}: {state['selectionEvidence']['frameCount']} validation frames...")
            try:
                if conditionId in previous:
                    reuseCondition(root, state, condition, previous, resumeFrom, tracker)
                else:
                    executeCondition(root, state, condition, model, images)
            except Exception as error:
                condition.update(status="failed", error=f"{type(error).__name__}: {error}")
                folder = root / condition["evidenceDirectory"]
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
            progress(f"{conditionId}: {condition['status']}")
            saveEvaluation(root, state)
            verifyTrackedInputs(datasetDir, mediaStoreDir, state["selectionEvidence"], models, full=False)


def finalTaskId(runDirs):
    taskIds = {readJson(root / "run.json")["configuration"]["stages"][-1]["taskId"] for root in runDirs.values()}
    if len(taskIds) != 1:
        raise ValueError(f"All runs must share one final task; found {sorted(taskIds)}.")
    return taskIds.pop()


def trackRuns(runSpecs, datasetDir, mediaStoreDir, outputDir, *, configPath="configs/evaluation/tracking.yaml",
              device="cpu", resumeFrom=None, progress=print):
    datasetDir, mediaStoreDir = Path(datasetDir).resolve(), Path(mediaStoreDir).resolve()
    runDirs = resolveTrackedRuns(runSpecs)
    config = loadTrackingConfig(configPath)
    resumeFrom = Path(resumeFrom).resolve() if resumeFrom is not None else None
    roots = [datasetDir, mediaStoreDir, *runDirs.values(), *([resumeFrom] if resumeFrom else [])]
    root = validateResearchOutput(outputDir, roots, [Path(configPath)])
    protectMediaOutput(root, mediaStoreDir)
    work = root.parent / ".trackingWork" / hashJson(str(root))[:16]
    validateResearchOutput(work, [*roots, root], [Path(configPath)])
    protectMediaOutput(work, mediaStoreDir)
    root.mkdir(parents=True)
    (root / "evidence").mkdir()
    state = {"schemaVersion": 1, "kind": "trackingComparison", "status": "running", "startedAt": utcNow(),
             "purpose": "operationalTracking", "selection": "fullValidation", "testEvaluated": False,
             "protocol": config, "requestedDevice": device, "conditions": [], "workDirectory": str(work)}
    models = {}
    try:
        selection, images = selectTrackedData(datasetDir, mediaStoreDir, finalTaskId(runDirs))
        state["selectionEvidence"] = selection
        writeJson(root / "evidence" / "selection.json", selection)
        models = loadTrackedRuns(runDirs, selection, datasetDir)
        state["models"] = models
        state["environment"], nativeConfigs = preflightTracking(config, device, work / "preflight")
        inspectTrackedModels(models, config, device)
        trackerDir = root / "evidence" / "trackers"
        trackerDir.mkdir()
        for name, content in nativeConfigs.items():
            (trackerDir / f"{name}.yaml").write_text(content, encoding="utf-8", newline="\n")
        state["trackerHashes"] = {name: hashFile(trackerDir / f"{name}.yaml") for name in nativeConfigs}
        state["code"] = snapshotCode(root / "evidence")
        for label, directory in runDirs.items():
            target = root / "evidence" / "training" / label
            target.mkdir(parents=True)
            for name in ("run.json", "selection.json"):
                shutil.copyfile(directory / name, target / name)
        state["campaignId"] = campaignIdentity(selection, models, config, state["environment"], state["code"],
                                               state["trackerHashes"])
        previous = {}
        if resumeFrom is not None:
            previous, state["resumeContractId"] = resumeTrackedConditions(resumeFrom, state["campaignId"])
        runConditions(root, state, models, images, previous, resumeFrom, progress, datasetDir, mediaStoreDir)
        state["status"] = "completed" if all(c["status"] == "completed" for c in state["conditions"]) else "partial"
    except (Exception, KeyboardInterrupt) as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     error=f"{type(error).__name__}: {error}")
        for condition in state["conditions"]:
            if condition["status"] == "running":
                condition["status"] = state["status"]
        (root / "evidence" / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        if "selectionEvidence" in state:
            try:
                verifyTrackedInputs(datasetDir, mediaStoreDir, state["selectionEvidence"], models)
            except Exception as error:
                state.update(status="failed", integrityError=f"{type(error).__name__}: {error}")
        sealEvaluation(root, state)
    return state
