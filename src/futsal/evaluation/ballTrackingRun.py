import shutil
import subprocess
import sys
import traceback
from pathlib import Path

from futsal.datasets.datasetFiles import hashJson, readJson, writeJson
from futsal.datasets.mediaStore import protectMediaOutput
from futsal.evaluation.ballTrackingMetrics import scoreBallTracking
from futsal.evaluation.evaluationData import readJsonLines, validateResearchOutput
from futsal.evaluation.evaluationRun import saveEvaluation, sealEvaluation
from futsal.inventory.catalog import hashFile, resolveSourcePath
from futsal.tracking.trackingBackend import preflightTracking
from futsal.tracking.trackingConfig import loadTrackingConfig, loadTrackingRuns
from futsal.tracking.trackingData import selectTrackingData
from futsal.tracking.trackingModels import (
    inspectTrackingModels,
    loadTrackingModels,
    verifyTrackingModels,
)
from futsal.training.trainingRun import snapshotCode, utcNow


def campaignIdentity(selection, models, config, environment, code, trackerHashes):
    environmentKeys = ("pythonVersion", "platform", "packages", "requestedDevice", "gpus", "cudaBuild", "cudnnVersion", "nvidiaKernelModule", "nvidiaSmi")
    return hashJson({"selectionId": selection["selectionId"], "config": config,
                     "models": {size: {key: model[key] for key in ("runSha256", "selectionSha256", "checkpointSha256")}
                                for size, model in models.items()},
                     "environment": {key: environment.get(key) for key in environmentKeys},
                     "code": code["files"], "trackers": trackerHashes})


def resumeConditions(source, identity):
    from futsal.reporting.researchReport import verifyEvaluation

    state, contract = verifyEvaluation(source)
    if state.get("kind") != "ballTrackingComparison" or state.get("campaignId") != identity or state.get("integrityError"):
        raise ValueError("Resume requires identical verified tracking campaign inputs and environment.")
    return {item["conditionId"]: item for item in state["conditions"] if item["status"] == "completed"}, contract["contractId"]


def launchTrackingWorker(requestPath):
    root = requestPath.parent
    with (root / "console.log").open("w", encoding="utf-8") as stream:
        result = subprocess.run([sys.executable, "-m", "futsal.tracking.trackingWorker", "--request", str(requestPath)],
                                stdout=stream, stderr=subprocess.STDOUT, check=False)
    state = readJson(root / "worker.json") if (root / "worker.json").is_file() else {
        "status": "failed", "frames": 0, "error": f"Worker exited without state: {result.returncode}."}
    if result.returncode and state["status"] == "completed":
        state.update(status="failed", error=f"Worker exited with code {result.returncode}.")
    return state


def executeTrackingCondition(root, state, condition, model, images):
    work = Path(state["workDirectory"]) / condition["conditionId"]
    work.mkdir(parents=True)
    evidence = root / condition["evidenceDirectory"]
    evidence.mkdir(parents=True)
    tracker = condition["tracker"]
    trackerPath = None if tracker == "detector" else str(root / "evidence" / "trackers" / f"{tracker}.yaml")
    request = {"size": condition["size"], "config": state["protocol"], "device": state["requestedDevice"],
               "checkpoint": model["checkpointPath"], "checkpointSha256": model["checkpointSha256"],
               "trackerPath": trackerPath, "references": state["selectionEvidence"]["references"], "images": images}
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
        metrics = scoreBallTracking(request["references"], readJsonLines(evidence / "predictions.jsonl"),
                                   state["protocol"], tracked=tracker != "detector")
        writeJson(evidence / "metrics.json", metrics)


def verifyTrackingInputs(datasetDir, mediaStoreDir, selection, models, *, full=True):
    if hashFile(datasetDir / "contract.json") != selection["datasetContractSha256"] or any(
        hashFile(datasetDir / item["path"]) != item["sha256"] for item in selection["metadataEvidence"]
    ):
        raise ValueError("Tracking validation metadata changed during the campaign.")
    if full:
        verifyTrackingModels(models)
        current, _ = selectTrackingData(datasetDir, mediaStoreDir)
        if current["selectionId"] != selection["selectionId"]:
            raise ValueError("Tracking validation inputs changed during the campaign.")


def runTrackingConditions(root, state, models, images, previous, resumeFrom, progress, datasetDir, mediaStoreDir):
    for size, model in models.items():
        for tracker in ("detector", *state["protocol"]["trackers"]):
            conditionId = f"yolo26{size}_{tracker}"
            condition = {"conditionId": conditionId, "size": size, "tracker": tracker,
                         "status": "running", "evidenceDirectory": f"evidence/conditions/{conditionId}"}
            state["conditions"].append(condition)
            saveEvaluation(root, state)
            progress(f"Evaluating {conditionId}: {state['selectionEvidence']['frameCount']} validation frames...")
            try:
                if conditionId in previous:
                    old = previous[conditionId]
                    source = resolveSourcePath(resumeFrom, old["evidenceDirectory"])
                    shutil.copytree(source, root / condition["evidenceDirectory"])
                    condition.update(old, reusedFromContractId=state["resumeContractId"])
                    metrics = scoreBallTracking(state["selectionEvidence"]["references"],
                        readJsonLines(root / condition["evidenceDirectory"] / "predictions.jsonl"),
                        state["protocol"], tracked=tracker != "detector")
                    if metrics != readJson(root / condition["evidenceDirectory"] / "metrics.json") or condition["frames"] != state["selectionEvidence"]["frameCount"]:
                        raise ValueError("Reused tracking evidence differs from recomputed metrics or cohort.")
                else:
                    executeTrackingCondition(root, state, condition, model, images)
            except Exception as error:
                condition.update(status="failed", error=f"{type(error).__name__}: {error}")
                folder = root / condition["evidenceDirectory"]
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
            progress(f"{conditionId}: {condition['status']}")
            saveEvaluation(root, state)
            verifyTrackingInputs(datasetDir, mediaStoreDir, state["selectionEvidence"], models, full=False)


def compareBallTrackers(runsConfig, datasetDir, mediaStoreDir, outputDir, *, configPath="configs/evaluation/ballTracking.yaml", device="cpu", resumeFrom=None, progress=print):
    datasetDir, mediaStoreDir = Path(datasetDir).resolve(), Path(mediaStoreDir).resolve()
    runDirs = loadTrackingRuns(runsConfig)
    config = loadTrackingConfig(configPath)
    resumeFrom = Path(resumeFrom).resolve() if resumeFrom is not None else None
    roots = [datasetDir, mediaStoreDir, *runDirs.values(), *([resumeFrom] if resumeFrom else [])]
    root = validateResearchOutput(outputDir, roots, [Path(runsConfig), Path(configPath)])
    protectMediaOutput(root, mediaStoreDir)
    work = root.parent / ".ballTrackingWork" / hashJson(str(root))[:16]
    validateResearchOutput(work, [*roots, root], [Path(runsConfig), Path(configPath)])
    protectMediaOutput(work, mediaStoreDir)
    root.mkdir(parents=True)
    (root / "evidence").mkdir()
    state = {"schemaVersion": 1, "kind": "ballTrackingComparison", "status": "running", "startedAt": utcNow(),
             "purpose": "operationalBallTracking", "selection": "fullValidation", "testEvaluated": False,
             "protocol": config, "requestedDevice": device, "conditions": [], "workDirectory": str(work)}
    models = {}
    try:
        selection, images = selectTrackingData(datasetDir, mediaStoreDir)
        state["selectionEvidence"] = selection
        writeJson(root / "evidence" / "selection.json", selection)
        models = loadTrackingModels(runDirs, selection, datasetDir, config["seed"])
        state["models"] = models
        state["environment"], nativeConfigs = preflightTracking(config, device, work / "preflight")
        inspectTrackingModels(models, config, device)
        trackerDir = root / "evidence" / "trackers"
        trackerDir.mkdir()
        for name, content in nativeConfigs.items():
            (trackerDir / f"{name}.yaml").write_text(content, encoding="utf-8", newline="\n")
        state["trackerHashes"] = {name: hashFile(trackerDir / f"{name}.yaml") for name in nativeConfigs}
        state["code"] = snapshotCode(root / "evidence")
        for size, directory in runDirs.items():
            target = root / "evidence" / "training" / size
            target.mkdir(parents=True)
            for name in ("run.json", "selection.json"):
                shutil.copyfile(directory / name, target / name)
        state["campaignId"] = campaignIdentity(selection, models, config, state["environment"], state["code"], state["trackerHashes"])
        previous = {}
        if resumeFrom is not None:
            previous, state["resumeContractId"] = resumeConditions(resumeFrom, state["campaignId"])
        runTrackingConditions(root, state, models, images, previous, resumeFrom, progress, datasetDir, mediaStoreDir)
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
                verifyTrackingInputs(datasetDir, mediaStoreDir, state["selectionEvidence"], models)
            except Exception as error:
                state.update(status="failed", integrityError=f"{type(error).__name__}: {error}")
        sealEvaluation(root, state)
    return state
