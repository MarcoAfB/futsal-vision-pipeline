import json
import shutil
import traceback
from datetime import UTC, datetime
from pathlib import Path

from futsal.catalog import hashFile, resolveSourcePath
from futsal.datasetFiles import hashJson, readJson, verifyEvidence, writeJson
from futsal.evaluationBackend import predictValidation
from futsal.evaluationData import (
    buildReferences,
    evaluationTasks,
    loadEvaluationConfig,
    validatePredictions,
    validateResearchOutput,
    writeJsonLines,
)
from futsal.evaluationMetrics import scoreDetections
from futsal.pointMetrics import scorePoints
from futsal.report import atomicWrite
from futsal.researchExamples import fixedExamples, selectExamples
from futsal.trainingBackend import initializeBackend
from futsal.trainingData import copyRuntimeData
from futsal.trainingEnvironment import packageVersions
from futsal.trainingRun import fileEvidence, snapshotCode, utcNow, verifyBundleUnchanged


def saveEvaluation(outputDir, state):
    atomicWrite(outputDir / "evaluation.json", json.dumps(state, sort_keys=True, allow_nan=False,
                                                         ensure_ascii=False, indent=2) + "\n")


def newEvaluationPath(runDir, selection="runValidation"):
    runDir = Path(runDir).resolve()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return runDir.parent.parent / "reports" / runDir.name / f"{selection}-{stamp}"


def archiveExamples(datasetDir, outputDir, selected, references):
    byId = {row["recordId"]: row for row in references}
    root = outputDir / "evidence" / "exampleImages"
    root.mkdir(exist_ok=True)
    result = []
    for item in selected:
        reference = byId[item["recordId"]]
        source = resolveSourcePath(datasetDir, reference["imagePath"])
        target = root / f"{reference['imageSha256']}{source.suffix.lower()}"
        if not target.exists():
            shutil.copyfile(source, target)
        if hashFile(target) != reference["imageSha256"]:
            raise ValueError("Example image changed while archiving report evidence.")
        result.append({**item, "image": target.relative_to(outputDir).as_posix(),
                       "imageSha256": reference["imageSha256"]})
    return result


def evaluateStage(datasetDir, runDir, outputDir, workDir, task, references, stage, state, fixedIds):
    taskId, config = task["taskId"], state["protocol"]
    evidenceDir = outputDir / "evidence" / taskId
    evidenceDir.mkdir()
    best = stage["results"]["checkpoints"]["best"]
    checkpoint = resolveSourcePath(runDir, best["path"])
    if hashFile(checkpoint) != best["sha256"]:
        raise ValueError("Selected best checkpoint changed since training.")
    writeJsonLines(evidenceDir / "references.jsonl", references)
    selection = {"tasks": {taskId: {"classes": task["classes"], "training": [], "validation": task["rows"]}}}
    runtimeEvidence = copyRuntimeData(datasetDir, workDir, selection)
    writeJson(evidenceDir / "runtimeEvidence.json", runtimeEvidence)
    predictions, native = predictValidation(checkpoint, task, references, stage,
                                            state["training"]["configuration"], config, workDir)
    verifyEvidence(workDir, runtimeEvidence)
    if hashFile(checkpoint) != best["sha256"]:
        raise ValueError("Evaluation changed the input checkpoint.")
    predictions = validatePredictions(references, predictions, task["classes"], config)
    writeJsonLines(evidenceDir / "predictions.jsonl", predictions)
    detection = scoreDetections(references, predictions, task["classes"], state["pointClasses"], config)
    points = scorePoints(references, predictions, task["classes"], state["pointClasses"], config)
    writeJson(evidenceDir / "detection.json", detection)
    writeJson(evidenceDir / "points.json", points)
    selected = selectExamples(fixedIds, detection, points, config)
    examples = archiveExamples(datasetDir, outputDir, selected, references)
    for name in ("results.csv", "transferAudit.json", "loaderAudit.json", "optimizer.json"):
        source = resolveSourcePath(runDir, f"{stage['directory']}/{name}")
        if source.is_file():
            shutil.copyfile(source, evidenceDir / name)
    return {"taskId": taskId, "status": "completed", "classes": task["classes"], "checkpoint": best,
            "imageCount": len(references), "recordIds": [row["recordId"] for row in references],
            "evidenceDirectory": evidenceDir.relative_to(outputDir).as_posix(), "examples": examples,
            "nativeValidationMetrics": native, "apMatchingSpace": "nativeLetterboxedInput",
            "predictionSpace": "originalImagePixels", "runtimeHashesVerified": True}


def sealEvaluation(outputDir, state):
    artifacts = [fileEvidence(path, outputDir) for path in sorted((outputDir / "evidence").rglob("*"))
                 if path.is_file()]
    state["evidence"] = artifacts
    state["endedAt"] = utcNow()
    saveEvaluation(outputDir, state)
    contract = {"schemaVersion": 1, "evaluationSha256": hashFile(outputDir / "evaluation.json"),
                "artifacts": artifacts}
    contract["contractId"] = hashJson(contract)
    writeJson(outputDir / "evaluationContract.json", contract)


def evaluateRun(runDir, datasetDir, configPath, selection, outputDir, progress=print):
    runDir, datasetDir = Path(runDir).resolve(), Path(datasetDir).resolve()
    outputDir = validateResearchOutput(outputDir, [datasetDir, runDir], [Path(configPath)])
    config = loadEvaluationConfig(configPath)
    run = readJson(runDir / "run.json")
    if not run.get("contractId") or not run.get("selectionSha256"):
        raise ValueError("Run has no verified dataset selection; use reportRun --runDir for execution diagnostics.")
    plan, tasks = evaluationTasks(datasetDir, runDir, run, selection)
    contract = readJson(datasetDir / "contract.json")
    if contract["contractId"] != run["contractId"] or contract["contractId"] != hashJson({
        key: value for key, value in contract.items() if key != "contractId"
    }):
        raise ValueError("Evaluation dataset contract differs from the verified training input.")
    workDir = Path(__file__).resolve().parents[2] / "outputs" / "evaluationWork" / hashJson(str(outputDir))[:16]
    validateResearchOutput(workDir, [datasetDir, runDir], [Path(configPath)])
    outputDir.mkdir(parents=True)
    (outputDir / "evidence").mkdir()
    workDir.mkdir(parents=True)
    pointClasses = [item["name"] for item in plan["catalog"]["labels"]["canonicalClasses"]
                    if item["geometry"] == "point"]
    state = {"schemaVersion": 1, "status": "running", "startedAt": utcNow(), "purpose": run["purpose"],
             "selection": selection, "protocol": config, "protocolId": hashJson(config),
             "training": run, "trainingDirectory": str(runDir), "trainingRunId": runDir.name,
             "trainingRunSha256": hashFile(runDir / "run.json"),
             "contractId": contract["contractId"], "pointClasses": pointClasses, "testEvaluated": False,
             "stages": [{"taskId": taskId, "status": "notEvaluated"} for taskId in tasks],
             "datasetSummary": contract["summary"], "ownSplitSummaries": plan["ownSplitSummaries"],
             "reviewedOwnRecordCount": plan["catalog"]["data"]["expected"]["reviewedFrames"],
             "environment": packageVersions(), "workDirectory": str(workDir)}
    saveEvaluation(outputDir, state)
    contractHash = hashFile(datasetDir / "contract.json")
    current = None
    try:
        state["code"] = snapshotCode(workDir)
        shutil.copytree(workDir / "code", outputDir / "evidence" / "code")
        writeJson(outputDir / "evidence" / "codeSnapshot.json", state["code"])
        shutil.copyfile(runDir / "run.json", outputDir / "evidence" / "trainingRun.json")
        shutil.copyfile(runDir / "selection.json", outputDir / "evidence" / "trainingSelection.json")
        progress("Checking the prepared bundle and archived training artifacts before evaluation...")
        state["bundleBefore"] = verifyBundleUnchanged(datasetDir, contract, contractHash)
        verifyEvidence(runDir, run["artifacts"])
        initializeBackend(workDir / "backend", run["configuration"]["numThreads"])
        fixedIds = fixedExamples(tasks, config)
        stages = {stage["taskId"]: stage for stage in run["stages"]}
        for current in state["stages"]:
            taskId = current["taskId"]
            stage = stages.get(taskId)
            if stage is None or stage["status"] != "completed":
                current["reason"] = "Training stage is not completed."
                continue
            current["status"] = "running"
            saveEvaluation(outputDir, state)
            progress(f"Evaluating {taskId}: {len(tasks[taskId]['rows'])} {selection} images...")
            references = buildReferences(plan, tasks[taskId])
            current.update(evaluateStage(datasetDir, runDir, outputDir, workDir, tasks[taskId], references,
                                         stage, state, fixedIds))
            saveEvaluation(outputDir, state)
        state["status"] = "completed" if all(s["status"] == "completed" for s in state["stages"]) else "partial"
    except (Exception, KeyboardInterrupt) as error:
        state["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        state["error"] = {"type": type(error).__name__, "message": str(error)}
        (outputDir / "evidence" / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        if current is not None and current["status"] == "running":
            current["status"] = state["status"]
    finally:
        try:
            progress("Checking prepared bundle integrity after evaluation...")
            state["bundleAfter"] = verifyBundleUnchanged(datasetDir, contract, contractHash)
            for path in (outputDir / "evidence").glob("*/runtimeEvidence.json"):
                verifyEvidence(workDir, readJson(path))
            state["runtimeAfter"] = True
            verifyEvidence(runDir, run["artifacts"])
            if hashFile(runDir / "run.json") != state["trainingRunSha256"]:
                raise ValueError("Archived training run metadata changed during evaluation.")
        except (OSError, ValueError) as error:
            state["status"] = "failed"
            state["integrityError"] = str(error)
        sealEvaluation(outputDir, state)
    return state
