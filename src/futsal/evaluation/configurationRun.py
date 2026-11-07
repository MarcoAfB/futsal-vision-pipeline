import shutil
import traceback
from pathlib import Path

import futsal
from futsal.datasets.datasetFiles import hashJson, readJson, verifyEvidence, writeJson
from futsal.datasets.datasetPlan import loadDatasetPlan, planRecipe
from futsal.datasets.projectionPlan import buildProjectionPlan
from futsal.evaluation.configurationBackend import predictConfiguration
from futsal.evaluation.evaluationData import (
    buildReferences,
    evaluationTasks,
    loadEvaluationConfig,
    validatePredictions,
    validateResearchOutput,
    writeJsonLines,
)
from futsal.evaluation.evaluationMetrics import scoreDetections
from futsal.evaluation.evaluationRun import archiveExamples, saveEvaluation, sealEvaluation
from futsal.evaluation.pointMetrics import scorePoints
from futsal.inventory.catalog import hashFile, resolveSourcePath
from futsal.reporting.researchExamples import fixedExamples, selectExamples
from futsal.training.modelWeights import verifyWeights
from futsal.training.trainingBackend import initializeBackend
from futsal.training.trainingData import copyRuntimeData, selectTrainingData
from futsal.training.trainingEnvironment import inspectEnvironment, packageVersions
from futsal.training.trainingRun import snapshotCode, utcNow, verifyBundleUnchanged


def loadComponent(runDir, taskId, datasetDir, classes):
    root = Path(runDir).resolve()
    run = readJson(root / "run.json")
    config = run["configuration"]
    allowed = {"realOnly", "realWithSyntheticCourt"} if taskId == "courtPoints" else {"realOnly"}
    if run["status"] != "completed" or run["purpose"] != "scientificTraining" or config.get("sourcePolicy") not in allowed:
        raise ValueError("Joint evaluation requires completed scientific runs with an allowed source policy.")
    if len(config["stages"]) != 1 or config["stages"][0]["taskId"] != taskId or len(run["stages"]) != 1 or run["stages"][0]["taskId"] != taskId or run["stages"][0]["status"] != "completed":
        raise ValueError("Each component must be its own independent single-stage run.")
    if run.get("integrityErrors") or not run.get("runtimeIntegrityAfterRun") or not run.get("bundleIntegrityAfterRun", {}).get("hashesVerified"):
        raise ValueError("Component run has no successful input/runtime integrity record.")
    if run["contractId"] != readJson(Path(datasetDir) / "contract.json")["contractId"] or run["configurationId"] != hashJson(config):
        raise ValueError("Component dataset/configuration identity mismatch.")
    for evidence in (run["artifacts"], run["inputEvidence"], run["code"]["files"]):
        verifyEvidence(root, evidence)
    selection = readJson(root / "selection.json")
    from futsal.training.syntheticTraining import loadSyntheticSnapshot

    snapshot = loadSyntheticSnapshot(root, config, classes)
    if snapshot is not None and not run.get("syntheticIntegrityAfterRun"):
        raise ValueError("Mixed component has no successful synthetic source integrity record.")
    if selection != selectTrainingData(datasetDir, config, snapshot):
        raise ValueError("Component did not use its full declared training/validation cohort.")
    _, tasks = evaluationTasks(datasetDir, root, run, "runValidation")
    if tasks[taskId]["classes"] != classes:
        raise ValueError("Component classes differ from their canonical partition.")
    weights = verifyWeights(root / "inputs" / run["initialWeights"]["filename"], config["modelSize"])
    if weights != run["initialWeights"] or run["stages"][0]["inputWeights"]["sha256"] != weights["sha256"]:
        raise ValueError("Component did not initialize directly from the official checkpoint.")
    checkpoint = run["stages"][0]["results"]["checkpoints"]["best"]
    verifyEvidence(root, [checkpoint])
    return {"taskId": taskId, "classes": classes, "runDirectory": str(root), "run": run,
            "runSha256": hashFile(root / "run.json"), "checkpoint": checkpoint,
            "checkpointPath": str(resolveSourcePath(root, checkpoint["path"])), "rows": tasks[taskId]["rows"]}


def configurationInputs(objectsRunDir, courtRunDir, datasetDir, referenceDatasetDir):
    parent = loadDatasetPlan(Path(referenceDatasetDir) / "plan.json")
    plan = loadDatasetPlan(Path(datasetDir) / "plan.json")
    if plan != buildProjectionPlan(parent, planRecipe(plan)):
        raise ValueError("Configuration data do not derive from the reference native plan.")
    canonical = next(task for task in parent["recipe"]["tasks"] if task["taskId"] == parent["recipe"]["coverageTask"])
    names = canonical["classes"]
    components = [loadComponent(objectsRunDir, "detailedObjects", datasetDir, names[:5]),
                  loadComponent(courtRunDir, "courtPoints", datasetDir, names[5:])]
    first, second = [item["run"] for item in components]
    fields = ("modelSize", "seed", "device", "evaluationDevice", "parameters", "numThreads")
    if any(first["configuration"][key] != second["configuration"][key] for key in fields) or first["initialWeights"] != second["initialWeights"]:
        raise ValueError("Independent components require identical official initialization and shared protocol.")
    if first["configuration"]["stages"][0]["epochs"] != second["configuration"]["stages"][0]["epochs"]:
        raise ValueError("Independent component epoch budgets differ.")
    if first["code"]["commit"] != second["code"]["commit"] or first["code"]["files"] != second["code"]["files"]:
        raise ValueError("Independent component code snapshots differ.")
    common = set.intersection(*({row["recordId"] for row in item["rows"]} for item in components))
    rows = [row for row in readJson(Path(referenceDatasetDir) / "manifest.json")["samples"]
            if row["taskId"] == canonical["taskId"] and row["split"] == "validation" and row["recordId"] in common]
    if not rows:
        raise ValueError("No common eligible own validation records.")
    for component in components:
        images = {row["recordId"]: row["imageSha256"] for row in component["rows"]}
        if any(images[row["recordId"]] != row["imageSha256"] for row in rows):
            raise ValueError("Common validation source image bytes differ.")
    task = {**canonical, "rows": sorted(rows, key=lambda row: row["recordId"])}
    return parent, task, components


def evaluateConfiguration(objectsRunDir, courtRunDir, datasetDir, referenceDatasetDir, configPath, outputDir,
                          progress=print, *, workRoot=None):
    datasetDir, referenceDir = Path(datasetDir).resolve(), Path(referenceDatasetDir).resolve()
    inputs = [Path(objectsRunDir), Path(courtRunDir), datasetDir, referenceDir]
    output = validateResearchOutput(outputDir, inputs, [Path(configPath)])
    config = loadEvaluationConfig(configPath)
    parent, task, components = configurationInputs(objectsRunDir, courtRunDir, datasetDir, referenceDir)
    if readJson(datasetDir / "contract.json").get("parentContractId") != readJson(referenceDir / "contract.json")["contractId"]:
        raise ValueError("Materialized dataset parent contract mismatch.")
    workRoot = Path(workRoot).resolve() if workRoot is not None else Path(futsal.__file__).resolve().parents[2] / "outputs/evaluationWork"
    work = validateResearchOutput(workRoot / hashJson(str(output))[:16], [*inputs, output], [Path(configPath)])
    validateResearchOutput(output, [*inputs, work], [Path(configPath)])
    output.mkdir(parents=True)
    (output / "evidence/joint").mkdir(parents=True)
    work.mkdir(parents=True)
    trainingConfig = components[0]["run"]["configuration"]
    state = {"schemaVersion": 1, "kind": "configurationEvaluation", "configurationId": "separatedDetection",
        "status": "running", "startedAt": utcNow(), "purpose": "scientificTraining", "selection": "commonValidation",
        "protocol": config, "protocolId": hashJson(config), "classes": task["classes"], "pointClasses": task["classes"][5:],
        "components": [{key: value for key, value in component.items() if key != "rows"} for component in components],
        "referenceContractId": readJson(referenceDir / "contract.json")["contractId"],
        "contractId": readJson(datasetDir / "contract.json")["contractId"], "recordIds": [row["recordId"] for row in task["rows"]],
        "testEvaluated": False, "requestedDevice": trainingConfig["evaluationDevice"], "environment": packageVersions(),
        "workDirectory": str(work), "datasetDirectory": str(datasetDir), "referenceDatasetDirectory": str(referenceDir),
        "ownSplitSummaries": readJson(datasetDir / "plan.json")["ownSplitSummaries"]}
    saveEvaluation(output, state)
    contracts = [(root, readJson(root / "contract.json"), hashFile(root / "contract.json")) for root in (referenceDir, datasetDir)]
    runtime = None
    try:
        state["code"] = snapshotCode(work)
        shutil.copytree(work / "code", output / "evidence/code")
        for component in components:
            target = output / "evidence/components" / component["taskId"]
            target.mkdir(parents=True)
            for name in ("run.json", "selection.json"):
                shutil.copyfile(Path(component["runDirectory"]) / name, target / name)
            if component["run"].get("syntheticSource"):
                for name in ("syntheticContract.json", "syntheticManifest.jsonl"):
                    shutil.copyfile(Path(component["runDirectory"]) / "inputs" / name, target / name)
        progress("Checking reference/projected bundles before joint validation...")
        state["bundlesBefore"] = [verifyBundleUnchanged(root, contract, digest) for root, contract, digest in contracts]
        if state["requestedDevice"] != "cpu":
            state["deviceInspection"] = inspectEnvironment(state["requestedDevice"])
            if not state["deviceInspection"]["ready"]:
                raise ValueError("Joint evaluation GPU is not ready; no fallback is allowed.")
        runtime = copyRuntimeData(referenceDir, work, {"tasks": {task["taskId"]: {
            "classes": task["classes"], "training": [], "validation": task["rows"]}}})
        writeJson(output / "evidence/runtimeEvidence.json", runtime)
        initializeBackend(work / "backend", trainingConfig["numThreads"])
        references = buildReferences(parent, task)
        writeJsonLines(output / "evidence/joint/references.jsonl", references)
        predictions, native = predictConfiguration(components, task, references, trainingConfig, config, work)
        predictions = validatePredictions(references, predictions, task["classes"], config)
        writeJsonLines(output / "evidence/joint/predictions.jsonl", predictions)
        detection = scoreDetections(references, predictions, task["classes"], state["pointClasses"], config)
        points = scorePoints(references, predictions, task["classes"], state["pointClasses"], config)
        writeJson(output / "evidence/joint/detection.json", detection)
        writeJson(output / "evidence/joint/points.json", points)
        fixed = fixedExamples({task["taskId"]: task}, config)
        state["examples"] = archiveExamples(referenceDir, output, selectExamples(fixed, detection, points, config), references)
        state.update(status="completed", nativeValidationMetrics=native, apMatchingSpace="nativeLetterboxedInput",
                     predictionSpace="originalImagePixels", globalDetectionLimit=config["maxDetections"])
        for name in ("configurationArguments.json", "configurationLoaderAudit.json"):
            shutil.copyfile(work / name, output / "evidence" / name)
    except (Exception, KeyboardInterrupt) as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     error={"type": type(error).__name__, "message": str(error)})
        (output / "evidence/failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        try:
            state["bundlesAfter"] = [verifyBundleUnchanged(root, contract, digest) for root, contract, digest in contracts]
            if runtime is not None:
                verifyEvidence(work, runtime)
            for component in components:
                root = Path(component["runDirectory"])
                verifyEvidence(root, component["run"]["artifacts"])
                if hashFile(root / "run.json") != component["runSha256"]:
                    raise ValueError("Component run metadata changed during joint evaluation.")
            state["runtimeAfter"] = True
        except (OSError, ValueError) as error:
            state.update(status="failed", integrityError=str(error))
        sealEvaluation(output, state)
    return state
