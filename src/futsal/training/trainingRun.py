import json
import shutil
import subprocess
import traceback
from datetime import UTC, datetime
from pathlib import Path

import futsal
from futsal.datasets.datasetFiles import (
    hashJson,
    readJson,
    validateNewOutput,
    verifyEvidence,
    writeJson,
)
from futsal.datasets.datasetVerify import verifyDataset
from futsal.inventory.catalog import hashFile
from futsal.inventory.yoloInventory import scanFiles
from futsal.reporting.report import atomicWrite
from futsal.training.modelWeights import verifyWeights
from futsal.training.trainingBackend import initializeBackend, trainStage
from futsal.training.trainingConfig import loadTrainingConfig
from futsal.training.trainingContinuation import continueCompletedStages
from futsal.training.trainingData import copyRuntimeData, selectTrainingData
from futsal.training.trainingEnvironment import inspectEnvironment


def utcNow():
    return datetime.now(UTC).isoformat()


def saveRun(runDir, run):
    atomicWrite(runDir / "run.json", json.dumps(run, sort_keys=True, ensure_ascii=False,
                                               allow_nan=False, indent=2) + "\n")


def fileEvidence(path, root):
    return {"path": path.relative_to(root).as_posix(), "byteSize": path.stat().st_size,
            "sha256": hashFile(path)}


def snapshotCode(runDir):
    projectRoot = Path(futsal.__file__).resolve().parents[2]
    def gitOutput(*arguments):
        return subprocess.run(["git", *arguments], cwd=projectRoot, capture_output=True,
                              text=True, check=True).stdout.strip()

    files = [*sorted((projectRoot / "src").rglob("*.py")),
             projectRoot / "pyproject.toml", projectRoot / "uv.lock"]
    evidence = []
    for source in files:
        target = runDir / "code" / source.relative_to(projectRoot)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        evidence.append(fileEvidence(target, runDir))
    status = gitOutput("status", "--porcelain")
    (runDir / "code" / "changes.patch").write_text(gitOutput("diff", "--binary", "HEAD"),
                                                  encoding="utf-8")
    return {"commit": gitOutput("rev-parse", "HEAD"), "dirty": bool(status), "status": status,
            "files": evidence}


def verifyBundleUnchanged(datasetDir, contract, contractHash, *, mediaStoreDir=None):
    if hashFile(datasetDir / "contract.json") != contractHash:
        raise ValueError("Prepared bundle contract changed during training.")
    expected = {item["path"] for item in contract["artifactEvidence"]} | {"contract.json"}
    if {path.relative_to(datasetDir).as_posix() for path in scanFiles(datasetDir)} != expected:
        raise ValueError("Prepared bundle file membership changed during training.")
    verifyEvidence(datasetDir, contract["artifactEvidence"])
    if contract.get("kind") == "sharedTaskView":
        from futsal.datasets.mediaStore import verifyAssets

        if mediaStoreDir is None:
            raise ValueError("Shared view integrity requires --mediaStoreDir.")
        verifyAssets(mediaStoreDir, contract["assets"])
    return {"artifactCount": len(expected), "hashesVerified": True, "membershipVerified": True}


def executeStages(config, selection, initialWeights, runDir, run, evidence, progress):
    inputWeights = initialWeights
    for index, stage in enumerate(config["stages"], start=1):
        if index <= len(run["stages"]) and run["stages"][index - 1]["status"] == "completed":
            inputWeights = runDir / run["stages"][index - 1]["results"]["checkpoints"]["best"]["path"]
            continue
        taskId = stage["taskId"]
        stageDir = runDir / "stages" / f"stage{index:02d}{taskId[0].upper()}{taskId[1:]}"
        state = {"taskId": taskId, "status": "running", "startedAt": utcNow(),
                 "directory": stageDir.relative_to(runDir).as_posix(),
                 "inputWeights": fileEvidence(inputWeights, runDir)}
        run["stages"].append(state)
        saveRun(runDir, run)
        progress(f"Training stage {index}/{len(config['stages'])}: {taskId}...")
        result = trainStage(config, selection["tasks"][taskId], stage, inputWeights,
                            stageDir, runDir, evidence)
        if hashFile(inputWeights) != state["inputWeights"]["sha256"]:
            raise ValueError("Input checkpoint changed during training.")
        verifyEvidence(runDir, evidence)
        state.update(status="completed", completedAt=utcNow(), results=result)
        saveRun(runDir, run)
        inputWeights = runDir / result["checkpoints"]["best"]["path"]


def runTraining(datasetDir, configPath, weightsPath, outputDir, progress=print, *, seed=None,
                continueFrom=None, syntheticDir=None, mediaStoreDir=None):
    datasetDir, configPath, weightsPath = (Path(path).resolve() for path in
                                           (datasetDir, configPath, weightsPath))
    projectRoot = Path(futsal.__file__).resolve().parents[2]
    protected = [configPath, weightsPath, projectRoot / "src", projectRoot / "configs",
                 projectRoot / "pyproject.toml", projectRoot / "uv.lock"]
    runDir = validateNewOutput(outputDir, datasetDir, protected)
    from futsal.datasets.mediaStore import isSharedView

    shared = isSharedView(datasetDir) or mediaStoreDir is not None
    if shared:
        if mediaStoreDir is None:
            raise ValueError("Shared task views require --mediaStoreDir.")
        mediaStoreDir = Path(mediaStoreDir).resolve()
        validateNewOutput(runDir, mediaStoreDir, [mediaStoreDir])
        from futsal.datasets.mediaStore import protectMediaOutput

        protectMediaOutput(runDir, mediaStoreDir)
    validateNewOutput(runDir, projectRoot / "data" / "raw")
    if continueFrom is not None:
        validateNewOutput(runDir, continueFrom, [continueFrom])
    config = loadTrainingConfig(configPath, readJson(datasetDir / "plan.json")["recipe"])
    sourceConfig = dict(config)
    if syntheticDir is not None:
        from futsal.training.syntheticTraining import requireCourtTraining

        requireCourtTraining(config)
        syntheticDir = Path(syntheticDir).resolve()
        validateNewOutput(runDir, syntheticDir, [syntheticDir])
        config = {**config, "sourcePolicy": "realWithSyntheticCourt"}
    if seed is not None:
        if type(seed) is not int or seed < 0:
            raise ValueError("Seed override must be a nonnegative integer.")
        config = {**config, "seed": seed}
    weights = verifyWeights(weightsPath, config["modelSize"])
    runDir.mkdir(parents=True, exist_ok=False)
    run = {"schemaVersion": 1, "purpose": config["purpose"], "status": "running",
           "startedAt": utcNow(), "configuration": config, "initialWeights": weights,
           "datasetDirectory": str(datasetDir), "stages": [], "testEvaluated": False}
    run["configurationId"] = hashJson(config)
    run["configurationOverrides"] = {} if seed is None else {"seed": seed}
    if syntheticDir is not None:
        run["configurationOverrides"]["sourcePolicy"] = "realWithSyntheticCourt"
    saveRun(runDir, run)
    contract, contractHash, evidence, syntheticSnapshot = None, None, None, None
    try:
        run["code"] = snapshotCode(runDir)
        inputs = runDir / "inputs"
        inputs.mkdir()
        for source, name in ((configPath, "training.yaml"), (weightsPath, weightsPath.name),
                             (weightsPath.with_name("weights.json"), "weights.json")):
            shutil.copyfile(source, inputs / name)
        if shared:
            for name in ("contract.json", "plan.json", "manifest.json"):
                shutil.copyfile(datasetDir / name, inputs / f"dataset{name[0].upper()}{name[1:]}")
        if loadTrainingConfig(inputs / "training.yaml", readJson(datasetDir / "plan.json")["recipe"]) != sourceConfig:
            raise ValueError("Training configuration changed during input snapshot.")
        writeJson(inputs / "resolvedTraining.json", config)
        if syntheticDir is not None:
            from futsal.training.syntheticTraining import archiveSyntheticSource

            recipe = readJson(datasetDir / "plan.json")["recipe"]
            classes = next(task["classes"] for task in recipe["tasks"] if task["taskId"] == "courtPoints")
            syntheticSnapshot = archiveSyntheticSource(syntheticDir, inputs, classes)
            if shared:
                from futsal.datasets.mediaRegistration import registerMedia

                registerMedia(mediaStoreDir, syntheticDir, progress=progress)
            run["syntheticSource"] = {"contractId": syntheticSnapshot[0]["contractId"],
                                      "imageCount": len(syntheticSnapshot[1])}
        if verifyWeights(inputs / weightsPath.name, config["modelSize"]) != weights:
            raise ValueError("Initial checkpoint changed during input snapshot.")
        run["inputEvidence"] = [fileEvidence(path, runDir) for path in sorted(inputs.iterdir())]
        run["environment"] = inspectEnvironment(config["device"])
        saveRun(runDir, run)
        if not run["environment"]["ready"]:
            raise ValueError("Training environment is not ready; inspect run.json findings.")
        progress("Verifying the complete immutable prepared bundle before training...")
        verificationOptions = {"mediaStoreDir": mediaStoreDir} if shared else {}
        run["datasetVerification"] = verifyDataset(datasetDir, progress=progress, **verificationOptions)
        contract = readJson(datasetDir / "contract.json")
        contractHash = hashFile(datasetDir / "contract.json")
        run["contractId"] = contract["contractId"]
        method = {key: value for key, value in config.items() if key not in
                  {"seed", "device", "evaluationDevice", "numThreads"}}
        methodInputs = {"configuration": method, "contractId": run["contractId"],
                        "initialWeightsSha256": weights["sha256"]}
        if syntheticSnapshot is not None:
            methodInputs["syntheticContractId"] = syntheticSnapshot[0]["contractId"]
        run["methodId"] = hashJson(methodInputs)
        selection = selectTrainingData(datasetDir, config, syntheticSnapshot)
        if shared:
            from futsal.training.sharedRuntime import prepareSharedRuntime

            evidence = prepareSharedRuntime(mediaStoreDir, runDir, selection)
        else:
            evidence = copyRuntimeData(datasetDir, runDir, selection, syntheticDir=syntheticDir)
        run["selectionSha256"] = hashFile(runDir / "selection.json")
        if continueFrom is not None:
            continueCompletedStages(continueFrom, runDir, run)
            run["inputEvidence"] = [fileEvidence(path, runDir) for path in sorted(inputs.iterdir())]
        saveRun(runDir, run)
        initializeBackend(runDir / "backend", config["numThreads"])
        executeStages(config, selection, inputs / weightsPath.name, runDir, run, evidence, progress)
        run["status"] = "completed"
    except (Exception, KeyboardInterrupt) as error:
        run["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        run["error"] = {"type": type(error).__name__, "message": str(error)}
        (runDir / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        if run["stages"] and run["stages"][-1]["status"] == "running":
            run["stages"][-1].update(status=run["status"], endedAt=utcNow())
        saveRun(runDir, run)
    finally:
        integrityErrors = []
        if syntheticSnapshot is not None:
            from futsal.datasets.syntheticCourt import verifySyntheticCourt

            try:
                if verifySyntheticCourt(syntheticDir) != syntheticSnapshot[0]:
                    raise ValueError("Synthetic source contract changed during training.")
                run["syntheticIntegrityAfterRun"] = True
            except (OSError, ValueError, KeyError) as error:
                integrityErrors.append(str(error))
        if contract is not None:
            progress("Checking prepared bundle hashes and membership after the run...")
            try:
                integrityOptions = {"mediaStoreDir": mediaStoreDir} if shared else {}
                run["bundleIntegrityAfterRun"] = verifyBundleUnchanged(datasetDir, contract, contractHash, **integrityOptions)
            except (OSError, ValueError) as error:
                integrityErrors.append(str(error))
        if evidence is not None:
            try:
                verifyEvidence(runDir, evidence)
                run["runtimeIntegrityAfterRun"] = True
            except (OSError, ValueError) as error:
                integrityErrors.append(str(error))
        if "inputEvidence" in run:
            try:
                verifyEvidence(runDir, run["inputEvidence"])
            except (OSError, ValueError) as error:
                integrityErrors.append(str(error))
        if integrityErrors:
            run["status"] = "failed"
            run["integrityErrors"] = integrityErrors
        run["endedAt"] = utcNow()
        run["artifacts"] = [fileEvidence(path, runDir) for folder in ("stages", "logs")
                            for path in sorted((runDir / folder).rglob("*")) if path.is_file()]
        saveRun(runDir, run)
        writeJson(runDir / "summary.json", {key: run[key] for key in
                  ("purpose", "status", "startedAt", "endedAt", "testEvaluated")})
    return run
