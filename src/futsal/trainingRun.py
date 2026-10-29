import json
import shutil
import subprocess
import traceback
from datetime import UTC, datetime
from pathlib import Path

from futsal.catalog import hashFile
from futsal.datasetFiles import readJson, validateNewOutput, verifyEvidence, writeJson
from futsal.datasetVerify import verifyDataset
from futsal.modelWeights import verifyWeights
from futsal.report import atomicWrite
from futsal.trainingBackend import initializeBackend, trainStage
from futsal.trainingConfig import loadTrainingConfig
from futsal.trainingData import copyRuntimeData, selectTrainingData
from futsal.trainingEnvironment import inspectEnvironment
from futsal.yoloInventory import scanFiles


def utcNow():
    return datetime.now(UTC).isoformat()


def saveRun(runDir, run):
    atomicWrite(runDir / "run.json", json.dumps(run, sort_keys=True, ensure_ascii=False,
                                               allow_nan=False, indent=2) + "\n")


def fileEvidence(path, root):
    return {"path": path.relative_to(root).as_posix(), "byteSize": path.stat().st_size,
            "sha256": hashFile(path)}


def snapshotCode(runDir):
    projectRoot = Path(__file__).resolve().parents[2]
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


def verifyBundleUnchanged(datasetDir, contract, contractHash):
    if hashFile(datasetDir / "contract.json") != contractHash:
        raise ValueError("Prepared bundle contract changed during training.")
    expected = {item["path"] for item in contract["artifactEvidence"]} | {"contract.json"}
    if {path.relative_to(datasetDir).as_posix() for path in scanFiles(datasetDir)} != expected:
        raise ValueError("Prepared bundle file membership changed during training.")
    verifyEvidence(datasetDir, contract["artifactEvidence"])
    return {"artifactCount": len(expected), "hashesVerified": True, "membershipVerified": True}


def executeStages(config, selection, initialWeights, runDir, run, evidence, progress):
    inputWeights = initialWeights
    for index, stage in enumerate(config["stages"], start=1):
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


def runTraining(datasetDir, configPath, weightsPath, outputDir, progress=print):
    datasetDir, configPath, weightsPath = (Path(path).resolve() for path in
                                           (datasetDir, configPath, weightsPath))
    projectRoot = Path(__file__).resolve().parents[2]
    protected = [configPath, weightsPath, projectRoot / "src", projectRoot / "configs",
                 projectRoot / "pyproject.toml", projectRoot / "uv.lock"]
    runDir = validateNewOutput(outputDir, datasetDir, protected)
    validateNewOutput(runDir, projectRoot.parent / "Python")
    config = loadTrainingConfig(configPath, readJson(datasetDir / "plan.json")["recipe"])
    weights = verifyWeights(weightsPath, config["modelSize"])
    runDir.mkdir(parents=True, exist_ok=False)
    run = {"schemaVersion": 1, "purpose": config["purpose"], "status": "running",
           "startedAt": utcNow(), "configuration": config, "initialWeights": weights,
           "datasetDirectory": str(datasetDir), "stages": [], "testEvaluated": False}
    saveRun(runDir, run)
    contract, contractHash, evidence = None, None, None
    try:
        run["code"] = snapshotCode(runDir)
        inputs = runDir / "inputs"
        inputs.mkdir()
        for source, name in ((configPath, "training.yaml"), (weightsPath, weightsPath.name),
                             (weightsPath.with_name("weights.json"), "weights.json")):
            shutil.copyfile(source, inputs / name)
        if loadTrainingConfig(inputs / "training.yaml", readJson(datasetDir / "plan.json")["recipe"]) != config:
            raise ValueError("Training configuration changed during input snapshot.")
        if verifyWeights(inputs / weightsPath.name, config["modelSize"]) != weights:
            raise ValueError("Initial checkpoint changed during input snapshot.")
        run["inputEvidence"] = [fileEvidence(path, runDir) for path in sorted(inputs.iterdir())]
        run["environment"] = inspectEnvironment(config["device"])
        saveRun(runDir, run)
        if not run["environment"]["ready"]:
            raise ValueError("Training environment is not ready; inspect run.json findings.")
        progress("Verifying the complete immutable prepared bundle before training...")
        run["datasetVerification"] = verifyDataset(datasetDir, progress=progress)
        contract = readJson(datasetDir / "contract.json")
        contractHash = hashFile(datasetDir / "contract.json")
        run["contractId"] = contract["contractId"]
        selection = selectTrainingData(datasetDir, config)
        evidence = copyRuntimeData(datasetDir, runDir, selection)
        run["selectionSha256"] = hashFile(runDir / "selection.json")
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
        if contract is not None:
            progress("Checking prepared bundle hashes and membership after the run...")
            try:
                run["bundleIntegrityAfterRun"] = verifyBundleUnchanged(datasetDir, contract, contractHash)
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
