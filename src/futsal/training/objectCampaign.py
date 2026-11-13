import copy
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import yaml

from futsal.datasets.datasetFiles import readJson, verifyEvidence, writeJson
from futsal.inventory.catalog import hashFile
from futsal.reporting.report import atomicWrite
from futsal.training.memoryRecovery import isMemoryFailure
from futsal.training.trainingConfig import loadTrainingConfig, stageParameters
from futsal.training.trainingRun import utcNow


@contextmanager
def campaignLock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "campaign.lock"
    try:
        stream = path.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise ValueError("Campaign is locked; verify that its worker stopped before removing campaign.lock.") from error
    try:
        with stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        path.unlink()


def saveCampaign(directory, state):
    atomicWrite(directory / "campaign.json", json.dumps(state, indent=2, sort_keys=True) + "\n")


def verifyAttempt(directory):
    run = readJson(directory / "run.json")
    if run["status"] not in {"completed", "failed", "interrupted"}:
        raise ValueError("Unfinished attempt requires inspection; no concurrent or unsealed retry is allowed.")
    for evidence in (run["artifacts"], run["inputEvidence"], run["code"]["files"]):
        verifyEvidence(directory, evidence)
    if hashFile(directory / "selection.json") != run["selectionSha256"]:
        raise ValueError("Attempt selection changed.")
    return run


def nextAttempt(runRoot, size):
    index = 1
    while (runRoot / f"yolo26{size}RealAttempt{index:02d}").exists():
        index += 1
    return runRoot / f"yolo26{size}RealAttempt{index:02d}"


def retryConfiguration(run):
    if run["status"] != "failed" or run.get("integrityErrors") or not (
        run.get("batchRecovery") == "fixed" and isMemoryFailure(run.get("error", {}).get("causes", []))
    ):
        return None
    index = sum(stage["status"] == "completed" for stage in run["stages"])
    if index >= len(run["configuration"]["stages"]) or len(run["stages"]) != index + 1:
        return None
    config = copy.deepcopy(run["configuration"])
    stage = config["stages"][index]
    batch = stageParameters(config, stage)["batch"]
    if batch not in {8, 4, 2}:
        return None
    stage.setdefault("parameters", {})["batch"] = batch // 2
    return config


def runWorker(script, project, campaignDir, payload, name):
    request = campaignDir / f"{name}.json"
    if request.exists():
        raise ValueError("Worker request already exists; allocate a fresh worker identity.")
    writeJson(request, payload)
    env = {**os.environ, "YOLO_OFFLINE": "true", "YOLO_AUTOINSTALL": "false", "UV_OFFLINE": "1"}
    command = [sys.executable, str(script), "worker", "--request", str(request)]
    print(f"Running {name}; log: {campaignDir / (name + '.log')}", flush=True)
    with (campaignDir / f"{name}.log").open("xb") as log:
        process = subprocess.Popen(command, cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            return process.wait()
        except KeyboardInterrupt:
            # The terminal also delivers Ctrl-C to the worker. Wait for its run seal.
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=15)
            raise


def worker(payload):
    if any(hashFile(Path(path)) != digest for path, digest in payload.get("frozenFiles", {}).items()):
        raise ValueError("Frozen worker source or configuration changed.")
    if payload["action"] == "training" and payload.get("configSha256"):
        if hashFile(Path(payload["config"])) != payload["configSha256"]:
            raise ValueError("Worker training configuration changed.")
    if payload["action"] == "environment":
        from futsal.training.trainingEnvironment import inspectEnvironment

        result = inspectEnvironment(payload["device"])
        writeJson(Path(payload["output"]), result)
        return 0 if result["ready"] else 2
    if payload["action"] == "training":
        from futsal.training.trainingRun import runTraining

        result = runTraining(payload["datasetDir"], payload["config"], payload["weights"], payload["runDir"],
            mediaStoreDir=payload["mediaStoreDir"], batchRecovery="fixed",
            memoryRetryFrom=payload.get("memoryRetryFrom"), continueFrom=payload.get("continueFrom"))
        return 130 if result["status"] == "interrupted" else 0 if result["status"] == "completed" else 1
    if payload["action"] != "report":
        raise ValueError("Unsupported worker action.")
    from futsal.reporting.researchCli import generateRunReport
    from futsal.reporting.researchReport import renderReport, verifyEvaluation

    root = Path(payload["reportRoot"]) / Path(payload["runDir"]).name
    for directory in sorted(root.glob("runValidation-*"), reverse=True):
        if (directory / "evaluationContract.json").is_file() and readJson(directory / "evaluation.json")["status"] == "completed":
            verifyEvaluation(directory)
            if (directory / "reportState.json").exists():
                state = readJson(directory / "reportState.json")
                if state["status"] == "completed":
                    verifyEvidence(directory, state["artifacts"])
                    return 0
            if (directory / "reportState.json").exists() or (directory / "report.md").exists():
                from futsal.evaluation.evaluationRun import newEvaluationPath

                renderReport(directory, newEvaluationPath(payload["runDir"], reportRoot=payload["reportRoot"]))
            else:
                renderReport(directory)
            return 0
    completed, path = generateRunReport(payload["runDir"], payload["datasetDir"], payload["evaluationConfig"],
        reportRoot=payload["reportRoot"], workRoot=payload["workRoot"], mediaStoreDir=payload["mediaStoreDir"])
    diagnostic = Path(path).name == "report.md" and readJson(Path(payload["runDir"]) / "run.json")["status"] != "completed"
    return 0 if completed or diagnostic else 1


def campaignIdentity(project, dataset, store, configPath, evaluationConfig, sizes, weightsRoot, workDir, campaignDir):
    from futsal.datasets.sharedViews import verifyView
    from futsal.training.modelWeights import verifyWeights
    from futsal.training.trainingData import selectTrainingData

    def git(*arguments):
        return subprocess.run(["git", *arguments], cwd=project, capture_output=True, text=True, check=True).stdout.strip()

    if git("status", "--porcelain"):
        raise ValueError("Campaign requires a clean frozen checkout.")
    verification = verifyView(dataset, store)
    config = loadTrainingConfig(configPath, readJson(dataset / "plan.json")["recipe"])
    if config["purpose"] != "scientificTraining" or config["seed"] != 42 or config["device"] == "cpu" or (
        [stage["taskId"] for stage in config["stages"]] != ["genericObjects", "roleObjects", "detailedObjects"]
        or any(stageParameters(config, stage)["batch"] != 8 for stage in config["stages"])
    ):
        raise ValueError("Campaign requires the scientific 2 -> 3 -> 5 seed42 GPU profile, starting each stage at batch8.")
    if not sizes or len(sizes) != len(set(sizes)):
        raise ValueError("Campaign model sizes must be a nonempty unique sequence.")
    selectTrainingData(dataset, config)
    files = [*sorted((project / "src").rglob("*.py")), *sorted((project / "scripts").glob("*.py")),
             project / "pyproject.toml", project / "uv.lock", configPath, evaluationConfig]
    identity = {"commit": git("rev-parse", "HEAD"), "files": {str(path): hashFile(path) for path in files},
        "contractSha256": hashFile(dataset / "contract.json"),
        "contractId": verification["contractId"], "datasetDir": str(dataset), "mediaStoreDir": str(store),
        "configuration": config, "modelSizes": sizes, "workDir": str(workDir), "campaignDir": str(campaignDir),
        "weights": {size: {"path": str(weightsRoot / f"yolo26{size}" / f"yolo26{size}.pt"),
            "metadata": verifyWeights(weightsRoot / f"yolo26{size}" / f"yolo26{size}.pt", size)} for size in sizes}}
    return identity


def trainCampaign(project, script, *, datasetDir, mediaStoreDir, workDir, campaignDir, configPath,
                  evaluationConfig, weightsRoot, modelSizes, resume=False):
    from futsal.datasets.mediaStore import readBindings

    dataset, store, work, directory, configPath, evaluationConfig, weightsRoot = (Path(path).resolve() for path in
        (datasetDir, mediaStoreDir, workDir, campaignDir, configPath, evaluationConfig, weightsRoot))
    runRoot, reportRoot, workRoot = (work / name / directory.name for name in ("runs", "reports", "evaluationWork"))
    protected = [dataset, store, weightsRoot, project / "src", project / "configs", project / "scripts", project / "data/raw"]
    protected += [Path(path).resolve() for path in readBindings(store).get("providers", [])]
    outputs = [directory, runRoot, reportRoot, workRoot]
    for target in outputs:
        for source in [*protected, *(other for other in outputs if other != target)]:
            if target.is_relative_to(source) or source.is_relative_to(target):
                raise ValueError("Campaign outputs must be separate from inputs and each other.")
        if target.exists() and not target.is_dir():
            raise ValueError("Campaign output roots must be directories.")
    with campaignLock(directory):
        identity = campaignIdentity(project, dataset, store, configPath, evaluationConfig, modelSizes, weightsRoot, work, directory)
        statePath = directory / "campaign.json"
        if statePath.exists():
            if not resume:
                raise ValueError("Campaign exists; use --resume with the same inputs.")
            state = readJson(statePath)
            if state["identity"] != identity:
                raise ValueError("Campaign code, configuration, data or weights changed.")
            for model in state["models"].values():
                for attempt in model["attempts"]:
                    if hashFile(Path(attempt["config"])) != attempt["configSha256"]:
                        raise ValueError("Prior attempt configuration changed.")
                    if attempt.get("runSha256"):
                        verifyAttempt(Path(attempt["runDir"]))
                        if hashFile(Path(attempt["runDir"]) / "run.json") != attempt["runSha256"]:
                            raise ValueError("Prior attempt record changed.")
        else:
            if resume:
                raise ValueError("No campaign to resume.")
            if any(path.exists() for path in (runRoot, reportRoot, workRoot)):
                raise ValueError("Campaign run/report/work roots already exist; choose a new campaign name.")
            state = {"schemaVersion": 1, "startedAt": utcNow(), "identity": identity,
                     "roots": {"runs": str(runRoot), "reports": str(reportRoot), "evaluationWork": str(workRoot)},
                     "models": {size: {"status": "pending", "attempts": []} for size in modelSizes}}
        saveCampaign(directory, state)
        sequence = len(list(directory.glob("worker*.json")))

        def dispatch(payload):
            nonlocal sequence
            if any(hashFile(Path(path)) != digest for path, digest in identity["files"].items()):
                raise ValueError("Frozen campaign source or configuration changed.")
            if hashFile(dataset / "contract.json") != identity["contractSha256"]:
                raise ValueError("Frozen campaign dataset contract changed.")
            if payload["action"] == "training":
                from futsal.training.modelWeights import verifyWeights

                size = yaml.safe_load(Path(payload["config"]).read_text(encoding="utf-8"))["modelSize"]
                if verifyWeights(payload["weights"], size) != identity["weights"][size]["metadata"]:
                    raise ValueError("Frozen campaign initial weights changed.")
            sequence += 1
            return runWorker(script, project, directory, {**payload, "frozenFiles": identity["files"]}, f"worker{sequence:04d}")

        try:
            state["status"] = "preflight"
            saveCampaign(directory, state)
            for index, device in enumerate(dict.fromkeys((identity["configuration"]["device"], identity["configuration"]["evaluationDevice"]))):
                if dispatch({"action": "environment", "device": device, "output": str(directory / f"environment{index}.json")}):
                    raise ValueError("Offline environment preflight failed; inspect campaign logs.")
            state["status"] = "running"
            for size in modelSizes:
                model = state["models"][size]
                processModel(model, size, identity, directory, runRoot, reportRoot, workRoot,
                             evaluationConfig, dispatch, lambda: saveCampaign(directory, state))
            state["status"] = "completed" if all(model["status"] == "completed" for model in state["models"].values()) else "completedWithFailures"
        except KeyboardInterrupt:
            state["status"] = "interrupted"
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            state.update(status="failed", error=str(error))
            raise
        finally:
            state["endedAt"] = utcNow()
            saveCampaign(directory, state)
        return 130 if state["status"] == "interrupted" else 0 if state["status"] == "completed" else 1


def processModel(model, size, identity, campaignDir, runRoot, reportRoot, workRoot, evaluationConfig, dispatch, save):
    config = copy.deepcopy(identity["configuration"])
    config["modelSize"] = size
    continuation = {}
    while True:
        if model["attempts"]:
            attempt = model["attempts"][-1]
            previousDir = Path(attempt["runDir"])
            if not (previousDir / "run.json").is_file():
                raise ValueError("Allocated attempt has no run record; inspect worker logs before resuming.")
            run = verifyAttempt(previousDir)
            if hashFile(Path(attempt["config"])) != attempt["configSha256"]:
                raise ValueError("Attempt configuration changed.")
            if attempt.get("runSha256") and attempt["runSha256"] != hashFile(previousDir / "run.json"):
                raise ValueError("Prior attempt metadata changed.")
            attempt.update(trainingStatus=run["status"], runSha256=hashFile(previousDir / "run.json"))
            save()
            if run["status"] == "completed":
                if attempt.get("reportStatus") != "completed":
                    code = dispatch({"action": "report", "runDir": str(previousDir), "datasetDir": identity["datasetDir"],
                        "mediaStoreDir": identity["mediaStoreDir"], "evaluationConfig": str(evaluationConfig),
                        "reportRoot": str(reportRoot), "workRoot": str(workRoot)})
                    if code in {130, -2}:
                        raise KeyboardInterrupt
                    attempt["reportStatus"] = "completed" if code == 0 else "failed"
                model["status"] = "completed" if attempt["reportStatus"] == "completed" else "reportFailed"
                save()
                return
            if attempt.get("diagnosticReportStatus") != "completed":
                code = dispatch({"action": "report", "runDir": str(previousDir), "datasetDir": identity["datasetDir"],
                    "mediaStoreDir": identity["mediaStoreDir"], "evaluationConfig": str(evaluationConfig),
                    "reportRoot": str(reportRoot), "workRoot": str(workRoot)})
                if code in {130, -2}:
                    raise KeyboardInterrupt
                attempt["diagnosticReportStatus"] = "completed" if code == 0 else "failed"
                save()
            if run["status"] == "interrupted":
                config = copy.deepcopy(run["configuration"])
                continuation = {"continueFrom": str(previousDir)} if any(stage["status"] == "completed" for stage in run["stages"]) else {}
            else:
                config = retryConfiguration(run)
                if config is None:
                    model["status"] = "failed"
                    save()
                    return
                continuation = {"memoryRetryFrom": str(previousDir)}
        output = nextAttempt(runRoot, size)
        configFile = campaignDir / f"{output.name}.yaml"
        with configFile.open("x", encoding="utf-8") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
        attempt = {"runDir": str(output), "config": str(configFile), "configSha256": hashFile(configFile),
                   "requestedBatches": {stage["taskId"]: stageParameters(config, stage)["batch"] for stage in config["stages"]},
                   **continuation}
        model["attempts"].append(attempt)
        model["status"] = "running"
        save()
        code = dispatch({"action": "training", "runDir": str(output), "config": str(configFile),
            "configSha256": attempt["configSha256"],
            "weights": identity["weights"][size]["path"], "datasetDir": identity["datasetDir"],
            "mediaStoreDir": identity["mediaStoreDir"], **continuation})
        if code in {130, -2}:
            raise KeyboardInterrupt
