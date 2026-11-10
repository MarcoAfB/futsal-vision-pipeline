import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from resultTransfer import verifyRun
from transferArchive import hashFile, readJson, requireNewPath


def gitOutput(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()


def validatePilot(pilotDir, config, selection, weights, contractId, commit):
    from futsal.training.trainingConfig import stageParameters

    root = Path(pilotDir).resolve()
    pilot = verifyRun(root)
    if pilot["status"] != "completed" or pilot["purpose"] != "scientificPilot" or (
        pilot.get("contractId") != contractId or pilot["initialWeights"] != weights
        or pilot["code"]["commit"] != commit or pilot["code"]["dirty"]
        or not pilot.get("runtimeIntegrityAfterRun") or pilot.get("integrityErrors")
        or not pilot.get("bundleIntegrityAfterRun", {}).get("hashesVerified")
    ):
        raise ValueError("Scientific launch requires a completed, verified pilot of the same code/data/weights.")
    if pilot["configuration"]["device"] != config["device"] or pilot["configuration"].get("evaluationDevice") != config["evaluationDevice"]:
        raise ValueError("Scientific and pilot devices must agree.")
    priorSelection = readJson(root / "selection.json")
    if priorSelection.get("selectionMethod") != "fullPerTask" or priorSelection["tasks"] != selection["tasks"]:
        raise ValueError("Scientific and pilot full per-task selections must agree.")
    previousStages = pilot["configuration"]["stages"]
    if [stage["taskId"] for stage in previousStages] != [stage["taskId"] for stage in config["stages"]] or [
        stage["taskId"] for stage in pilot["stages"]
    ] != [stage["taskId"] for stage in previousStages] or any(
        stage["status"] != "completed" for stage in pilot["stages"]
    ):
        raise ValueError("Pilot must complete the entire declared curriculum.")
    for before, after in zip(previousStages, config["stages"], strict=True):
        first, second = stageParameters(pilot["configuration"], before), stageParameters(config, after)
        for key in ("patience", "warmup_epochs"):
            first.pop(key)
            second.pop(key)
        if first != second:
            raise ValueError("Resource/optimizer/augmentation parameters changed after the pilot; repeat it.")
    if pilot["configuration"]["modelSize"] != config["modelSize"] or pilot["configuration"].get("sourcePolicy") != "realOnly":
        raise ValueError("Pilot model scale and real-only source policy must agree.")
    return {"directory": str(root), "runSha256": hashFile(root / "run.json")}


def launch(arguments):
    from futsal.datasets.datasetFiles import readJson as readProjectJson
    from futsal.inventory.catalog import resolveSourcePath
    from futsal.training.modelWeights import verifyWeights
    from futsal.training.trainingConfig import loadTrainingConfig, resolveTrainingConfig
    from futsal.training.trainingData import requireDiskSpace, selectTrainingData

    project = Path(__file__).resolve().parents[1]
    dataset, configPath, weightsPath = (path.resolve() for path in
                                      (arguments.datasetDir, arguments.config, arguments.weights))
    runDir = requireNewPath(arguments.outputDir, [dataset, project / "src", project / "configs", configPath, weightsPath])
    for outputRoot in (arguments.reportRoot.resolve(), arguments.workRoot.resolve()):
        for source in (dataset, runDir, project / "src", project / "configs"):
            if outputRoot.is_relative_to(source) or source.is_relative_to(outputRoot):
                raise ValueError("Report/work roots must be separate from inputs and the training attempt.")
    protected = [dataset, project / "src", project / "configs", configPath, weightsPath, runDir,
                 arguments.reportRoot, arguments.workRoot]
    logDir = requireNewPath(arguments.logDir, protected)
    logDir.mkdir(parents=True)
    state = {"schemaVersion": 1, "status": "preflight", "startedAt": datetime.now(UTC).isoformat(),
             "pythonExecutable": sys.executable, "pythonVersion": sys.version,
             "expectedCommit": arguments.expectedCommit, "runDirectory": str(runDir), "commands": []}
    env = {**os.environ, "YOLO_OFFLINE": "true", "YOLO_AUTOINSTALL": "false", "UV_OFFLINE": "1"}

    def runCommand(command, name):
        state["commands"].append(command)
        print(f"Running {name}; logs: {logDir}", flush=True)
        with (logDir / f"{name}.stdout.log").open("xb") as output, (logDir / f"{name}.stderr.log").open("xb") as error:
            result = subprocess.run(command, cwd=project, env=env, stdout=output, stderr=error, check=False)
        state[f"{name}ExitCode"] = result.returncode
        return result.returncode

    try:
        actualCommit = gitOutput(project, "rev-parse", "HEAD")
        if actualCommit != arguments.expectedCommit or gitOutput(project, "status", "--porcelain"):
            raise ValueError("Launch requires the expected commit and a clean checkout.")
        state["codeCommit"] = actualCommit
        sourceDir = logDir / "source"
        sourceDir.mkdir()
        state["launcherSourceHashes"] = {}
        for name in ("runServerTraining.py", "resultTransfer.py", "transferArchive.py"):
            target = sourceDir / name
            shutil.copyfile(Path(__file__).with_name(name), target)
            state["launcherSourceHashes"][name] = hashFile(target)
        plan = readProjectJson(dataset / "plan.json")
        config = resolveTrainingConfig(loadTrainingConfig(configPath, plan["recipe"]),
                                       seed=arguments.seed, modelSize=arguments.modelSize)
        if config["schemaVersion"] != 2 or config["device"] == "cpu":
            raise ValueError("Server launcher requires an explicit scientific/pilot single-GPU profile.")
        weights = verifyWeights(weightsPath, config["modelSize"])
        selection = selectTrainingData(dataset, config)
        required = sum(resolveSourcePath(dataset, row[f"{kind}Path"]).stat().st_size
                       for task in selection["tasks"].values() for split in ("training", "validation")
                       for row in task[split] for kind in ("image", "label"))
        state["runtimeDisk"] = requireDiskSpace(runDir, required)
        state["configSha256"] = hashFile(configPath)
        state["initialWeightsSha256"] = weights["sha256"]
        state["seed"] = config["seed"]
        state["modelSize"] = config["modelSize"]
        if config["purpose"] == "scientificTraining":
            if arguments.pilotRun is None:
                raise ValueError("Provide --pilotRun after the resource pilot and profile freeze.")
            state["pilot"] = validatePilot(arguments.pilotRun, config, selection, weights,
                                            readJson(dataset / "contract.json")["contractId"], actualCommit)
        for index, device in enumerate(dict.fromkeys((config["device"], config["evaluationDevice"]))):
            code = runCommand([sys.executable, "-m", "futsal", "inspectEnvironment", "--device", device,
                               "--outputDir", str(logDir / f"environment{index}")], f"environment{index}")
            if code:
                state["status"] = "environmentFailed"
                return code
        command = [sys.executable, "-m", "futsal", "train", "--datasetDir", str(dataset),
                   "--config", str(configPath), "--weights", str(weightsPath), "--outputDir", str(runDir),
                   "--reportRoot", str(arguments.reportRoot.resolve()), "--workRoot", str(arguments.workRoot.resolve())]
        if arguments.seed is not None:
            command += ["--seed", str(arguments.seed)]
        if arguments.modelSize is not None:
            command += ["--modelSize", arguments.modelSize]
        if arguments.continueFrom is not None:
            command += ["--continueFrom", str(arguments.continueFrom.resolve())]
        state["status"] = "running"
        (logDir / "launchStarted.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        code = runCommand(command, "training")
        state["status"] = "completed" if code == 0 else "trainingOrPostprocessingFailed"
        return code
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        state.update(status="preflightFailed", error=str(error))
        print(f"Launch failed: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        state["status"] = "interrupted"
        return 130
    finally:
        state["endedAt"] = datetime.now(UTC).isoformat()
        (logDir / "launch.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    from futsal.training.modelWeights import modelSizes

    parser = argparse.ArgumentParser(description="Launch a prepared offline GPU curriculum from a clean, identified checkout.")
    for name in ("datasetDir", "config", "weights", "outputDir", "reportRoot", "workRoot", "logDir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expectedCommit", required=True)
    parser.add_argument("--pilotRun", type=Path)
    parser.add_argument("--continueFrom", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--modelSize", choices=modelSizes)
    return launch(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
