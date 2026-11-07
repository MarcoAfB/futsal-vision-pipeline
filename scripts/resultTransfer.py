import argparse
import sys
import tarfile
from pathlib import Path

from transferArchive import (
    checkArchive,
    filePath,
    hashFile,
    hashJson,
    packArchive,
    readJson,
    regularFiles,
    safeRelative,
    unpackArchive,
)


def verifyFiles(root, evidence):
    for item in evidence:
        path = filePath(Path(root) / str(safeRelative(item["path"])))
        if path.is_symlink() or not path.is_file() or path.stat().st_size != item["byteSize"] or hashFile(path) != item["sha256"]:
            raise ValueError(f"Archived evidence changed: {item['path']}.")


def verifyRun(root):
    root = Path(root)
    run = readJson(root / "run.json")
    if run["status"] not in {"completed", "failed", "interrupted"}:
        raise ValueError("Results require a finished training attempt.")
    for evidence in (run.get("artifacts", []), run.get("inputEvidence", []), run.get("code", {}).get("files", [])):
        verifyFiles(root, evidence)
    if run.get("selectionSha256") and hashFile(root / "selection.json") != run["selectionSha256"]:
        raise ValueError("Archived training selection changed.")
    return run


def verifyReport(root):
    root = Path(root)
    if not (root / "reportState.json").is_file() and not (root / "evaluationContract.json").is_file():
        raise ValueError("Report directory has no evaluation seal or presentation state.")
    state = readJson(root / "reportState.json") if (root / "reportState.json").is_file() else {}
    if state.get("status") == "running":
        raise ValueError("An active presentation cannot be transported.")
    if (root / "evaluationContract.json").is_file():
        contract = readJson(root / "evaluationContract.json")
        if contract["contractId"] != hashJson({key: value for key, value in contract.items() if key != "contractId"}) or (
            contract["evaluationSha256"] != hashFile(root / "evaluation.json")
        ):
            raise ValueError("Evaluation seal mismatch.")
        expected = {item["path"] for item in contract["artifacts"]}
        if expected != {f"evidence/{name}" for name in regularFiles(root / "evidence")}:
            raise ValueError("Evaluation evidence membership mismatch.")
        verifyFiles(root, contract["artifacts"])
        evaluation = readJson(root / "evaluation.json")
        if evaluation["status"] == "running":
            raise ValueError("An active evaluation cannot be transported.")
        trainingHash = evaluation.get("trainingRunSha256")
    elif (root / "trainingRun.json").is_file():
        trainingHash = hashFile(root / "trainingRun.json")
    else:
        trainingHash = None
    if state.get("artifacts"):
        verifyFiles(root, state["artifacts"])
        if set(regularFiles(root)) != {item["path"] for item in state["artifacts"]} | {"reportState.json"}:
            raise ValueError("Presentation membership mismatch.")
    return trainingHash


def packResults(runDir, reportDirs, archive, receipt, compression="gzip", launchDir=None):
    runDir = Path(runDir).resolve()
    run = verifyRun(runDir)
    included = {}
    for name in ("run.json", "summary.json", "selection.json", "runtimeEvidence.json", "failure.txt"):
        if (runDir / name).is_file():
            included[f"run/{name}"] = runDir / name
    for folder in ("inputs", "code", "stages", "logs"):
        included.update({f"run/{folder}/{name}": path for name, path in regularFiles(runDir / folder).items()})
    runHash = hashFile(runDir / "run.json")
    reports = []
    for index, directory in enumerate(reportDirs, start=1):
        directory = Path(directory).resolve()
        if directory.is_relative_to(runDir) or runDir.is_relative_to(directory):
            raise ValueError("Reports and run roots must be separate.")
        trainingHash = verifyReport(directory)
        if trainingHash is not None and trainingHash != runHash:
            raise ValueError("Report belongs to different training run metadata.")
        prefix = f"reports/report{index:02d}"
        included.update({f"{prefix}/{name}": path for name, path in regularFiles(directory).items()})
        reports.append({"path": prefix, "originalName": directory.name})
    if launchDir is not None:
        included.update({f"launch/{name}": path for name, path in regularFiles(Path(launchDir)).items()})
    sources = {name: (path, {"path": name, "byteSize": path.stat().st_size, "sha256": hashFile(path)})
               for name, path in included.items()}
    metadata = {"runSha256": runHash, "runStatus": run["status"], "purpose": run["purpose"], "reports": reports,
                "excluded": ["run/runtimeData", "run/backend", "evaluationWork"],
                "excludedRuntimeBytesVerifiedAtReceiver": False,
                "toolSourceHashes": {name: hashFile(Path(__file__).with_name(name))
                                     for name in ("resultTransfer.py", "transferArchive.py")}}
    return packArchive(sources, archive, receipt, "curriculumResultsTransfer", metadata, compression,
                       protected=[runDir, *reportDirs, *([launchDir] if launchDir else [])])


def verifyAttachedReport(root, configurationSeal, runHashes):
    trainingHash = verifyReport(root)
    if trainingHash is not None:
        if trainingHash not in runHashes:
            raise ValueError("Attached report belongs to another component run.")
        return
    evaluation = readJson(Path(root) / "evaluation.json")
    if evaluation.get("kind") != "configurationComparison" or evaluation.get("candidateContractId") != configurationSeal:
        raise ValueError("Attached comparison belongs to a different configuration evaluation.")
    nested = Path(root) / "evidence/candidate"
    verifyReport(nested)
    if readJson(nested / "evaluationContract.json")["contractId"] != configurationSeal:
        raise ValueError("Comparison candidate evidence differs from the joint evaluation.")
    baseline = Path(root) / "evidence/baseline"
    verifyReport(baseline)
    if readJson(baseline / "evaluationContract.json")["contractId"] != evaluation["baselineContractId"]:
        raise ValueError("Comparison baseline evidence differs from its recorded seal.")


def configurationRuns(configurationDir):
    root = Path(configurationDir)
    verifyReport(root)
    state = readJson(root / "evaluation.json")
    if state.get("kind") != "configurationEvaluation" or state["status"] != "completed":
        raise ValueError("Configuration results require a completed joint evaluation.")
    if [item["taskId"] for item in state["components"]] != ["detailedObjects", "courtPoints"]:
        raise ValueError("Configuration component partition differs from independent objects/court.")
    return state


def packConfigurationResults(configurationDir, reportDirs, archive, receipt, compression="gzip", launchDir=None):
    root = Path(configurationDir).resolve()
    state = configurationRuns(root)
    included = {f"configuration/{name}": path for name, path in regularFiles(root).items()}
    runs, protected = [], [root, *reportDirs]
    for component in state["components"]:
        directory = Path(component["runDirectory"]).resolve()
        run = verifyRun(directory)
        if run != component["run"] or hashFile(directory / "run.json") != component["runSha256"]:
            raise ValueError("Component metadata differ from the joint evaluation.")
        prefix = f"runs/{component['taskId']}"
        for name in ("run.json", "summary.json", "selection.json", "runtimeEvidence.json", "failure.txt"):
            if (directory / name).is_file():
                included[f"{prefix}/{name}"] = directory / name
        for folder in ("inputs", "code", "stages", "logs"):
            included.update({f"{prefix}/{folder}/{name}": path for name, path in regularFiles(directory / folder).items()})
        runs.append({"taskId": component["taskId"], "path": prefix, "runSha256": component["runSha256"]})
        protected.append(directory)
    configurationSeal = readJson(root / "evaluationContract.json")["contractId"]
    reports = []
    for index, directory in enumerate(map(Path, reportDirs), start=1):
        verifyAttachedReport(directory, configurationSeal, {item["runSha256"] for item in runs})
        prefix = f"reports/report{index:02d}"
        included.update({f"{prefix}/{name}": path for name, path in regularFiles(directory).items()})
        reports.append({"path": prefix, "originalName": directory.name})
    if launchDir is not None:
        included.update({f"launch/{name}": path for name, path in regularFiles(Path(launchDir)).items()})
        protected.append(launchDir)
    sources = {name: (path, {"path": name, "byteSize": path.stat().st_size, "sha256": hashFile(path)}) for name, path in included.items()}
    metadata = {"configurationContractId": configurationSeal, "runs": runs, "reports": reports,
        "excluded": ["runs/*/runtimeData", "runs/*/backend", "evaluationWork"],
        "excludedRuntimeBytesVerifiedAtReceiver": False,
        "toolSourceHashes": {name: hashFile(Path(__file__).with_name(name)) for name in ("resultTransfer.py", "transferArchive.py")}}
    return packArchive(sources, archive, receipt, "configurationResultsTransfer", metadata, compression, protected)


def validateConfigurationResults(root, receipt):
    state = configurationRuns(root / "configuration")
    seal = readJson(root / "configuration/evaluationContract.json")["contractId"]
    if seal != receipt["metadata"]["configurationContractId"]:
        raise ValueError("Extracted configuration identity mismatch.")
    byTask = {item["taskId"]: item for item in state["components"]}
    if {item["taskId"] for item in receipt["metadata"]["runs"]} != set(byTask):
        raise ValueError("Extracted component membership mismatch.")
    for item in receipt["metadata"]["runs"]:
        directory = root / str(safeRelative(item["path"]))
        run = verifyRun(directory)
        if run != byTask[item["taskId"]]["run"] or hashFile(directory / "run.json") != item["runSha256"] or item["runSha256"] != byTask[item["taskId"]]["runSha256"]:
            raise ValueError("Extracted component metadata changed.")
    for report in receipt["metadata"]["reports"]:
        verifyAttachedReport(root / str(safeRelative(report["path"])), seal,
                             {item["runSha256"] for item in receipt["metadata"]["runs"]})


def validateResults(root, receipt):
    run = verifyRun(root / "run")
    if hashFile(root / "run/run.json") != receipt["metadata"]["runSha256"] or run["status"] != receipt["metadata"]["runStatus"]:
        raise ValueError("Extracted run identity mismatch.")
    for report in receipt["metadata"]["reports"]:
        trainingHash = verifyReport(root / str(safeRelative(report["path"])))
        if trainingHash is not None and trainingHash != receipt["metadata"]["runSha256"]:
            raise ValueError("Extracted report/run identity mismatch.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Transport run/checkpoint/report evidence without runtime image copies.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("pack", "check", "unpack"):
        command = commands.add_parser(name)
        command.add_argument("--archive", type=Path, required=True)
        command.add_argument("--receipt", type=Path, required=True)
        if name == "pack":
            source = command.add_mutually_exclusive_group(required=True)
            source.add_argument("--runDir", type=Path)
            source.add_argument("--configurationDir", type=Path)
            command.add_argument("--reportDir", type=Path, action="append", default=[])
            command.add_argument("--launchDir", type=Path)
            command.add_argument("--compression", choices=["gzip", "none"], default="gzip")
        if name == "unpack":
            command.add_argument("--outputDir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "pack":
            if arguments.configurationDir is not None:
                receipt = packConfigurationResults(arguments.configurationDir, arguments.reportDir, arguments.archive,
                                                  arguments.receipt, arguments.compression, arguments.launchDir)
            else:
                receipt = packResults(arguments.runDir, arguments.reportDir, arguments.archive, arguments.receipt,
                                      arguments.compression, arguments.launchDir)
        elif arguments.command == "unpack":
            kind = readJson(arguments.receipt)["kind"]
            if kind not in {"curriculumResultsTransfer", "configurationResultsTransfer"}:
                raise ValueError("Unsupported results transfer kind.")
            receipt = unpackArchive(arguments.archive, arguments.receipt, arguments.outputDir,
                                    kind, validateConfigurationResults if kind == "configurationResultsTransfer" else validateResults)
        else:
            receipt = readJson(arguments.receipt)
            if receipt["kind"] not in {"curriculumResultsTransfer", "configurationResultsTransfer"}:
                raise ValueError("Unsupported results transfer kind.")
            checkArchive(arguments.archive, receipt, kind=receipt["kind"])
        print(f"Verified {receipt['fileCount']} result files; archive SHA-256 {receipt['archiveSha256']}", flush=True)
        return 0
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError) as error:
        print(f"Results transfer failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
