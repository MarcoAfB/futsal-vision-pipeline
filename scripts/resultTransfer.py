import argparse
import sys
import tarfile
from pathlib import Path

from transferArchive import (
    checkArchive,
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
        path = Path(root) / str(safeRelative(item["path"]))
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
            command.add_argument("--runDir", type=Path, required=True)
            command.add_argument("--reportDir", type=Path, action="append", default=[])
            command.add_argument("--launchDir", type=Path)
            command.add_argument("--compression", choices=["gzip", "none"], default="gzip")
        if name == "unpack":
            command.add_argument("--outputDir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "pack":
            receipt = packResults(arguments.runDir, arguments.reportDir, arguments.archive, arguments.receipt,
                                  arguments.compression, arguments.launchDir)
        elif arguments.command == "unpack":
            receipt = unpackArchive(arguments.archive, arguments.receipt, arguments.outputDir,
                                    "curriculumResultsTransfer", validateResults)
        else:
            receipt = readJson(arguments.receipt)
            checkArchive(arguments.archive, receipt, kind="curriculumResultsTransfer")
        print(f"Verified {receipt['fileCount']} result files; archive SHA-256 {receipt['archiveSha256']}", flush=True)
        return 0
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError) as error:
        print(f"Results transfer failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
