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
    unpackArchive,
    validateFiles,
)


def inspectDataset(datasetDir, contractId):
    root = Path(datasetDir).absolute()
    observed = regularFiles(root)
    contract = readJson(root / "contract.json")
    if contract.get("kind") != "preparedCurriculumDataset" or contract.get("schemaVersion") != 1:
        raise ValueError("Unsupported prepared dataset contract.")
    if contract["contractId"] != contractId or contractId != hashJson({k: v for k, v in contract.items() if k != "contractId"}):
        raise ValueError("Dataset contract identity mismatch.")
    if contract["testExported"] is not False or contract["sourcePreservationVerified"] is not True:
        raise ValueError("Transport requires reserved test images and preserved sources.")
    evidence = [*contract["artifactEvidence"], {"path": "contract.json", "byteSize": (root / "contract.json").stat().st_size,
                                              "sha256": hashFile(root / "contract.json")}]
    expected = validateFiles(evidence)
    if set(observed) != set(expected):
        raise ValueError("Dataset has missing or unexpected files.")
    plan, manifest = readJson(root / "plan.json"), readJson(root / "manifest.json")
    if plan["planId"] != manifest["planId"] or plan["planId"] != contract["planId"]:
        raise ValueError("Dataset plan identities disagree.")
    catalog = plan["catalog"]["data"]
    real = {"public": {source["sourceId"] for source in catalog["publicSources"]},
            "own": {source["videoId"] for source in catalog["ownVideos"]}}
    synthetic = {source["sourceId"] for source in catalog["syntheticSources"]}
    tasks = {task["taskId"]: task for task in plan["recipe"]["tasks"]}
    for taskId in plan["recipe"]["curriculum"]:
        sources = set(tasks[taskId]["trainingSources"])
        if not sources <= real["public"] | {"own"} or sources & synthetic:
            raise ValueError("realOnly transport rejects synthetic or unregistered task sources.")
    native = {sample["recordId"]: sample for sample in plan["samples"]}
    for row in manifest["samples"]:
        if row["origin"] not in real or row["sourceId"] not in real[row["origin"]] or row["sourceId"] in synthetic:
            raise ValueError("realOnly transport rejects synthetic or unregistered provenance.")
        if row["split"] not in {"training", "validation"} or (row["split"] == "validation" and row["origin"] != "own"):
            raise ValueError("Only real development data may be transported.")
        if any(row[key] != native[row["recordId"]][key] for key in ("origin", "sourceId", "mediaPath", "annotationSha256")):
            raise ValueError("Exported source provenance differs from native records.")
    return contract, {name: (observed[name], item) for name, item in expected.items()}


def packDataset(datasetDir, archive, receipt, contractId, verificationPath, compression="gzip"):
    contract, sources = inspectDataset(datasetDir, contractId)
    verification = readJson(verificationPath)
    result = verification.get("verification", {})
    if verification.get("kind") != "datasetVerification" or verification.get("status") != "passed" or (
        verification.get("contractSha256") != hashFile(Path(datasetDir) / "contract.json")
        or result.get("contractId") != contractId or result.get("planId") != contract["planId"]
        or result.get("verifiedArtifactCount") != len(contract["artifactEvidence"])
    ):
        raise ValueError("Packing requires a successful full semantic verification receipt for this contract.")
    metadata = {"contractId": contractId, "planId": contract["planId"], "sourcePolicy": "realOnly",
                "contractSha256": hashFile(Path(datasetDir) / "contract.json"),
                "semanticVerification": verification, "verificationReceiptSha256": hashFile(verificationPath),
                "toolSourceHashes": {name: hashFile(Path(__file__).with_name(name))
                                     for name in ("datasetTransfer.py", "transferArchive.py")}}
    return packArchive(sources, archive, receipt, "curriculumDatasetTransfer", metadata, compression,
                       protected=[datasetDir, verificationPath])


def validateDataset(root, receipt):
    metadata = receipt["metadata"]
    contract, sources = inspectDataset(root, metadata["contractId"])
    expected = {name: item for name, (_, item) in sources.items()}
    if metadata["sourcePolicy"] != "realOnly" or metadata["planId"] != contract["planId"] or (
        metadata["contractSha256"] != hashFile(root / "contract.json")
        or expected != {item["path"]: item for item in receipt["files"]}
    ):
        raise ValueError("Extracted dataset differs from transfer identities or membership.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Transport an immutable real-only curriculum bundle using the standard library.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("pack", "check", "unpack"):
        command = commands.add_parser(name)
        command.add_argument("--archive", type=Path, required=True)
        command.add_argument("--receipt", type=Path, required=True)
        if name == "pack":
            command.add_argument("--datasetDir", type=Path, required=True)
            command.add_argument("--contractId", required=True)
            command.add_argument("--verification", type=Path, required=True)
            command.add_argument("--compression", choices=["gzip", "none"], default="gzip")
        if name == "unpack":
            command.add_argument("--outputDir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "pack":
            receipt = packDataset(arguments.datasetDir, arguments.archive, arguments.receipt,
                                  arguments.contractId, arguments.verification, arguments.compression)
        elif arguments.command == "unpack":
            receipt = unpackArchive(arguments.archive, arguments.receipt, arguments.outputDir,
                                    "curriculumDatasetTransfer", validateDataset)
        else:
            receipt = readJson(arguments.receipt)
            checkArchive(arguments.archive, receipt, kind="curriculumDatasetTransfer")
        print(f"Verified {receipt['fileCount']} files; contract {receipt['metadata']['contractId']}; archive SHA-256 {receipt['archiveSha256']}", flush=True)
        return 0
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError) as error:
        print(f"Dataset transfer failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
