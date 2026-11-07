import argparse
import json
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


def inspectSynthetic(datasetDir):
    root = Path(datasetDir)
    contract = readJson(root / "contract.json")
    if (
        contract.get("schemaVersion") != 1
        or contract.get("kind") != "syntheticCourtDataset"
        or contract.get("trainingOnly") is not True
    ):
        raise ValueError("Transport requires a training-only synthetic court dataset.")
    if contract["contractId"] != hashJson({k: v for k, v in contract.items() if k != "contractId"}):
        raise ValueError("Synthetic contract identity mismatch.")
    files = regularFiles(root)
    evidence = [
        *contract["artifactEvidence"],
        {
            "path": "contract.json",
            "byteSize": (root / "contract.json").stat().st_size,
            "sha256": hashFile(root / "contract.json"),
        },
    ]
    expected = validateFiles(evidence)
    if set(files) != set(expected):
        raise ValueError("Synthetic dataset has missing or unexpected files.")
    rows = [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if len(rows) != contract["configuration"]["count"] or not rows:
        raise ValueError("Synthetic manifest count mismatch.")
    for row in rows:
        if (
            row["origin"] != "synthetic"
            or row["taskId"] != "courtPoints"
            or row["split"] != "training"
        ):
            raise ValueError("Synthetic transport cannot include validation or test records.")
        for kind in ("image", "label"):
            if (
                row[f"{kind}Path"] not in expected
                or expected[row[f"{kind}Path"]]["sha256"] != row[f"{kind}Sha256"]
            ):
                raise ValueError("Synthetic manifest and contract hashes disagree.")
    return contract, {name: (files[name], item) for name, item in expected.items()}


def packSynthetic(datasetDir, archive, receipt, compression="gzip"):
    contract, sources = inspectSynthetic(datasetDir)
    return packArchive(
        sources,
        archive,
        receipt,
        "syntheticCourtTransfer",
        {"contractId": contract["contractId"], "trainingOnly": True},
        compression,
        protected=[datasetDir],
    )


def validateSyntheticTransfer(root, receipt):
    contract, sources = inspectSynthetic(root)
    if receipt["metadata"] != {"contractId": contract["contractId"], "trainingOnly": True} or {
        name: item for name, (_, item) in sources.items()
    } != {item["path"]: item for item in receipt["files"]}:
        raise ValueError("Synthetic receipt and extracted source disagree.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Transport a sealed synthetic court source using the standard library."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("pack", "check", "unpack"):
        command = commands.add_parser(name)
        command.add_argument("--archive", type=Path, required=True)
        command.add_argument("--receipt", type=Path, required=True)
        if name == "pack":
            command.add_argument("--datasetDir", type=Path, required=True)
            command.add_argument("--compression", choices=["gzip", "none"], default="gzip")
        elif name == "unpack":
            command.add_argument("--outputDir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "pack":
            receipt = packSynthetic(args.datasetDir, args.archive, args.receipt, args.compression)
        elif args.command == "unpack":
            receipt = unpackArchive(
                args.archive,
                args.receipt,
                args.outputDir,
                "syntheticCourtTransfer",
                validateSyntheticTransfer,
            )
        else:
            receipt = readJson(args.receipt)
            checkArchive(args.archive, receipt, kind="syntheticCourtTransfer")
        print(
            f"Verified {receipt['fileCount']} files; synthetic contract: {receipt['metadata']['contractId']}"
        )
        return 0
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError) as error:
        print(f"Synthetic transport failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
