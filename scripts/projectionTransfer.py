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


def projectionSources(root, expectedId):
    root = Path(root)
    files = regularFiles(root)
    projection = readJson(root / "projection.json")
    if projection.get("schemaVersion") != 1 or projection.get("kind") != "preparedTaskProjection" or projection.get("sourcePolicy") != "realOnly":
        raise ValueError("Unsupported projection transport contract.")
    if projection["projectionId"] != expectedId or expectedId != hashJson({k: v for k, v in projection.items() if k != "projectionId"}):
        raise ValueError("Projection identity mismatch.")
    evidence = [*projection["artifactEvidence"], {"path": "projection.json", "byteSize": (root / "projection.json").stat().st_size,
                                                "sha256": hashFile(root / "projection.json")}]
    expected = validateFiles(evidence)
    if set(files) != set(expected):
        raise ValueError("Projection has missing or unexpected files.")
    return projection, {name: (files[name], item) for name, item in expected.items()}


def validateProjection(root, receipt):
    projection, sources = projectionSources(root, receipt["metadata"]["projectionId"])
    if projection["parentContractId"] != receipt["metadata"]["parentContractId"] or (
        {name: item for name, (_, item) in sources.items()} != {item["path"]: item for item in receipt["files"]}
    ):
        raise ValueError("Projection transfer parent or membership mismatch.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Transport incremental real task projections without duplicating parent images.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("pack", "check", "unpack"):
        command = commands.add_parser(name)
        command.add_argument("--archive", type=Path, required=True)
        command.add_argument("--receipt", type=Path, required=True)
        if name == "pack":
            command.add_argument("--projectionDir", type=Path, required=True)
            command.add_argument("--projectionId", required=True)
        if name == "unpack":
            command.add_argument("--outputDir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "pack":
            projection, sources = projectionSources(args.projectionDir, args.projectionId)
            receipt = packArchive(sources, args.archive, args.receipt, "taskProjectionTransfer",
                {"projectionId": projection["projectionId"], "parentContractId": projection["parentContractId"],
                 "sourcePolicy": "realOnly", "toolSourceHashes": {name: hashFile(Path(__file__).with_name(name))
                    for name in ("projectionTransfer.py", "transferArchive.py")}}, protected=[args.projectionDir])
        elif args.command == "unpack":
            receipt = unpackArchive(args.archive, args.receipt, args.outputDir, "taskProjectionTransfer", validateProjection)
        else:
            receipt = readJson(args.receipt)
            checkArchive(args.archive, receipt, kind="taskProjectionTransfer")
        print(f"Verified projection {receipt['metadata']['projectionId']}; {receipt['fileCount']} files.")
        return 0
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError) as error:
        print(f"Projection transfer failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
