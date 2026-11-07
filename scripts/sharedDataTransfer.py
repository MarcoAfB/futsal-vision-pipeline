import argparse
import sys
import tarfile
from pathlib import Path

from transferArchive import hashFile, packArchive, readJson, regularFiles, unpackArchive

from futsal.datasets.datasetFiles import writeJson
from futsal.datasets.mediaRegistration import registerMedia
from futsal.datasets.mediaStore import (
    atomicJson,
    bindAsset,
    protectMediaOutput,
    readBindings,
    resolveAsset,
    storeLock,
    verifyAssets,
)
from futsal.datasets.sharedCli import mediaInventory
from futsal.datasets.sharedViews import verifyView


def packView(datasetDir, storeDir, receiverInventory, archive, receipt):
    datasetDir = Path(datasetDir).resolve()
    verifyView(datasetDir, storeDir)
    protectMediaOutput(archive, storeDir)
    protectMediaOutput(receipt, storeDir)
    contract = readJson(datasetDir / "contract.json")
    inventory = readJson(receiverInventory)
    if inventory.get("kind") != "sharedMediaInventory" or inventory.get("schemaVersion") != 1:
        raise ValueError("Expected a verified receiver media inventory.")
    sources = {}

    def add(name, path):
        sources[name] = (path, {"byteSize": path.stat().st_size, "sha256": hashFile(path)})

    for name, path in regularFiles(datasetDir).items():
        add(f"view/{name}", path)
    transferred = {}
    bindings = readBindings(storeDir)
    for assetId, item in contract["assets"].items():
        if assetId in inventory["assets"]:
            if inventory["assets"][assetId] != item:
                raise ValueError("Receiver inventory disagrees on an asset size.")
            continue
        path = resolveAsset(storeDir, assetId, bindings=bindings)
        name = f"assets/{assetId}{path.suffix.lower()}"
        add(name, path)
        transferred[assetId] = name
    return packArchive(sources, archive, receipt, "sharedTaskViewTransfer",
                       {"contractId": contract["contractId"], "transferred": transferred},
                       protected=[datasetDir, storeDir, receiverInventory])


def receiveView(storeDir, archive, receipt, outputDir):
    def validate(root, transfer):
        verifyView(root / "view")
        contract = readJson(root / "view/contract.json")
        metadata = transfer["metadata"]
        if contract["contractId"] != metadata["contractId"] or not metadata["transferred"].keys() <= contract["assets"].keys():
            raise ValueError("Transferred view identity or asset membership mismatch.")
        if set(regularFiles(root)) != {f"view/{name}" for name in regularFiles(root / "view")} | set(metadata["transferred"].values()):
            raise ValueError("Unexpected files in shared view package.")
        for assetId, name in metadata["transferred"].items():
            from transferArchive import safeRelative

            safeRelative(name)
            path = root / name
            if not name.startswith(f"assets/{assetId}.") or hashFile(path) != assetId or path.stat().st_size != contract["assets"][assetId]["byteSize"]:
                raise ValueError("Transferred media differs from the selected view.")
        verifyAssets(storeDir, {key: value for key, value in contract["assets"].items() if key not in metadata["transferred"]})

    from futsal.evaluation.evaluationData import validateResearchOutput

    validateResearchOutput(outputDir, [storeDir, archive, receipt])
    protectMediaOutput(outputDir, storeDir)
    transfer = unpackArchive(archive, receipt, outputDir, "sharedTaskViewTransfer", validate)
    with storeLock(storeDir) as root:
        bindings = readBindings(root)
        for assetId, name in transfer["metadata"]["transferred"].items():
            bindAsset(bindings, assetId, Path(outputDir) / name)
        atomicJson(root / "bindings.json", bindings)
    registerMedia(storeDir, Path(outputDir) / "view")
    verifyView(Path(outputDir) / "view", storeDir)
    return transfer


def main(argv=None):
    parser = argparse.ArgumentParser(description="Transfer shared task metadata and only missing image assets.")
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("inventory")
    inventory.add_argument("--storeDir", type=Path, required=True)
    inventory.add_argument("--output", type=Path, required=True)
    for name in ("pack", "unpack"):
        command = commands.add_parser(name)
        for field in ("storeDir", "archive", "receipt"):
            command.add_argument(f"--{field}", type=Path, required=True)
        if name == "pack":
            command.add_argument("--datasetDir", type=Path, required=True)
            command.add_argument("--receiverInventory", type=Path, required=True)
        else:
            command.add_argument("--outputDir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "inventory":
            from futsal.evaluation.evaluationData import validateResearchOutput

            path = validateResearchOutput(args.output, [args.storeDir])
            protectMediaOutput(path, args.storeDir)
            value = mediaInventory(args.storeDir)
            path.parent.mkdir(parents=True, exist_ok=True)
            writeJson(path, value)
            print(f"Verified receiver assets: {len(value['assets'])}")
        else:
            value = (packView(args.datasetDir, args.storeDir, args.receiverInventory, args.archive, args.receipt)
                     if args.command == "pack" else receiveView(args.storeDir, args.archive, args.receipt, args.outputDir))
            print(f"Verified view {value['metadata']['contractId']}; transferred assets: {len(value['metadata']['transferred'])}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError) as error:
        print(f"Shared transfer failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
