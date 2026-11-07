from pathlib import Path

from futsal.datasets.datasetFiles import writeJson
from futsal.datasets.mediaRegistration import registerMedia
from futsal.datasets.mediaStore import (
    protectMediaOutput,
    readBindings,
    rebindMedia,
    storeCollections,
    verifyAssets,
)
from futsal.datasets.sharedViews import prepareView


def addSharedCommands(commands):
    register = commands.add_parser("registerMedia", help="Register existing immutable media without copying images.")
    register.add_argument("--storeDir", type=Path, required=True)
    register.add_argument("--sourceDir", type=Path, required=True)
    view = commands.add_parser("prepareView", help="Prepare task selections and labels over shared media.")
    for name in ("storeDir", "config", "outputDir"):
        view.add_argument(f"--{name}", type=Path, required=True)
    view.add_argument("--collectionId")
    view.add_argument("--dataRoot", type=Path)
    inventory = commands.add_parser("mediaInventory", help="Verify registered assets and export receiver inventory.")
    inventory.add_argument("--storeDir", type=Path, required=True)
    inventory.add_argument("--output", type=Path, required=True)
    rebind = commands.add_parser("rebindMedia", help="Bind a registered asset to verified bytes at a new location.")
    rebind.add_argument("--storeDir", type=Path, required=True)
    rebind.add_argument("--assetId", required=True)
    rebind.add_argument("--root", type=Path, required=True)
    rebind.add_argument("--relativePath", required=True)


def mediaInventory(storeDir):
    assets = {}
    for collection in storeCollections(storeDir):
        assets.update(collection["assets"])
    bindings = readBindings(storeDir)["assets"]
    assets = {key: item for key, item in assets.items() if key in bindings}
    verifyAssets(storeDir, assets)
    return {"schemaVersion": 1, "kind": "sharedMediaInventory", "assets": assets}


def handleSharedCommand(args):
    if args.command == "registerMedia":
        return registerMedia(args.storeDir, args.sourceDir)
    if args.command == "prepareView":
        contract = prepareView(args.storeDir, args.config, args.outputDir,
                               collectionId=args.collectionId, dataRoot=args.dataRoot)
        return {"contractId": contract["contractId"], "summary": contract["summary"],
                "viewDirectory": str(args.outputDir), "copiedImageCount": 0}
    if args.command == "rebindMedia":
        rebindMedia(args.storeDir, args.assetId, args.root, args.relativePath)
        return {"assetId": args.assetId, "rebound": True}
    from futsal.evaluation.evaluationData import validateResearchOutput

    output = validateResearchOutput(args.output, [args.storeDir])
    protectMediaOutput(output, args.storeDir)
    result = mediaInventory(args.storeDir)
    output.parent.mkdir(parents=True, exist_ok=True)
    writeJson(output, result)
    return {"assetCount": len(result["assets"]), "output": str(output)}
