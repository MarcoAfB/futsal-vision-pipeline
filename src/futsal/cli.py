import argparse
from pathlib import Path

from futsal import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Futsal dataset research tools.")
    parser.add_argument("--version", action="version", version=f"futsal {__version__}")
    commands = parser.add_subparsers(dest="command")
    inventoryParser = commands.add_parser("inventory", help="Audit registered source data.")
    inventoryParser.add_argument(
        "--dataRoot",
        type=Path,
        required=True,
        help="Read-only root containing registered source files.",
    )
    inventoryParser.add_argument(
        "--outputDir", type=Path, required=True, help="Report directory outside dataRoot."
    )
    inventoryParser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dataSources.yaml"),
        help="Source catalog, relative to the working directory.",
    )
    planParser = commands.add_parser("planDataset", help="Plan curriculum datasets and frozen splits.")
    planParser.add_argument("--dataRoot", type=Path, required=True)
    planParser.add_argument("--outputDir", type=Path, required=True)
    planParser.add_argument("--config", type=Path, default=Path("configs/curriculum.yaml"))
    prepareParser = commands.add_parser("prepareDataset", help="Export development data from a verified plan.")
    prepareParser.add_argument("--dataRoot", type=Path, required=True)
    prepareParser.add_argument("--plan", type=Path, required=True)
    prepareParser.add_argument("--outputDir", type=Path, required=True)
    verifyParser = commands.add_parser("verifyDataset", help="Verify a portable prepared dataset bundle.")
    verifyParser.add_argument("--datasetDir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.command is None:
        parser.print_help()
        return 0
    if arguments.command == "verifyDataset":
        from futsal.datasetVerify import verifyDataset

        try:
            result = verifyDataset(arguments.datasetDir,
                                   progress=lambda message: print(message, flush=True))
        except (OSError, ValueError) as error:
            parser.exit(2, f"Dataset verification failed: {error}\n")
        print(f"Verified {result['exportedImageCount']} development images; "
              f"reserved test records: {result['reservedTestRecordCount']}.")
        print(f"Contract: {result['contractId']}")
        return 0
    if arguments.command == "planDataset":
        from futsal.datasetFiles import validateNewOutput
        from futsal.datasetPlan import buildDatasetPlan, writeDatasetPlan

        try:
            validateNewOutput(arguments.outputDir, arguments.dataRoot, [arguments.config])
            print("Reading source inventory and preparing task eligibility...", flush=True)
            plan = buildDatasetPlan(arguments.config, arguments.dataRoot)
            path = writeDatasetPlan(plan, arguments.outputDir, arguments.dataRoot, [arguments.config])
        except (OSError, ValueError) as error:
            parser.exit(2, f"Dataset planning could not be completed: {error}\n")
        print(f"Plan: {path}\nAll canonical classes have eligible training and test annotations.")
        return 0
    if arguments.command == "prepareDataset":
        import cv2

        from futsal.datasetExport import prepareDataset

        try:
            contract = prepareDataset(arguments.plan, arguments.dataRoot, arguments.outputDir,
                                      progress=lambda message: print(message, flush=True))
        except (OSError, ValueError, cv2.error) as error:
            parser.exit(2, f"Dataset preparation could not be completed: {error}\n")
        print(f"Dataset: {arguments.outputDir}\nContract: {contract['contractId']}")
        return 0
    from futsal.catalog import loadCatalog, resolveSourcePath
    from futsal.inventory import buildInventory
    from futsal.report import validateOutputDirectory, writeInventory

    try:
        catalog = loadCatalog(arguments.config)
        protectedPaths = (
            arguments.config,
            resolveSourcePath(arguments.config.parent, catalog.data["labelMapPath"]),
        )
        validateOutputDirectory(arguments.outputDir, arguments.dataRoot, protectedPaths)
        inventory = buildInventory(arguments.dataRoot, catalog)
        inventoryPath, reportPath = writeInventory(
            inventory, arguments.outputDir, arguments.dataRoot, protectedPaths
        )
    except (OSError, ValueError) as error:
        parser.exit(2, f"Inventory could not be completed: {error}\n")
    summary = inventory["summary"]
    print(f"Reviewed frame records: {summary['reviewedFrames']}; matches: {summary['matchCount']}.")
    print(f"Findings: {summary['errorCount']} errors, {summary['warningCount']} warnings.")
    print(f"JSON: {inventoryPath}\nReport: {reportPath}")
    return 1 if summary["errorCount"] else 0
