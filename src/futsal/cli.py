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
    arguments = parser.parse_args(argv)
    if arguments.command is None:
        parser.print_help()
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
