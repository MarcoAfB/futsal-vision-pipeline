import argparse
from pathlib import Path

from futsal import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Futsal dataset research tools.")
    parser.add_argument("--version", action="version", version=f"futsal {__version__}")
    commands = parser.add_subparsers(dest="command")
    rawParser = commands.add_parser("importRawSources", help="Copy and verify registered real raw inputs.")
    rawParser.add_argument("--sourceRoot", type=Path, required=True)
    rawParser.add_argument("--config", type=Path, default=Path("configs/dataSources.yaml"))
    rawParser.add_argument("--outputDir", type=Path, default=Path("data/raw"))
    rawParser.add_argument("--catalogOutput", type=Path, required=True)
    rawVerifyParser = commands.add_parser("verifyRawSources", help="Verify local raw source membership and hashes.")
    rawVerifyParser.add_argument("--dataRoot", type=Path, default=Path("data/raw"))
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
    verifyParser.add_argument("--receipt", type=Path)
    projectionParser = commands.add_parser("prepareProjection", help="Prepare native task labels and missing development images.")
    projectionParser.add_argument("--sourceDatasetDir", type=Path, required=True)
    projectionParser.add_argument("--config", type=Path, default=Path("configs/separatedDetection.yaml"))
    projectionParser.add_argument("--dataRoot", type=Path, required=True)
    projectionParser.add_argument("--outputDir", type=Path, required=True)
    materializeParser = commands.add_parser("materializeProjection", help="Materialize a verified task projection from its parent and supplement.")
    materializeParser.add_argument("--sourceDatasetDir", type=Path, required=True)
    materializeParser.add_argument("--projectionDir", type=Path, required=True)
    materializeParser.add_argument("--outputDir", type=Path, required=True)
    environmentParser = commands.add_parser("inspectEnvironment", help="Inspect the training device and packages.")
    environmentParser.add_argument("--device", default="cpu")
    environmentParser.add_argument("--outputDir", type=Path, required=True)
    weightsParser = commands.add_parser("prepareWeights", help="Obtain official YOLO26 pretrained weights.")
    weightsParser.add_argument("--modelSize", choices=["n", "s", "m", "l", "x"], default="n")
    weightsParser.add_argument("--outputDir", type=Path, required=True)
    trainParser = commands.add_parser("train", help="Run an audited YOLO26 smoke, pilot or scientific curriculum.")
    trainParser.add_argument("--datasetDir", type=Path, required=True)
    trainParser.add_argument("--config", type=Path, default=Path("configs/training/yolo26Smoke.yaml"))
    trainParser.add_argument("--weights", type=Path, required=True)
    trainParser.add_argument("--outputDir", type=Path, required=True)
    trainParser.add_argument("--seed", type=int)
    trainParser.add_argument("--continueFrom", type=Path)
    trainParser.add_argument("--syntheticDir", type=Path, help="Previously generated court training supplement.")
    trainParser.add_argument("--reportRoot", type=Path)
    trainParser.add_argument("--workRoot", type=Path)
    trainParser.add_argument("--evaluationConfig", type=Path, default=Path("configs/evaluation/researchValidation.yaml"))
    from futsal.reporting.researchCli import addResearchCommands

    addResearchCommands(commands)
    arguments = parser.parse_args(argv)
    if arguments.command is None:
        parser.print_help()
        return 0
    if arguments.command in {"importRawSources", "verifyRawSources"}:
        from futsal.datasets.rawSources import importRawSources, verifyRawSources
        from futsal.inventory.catalog import loadCatalog

        try:
            if arguments.command == "importRawSources":
                import yaml

                from futsal.datasets.datasetFiles import validateNewOutput

                catalogOutput = arguments.catalogOutput.resolve()
                project = Path(__file__).resolve().parents[2]
                for protected in (arguments.sourceRoot, arguments.outputDir, project / "src", project / "configs"):
                    validateNewOutput(catalogOutput, protected)
                result = importRawSources(loadCatalog(arguments.config), arguments.sourceRoot, arguments.outputDir)
                catalogOutput.parent.mkdir(parents=True, exist_ok=True)
                with catalogOutput.open("x", encoding="utf-8", newline="\n") as stream:
                    yaml.safe_dump(result["catalog"], stream, sort_keys=False, allow_unicode=True)
            else:
                result = verifyRawSources(arguments.dataRoot)
            print(f"Raw source contract: {result['contractId']}; files: {len(result['artifacts'])}")
            return 0
        except (OSError, ValueError, KeyError) as error:
            parser.exit(2, f"Raw source operation could not complete: {error}\n")
    if arguments.command in {"prepareProjection", "materializeProjection"}:
        from futsal.datasets.datasetProjection import materializeProjection, prepareProjection

        try:
            def progress(message):
                print(message, flush=True)
            if arguments.command == "prepareProjection":
                result = prepareProjection(arguments.sourceDatasetDir, arguments.config,
                                           arguments.dataRoot, arguments.outputDir, progress=progress)
                print(f"Projection: {result['projectionId']}; supplement records: "
                      f"{sum(item['provider'] == 'supplement' for item in result['imageSources'].values())}")
            else:
                result = materializeProjection(arguments.sourceDatasetDir, arguments.projectionDir,
                                               arguments.outputDir, progress=progress)
                print(f"Materialized dataset contract: {result['contractId']}")
            return 0
        except (OSError, ValueError, KeyError) as error:
            parser.exit(2, f"Projection could not complete: {error}\n")
    if arguments.command in {"evaluateRun", "reportRun", "evaluateConfiguration", "compareConfigurations"}:
        from futsal.reporting.researchCli import handleResearchCommand

        try:
            return handleResearchCommand(arguments)
        except (OSError, ValueError, ImportError) as error:
            parser.exit(2, f"Research command could not complete: {error}\n")
    if arguments.command == "train":
        from futsal.training.trainingRun import runTraining

        try:
            run = runTraining(arguments.datasetDir, arguments.config, arguments.weights,
                              arguments.outputDir, progress=lambda message: print(message, flush=True),
                              seed=arguments.seed, continueFrom=arguments.continueFrom,
                              syntheticDir=arguments.syntheticDir)
        except (OSError, ValueError) as error:
            parser.exit(2, f"Training could not start: {error}\n")
        print(f"Training status: {run['status']}; record: {arguments.outputDir / 'run.json'}")
        from futsal.reporting.researchCli import generateRunReport

        try:
            complete, path = generateRunReport(arguments.outputDir, arguments.datasetDir, arguments.evaluationConfig,
                                               reportRoot=arguments.reportRoot, workRoot=arguments.workRoot)
        except (OSError, ValueError, ImportError) as error:
            print(f"Research report could not start: {error}")
            return 1
        print(f"Research report: {path}")
        return 0 if run["status"] == "completed" and complete else 1
    if arguments.command == "prepareWeights":
        from futsal.training.modelWeights import prepareWeights

        try:
            metadata = prepareWeights(arguments.modelSize, arguments.outputDir)
        except (OSError, ValueError) as error:
            parser.exit(2, f"Weight preparation failed: {error}\n")
        print(f"Weights: {arguments.outputDir / metadata['filename']}\nSHA-256: {metadata['sha256']}")
        return 0
    if arguments.command == "inspectEnvironment":
        from futsal.training.trainingEnvironment import inspectEnvironment, writeEnvironmentReport

        try:
            report = inspectEnvironment(arguments.device)
            writeEnvironmentReport(report, arguments.outputDir)
        except (OSError, ValueError) as error:
            parser.exit(2, f"Environment inspection failed: {error}\n")
        print(f"Training environment ready: {report['ready']}; report: {arguments.outputDir}")
        return 0 if report["ready"] else 1
    if arguments.command == "verifyDataset":
        from futsal.datasets.datasetFiles import writeJson
        from futsal.datasets.datasetVerify import verifyDataset
        from futsal.evaluation.evaluationData import validateResearchOutput
        from futsal.inventory.catalog import hashFile
        from futsal.training.trainingRun import utcNow

        try:
            if arguments.receipt is not None:
                receipt = validateResearchOutput(arguments.receipt, [arguments.datasetDir])
            result = verifyDataset(arguments.datasetDir,
                                   progress=lambda message: print(message, flush=True))
            if arguments.receipt is not None:
                receipt.parent.mkdir(parents=True, exist_ok=True)
                writeJson(receipt, {"schemaVersion": 1, "kind": "datasetVerification", "status": "passed",
                                    "verifiedAt": utcNow(), "contractSha256": hashFile(arguments.datasetDir / "contract.json"),
                                    "verification": result})
        except (OSError, ValueError) as error:
            parser.exit(2, f"Dataset verification failed: {error}\n")
        print(f"Verified {result['exportedImageCount']} development images; "
              f"reserved test records: {result['reservedTestRecordCount']}.")
        print(f"Contract: {result['contractId']}")
        return 0
    if arguments.command == "planDataset":
        from futsal.datasets.datasetFiles import validateNewOutput
        from futsal.datasets.datasetPlan import buildDatasetPlan, writeDatasetPlan

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

        from futsal.datasets.datasetExport import prepareDataset

        try:
            contract = prepareDataset(arguments.plan, arguments.dataRoot, arguments.outputDir,
                                      progress=lambda message: print(message, flush=True))
        except (OSError, ValueError, cv2.error) as error:
            parser.exit(2, f"Dataset preparation could not be completed: {error}\n")
        print(f"Dataset: {arguments.outputDir}\nContract: {contract['contractId']}")
        return 0
    from futsal.inventory.catalog import loadCatalog, resolveSourcePath
    from futsal.inventory.inventory import buildInventory
    from futsal.reporting.report import validateOutputDirectory, writeInventory

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
