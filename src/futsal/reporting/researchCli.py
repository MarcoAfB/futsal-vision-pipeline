import traceback
from pathlib import Path

from futsal.datasets.datasetFiles import readJson, writeJson


def addResearchCommands(commands):
    courtComparison = commands.add_parser("compareCourtTraining", help="Compare real-only and supplemented court training.")
    for name in ("baselineEvaluationDir", "candidateEvaluationDir", "outputDir"):
        courtComparison.add_argument(f"--{name}", type=Path, required=True)
    comparison = commands.add_parser("compareConfigurations", help="Compare sealed scientific evaluations without inference.")
    for name in ("baselineEvaluationDir", "candidateEvaluationDir", "outputDir"):
        comparison.add_argument(f"--{name}", type=Path, required=True)
    configuration = commands.add_parser("evaluateConfiguration", help="Evaluate independent object/court checkpoints jointly.")
    for name in ("objectsRunDir", "courtRunDir", "datasetDir", "referenceDatasetDir", "outputDir"):
        configuration.add_argument(f"--{name}", type=Path, required=True)
    configuration.add_argument("--config", type=Path, default=Path("configs/evaluation/researchValidation.yaml"))
    configuration.add_argument("--workRoot", type=Path)
    configuration.add_argument("--mediaStoreDir", type=Path)
    evaluate = commands.add_parser("evaluateRun", help="Evaluate archived best checkpoints on own validation data.")
    evaluate.add_argument("--runDir", type=Path, required=True)
    evaluate.add_argument("--datasetDir", type=Path, required=True)
    evaluate.add_argument("--selection", choices=["runValidation", "fullValidation"], default="runValidation")
    evaluate.add_argument("--config", type=Path, default=Path("configs/evaluation/researchValidation.yaml"))
    evaluate.add_argument("--outputDir", type=Path, required=True)
    evaluate.add_argument("--workRoot", type=Path)
    evaluate.add_argument("--mediaStoreDir", type=Path)
    report = commands.add_parser("reportRun", help="Render archived evaluation or training evidence without inference.")
    source = report.add_mutually_exclusive_group(required=True)
    source.add_argument("--evaluationDir", type=Path)
    source.add_argument("--runDir", type=Path)
    report.add_argument("--outputDir", type=Path, required=True)


def handleResearchCommand(arguments):
    from futsal.evaluation.evaluationRun import evaluateRun
    from futsal.reporting.researchReport import renderReport, renderTrainingReport

    if arguments.command == "compareCourtTraining":
        from futsal.evaluation.courtComparison import compareCourtTraining

        result = compareCourtTraining(arguments.baselineEvaluationDir, arguments.candidateEvaluationDir, arguments.outputDir)
        path = renderReport(arguments.outputDir)
        print(f"Court training comparison: {result['status']}; report: {path}")
        return 0 if result["status"] == "completed" else 1
    if arguments.command == "compareConfigurations":
        from futsal.evaluation.configurationComparison import compareConfigurations

        result = compareConfigurations(arguments.baselineEvaluationDir, arguments.candidateEvaluationDir, arguments.outputDir)
        path = renderReport(arguments.outputDir)
        print(f"Comparison: {result['status']}; report: {path}")
        return 0 if result["status"] == "completed" else 1
    if arguments.command == "evaluateConfiguration":
        from futsal.evaluation.configurationRun import evaluateConfiguration

        result = evaluateConfiguration(arguments.objectsRunDir, arguments.courtRunDir, arguments.datasetDir,
            arguments.referenceDatasetDir, arguments.config, arguments.outputDir,
            progress=lambda message: print(message, flush=True), workRoot=arguments.workRoot,
            mediaStoreDir=arguments.mediaStoreDir)
        path = renderReport(arguments.outputDir)
        print(f"Configuration evaluation: {result['status']}; report: {path}")
        return 0 if result["status"] == "completed" else 1
    if arguments.command == "reportRun":
        path = (renderReport(arguments.evaluationDir, arguments.outputDir) if arguments.evaluationDir else
                renderTrainingReport(arguments.runDir, arguments.outputDir))
        print(f"Report: {path}")
        return 0
    result = evaluateRun(arguments.runDir, arguments.datasetDir, arguments.config,
                         arguments.selection, arguments.outputDir,
                         progress=lambda message: print(message, flush=True), workRoot=arguments.workRoot,
                         mediaStoreDir=arguments.mediaStoreDir)
    path = renderReport(arguments.outputDir)
    print(f"Evaluation status: {result['status']}; report: {path}")
    return 0 if result["status"] == "completed" else 1


def generateRunReport(runDir, datasetDir, configPath, reportRoot=None, workRoot=None, mediaStoreDir=None):
    from futsal.evaluation.evaluationData import validateResearchOutput
    from futsal.evaluation.evaluationRun import evaluateRun, newEvaluationPath
    from futsal.reporting.researchReport import (
        renderReport,
        renderTrainingReport,
        updateReportIndex,
    )
    from futsal.training.trainingRun import utcNow

    runDir = Path(runDir).resolve()
    run = readJson(runDir / "run.json")
    output = newEvaluationPath(runDir) if reportRoot is None else newEvaluationPath(runDir, reportRoot=reportRoot)
    validateResearchOutput(output, [runDir, Path(datasetDir)], [Path(configPath)])
    startedAt = utcNow()
    try:
        if run["status"] == "completed":
            result = evaluateRun(runDir, datasetDir, configPath, "runValidation", output,
                                 progress=lambda message: print(message, flush=True), workRoot=workRoot,
                                 mediaStoreDir=mediaStoreDir)
            path = renderReport(output)
            return result["status"] == "completed", path
        return False, renderTrainingReport(runDir, output)
    except (Exception, KeyboardInterrupt) as error:
        output.mkdir(parents=True, exist_ok=True)
        (output / "pipelineFailure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        record = readJson(output / "reportState.json") if (output / "reportState.json").is_file() else {}
        record.update({"schemaVersion": 1, "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                  "startedAt": startedAt, "endedAt": utcNow(), "runId": runDir.name,
                  "purpose": run["purpose"], "selection": "runValidation", "trainingStatus": run["status"],
                  "evaluationStatus": readJson(output / "evaluation.json")["status"]
                  if (output / "evaluation.json").is_file() else "couldNotStart",
                  "error": {"type": type(error).__name__, "message": str(error)}})
        writeJson(output / "reportState.json", record)
        updateReportIndex(output.parent.parent)
        print(f"Research report failed: {error}; details: {output / 'pipelineFailure.txt'}", flush=True)
        return False, output / "reportState.json"
