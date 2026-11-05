import os
import shutil
import traceback
from pathlib import Path

from futsal.datasets.datasetFiles import readJson, writeJson
from futsal.evaluation.evaluationData import readJsonLines, validateResearchOutput
from futsal.reporting.reportExamples import renderExamples
from futsal.reporting.researchFigures import classFigure, pointFigure
from futsal.reporting.researchReport import (
    archiveRenderer,
    markdownTable,
    stageTables,
    updateReportIndex,
    verifyEvaluation,
)
from futsal.training.trainingRun import fileEvidence, utcNow


def configurationMarkdown(root, state):
    lines = ["# Independent object and court detection", "",
             f"Evaluation: **{state['status']}**. Selection: **commonValidation**. Test evaluated: **False**.", "",
             "Two independent detectors start from the same official checkpoint. Object and landmark outputs "
             "are mapped to canonical classes and capped globally before native AP matching. This is a single-seed "
             "quality result; no inference benchmark or significance test is reported.", ""]
    if state["status"] != "completed":
        return "\n".join(lines + [f"Evaluation failure: {state.get('error', state.get('integrityError', 'See evaluation.json'))}.", ""])
    components = []
    for item in state["components"]:
        selected = readJson(root / "evidence/components" / item["taskId"] / "selection.json")["tasks"][item["taskId"]]
        components.append([item["taskId"], len(item["classes"]), item["run"]["configuration"]["seed"],
                           len(selected["training"]), len(selected["validation"]), item["run"]["initialWeights"]["sha256"]])
    lines += markdownTable(["Component", "Classes", "Seed", "Train", "Validation", "Initial SHA-256"], components)
    evidence = root / "evidence/joint"
    detection, points = readJson(evidence / "detection.json"), readJson(evidence / "points.json")
    stageTables(detection, points, root / "tables")
    figures = root / "figures"
    figures.mkdir()
    classPlot = classFigure(detection, figures)
    pointPlot = pointFigure(points, state["protocol"], figures)
    lines += [f"Common validation images: **{detection['imageCount']}**. Global limit: {state['globalDetectionLimit']}.", "",
              "## Detection by target group", "",
              *markdownTable(["Group", "Supported classes", "AP50-95", "Precision", "Recall", "F1"],
                [[group, f"{item['supportedClassCount']}/{item['classCount']}", item["ap50To95"], item["precision"], item["recall"], item["f1"]]
                 for group, item in detection["groups"].items()]),
              "[Per-class detection and support](tables/classDetection.csv). "
              "[Original-match detection](tables/matchDetection.csv).", "",
              f"![Class detection and support](figures/{classPlot})", "",
              "## Native court-point localization", "",
              "Missing points count as failures. Conditional localization errors are accompanied by support. "
              "Unsupported lowerCornerSideB remains N/A; no point annotations are invented.", "",
              "[Localization counts and errors](tables/pointLocalization.csv). "
              "[Success at fixed tolerances](tables/pointToleranceSuccess.csv).", ""]
    if pointPlot:
        lines += [f"![Point localization](figures/{pointPlot})", ""]
    refs, preds = readJsonLines(evidence / "references.jsonl"), readJsonLines(evidence / "predictions.jsonl")
    examples = renderExamples(root, {"examples": state["examples"]}, refs, preds, points,
                              state["classes"], state["protocol"], state["pointClasses"], root / "examples")
    lines += ["## Selected validation examples", ""]
    for example in examples:
        lines += [f"{example['recordId']} — {example['reason']}", "", f"![Reference and prediction](examples/{example['figure']})", ""]
    lines += ["## Evidence", "", "[Sealed evaluation](evaluation.json), [contract](evaluationContract.json). "
              "Component runs, code and exact references/predictions are archived under evidence/. "
              "Component full-validation metrics and this common-cohort result describe different selections.", ""]
    return "\n".join(lines)


def renderConfigurationReport(evaluationDir, outputDir=None):
    source = Path(evaluationDir).resolve()
    state, contract = verifyEvaluation(source)
    inputs = [source, Path(state["datasetDirectory"]), Path(state["referenceDatasetDirectory"]),
              *[Path(item["runDirectory"]) for item in state["components"]]]
    root = source if outputDir is None else validateResearchOutput(outputDir, inputs)
    if (root / "report.md").exists() or (root / "reportState.json").exists():
        raise ValueError("Report output must be new.")
    if root != source:
        root.mkdir(parents=True)
        shutil.copytree(source / "evidence", root / "evidence")
        for name in ("evaluation.json", "evaluationContract.json"):
            shutil.copyfile(source / name, root / name)
    cache = root.parent / f".{root.name}PlotCache"
    cache.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    report = {"schemaVersion": 1, "status": "running", "startedAt": utcNow(), "runId": state["configurationId"],
        "purpose": state["purpose"], "selection": state["selection"], "trainingStatus": "completed",
        "evaluationStatus": state["status"], "evaluationContractId": contract["contractId"]}
    try:
        report["renderingEnvironment"] = archiveRenderer(root)
        (root / "report.md").write_text(configurationMarkdown(root, state), encoding="utf-8")
        report["status"] = "completed"
        report["artifacts"] = [fileEvidence(path, root) for path in sorted(root.rglob("*")) if path.is_file() and path.name != "reportState.json"]
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="failed", error=str(error))
        (root / "presentationFailure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        report["endedAt"] = utcNow()
        writeJson(root / "reportState.json", report)
        catalog = next((parent for parent in root.parents if parent.name == "reports"), None)
        if catalog:
            updateReportIndex(catalog)
    return root / "report.md"
