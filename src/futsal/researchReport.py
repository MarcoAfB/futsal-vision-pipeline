import csv
import json
import os
import platform
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from futsal.catalog import hashFile, resolveSourcePath
from futsal.datasetFiles import hashJson, readJson, verifyEvidence, writeJson
from futsal.evaluationData import readJsonLines, validateResearchOutput
from futsal.report import atomicWrite, markdownCell
from futsal.reportExamples import renderExamples
from futsal.researchFigures import stageFigures
from futsal.trainingRun import fileEvidence, utcNow


def number(value):
    return "N/A" if value is None else f"{value:.6g}" if isinstance(value, float) else str(value)


def elapsedSeconds(record, endField="endedAt"):
    if not record.get("startedAt") or not record.get(endField):
        return None
    return (datetime.fromisoformat(record[endField]) - datetime.fromisoformat(record["startedAt"])).total_seconds()


def markdownTable(headers, rows):
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
            *["| " + " | ".join(markdownCell(number(value)) for value in row) + " |" for row in rows], ""]


def writeCsv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def verifyEvaluation(root):
    contract = readJson(root / "evaluationContract.json")
    if contract.get("schemaVersion") != 1 or contract["contractId"] != hashJson({
        key: value for key, value in contract.items() if key != "contractId"
    }) or hashFile(root / "evaluation.json") != contract["evaluationSha256"]:
        raise ValueError("Evaluation metadata or evidence contract changed.")
    verifyEvidence(root, contract["artifacts"])
    state = readJson(root / "evaluation.json")
    if state["status"] == "running":
        raise ValueError("An unfinished evaluation cannot produce a sealed research report.")
    return state, contract


def archiveRenderer(root):
    directory = root / "presentation" / "src" / "futsal"
    directory.mkdir(parents=True)
    for source in sorted(Path(__file__).parent.glob("*.py")):
        shutil.copyfile(source, directory / source.name)
    return {"python": platform.python_version(), "matplotlib": version("matplotlib"),
            "numpy": version("numpy"), "opencv": version("opencv-python-headless"),
            "files": [fileEvidence(path, root) for path in sorted(directory.glob("*.py"))]}


def stageTables(detection, points, directory):
    directory.mkdir(parents=True)
    writeCsv(directory / "classDetection.csv", detection["perClass"])
    writeCsv(directory / "groupDetection.csv", [{"group": key, **row} for key, row in detection["groups"].items()])
    writeCsv(directory / "matchDetection.csv", [{"matchId": matchId, "imageCount": data["imageCount"], **row}
             for matchId, data in detection["byMatch"].items() for row in data["perClass"]])
    labels = detection["confusionLabels"]
    counts, normalized = [], []
    for index, row in enumerate(detection["confusionMatrix"]):
        counts.append({"trueClass": labels[index], **dict(zip(labels, row, strict=True))})
        total = sum(row)
        normalized.append({"trueClass": labels[index], **{label: value / total if total else None
                                                          for label, value in zip(labels, row, strict=True)}})
    writeCsv(directory / "confusionCounts.csv", counts)
    writeCsv(directory / "confusionNormalizedByTrueClass.csv", normalized)
    writeCsv(directory / "detectionAssociations.csv", detection["matches"],
             ["recordId", "referenceIndex", "predictionIndex", "iou", "kind"])
    writeCsv(directory / "imageErrors.csv", detection["imageErrors"])
    curves = [{"className": name, "recall": recall, "precision": precision}
              for name, curve in detection["prCurves"].items()
              for recall, precision in zip(curve["recall"], curve["precision"], strict=True)]
    writeCsv(directory / "precisionRecall.csv", curves, ["className", "recall", "precision"])
    if points["perClass"]:
        summaries = [{key: value for key, value in row.items() if key != "successRates"} for row in points["perClass"]]
        writeCsv(directory / "pointLocalization.csv", summaries)
        rates = [{"className": row["className"], "support": row["support"], "toleranceDiagonal": tolerance,
                  "successRate": rate} for row in points["perClass"] for tolerance, rate in row["successRates"].items()]
        writeCsv(directory / "pointToleranceSuccess.csv", rates)
        errors = [{key: value for key, value in row.items() if key not in {"nativePoint", "predictedPoint"}}
                  | {f"{prefix}{axis}": row[field][index] if row[field] is not None else None
                     for prefix, field in (("native", "nativePoint"), ("predicted", "predictedPoint"))
                     for index, axis in enumerate(("X", "Y"))} for row in points["errors"]]
        writeCsv(directory / "pointErrors.csv", errors)
        writeCsv(directory / "matchPointLocalization.csv", [{"matchId": matchId, **{key: value for key, value in row.items()
                    if key != "successRates"}} for matchId, values in points["byMatch"].items() for row in values["perClass"]])
        writeCsv(directory / "matchPointToleranceSuccess.csv", [{"matchId": matchId, "className": row["className"],
                 "support": row["support"], "toleranceDiagonal": tolerance, "successRate": rate}
                 for matchId, values in points["byMatch"].items() for row in values["perClass"]
                 for tolerance, rate in row["successRates"].items()])


def stageMarkdown(root, state, stage):
    taskId = stage["taskId"]
    lines = [f"## {taskId}", "", f"Evaluation status: **{stage['status']}**.", ""]
    if stage["status"] != "completed":
        return lines + [stage.get("reason", "No complete research metrics are available for this stage."), ""]
    evidenceDir = resolveSourcePath(root, stage["evidenceDirectory"])
    detection, points = readJson(evidenceDir / "detection.json"), readJson(evidenceDir / "points.json")
    references, predictions = readJsonLines(evidenceDir / "references.jsonl"), readJsonLines(evidenceDir / "predictions.jsonl")
    classRoot = root / taskId
    stageTables(detection, points, classRoot / "tables")
    figures = stageFigures(detection, points, state["protocol"], state["pointClasses"],
                           evidenceDir / "results.csv", classRoot / "figures")
    examples = renderExamples(root, stage, references, predictions, points, stage["classes"], state["protocol"],
                              state["pointClasses"], classRoot / "examples")
    training = next((item for item in state["training"]["stages"] if item["taskId"] == taskId), {})
    loader = readJson(evidenceDir / "loaderAudit.json") if (evidenceDir / "loaderAudit.json").is_file() else {}
    trainCount = loader.get("training", {}).get("imageCount")
    available = state["datasetSummary"][taskId]
    lines += ["### Data and checkpoint", "", *markdownTable(
        ["Training images used", "Eligible training images", "Evaluated validation images", "Eligible validation images", "Declared classes"],
        [[trainCount, available["training"]["imageCount"], stage["imageCount"],
          available["validation"]["imageCount"], len(stage["classes"])]]),
        f"Best checkpoint SHA-256: `{stage['checkpoint']['sha256']}`. Size: "
        f"{stage['checkpoint']['byteSize'] / 1024 ** 2:.2f} MiB. Selection criterion: training validation mAP50-95.", "",
        *markdownTable(["Original match", "Evaluated images"], sorted(detection["matchCounts"].items()))]
    selectionPath = root / "evidence" / "trainingSelection.json"
    if selectionPath.is_file():
        selected = readJson(selectionPath)["tasks"][taskId]["training"]
        lines += markdownTable(["Training source", "Selected images"], sorted(Counter(row["sourceId"] for row in selected).items()))
    if training:
        lines += [f"Training stage elapsed: {number(elapsedSeconds(training, 'completedAt'))} s "
                  "(including setup and validation).", ""]
    if (evidenceDir / "transferAudit.json").is_file():
        audit = readJson(evidenceDir / "transferAudit.json")
        optimizer = readJson(evidenceDir / "optimizer.json")
        lines += [f"Transfer audit: {audit['verifiedCompatibleTensorCount']} compatible tensors; "
                  f"{len(audit['verifiedRows'])} classification rows; matched classes: "
                  f"{', '.join(audit['matchedClasses']) or 'none'}. "
                  f"Initial optimizer state entries: {optimizer['initialStateEntries']}.", ""]
    lines += ["### Detection", "", "Group values are macro averages over classes with references. "
              "Precision/recall/F1 use the fixed operational threshold; AP integrates the confidence ranking.", "",
              *markdownTable(["Group", "Supported / declared classes", "Instances", "AP50", "AP50-95", "Precision", "Recall", "F1"],
               [[name, f"{row['supportedClassCount']} / {row['classCount']}", row["support"], row["ap50"],
                 row["ap50To95"], row["precision"], row["recall"], row["f1"]] for name, row in detection["groups"].items()]),
              *markdownTable(["ID", "Class", "Instances", "Images", "AP50-95", "TP", "FP", "FN", "Precision", "Recall"],
               [[row["classId"], row["className"], row["support"], row["imageSupport"], row["ap50To95"], row["truePositives"],
                 row["falsePositives"], row["falseNegatives"], row["precision"], row["recall"]] for row in detection["perClass"]]),
              f"![Class detection and support]({taskId}/figures/{figures['classes']})", "",
              f"[Full detection table]({taskId}/tables/classDetection.csv). Unsupported scores remain N/A; counts are retained.", "",
              "### Original-match diagnostics", "",
              *markdownTable(["Match", "Images", "Supported classes", "Macro AP50-95", "Macro precision", "Macro recall"],
               [[matchId, values["imageCount"], values["groups"]["all"]["supportedClassCount"],
                 values["groups"]["all"]["ap50To95"], values["groups"]["all"]["precision"], values["groups"]["all"]["recall"]]
                for matchId, values in detection["byMatch"].items()]),
              f"![Original-match diagnostics]({taskId}/figures/{figures['matches']})", "",
              "These are descriptive results. Frames are not independent statistical replicates.", ""]
    if not sum(row["predictions"] for row in detection["perClass"]):
        lines += ["No detections met the operational confidence. Annotated instances remain false negatives. "
                  "AP may still be nonzero because it also uses lower-confidence detections.", ""]
    if "precisionRecall" in figures:
        lines += [f"![Object precision-recall]({taskId}/figures/{figures['precisionRecall']})", ""]
    else:
        lines += ["No informative object precision-recall curve: no positive precision values were observed.", ""]
    labels, matrix = detection["confusionLabels"], detection["confusionMatrix"]
    confusions = sorted([(matrix[i][j], labels[i], labels[j]) for i in range(len(labels) - 1)
                          for j in range(len(labels) - 1) if i != j and matrix[i][j]], reverse=True)[:10]
    if confusions:
        lines += ["Most frequent cross-class operational confusions:", "",
                  *markdownTable(["True class", "Predicted class", "Count"], [[a, b, n] for n, a, b in confusions])]
    else:
        lines += ["No cross-class associations at the operational threshold. Misses and false positives are retained in the tables.", ""]
    if points["perClass"]:
        key = str(state["protocol"]["primaryPointTolerance"])
        lines += ["### Court-point localization", "",
                  "References are native points. A predicted location is its detection-box center. "
                  "The highest-confidence detection is used, never the closest. "
                  "Distances are fractions of the original image diagonal. Error median/p90 include only points "
                  "with a prediction; success rates include every annotated point, including misses.", "",
                  *markdownTable(["Point", "Annotated", "Predicted", "Missing", "Extra predictions",
                                  f"Success at {100 * float(key):g}%", "Median error / diagonal", "P90 / diagonal",
                                  "Truncated targets", "Median proxy-center offset / diagonal"],
                   [[row["className"], row["support"], row["localizedCount"], row["missingCount"], row["extraPredictions"],
                     row["successRates"][key], row["medianDistanceDiagonal"], row["p90DistanceDiagonal"],
                     row["truncatedTargets"], row["medianTargetCenterOffsetDiagonal"]] for row in points["perClass"]]),
                  f"[Localization by original match]({taskId}/tables/matchPointLocalization.csv). "
                  f"[Success rates by match and tolerance]({taskId}/tables/matchPointToleranceSuccess.csv). "
                  "The proxy-center offset describes artificial-box representation; no offset correction is applied.", ""]
        if "points" in figures:
            lines += [f"![Point localization]({taskId}/figures/{figures['points']})", ""]
    lines += ["### Reproducible qualitative examples", "",
              "Fixed cases are shared across stages. Difficult cases are selected by the recorded error criterion, "
              "not as a representative accuracy estimate. Class IDs use the table above; line segments show native-point displacement.", ""]
    for example in examples:
        lines += [f"**{example['recordId']}** — {example['reason']}.", "",
                  f"![{example['recordId']}]({taskId}/examples/{example['figure']})", ""]
    if "history" in figures:
        lines += ["### Stage training history", "", f"![Training history]({taskId}/figures/{figures['history']})", ""]
    else:
        lines += ["Training history has fewer than two recorded epochs; no learning curve is drawn.", ""]
    lines += ["### Evidence and supplementary material", "",
              f"- [Pixel-coordinate predictions]({stage['evidenceDirectory']}/predictions.jsonl)",
              f"- [Native references]({stage['evidenceDirectory']}/references.jsonl)",
              f"- [Operational associations]({taskId}/tables/detectionAssociations.csv)",
              f"- [Full confusion counts]({taskId}/tables/confusionCounts.csv)",
              f"- [Confusion normalized by true class]({taskId}/tables/confusionNormalizedByTrueClass.csv)", ""]
    if "confusion" in figures:
        lines += [f"[Complete confusion figure (SVG)]({taskId}/figures/confusionCounts.svg). "
                  "Rows are true classes; columns are predictions; colour uses log(1 + count).", ""]
    return lines


def reportMarkdown(root, state):
    config = state["protocol"]
    trainingCode, evaluationCode = state["training"].get("code", {}), state.get("code", {})
    lines = ["# Validation research report", "", f"Run: **{state['trainingRunId']}**. Purpose: **{state['purpose']}**.", "",
             *markdownTable(["Training", "Evaluation", "Selection", "Test evaluated"],
                            [[state["training"]["status"], state["status"], state["selection"], False]]),
             *markdownTable(["Training command elapsed (s)", "Research evaluation elapsed (s)"],
                            [[elapsedSeconds(state["training"]), elapsedSeconds(state)]]),
             "Elapsed times include integrity verification and processing overhead; they are not inference benchmarks.", "",
             "Each curriculum stage is a different task. Stage scores are not a ranking or evidence that "
             "the curriculum improves performance. This technical smoke does not estimate scientific model quality.", "",
             "## Reading guide", "", *[f"- [{stage['taskId']}](#{stage['taskId'].lower()})" for stage in state["stages"]], "",
             "## Protocol and provenance", "",
             f"Confidence floor: {config['confidenceFloor']:g}; operational confidence >= {config['operationalConfidence']:g}; "
             f"operational IoU >= {config['operationalIou']:g}; AP IoU 0.50:0.95; max detections: {config['maxDetections']}.", "",
             "AP uses the recorded native matching masks in model-input coordinates. Operational counts and point "
             "localization use the archived original-image pixel coordinates. Operational matching is confidence-greedy "
             "and class-agnostic; a wrong-class match is an FP and an FN. Group averages exclude unsupported classes.", "",
             f"Dataset contract: `{state['contractId']}`. Evaluation protocol: `{state['protocolId']}`. "
             f"Reviewed own records: {state['reviewedOwnRecordCount']}.", "",
             f"Training code: `{trainingCode.get('commit', 'unavailable')}` "
             f"(dirty={trainingCode.get('dirty', 'unavailable')}). Evaluation code: `{evaluationCode.get('commit', 'unavailable')}` "
             f"(dirty={evaluationCode.get('dirty', 'unavailable')}).", "",
             "Input and evidence checks are recorded in [evaluation.json](evaluation.json) and "
             "[evaluationContract.json](evaluationContract.json). No automatic threshold calibration, test evaluation, "
             "experiment comparison, inferential significance test or hardware performance benchmark is performed.", ""]
    if state.get("ownSplitSummaries"):
        rows = [{"taskId": taskId, "split": split, "reviewedOwnRecords": values["reviewedFrameCount"],
                 "reviewedOwnFraction": values["reviewedFraction"], "eligibleOwnRecords": values["eligibleFrameCount"],
                 "eligibleOwnFraction": values["eligibleFraction"], "quarantinedOwnRecords": values["quarantinedFrameCount"]}
                for taskId, splits in state["ownSplitSummaries"].items()
                for split, values in sorted(splits.items(), key=lambda item: ("training", "validation", "test").index(item[0]))]
        (root / "tables").mkdir()
        writeCsv(root / "tables" / "ownSplitContext.csv", rows)
        lines += ["## Frozen own-data partition context", "",
                  "Reviewed proportions use the original reviewed-record denominator. Effective proportions use "
                  "eligible own records after each task's quarantine. Public training images are counted separately "
                  "in the stage sections. This is eligibility metadata, not each stage's source selection; "
                  "actual training sources follow in the stage chapters. Test rows are metadata only; test images are not read.", "",
                  *markdownTable(["Stage", "Split", "Reviewed own", "Reviewed %", "Eligible own", "Effective %", "Quarantined own"],
                   [[row["taskId"], row["split"], row["reviewedOwnRecords"], 100 * row["reviewedOwnFraction"],
                     row["eligibleOwnRecords"], 100 * row["eligibleOwnFraction"], row["quarantinedOwnRecords"]] for row in rows]),
                  "[Partition counts and fractions](tables/ownSplitContext.csv)", ""]
    if state.get("error"):
        lines += [f"Evaluation failure: `{state['error']['type']}: {state['error']['message']}`.", ""]
    if state.get("integrityError"):
        lines += [f"Integrity failure: **{state['integrityError']}**. Treat numerical outputs as invalid.", ""]
    for stage in state["stages"]:
        lines += stageMarkdown(root, state, stage)
    return "\n".join(lines)


def updateReportIndex(root):
    groups = defaultdict(list)
    for path in sorted(root.rglob("reportState.json")):
        report = readJson(path)
        target = path.parent / "report.md" if (path.parent / "report.md").is_file() else path
        link = target.relative_to(root).as_posix()
        groups[report["runId"]].append((report, path.parent.name, link))
    lines = ["# Research report catalogue", "",
             "Reports are grouped by training run, newest presentation first, without quality rankings. "
             "Different validation selections and regenerated presentations remain explicit.", ""]
    for runId in sorted(groups):
        items = sorted(groups[runId], key=lambda item: item[0].get("endedAt", item[0].get("startedAt", "")), reverse=True)
        rows = [[report["purpose"], report["selection"], name, report["trainingStatus"], report["evaluationStatus"],
                 report["status"], report.get("endedAt", report.get("startedAt", "unavailable"))[:19], f"[Open]({link})"]
                for report, name, link in items]
        lines += [f"## {runId}", "", *markdownTable(
            ["Purpose", "Selection", "Presentation", "Training", "Evaluation", "Report", "Generated UTC", "Link"], rows)]
    atomicWrite(root / "index.md", "\n".join(lines))


def renderReport(evaluationDir, outputDir=None):
    source = Path(evaluationDir).resolve()
    state, contract = verifyEvaluation(source)
    inputRoots = [source, *[Path(value) for value in
                  (state.get("trainingDirectory"), state["training"].get("datasetDirectory")) if value]]
    root = source if outputDir is None else validateResearchOutput(outputDir, inputRoots)
    if (root / "report.md").exists() or (root / "reportState.json").exists():
        raise ValueError("Report output must be new; regenerating requires a separate directory.")
    if root != source:
        root.mkdir(parents=True)
        shutil.copytree(source / "evidence", root / "evidence")
        for name in ("evaluation.json", "evaluationContract.json"):
            shutil.copyfile(source / name, root / name)
    cache = root.parent / f".{root.name}PlotCache"
    cache.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    report = {"schemaVersion": 1, "status": "running", "startedAt": utcNow(),
              "runId": state["trainingRunId"], "purpose": state["purpose"], "selection": state["selection"],
              "trainingStatus": state["training"]["status"], "evaluationStatus": state["status"],
              "evaluationContractId": contract["contractId"]}
    writeJson(root / "reportState.json", report)
    try:
        report["renderingEnvironment"] = archiveRenderer(root)
        atomicWrite(root / "report.md", reportMarkdown(root, state))
        report["status"] = "completed"
        report["artifacts"] = [fileEvidence(path, root) for path in sorted(root.rglob("*"))
                               if path.is_file() and path.name != "reportState.json"]
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        report["endedAt"] = utcNow()
        atomicWrite(root / "reportState.json", json.dumps(report, sort_keys=True, allow_nan=False, indent=2) + "\n")
        catalog = next((parent for parent in root.parents if parent.name == "reports"), None)
        if catalog:
            updateReportIndex(catalog)
    return root / "report.md"


def renderTrainingReport(runDir, outputDir):
    runDir = Path(runDir).resolve()
    run = readJson(runDir / "run.json")
    inputRoots = [runDir, *([Path(run["datasetDirectory"])] if run.get("datasetDirectory") else [])]
    root = validateResearchOutput(outputDir, inputRoots)
    root.mkdir(parents=True)
    shutil.copyfile(runDir / "run.json", root / "trainingRun.json")
    lines = ["# Training execution report", "", f"Run: **{runDir.name}**. Purpose: **{run['purpose']}**.", "",
             f"Training status: **{run['status']}**. Research evaluation: **not executed**.", "",
             "No research scores or diagnostic figures are inferred from incomplete or unevaluated stages.", "",
             *markdownTable(["Stage", "Status"], [[stage["taskId"], next((row["status"] for row in run["stages"]
                              if row["taskId"] == stage["taskId"]), "notExecuted")]
                              for stage in run["configuration"]["stages"]])]
    if run.get("error"):
        lines += [f"Failure: `{run['error']['type']}: {run['error']['message']}`.", ""]
    lines += ["[Archived execution metadata](trainingRun.json)", ""]
    atomicWrite(root / "report.md", "\n".join(lines))
    writeJson(root / "reportState.json", {"schemaVersion": 1, "status": "completed", "runId": runDir.name,
              "purpose": run["purpose"], "selection": "notEvaluated", "trainingStatus": run["status"],
              "evaluationStatus": "notExecuted", "endedAt": utcNow()})
    catalog = next((parent for parent in root.parents if parent.name == "reports"), None)
    if catalog:
        updateReportIndex(catalog)
    return root / "report.md"
