import shutil
from pathlib import Path

from futsal.datasets.datasetFiles import readJson, writeJson
from futsal.evaluation.evaluationData import validateResearchOutput
from futsal.reporting.researchReport import (
    archiveRenderer,
    markdownTable,
    updateReportIndex,
    verifyEvaluation,
    writeCsv,
)
from futsal.training.trainingRun import fileEvidence, utcNow

detectionFields = ["frames", "targets", "positiveFrames", "tp", "fp", "fn", "precision", "recall", "f1"]
stabilityFields = ["trackCount", "meanTrackLength", "medianTrackLength", "singleFrameTracks", "referenceChains",
                   "matchedChains", "chainsWithMultipleIds", "meanIdsPerMatchedChain", "adjacentMatchedPairs",
                   "adjacentIdChanges", "gapRecoveries", "gapIdChanges"]
baseFields = ["conditionId", "label", "size", "tracker", "status"]


def metricRow(values):
    return {**{key: values[key] for key in detectionFields},
            **{key: (values["stability"] or {}).get(key) for key in stabilityFields}}


def trackingTables(root, state):
    summary, byMatch, byClass, frames = [], [], [], []
    for condition in state["conditions"]:
        base = {key: condition[key] for key in baseFields}
        row = {**base, "framesProcessed": condition.get("frames"), "error": condition.get("error", ""),
               "reusedFromContractId": condition.get("reusedFromContractId", "")}
        if condition["status"] == "completed" and not state.get("integrityError"):
            metrics = readJson(root / condition["evidenceDirectory"] / "metrics.json")
            row.update(metricRow(metrics["overall"]))
            byMatch.extend({**base, "matchId": matchId, **metricRow(values)} for matchId, values in metrics["byMatch"].items())
            byClass.extend({**base, "className": name, **{key: values[key] for key in ("tp", "fp", "fn", "precision", "recall", "f1")}}
                           for name, values in metrics["overall"]["perClass"].items())
            frames.extend({**base, **{key: frame[key] for key in ("recordId", "matchId", "segmentId", "frameIndex",
                                                                  "targets", "detections", "tp", "fp", "fn")}}
                          for frame in metrics["frames"])
        summary.append(row)
    tables = root / "tables"
    tables.mkdir()
    fields = [*baseFields, *detectionFields, *stabilityFields]
    writeCsv(tables / "summary.csv", summary, [*fields, "framesProcessed", "error", "reusedFromContractId"])
    writeCsv(tables / "byMatch.csv", byMatch, [*baseFields, "matchId", *detectionFields, *stabilityFields])
    writeCsv(tables / "byClass.csv", byClass, [*baseFields, "className", "tp", "fp", "fn", "precision", "recall", "f1"])
    writeCsv(tables / "frameOutcomes.csv", frames, [*baseFields, "recordId", "matchId", "segmentId", "frameIndex",
                                                    "targets", "detections", "tp", "fp", "fn"])
    selected = state.get("selectionEvidence", {})
    writeCsv(tables / "segments.csv", selected.get("segments", []),
             ["segmentId", "matchId", "sourceId", "intervalId", "startFrame", "endFrame", "frameCount", "internalCutsStatus"])
    writeCsv(tables / "excluded.csv", selected.get("excluded", []), ["recordId", "matchId", "frameIndex", "reasons"])
    return summary


def trackingMarkdown(state, summary):
    selected = state.get("selectionEvidence", {})
    lines = ["# Tracking comparison", "",
             f"Status: **{state['status']}**. Task: {selected.get('taskId', 'N/A')} "
             f"({', '.join(selected.get('classes', []))}). Selection: own validation. Test evaluated: **False**.", "",
             f"Frames: {selected.get('frameCount', 'N/A')}; reference objects: {selected.get('targetCount', 'N/A')}; "
             f"frames without objects: {selected.get('negativeFrames', 'N/A')}; contiguous blocks: {len(selected.get('segments', []))}.", "",
             "Detection columns use fixed thresholds on emitted outputs with one-to-one same-class IoU matching; they are not detection AP, "
             "MOTA, IDF1 or HOTA. Annotation track IDs are not verified identities, so stability columns are exploratory: reference "
             "objects are chained across adjacent frames by greedy same-class IoU, and a tracker ID change is counted when the chain's "
             "matched detections carry different IDs (adjacent frames, or across a missed gap). Track counts and lengths describe the "
             "tracker output above the operational confidence. Detector-only stability is N/A. Native tracker parameters and "
             "ReID-disabled settings are preserved.", "",
             "No automatic winner is selected. Matches, rather than consecutive frames, are experimental units; the small validation "
             "cohort and single training seed limit interpretation, and validation also selected each checkpoint. Execution "
             "timestamps include setup and integrity checks and are not a latency/FPS benchmark.", ""]
    if state.get("error"):
        lines += [f"Campaign error: {state['error']}", ""]
    if state.get("integrityError"):
        lines += [f"Integrity failure: **{state['integrityError']}**. Numerical results are invalid and suppressed.", ""]
    columns = ["label", "tracker", "status", "tp", "fp", "fn", "precision", "recall", "f1", "trackCount",
               "chainsWithMultipleIds", "adjacentIdChanges", "gapIdChanges"]
    lines += markdownTable(columns, [[row.get(key) for key in columns] for row in summary])
    lines += ["## Evidence and tables", "", "- [Summary](tables/summary.csv)", "- [Per-match metrics](tables/byMatch.csv)",
              "- [Per-class detection counts](tables/byClass.csv)", "- [Frame outcomes](tables/frameOutcomes.csv)",
              "- [Sequence boundaries](tables/segments.csv)", "- [Quarantine exclusions](tables/excluded.csv)",
              "- [Evaluation metadata and protocol](evaluation.json)", "- [Evidence contract](evaluationContract.json)", "",
              "## Training provenance", ""]
    lines += markdownTable(["Run", "Size", "Training run SHA-256", "Checkpoint SHA-256", "Requested batch", "Seed"],
        [[label, model["size"], model["runSha256"], model["checkpointSha256"],
          model["trainingConfiguration"]["parameters"]["batch"], model["trainingConfiguration"]["seed"]]
         for label, model in state.get("models", {}).items()])
    lines += ["", "Training records and selections, native tracker YAMLs, code snapshot, worker logs, frame predictions and metrics are sealed under evidence/.", ""]
    return "\n".join(lines)


def renderTrackingReport(evaluationDir, outputDir=None):
    source = Path(evaluationDir).resolve()
    state, contract = verifyEvaluation(source)
    if state.get("kind") != "trackingComparison":
        raise ValueError("Expected a sealed tracking comparison.")
    protected = [source, *[Path(model["runDirectory"]) for model in state.get("models", {}).values()]]
    root = source if outputDir is None else validateResearchOutput(outputDir, protected)
    if (root / "reportState.json").exists() or (root / "report.md").exists():
        raise ValueError("Report output must be new; regenerate into a separate directory.")
    if root != source:
        root.mkdir(parents=True)
        shutil.copytree(source / "evidence", root / "evidence")
        for name in ("evaluation.json", "evaluationContract.json"):
            shutil.copyfile(source / name, root / name)
        verifyEvaluation(root)
    report = {"schemaVersion": 1, "status": "running", "startedAt": utcNow(), "runId": state.get("campaignId", source.name),
              "purpose": state["purpose"], "selection": state["selection"], "trainingStatus": "completed" if state.get("models") else "unverified",
              "evaluationStatus": state["status"], "evaluationContractId": contract["contractId"]}
    try:
        report["renderingEnvironment"] = archiveRenderer(root)
        summary = trackingTables(root, state)
        (root / "report.md").write_text(trackingMarkdown(state, summary), encoding="utf-8", newline="\n")
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["endedAt"] = utcNow()
        report["artifacts"] = [fileEvidence(path, root) for path in sorted(root.rglob("*"))
                               if path.is_file() and path.name != "reportState.json"]
        writeJson(root / "reportState.json", report)
        catalog = next((parent for parent in root.parents if parent.name == "reports"), None)
        if catalog:
            updateReportIndex(catalog)
    return root / "report.md"
