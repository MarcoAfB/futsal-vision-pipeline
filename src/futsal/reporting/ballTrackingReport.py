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

metricFields = ["frames", "positiveFrames", "tp", "fp", "fn", "precision", "recall", "f1",
                "lossCount", "maxLossFrames", "recoveredLossCount", "meanRecoveryFrames",
                "observedIdChanges", "adjacentMatchedPairs"]


def trackingTables(root, state):
    summary, byMatch, frames, losses = [], [], [], []
    for condition in state["conditions"]:
        base = {key: condition[key] for key in ("conditionId", "size", "tracker", "status")}
        row = {**base, "frames": condition.get("frames"), "error": condition.get("error", ""),
               "reusedFromContractId": condition.get("reusedFromContractId", "")}
        if condition["status"] == "completed" and not state.get("integrityError"):
            metrics = readJson(root / condition["evidenceDirectory"] / "metrics.json")
            row.update({key: metrics["overall"][key] for key in metricFields})
            byMatch.extend({**base, "matchId": matchId, **{key: values[key] for key in metricFields}}
                           for matchId, values in metrics["byMatch"].items())
            frames.extend({**base, **frame} for frame in metrics["frames"])
            losses.extend({**base, **loss} for loss in metrics["overall"]["lossRuns"])
        summary.append(row)
    tables = root / "tables"
    tables.mkdir()
    baseFields = ["conditionId", "size", "tracker", "status"]
    writeCsv(tables / "summary.csv", summary, [*baseFields, *metricFields, "error", "reusedFromContractId"])
    writeCsv(tables / "byMatch.csv", byMatch, [*baseFields, "matchId", *metricFields])
    writeCsv(tables / "frameOutcomes.csv", frames, [*baseFields, "recordId", "matchId", "segmentId", "frameIndex", "positive", "tp", "fp", "fn", "matchedTrackId"])
    writeCsv(tables / "lossRuns.csv", losses, [*baseFields, "segmentId", "startFrame", "length", "precededByHit", "recovered"])
    selected = state.get("selectionEvidence", {})
    writeCsv(tables / "segments.csv", selected.get("segments", []), ["segmentId", "matchId", "sourceId", "intervalId", "startFrame", "endFrame", "frameCount", "internalCutsStatus"])
    writeCsv(tables / "excluded.csv", selected.get("excluded", []), ["recordId", "matchId", "frameIndex", "reasons"])
    return summary


def trackingMarkdown(state, summary):
    selected = state.get("selectionEvidence", {})
    lines = ["# Ball tracking comparison", "", f"Status: **{state['status']}**. Selection: own validation. Test evaluated: **False**.", "",
             f"Frames: {selected.get('frameCount', 'N/A')}; positives: {selected.get('positiveFrames', 'N/A')}; "
             f"negatives: {selected.get('negativeFrames', 'N/A')}; contiguous blocks: {len(selected.get('segments', []))}.", "",
             "Results describe fixed-threshold frame quality on emitted tracker outputs. They are not detection AP, MOTA, IDF1 or HOTA. "
             "Internal cuts are unverified; temporal losses, recoveries and adjacent-match ID changes are exploratory diagnostics. "
             "Detector-only ID diagnostics are N/A. Native tracker parameters and ReID-disabled settings are preserved.", "",
             "No automatic winner is selected. Matches, rather than consecutive frames, are experimental units; "
             "the small validation cohort and single training seed limit interpretation. Requested training batches may differ; "
             "effective training batches are not established by folder names or this comparison. "
             "Execution timestamps include setup and integrity checks and are not a latency/FPS benchmark.", ""]
    if state.get("error"):
        lines += [f"Campaign error: {state['error']}", ""]
    if state.get("integrityError"):
        lines += [f"Integrity failure: **{state['integrityError']}**. Numerical results are invalid and suppressed.", ""]
    columns = ["size", "tracker", "status", "frames", "tp", "fp", "fn", "precision", "recall", "f1", "lossCount", "maxLossFrames"]
    lines += markdownTable(columns, [[row.get(key) for key in columns] for row in summary])
    lines += ["## Evidence and tables", "", "- [Summary](tables/summary.csv)", "- [Per-match metrics](tables/byMatch.csv)",
              "- [Frame outcomes](tables/frameOutcomes.csv)", "- [Loss runs](tables/lossRuns.csv)",
              "- [Sequence boundaries](tables/segments.csv)", "- [Quarantine exclusions](tables/excluded.csv)",
              "- [Evaluation metadata and protocol](evaluation.json)", "- [Evidence contract](evaluationContract.json)", "",
              "## Training provenance", ""]
    lines += markdownTable(["Size", "Training run SHA-256", "Checkpoint SHA-256", "Requested batch", "Seed"],
        [[size, model["runSha256"], model["checkpointSha256"], model["trainingConfiguration"]["parameters"]["batch"],
          model["trainingConfiguration"]["seed"]] for size, model in state.get("models", {}).items()])
    lines += ["", "Training records and selections, native tracker YAMLs, code snapshot, worker logs, frame predictions and metrics are sealed under evidence/.", ""]
    return "\n".join(lines)


def renderBallTrackingReport(evaluationDir, outputDir=None):
    source = Path(evaluationDir).resolve()
    state, contract = verifyEvaluation(source)
    if state.get("kind") != "ballTrackingComparison":
        raise ValueError("Expected a sealed ball tracking comparison.")
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
