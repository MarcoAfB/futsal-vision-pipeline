import os
import shutil
from pathlib import Path

import cv2

from futsal.datasets.datasetFiles import readJson, writeJson
from futsal.evaluation.configurationComparison import validateComparison
from futsal.evaluation.evaluationData import validateResearchOutput
from futsal.evaluation.modelComparison import comparisonModels
from futsal.inventory.catalog import hashFile
from futsal.reporting.modelComparisonReport import modelComparisonMarkdown
from futsal.reporting.reportExamples import examplePanel
from futsal.reporting.researchExamples import fixedExamples, selectExamples
from futsal.reporting.researchReport import (
    archiveRenderer,
    markdownTable,
    updateReportIndex,
    verifyEvaluation,
    writeCsv,
)
from futsal.training.trainingRun import fileEvidence, utcNow


def pairedExamples(root, state, baseline, candidate, values):
    names, protocol = state["classes"], state["protocol"]
    oldStage = next(stage for stage in baseline["stages"] if stage.get("classes") == names and stage["status"] == "completed")
    available = {}
    for prefix, examples in (("baseline", oldStage.get("examples", [])), ("candidate", candidate.get("examples", []))):
        for example in examples:
            available[example["recordId"]] = root / "evidence" / prefix / example["image"]
    refs = values[0]["references"]
    allowed = {row["recordId"] for row in refs} & set(available)
    rows = [row for row in refs if row["recordId"] in allowed]
    if not rows:
        return []
    fixed = fixedExamples({"common": {"rows": rows}}, protocol)
    errors = [{key: old[key] if key == "recordId" else max(old[key], new[key]) for key in old}
              for old, new in zip(values[0]["detection"]["imageErrors"], values[1]["detection"]["imageErrors"], strict=True)
              if old["recordId"] in allowed]
    pointErrors = []
    for old, new in zip(values[0]["points"]["errors"], values[1]["points"]["errors"], strict=True):
        if old["recordId"] in allowed:
            chosen = old if old["distanceDiagonal"] is None else new if new["distanceDiagonal"] is None else max((old, new), key=lambda row: row["distanceDiagonal"])
            pointErrors.append(chosen)
    selected = selectExamples(fixed, {"imageErrors": errors}, {"errors": pointErrors, "perClass": values[0]["points"]["perClass"]}, protocol)
    byId = {row["recordId"]: i for i, row in enumerate(refs)}
    destination = root / "examples"
    destination.mkdir()
    result = []
    for index, selectedRow in enumerate(selected, start=1):
        recordId = selectedRow["recordId"]
        reference = refs[byId[recordId]]
        if hashFile(available[recordId]) != reference["imageSha256"]:
            raise ValueError("Paired example image differs from the common canonical reference.")
        image = cv2.imread(str(available[recordId]))
        if image is None:
            raise ValueError("Cannot decode a sealed paired example.")
        panels = []
        for label, value in zip(("Baseline", "Candidate"), values, strict=True):
            errors = [row for row in value["points"]["errors"] if row["recordId"] == recordId]
            panel = examplePanel(image, reference, value["predictions"][byId[recordId]], names, protocol, state["pointClasses"], errors)
            filename = f"example{index:02d}{label}.png"
            if not cv2.imwrite(str(destination / filename), panel):
                raise OSError("Cannot write paired example panel.")
            panels.append(filename)
        result.append({**selectedRow, "figures": panels})
    return result


def comparisonMarkdown(root, state):
    lines = ["# Curriculum and independent detectors: validation comparison", "",
             f"Comparison status: **{state['status']}**. Seed: **{state['singleSeed']}**. "
             f"Common validation records: **{len(state['recordIds'])}**. Test evaluated: **False**.", "",
             "This compares complete configurations. Detector partition, public curriculum exposure and task-eligible "
             "training cohorts differ. Results do not isolate a curriculum effect or establish a statistical winner. "
             "Frames are not independent experimental replicates. No inference speed/memory benchmark is included.", ""]
    if state["status"] != "completed":
        return "\n".join(lines + [f"Integrity failure: {state.get('integrityError', 'See evaluation.json')}.", ""])
    mode = state.get("comparisonMode", "controlled")
    baseline, candidate, _, _, values = validateComparison(root / "evidence/baseline", root / "evidence/candidate",
                                                          comparisonMode=mode)
    models = comparisonModels(baseline["training"], [item["run"] for item in candidate["components"]])
    lines += modelComparisonMarkdown(mode, models)
    metrics = readJson(root / "evidence/comparisonMetrics.json")
    tables = root / "tables"
    tables.mkdir()
    for name in ("groups", "classes", "points"):
        writeCsv(tables / f"{name}.csv", metrics[name])
    rows = [row for row in metrics["groups"] if row["metric"] == "ap50To95" and row["group"] != "all"]
    lines += ["## Overview", "", "All deltas are candidate minus baseline. Higher AP/success is better; lower "
              "missing/error counts and localization distances are better. N/A remains N/A.", "",
              *markdownTable(["Target group", "Instances", "Curriculum AP50-95", "Independent AP50-95", "Delta"],
               [[row["group"], row["support"], row["baseline"], row["candidate"], row["delta"]] for row in rows])]
    components = []
    for item in candidate["components"]:
        selected = readJson(root / "evidence/candidate/evidence/components" / item["taskId"] / "selection.json")["tasks"][item["taskId"]]
        components.append([item["taskId"], len(selected["training"]), len(selected["validation"]), item["run"]["configuration"]["stages"][0]["epochs"]])
    baselineSelection = readJson(root / "evidence/baseline/evidence/trainingSelection.json")
    lastTask = baseline["training"]["configuration"]["stages"][-1]["taskId"]
    selected = baselineSelection["tasks"][lastTask]
    components.insert(0, ["curriculum final task", len(selected["training"]), len(selected["validation"]), baseline["training"]["configuration"]["stages"][-1]["epochs"]])
    lines += ["## Training exposure", "", *markdownTable(["Task", "Training records", "Full validation records", "Maximum epochs"], components),
              "The curriculum also has preceding public-data stages. The independent models each start directly "
              "from the official checkpoint. Maximum epochs are not necessarily completed epochs.", ""]
    for title, classes in (("Ball", state["classes"][:1]), ("Athletes", state["classes"][1:5])):
        lines += [f"## {title}", "", *markdownTable(["Class", "Metric", "Support", "Curriculum", "Independent", "Delta"],
            [[row["className"], row["metric"], row["support"], row["baseline"], row["candidate"], row["delta"]]
             for row in metrics["classes"] if row["className"] in classes and row["metric"] in {"ap50To95", "precision", "recall"}])]
    point = metrics["pointSuccess"]
    lines += ["## Court landmarks", "", *markdownTable(["Tolerance (image diagonal)", "Curriculum success", "Independent success", "Delta"],
               [[point["toleranceDiagonal"], point["baseline"], point["candidate"], point["delta"]]]),
              "Missing predictions count as failures; distance quantiles are conditional on localization. "
              "Unsupported lowerCornerSideB is retained with N/A scores.", "",
              *markdownTable(["Point", "Support", "Curriculum success", "Independent success", "Delta"],
                [[row["className"], row["support"], row["baseline"], row["candidate"], row["delta"]]
                 for row in metrics["points"] if row["metric"] == "successRate"]),
              "[Full point counts and conditional errors](tables/points.csv). "
              "[All class metrics](tables/classes.csv). [Group metrics](tables/groups.csv).", "",
              "## Paired examples", "", "Fixed shared records and a few larger errors from either configuration are "
              "selected among sealed example images. Each panel shows the same reference beside its prediction.", ""]
    for example in pairedExamples(root, state, baseline, candidate, values):
        lines += [f"### {example['recordId']} — {example['reason']}", "",
                  f"Curriculum: ![Reference and curriculum prediction](examples/{example['figures'][0]})", "",
                  f"Independent detectors: ![Reference and independent predictions](examples/{example['figures'][1]})", ""]
    lines += ["## Reproducibility", "", f"Baseline seal: `{state['baselineContractId']}`. "
              f"Candidate seal: `{state['candidateContractId']}`. Protocol: `{state['protocolId']}`.", "",
              "[Comparison metadata](evaluation.json) and [evidence seal](evaluationContract.json). "
              "Both input evaluations, native masks, references and component provenance are preserved under evidence/.", ""]
    return "\n".join(lines)


def renderComparisonReport(evaluationDir, outputDir=None):
    source = Path(evaluationDir).resolve()
    state, contract = verifyEvaluation(source)
    root = source if outputDir is None else validateResearchOutput(outputDir,
        [source, Path(state["baselineDirectory"]), Path(state["candidateDirectory"])])
    if (root / "report.md").exists() or (root / "reportState.json").exists():
        raise ValueError("Comparison report output must be new.")
    if root != source:
        root.mkdir(parents=True)
        shutil.copytree(source / "evidence", root / "evidence")
        for name in ("evaluation.json", "evaluationContract.json"):
            shutil.copyfile(source / name, root / name)
    cache = root.parent / f".{root.name}PlotCache"
    cache.mkdir(exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache)
    report = {"schemaVersion": 1, "status": "running", "startedAt": utcNow(), "runId": "configurationComparison",
        "purpose": state["purpose"], "selection": state["selection"], "trainingStatus": "completed",
        "evaluationStatus": state["status"], "evaluationContractId": contract["contractId"]}
    try:
        report["renderingEnvironment"] = archiveRenderer(root)
        (root / "report.md").write_text(comparisonMarkdown(root, state), encoding="utf-8")
        report["status"] = "completed"
        report["artifacts"] = [fileEvidence(path, root) for path in sorted(root.rglob("*")) if path.is_file() and path.name != "reportState.json"]
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["endedAt"] = utcNow()
        writeJson(root / "reportState.json", report)
        catalog = next((parent for parent in root.parents if parent.name == "reports"), None)
        if catalog:
            updateReportIndex(catalog)
    return root / "report.md"
