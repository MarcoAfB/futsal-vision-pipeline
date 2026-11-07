import shutil
from pathlib import Path

from futsal.datasets.datasetFiles import readJson, writeJson
from futsal.evaluation.courtComparison import courtComparisonInputs
from futsal.evaluation.evaluationData import validateResearchOutput
from futsal.reporting.researchReport import (
    archiveRenderer,
    markdownTable,
    updateReportIndex,
    verifyEvaluation,
    writeCsv,
)
from futsal.training.trainingRun import fileEvidence, utcNow


def renderCourtComparison(evaluationDir, outputDir=None):
    source = Path(evaluationDir).resolve()
    state, seal = verifyEvaluation(source)
    if state.get("kind") != "courtTrainingComparison" or state["status"] != "completed":
        raise ValueError("A completed court training comparison is required.")
    courtComparisonInputs(source / "evidence/baseline", source / "evidence/candidate")
    root = source if outputDir is None else validateResearchOutput(outputDir, [source])
    if (root / "report.md").exists():
        raise ValueError("Report output must be new.")
    if root != source:
        root.mkdir(parents=True)
        shutil.copytree(source / "evidence", root / "evidence")
        for name in ("evaluation.json", "evaluationContract.json"):
            shutil.copyfile(source / name, root / name)
    tables = root / "tables"
    tables.mkdir()
    metrics = readJson(root / "evidence/comparisonMetrics.json")
    for name in ("groups", "classes", "points"):
        writeCsv(tables / f"{name}.csv", metrics[name])
    matchRows = [
        {"matchId": matchId, **row}
        for matchId, measures in metrics["byMatch"].items()
        for row in measures["points"]
    ]
    writeCsv(tables / "matchPoints.csv", matchRows)
    matchDetection = [
        {"matchId": matchId, **row}
        for matchId, measures in metrics["byMatch"].items()
        for row in measures["classes"]
    ]
    writeCsv(tables / "matchDetection.csv", matchDetection)
    costs = [
        {"training": label, **value}
        for label, value in zip(
            ("realOnly", "realWithSyntheticCourt"), state["trainingCosts"], strict=True
        )
    ]
    writeCsv(tables / "trainingCosts.csv", costs)
    lines = [
        "# Court training: real data and synthetic supplementation",
        "",
        "The same twenty-class detector and training routine are used in both executions. "
        "Only synthetic training membership changes. Both use identical real validation images.",
        "",
        f"Single seed: {state['singleSeed']}. Test evaluated: False.",
        "",
        "The same epoch limit does not equalize optimizer updates or training time. "
        "A single seed does not establish statistical significance.",
        "",
        *markdownTable(
            ["Training", "Epochs", "Batches", "Optimizer updates", "Seconds"],
            [
                [
                    row["training"],
                    row.get("completedEpochs"),
                    row.get("batches"),
                    row.get("optimizerUpdates"),
                    row.get("elapsedSeconds"),
                ]
                for row in costs
            ],
        ),
        *markdownTable(
            ["Class", "Metric", "Real", "Real + synthetic", "Difference"],
            [
                [row["className"], row["metric"], row["baseline"], row["candidate"], row["delta"]]
                for row in metrics["points"]
            ],
        ),
        "[Detection by class](tables/classes.csv), [target groups](tables/groups.csv), "
        "[point localization](tables/points.csv), [points by match](tables/matchPoints.csv), "
        "[detection by match](tables/matchDetection.csv), [training costs](tables/trainingCosts.csv).",
        "",
        "Unsupported validation classes remain N/A. Missing points count as failures; "
        "localization errors are conditional on available predictions.",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    environment = archiveRenderer(root)
    writeJson(
        root / "reportState.json",
        {
            "schemaVersion": 1,
            "status": "completed",
            "runId": "courtTrainingComparison",
            "purpose": state["purpose"],
            "selection": state["selection"],
            "trainingStatus": "completed",
            "evaluationStatus": state["status"],
            "endedAt": utcNow(),
            "evaluationContractId": seal["contractId"],
            "renderingEnvironment": environment,
            "artifacts": [
                fileEvidence(p, root)
                for p in sorted(root.rglob("*"))
                if p.is_file() and p.name != "reportState.json"
            ],
        },
    )
    catalog = next((parent for parent in root.parents if parent.name == "reports"), None)
    if catalog:
        updateReportIndex(catalog)
    return root / "report.md"
