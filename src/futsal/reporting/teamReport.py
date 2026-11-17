import shutil
from pathlib import Path

from futsal.datasets.datasetFiles import readJson, writeJson
from futsal.evaluation.evaluationData import validateResearchOutput
from futsal.reporting.report import renderTable
from futsal.reporting.researchReport import archiveRenderer, verifyEvaluation
from futsal.training.trainingRun import utcNow


def percent(value):
    return "N/A" if value is None else f"{100 * value:.2f}"


def number(value, digits=2):
    return "N/A" if value is None else f"{value:.{digits}f}"


def variants(summary):
    return [(source, variant, summary[source][variant]) for source in ("pipeline", "baseline")
            for variant in ("named", "bestSwap")]


def overallTable(summary):
    rows = [[source, variant, item["assignment"]["matched"], percent(item["assignment"]["roleAccuracy"]),
             percent(item["assignment"]["teamAccuracy"]), percent(item["assignment"]["classAccuracy"]),
             percent(item["macroF1"]), ", ".join(item.get("swappedMatches", [])) or "-"]
            for source, variant, item in variants(summary)]
    return renderTable("Four-class assignment", ["Source", "Names", "Matched athletes", "Role accuracy %",
                                                 "Team accuracy %", "Class accuracy %", "Macro F1 %",
                                                 "Swapped matches"], rows)


def perClassTable(summary):
    rows = []
    for source, variant, item in variants(summary):
        rows += [[source, variant, row["className"], row["support"], row["predictions"], percent(row["precision"]),
                  percent(row["recall"]), percent(row["f1"])] for row in item["perClass"]]
    return renderTable("Per-class operational scores", ["Source", "Names", "Class", "References", "Predictions",
                                                        "Precision %", "Recall %", "F1 %"], rows)


def matchTable(summary):
    rows = []
    for source, variant, item in variants(summary):
        rows += [[source, variant, matchId, scores["matched"], percent(scores["roleAccuracy"]),
                  percent(scores["teamAccuracy"]), percent(scores["classAccuracy"])]
                 for matchId, scores in item["byMatch"].items()]
    return renderTable("Assignment by match", ["Source", "Names", "Match", "Matched athletes", "Role accuracy %",
                                               "Team accuracy %", "Class accuracy %"], rows)


def decisionTables(decisions):
    rows, clusters = [], []
    for matchId, decision in decisions.items():
        meanX = ", ".join(f"cluster {team}: {number(value)}" for team, value in decision["outfieldNaming"]["meanX"].items())
        rows.append([matchId, decision["outfieldNaming"]["source"], meanX,
                     ", ".join(f"cluster {team} = {name}" for team, name in decision["teams"].items()),
                     "/".join(decision["goalkeeperSides"]) or "none", decision["reassignedGoalkeepers"]])
        clusters += [[matchId, cluster, item["size"], percent(item["share"]), item["role"],
                      item.get("side") or item.get("team"), item["source"], item["projected"],
                      percent(item["goalZoneFraction"]), percent(item["sideConsistency"])]
                     for cluster, item in decision["clusters"].items()]
    return [*renderTable("Naming decisions", ["Match", "Outfield naming", "Mean court x (m)", "Team names",
                                             "Goalkeeper sides", "Goalkeepers reassigned by frame limit"], rows),
            *renderTable("Clusters", ["Match", "Cluster", "Crops", "Share %", "Role", "Side or team", "Source",
                                      "Projected", "Goal-zone %", "Side consistency %"], clusters)]


def confusionTable(summary):
    item = summary["pipeline"]["named"]
    labels = item["confusionLabels"]
    return renderTable("Pipeline confusion (reference rows, prediction columns, as named)", ["Reference", *labels],
                       [[label, *row] for label, row in zip(labels, item["confusionMatrix"], strict=True)])


def inputLines(state, summary):
    inputs, cohort = state["inputs"], summary["cohort"]
    features = state["featureConfiguration"]
    return ["## Inputs and protocol", "",
            *[f"- {name}: `{item['directory']}` (contract `{item['contractId']}`)" for name, item in inputs.items()],
            "",
            f"Cohort: {cohort['frames']} frames shared by the team features ({cohort['featureFrames']}) and the "
            f"detailed references ({cohort['referenceFrames']}); {cohort['framesWithHomographyRow']} have a "
            f"homography row; {cohort['athleteReferences']} annotated athletes; {cohort['unassignedDetections']} "
            "detections had no usable crop and are not predictions.", "",
            "Feature configuration: " + ", ".join(f"{key}={value}" for key, value in sorted(features.items())), "",
            "Assignment configuration: " + ", ".join(f"{key}={value}" for key, value in
                                                   sorted(state["configuration"].items())), ""]


def methodLines():
    return ["## Method", "",
            "Generic player detections are cropped to the torso and embedded. Each match is clustered once; no "
            "tracking is used. The two largest clusters are the outfield teams. A smaller cluster becomes a "
            "goalkeeper when its projected members lie mostly near one goal, or, without enough projections, when "
            "the side-specific landmarks of its frames agree on one side. Other clusters join the nearest team. "
            "At most one goalkeeper per side is kept in a frame.", "",
            "Names: a goalkeeper takes the side of the goal it defends; the outfield team with the lower mean "
            "court x is team A. Team A is the team defending Side A in the clip, which the validation labels "
            "followed for every goalkeeper. Naming source `none` means the names are arbitrary.", "",
            "Scoring matches boxes class-agnostically at the configured IoU. `named` uses the predicted names; "
            "`bestSwap` exchanges A and B jointly for outfield players and goalkeepers when that increases correct "
            "classes in a match. bestSwap reads the labels and is a diagnostic, not a deployable result. The "
            "baseline is the four-class detector of the reference evaluation, scored the same way.", ""]


def limitationLines():
    return ["## Limitations", "",
            "- Validation frames only; one seed and one detector checkpoint per input. No significance is claimed.",
            "- Few matches; goalkeepers of one team can be rare or absent in a match.",
            "- The side convention fails for clips that cross a change of ends.",
            "- Thresholds were frozen before results; tuning them on these frames would bias the result.", ""]


def teamMarkdown(root, state):
    summary = readJson(root / "evidence" / "summary.json")
    decisions = readJson(root / "evidence" / "matches.json")
    lines = ["# Team and role assignment report", "",
             f"Status: **{state['status']}**. Purpose: **{state['purpose']}**. Test evaluated: "
             f"**{state['testEvaluated']}**.", "", *inputLines(state, summary), *methodLines(),
             *overallTable(summary), *matchTable(summary), *perClassTable(summary), *confusionTable(summary),
             *decisionTables(decisions), *limitationLines()]
    return "\n".join(lines)


def renderTeamReport(evaluationDir, outputDir=None):
    source = Path(evaluationDir).resolve()
    state, contract = verifyEvaluation(source)
    if state.get("kind") != "teamEvaluation":
        raise ValueError("Expected a sealed team evaluation.")
    root = source if outputDir is None else validateResearchOutput(outputDir, [source])
    if (root / "report.md").exists() or (root / "reportState.json").exists():
        raise ValueError("Report output must be new; regenerate into a separate directory.")
    if root != source:
        root.mkdir(parents=True)
        shutil.copytree(source / "evidence", root / "evidence")
        for name in ("evaluation.json", "evaluationContract.json"):
            shutil.copyfile(source / name, root / name)
        verifyEvaluation(root)
    report = {"schemaVersion": 1, "status": "running", "startedAt": utcNow(), "evaluationStatus": state["status"],
              "evaluationContractId": contract["contractId"]}
    try:
        report["renderingEnvironment"] = archiveRenderer(root)
        text = teamMarkdown(root, state) if state["status"] == "completed" else \
            f"# Team and role assignment report\n\nStatus: **{state['status']}**: {state.get('error')}\n"
        (root / "report.md").write_text(text, encoding="utf-8", newline="\n")
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["endedAt"] = utcNow()
        writeJson(root / "reportState.json", report)
    return root / "report.md"
