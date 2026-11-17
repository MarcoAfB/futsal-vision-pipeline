import shutil
from pathlib import Path

from futsal.datasets.datasetFiles import readJson, writeJson
from futsal.evaluation.evaluationData import validateResearchOutput
from futsal.reporting.report import renderTable
from futsal.reporting.researchReport import archiveRenderer, verifyEvaluation
from futsal.training.trainingRun import utcNow

reasonText = {
    "insufficientPoints": "fewer than four landmarks",
    "degenerateCourtConfiguration": "no four landmarks without three collinear court points",
    "noConsistentSubset": "every four-landmark fit places its own points behind the camera",
    "estimationFailed": "least-squares estimation failed",
    "inlierRmseHigh": "inlier RMSE above limit",
    "insufficientImageSpread": "landmarks cover too little of the image",
    "pointsBehindCamera": "court points behind the camera",
    "orientationReversed": "mirrored court orientation",
    "farNearInverted": "far and near touchlines inverted",
}


def percent(value):
    return "N/A" if value is None else f"{100 * value:.2f}"


def number(value, digits=2):
    return "N/A" if value is None else f"{value:.{digits}f}"


def counts(mapping, keys):
    return [str(mapping.get(key, 0)) for key in keys]


def groups(summary):
    return [("all", summary["overall"]), *summary["byMatch"].items()]


def availabilityTables(summary):
    statuses = ("verified", "minimal", "rejected")
    reasons = sorted({reason for _, group in groups(summary)
                      for reason in (*group["referenceReasons"], *group["predictionReasons"])})
    lines = renderTable("Homography availability", ["Group", "Frames", "Reference verified/minimal/rejected",
                                                     "Prediction verified/minimal/rejected", "Median predicted points",
                                                     "Edge-touching candidates"],
                        [[name, group["frames"], "/".join(counts(group["referenceStatus"], statuses)),
                          "/".join(counts(group["predictionStatus"], statuses)),
                          number(group["predictedPoints"]["median"], 1), group["edgeTruncatedCandidates"]]
                         for name, group in groups(summary)])
    lines += renderTable("Rejection reasons", ["Reason", "Meaning", "Reference frames", "Predicted frames"],
                         [[reason, reasonText.get(reason, reason), summary["overall"]["referenceReasons"].get(reason, 0),
                           summary["overall"]["predictionReasons"].get(reason, 0)] for reason in reasons])
    return lines


def cohortTables(summary, cohort, config):
    thresholds = [str(value) for value in config["errorThresholdsMeters"]]
    tolerances = [str(value) for value in config["transferTolerancesDiagonal"]]
    rows, transfer = [], []
    for name, group in groups(summary):
        item = group["cohorts"][cohort]
        rows.append([name, item["frames"], percent(item["coverage"]),
                     *[percent(item["successRates"][value]) for value in thresholds],
                     number(item["acceptedP90Meters"]["median"]), item["falseValid"]["count"],
                     percent(item["falseValid"]["rate"]), item["rejectedWithinTolerance"],
                     number(item["visibleIou"]["median"], 3),
                     number(item["referenceLeaveOneOutPixels"]["median"], 1)])
        transfer.append([name, item["landmarkTransfer"]["landmarks"],
                         *[percent(item["landmarkTransfer"]["successRates"][value]) for value in tolerances],
                         number(item["landmarkTransfer"]["errorDiagonal"]["median"], 4)])
    lines = renderTable(f"Court-plane error, {cohort} reference frames",
                        ["Group", "Frames", "Accepted %", *[f"p90 <= {value} m %" for value in thresholds],
                         "Median accepted p90 (m)", "False valid", "False valid %", "Rejected but within tolerance",
                         "Median visible IoU", "Reference leave-one-out median (px)"], rows)
    lines += renderTable(f"Annotated landmark transfer, {cohort} reference frames (accepted predictions)",
                         ["Group", "Landmarks", *[f"<= {float(value) * 100:g}% diagonal %" for value in tolerances],
                          "Median error (diagonal)"], transfer)
    return lines


def statusTable(summary, config):
    thresholds = [str(value) for value in config["errorThresholdsMeters"]]
    rows = []
    for name, group in groups(summary):
        item = group["cohorts"]["verified"]
        quality, policy = item["acceptedByStatus"], item["verifiedOnly"]
        rows.append([name, *[f"{quality[status]['frames']} / {number(quality[status]['p90Meters']['median'])} / "
                              f"{quality[status]['falseValid']}" for status in ("verified", "minimal")],
                     percent(policy["coverage"]), *[percent(policy["successRates"][value]) for value in thresholds],
                     percent(policy["falseValidRate"])])
    return renderTable("Accepted prediction quality by status, verified reference frames", [
        "Group", "Verified: frames / median p90 (m) / false valid", "Minimal: frames / median p90 (m) / false valid",
        "Verified-only accepted %", *[f"Verified-only p90 <= {value} m %" for value in thresholds],
        "Verified-only false valid %"], rows)


def diagnosticTable(summary, config):
    rows = []
    for cohort in ("verified", "minimal"):
        for confidence, item in summary["overall"]["cohorts"][cohort]["diagnostic"].items():
            rows.append([cohort, confidence, percent(item["coverage"]),
                         *[percent(item["successRates"][str(value)]) for value in config["errorThresholdsMeters"]],
                         percent(item["falseValidRate"])])
    return renderTable("Landmark confidence sensitivity (diagnostic only)",
                       ["Reference cohort", "Point confidence", "Accepted %",
                        *[f"p90 <= {value} m %" for value in config["errorThresholdsMeters"]], "False valid %"], rows)


def athleteTable(summary, config):
    rows = []
    for name, group in groups(summary):
        for cohort in ("verified", "minimal"):
            item = group["cohorts"][cohort]["athletes"]
            rows.append([name, cohort, item["truthAvailable"], item["insideCourt"], item["matched"],
                         number(item["metresPerPixel"]["median"], 3),
                         number(item["courtOnlyMeters"]["median"]), number(item["courtOnlyMeters"]["p90"]),
                         number(item["detectionMeters"]["median"]), number(item["endToEndMeters"]["median"]),
                         number(item["endToEndMeters"]["p90"]), number(item["endToEndVerifiedOnlyMeters"]["p90"]),
                         *[percent(item["endToEndSuccessRates"][str(value)])
                           for value in config["errorThresholdsMeters"]],
                         *[percent(item["endToEndSuccessRatesVerifiedOnly"][str(value)])
                           for value in config["errorThresholdsMeters"]]])
    return renderTable("Athlete ground-position error", [
        "Group", "Reference cohort", "Annotated athletes", "Inside court + margin", "Matched detections",
        "Median m/px", "Court-only median (m)", "Court-only p90 (m)", "Detection-only median (m)",
        "End-to-end median (m)", "End-to-end p90 (m)", "End-to-end p90, verified only (m)",
        *[f"End-to-end <= {value} m %" for value in config["errorThresholdsMeters"]],
        *[f"Verified only <= {value} m %" for value in config["errorThresholdsMeters"]]], rows)


def inputLines(state):
    court, athletes = state["inputs"]["court"], state["inputs"]["athletes"]
    geometry = state["courtGeometry"]
    lines = ["## Inputs and protocol", "",
             f"Court evaluation: `{court['directory']}`; run `{court['trainingRunId']}`; task `{court['taskId']}`; "
             f"contract `{court['contractId']}`; {court['recordCount']} validation frames.",
             f"Court checkpoint SHA-256: `{(court.get('checkpoint') or {}).get('sha256', 'N/A')}`.", ""]
    if athletes:
        lines += [f"Athlete evaluation: `{athletes['directory']}`; run `{athletes['trainingRunId']}`; task "
                  f"`{athletes['taskId']}`; contract `{athletes['contractId']}`; {athletes['commonRecords']} "
                  "frames shared with the court evaluation.", ""]
    dimensions = geometry["defaultDimensions"]
    lines += [f"Court model: {dimensions['length']:g} x {dimensions['width']:g} m ({geometry['defaultSource']}); "
              f"per-match measured dimensions: {geometry['matchDimensions'] or 'none'}. Nominal dimensions "
              "make metric errors nominal: they assume the real court has these dimensions.", "",
              "Configuration: " + ", ".join(f"{key}={value}" for key, value in sorted(state["configuration"].items())),
              ""]
    return lines


def methodLines():
    return ["## Method", "",
            "Each predicted landmark is the highest-confidence box center of its class at the point confidence. "
            "Every four-landmark subset without three collinear court points is fitted exactly; the subset with "
            "most inliers (then lowest truncated squared residual) defines the inliers, which are refitted by "
            "least squares. The search is exhaustive and deterministic.",
            "",
            "Statuses: verified has at least five inliers and passes every gate; minimal has exactly four inliers "
            "(an exact fit with no internal evidence); rejected lists reason codes. The reference homography fits "
            "every annotated landmark with the same gates; it includes annotation error and lens distortion.",
            "",
            "Court-plane error maps the 0.5 m court grid visible under the reference homography back through the "
            "predicted homography; a frame's p90 uses the nearest rank and counts points behind the camera as "
            "failures. Success rates divide by all frames in the cohort, so rejected frames count as failures. "
            "A false valid frame is accepted with p90 above the false-valid limit. Athlete positions use the "
            "bottom center of the box, which assumes the feet touch the floor.",
            ""]


def limitationLines():
    return ["## Limitations", "",
            "- Validation frames also selected the court checkpoint; these are not test results.",
            "- A single seed and one court checkpoint are evaluated; no significance is claimed.",
            "- Minimal reference frames have four exact annotated points, so their reference is weaker.",
            "- Lens distortion is not modelled. Ball positions are not projected.",
            "- Confidence sensitivity is diagnostic and must not be used to tune thresholds on validation.",
            ""]


def homographyMarkdown(root, state):
    summary = readJson(root / "evidence" / "summary.json")
    config = state["configuration"]
    lines = ["# Homography validation report", "",
             f"Status: **{state['status']}**. Purpose: **{state['purpose']}**. Test evaluated: "
             f"**{state['testEvaluated']}**.", "", *inputLines(state), *methodLines(),
             *availabilityTables(summary), *cohortTables(summary, "verified", config),
             *statusTable(summary, config), *cohortTables(summary, "minimal", config),
             *diagnosticTable(summary, config)]
    if state["inputs"]["athletes"]:
        lines += athleteTable(summary, config)
    if state.get("examples"):
        lines += ["## Examples", "", "Magenta: reference court and annotated landmarks. Green/red: "
                  "accepted/rejected predicted court; circles are inlier/outlier landmark candidates. Inset: "
                  "annotated athletes (white), end-to-end predicted positions (yellow) and a red ring when the "
                  "predicted position falls outside the inset. Broadcast graphics belong to the source frames.",
                  ""]
        lines += [f"### {item['matchId']} {item['kind']} ({item['recordId']})\n\n![{item['recordId']}]({item['path']})\n"
                  for item in state["examples"]]
    return "\n".join([*lines, *limitationLines()])


def renderHomographyReport(evaluationDir, outputDir=None):
    source = Path(evaluationDir).resolve()
    state, contract = verifyEvaluation(source)
    if state.get("kind") != "homographyEvaluation":
        raise ValueError("Expected a sealed homography evaluation.")
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
        (root / "report.md").write_text(homographyMarkdown(root, state), encoding="utf-8", newline="\n")
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["endedAt"] = utcNow()
        writeJson(root / "reportState.json", report)
    return root / "report.md"
