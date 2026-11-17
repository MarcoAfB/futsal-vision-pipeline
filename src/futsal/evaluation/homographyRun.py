import math
import traceback
from pathlib import Path

from futsal.datasets.datasetFiles import hashJson, readJson, verifyEvidence, writeJson
from futsal.evaluation.evaluationData import readJsonLines, validateResearchOutput, writeJsonLines
from futsal.evaluation.evaluationRun import saveEvaluation, sealEvaluation
from futsal.evaluation.homographyMetrics import athleteClasses, evaluateFrame, summarizeEvaluation
from futsal.geometry.courtModel import landmarkNames, loadCourtGeometry, matchCourt
from futsal.inventory.catalog import hashFile, readYaml
from futsal.training.trainingRun import snapshotCode, utcNow

fractionFields = {"pointConfidence", "inlierThresholdDiagonal", "maxInlierRmseDiagonal",
                  "minImageHullFraction", "athleteConfidence", "athleteMatchIou"}
positiveFields = {"minTriangleAreaSquareMeters", "courtMarginMeters", "gridStepMeters",
                  "falseValidP90Meters", "iouSampleStepPixels"}
listFields = {"errorThresholdsMeters", "transferTolerancesDiagonal", "diagnosticConfidences"}


def finiteNumber(value):
    return type(value) in {int, float} and math.isfinite(value)


def loadHomographyConfig(path):
    config, evidence = readYaml(Path(path))
    if set(config) != {"schemaVersion"} | fractionFields | positiveFields | listFields:
        raise ValueError("Unexpected or missing homography configuration fields.")
    if config["schemaVersion"] != 1:
        raise ValueError("Only homography schemaVersion 1 is supported.")
    for key in fractionFields:
        if not finiteNumber(config[key]) or not 0 < config[key] <= 1:
            raise ValueError(f"{key} must be finite and in (0, 1].")
    for key in positiveFields:
        if not finiteNumber(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive finite number.")
    for key in listFields:
        values = config[key]
        if (not isinstance(values, list) or not values or not all(finiteNumber(value) and value > 0
                                                                  for value in values)
                or values != sorted(set(values))):
            raise ValueError(f"{key} must be a nonempty increasing list of positive numbers.")
    if config["pointConfidence"] not in config["diagnosticConfidences"]:
        raise ValueError("diagnosticConfidences must include pointConfidence.")
    if config["falseValidP90Meters"] not in config["errorThresholdsMeters"]:
        raise ValueError("errorThresholdsMeters must include falseValidP90Meters.")
    return config, evidence


def sealedStage(evaluationDir, accepts, description):
    root = Path(evaluationDir).resolve()
    contract = readJson(root / "evaluationContract.json")
    if contract.get("schemaVersion") != 1 or contract["contractId"] != hashJson(
            {key: value for key, value in contract.items() if key != "contractId"}) or \
            hashFile(root / "evaluation.json") != contract["evaluationSha256"]:
        raise ValueError(f"{description}: evaluation metadata or contract changed.")
    state = readJson(root / "evaluation.json")
    if state["status"] != "completed" or state.get("testEvaluated") is not False:
        raise ValueError(f"{description}: a completed validation-only evaluation is required.")
    stages = [stage for stage in state["stages"] if stage["status"] == "completed" and accepts(stage["classes"])]
    if len(stages) != 1:
        raise ValueError(f"{description}: expected exactly one completed stage with the required classes.")
    stage = stages[0]
    artifacts = {item["path"]: item for item in contract["artifacts"]}
    paths = {name: f"{stage['evidenceDirectory']}/{name}" for name in ("references.jsonl", "predictions.jsonl")}
    if not set(paths.values()) <= set(artifacts):
        raise ValueError(f"{description}: sealed references or predictions are missing.")
    # Only the consumed files are rehashed; other sealed members are not read.
    verifyEvidence(root, [artifacts[path] for path in paths.values()])
    references, predictions = (readJsonLines(root / paths[name]) for name in ("references.jsonl", "predictions.jsonl"))
    if [row["recordId"] for row in references] != [row["recordId"] for row in predictions] or any(
            reference["dimensions"] != prediction["dimensions"]
            for reference, prediction in zip(references, predictions, strict=True)):
        raise ValueError(f"{description}: predictions do not align with references.")
    identity = {"directory": str(root), "evaluationSha256": contract["evaluationSha256"],
                "contractId": contract["contractId"], "contractSha256": hashFile(root / "evaluationContract.json"),
                "trainingRunId": state.get("trainingRunId"), "protocolId": state.get("protocolId"),
                "taskId": stage["taskId"], "classes": stage["classes"], "checkpoint": stage.get("checkpoint"),
                "recordCount": len(references), "files": {name: artifacts[path] for name, path in paths.items()}}
    return identity, references, predictions


def athleteInputs(playerEvaluationDir, courtReferences):
    if playerEvaluationDir is None:
        return None, {}
    identity, references, predictions = sealedStage(
        playerEvaluationDir, lambda classes: bool(set(classes) & athleteClasses), "Athlete evaluation")
    byRecord = {reference["recordId"]: (reference, prediction)
                for reference, prediction in zip(references, predictions, strict=True)}
    for reference in courtReferences:
        pair = byRecord.get(reference["recordId"])
        if pair and (pair[0]["imageSha256"] != reference["imageSha256"] or
                     pair[0]["dimensions"] != reference["dimensions"]):
            raise ValueError(f"Athlete and court records disagree for {reference['recordId']}.")
    identity["commonRecords"] = sum(reference["recordId"] in byRecord for reference in courtReferences)
    return identity, byRecord


def evaluateHomography(courtEvaluationDir, outputDir, configPath, courtConfigPath, *, playerEvaluationDir=None,
                       courtLength=None, courtWidth=None, mediaStoreDir=None, progress=print):
    inputs = [Path(courtEvaluationDir)] + ([Path(playerEvaluationDir)] if playerEvaluationDir else [])
    outputDir = validateResearchOutput(outputDir, inputs, [Path(configPath), Path(courtConfigPath)])
    config, configEvidence = loadHomographyConfig(configPath)
    geometry = loadCourtGeometry(courtConfigPath, courtLength, courtWidth)
    courtIdentity, references, predictions = sealedStage(
        courtEvaluationDir, lambda classes: set(landmarkNames) <= set(classes), "Court evaluation")
    athleteIdentity, athletes = athleteInputs(playerEvaluationDir, references)
    if mediaStoreDir is not None:
        from futsal.datasets.mediaStore import protectMediaOutput

        protectMediaOutput(outputDir, mediaStoreDir)
    evidenceDir = outputDir / "evidence"
    evidenceDir.mkdir(parents=True)
    state = {"schemaVersion": 1, "kind": "homographyEvaluation", "status": "running", "startedAt": utcNow(),
             "purpose": "homographyValidation", "selection": "sealedCourtValidation", "testEvaluated": False,
             "inputs": {"court": courtIdentity, "athletes": athleteIdentity},
             "configuration": config, "configurationEvidence": configEvidence,
             "courtGeometry": {key: geometry[key] for key in
                               ("evidence", "markings", "defaultDimensions", "defaultSource", "matchDimensions")}}
    saveEvaluation(outputDir, state)
    try:
        state["code"] = snapshotCode(evidenceDir)
        rows, athleteRows = [], []
        for index, (reference, prediction) in enumerate(zip(references, predictions, strict=True), start=1):
            court = matchCourt(geometry, reference["matchId"])
            row, frameAthletes = evaluateFrame(reference, prediction, court, config,
                                               athletes.get(reference["recordId"]) if athletes else None)
            rows.append(row)
            athleteRows.extend(frameAthletes)
            if index % 100 == 0 or index == len(references):
                progress(f"Homography frames: {index}/{len(references)}")
        writeJsonLines(evidenceDir / "frames.jsonl", rows)
        writeJsonLines(evidenceDir / "athletes.jsonl", athleteRows)
        summary = summarizeEvaluation(rows, athleteRows, config)
        writeJson(evidenceDir / "summary.json", summary)
        if mediaStoreDir is not None:
            from futsal.reporting.homographyFigures import writeExamples

            state["examples"] = writeExamples(rows, athleteRows, geometry, mediaStoreDir, evidenceDir / "examples")
        state.update(status="completed", frameCount=len(rows), athleteCount=len(athleteRows))
    except (Exception, KeyboardInterrupt) as error:
        (outputDir / "pipelineFailure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     error={"type": type(error).__name__, "message": str(error)})
        sealEvaluation(outputDir, state)
        raise
    sealEvaluation(outputDir, state)
    return state
