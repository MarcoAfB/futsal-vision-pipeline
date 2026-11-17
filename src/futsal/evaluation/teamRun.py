import math
import traceback
from pathlib import Path

import numpy as np

from futsal.datasets.datasetFiles import writeJson
from futsal.evaluation.evaluationData import readJsonLines, validateResearchOutput, writeJsonLines
from futsal.evaluation.evaluationRun import saveEvaluation, sealEvaluation
from futsal.evaluation.homographyRun import sealedStage
from futsal.evaluation.teamMetrics import athleteClassNames, athleteRecord, scoreAssignment
from futsal.inventory.catalog import readYaml
from futsal.reporting.researchReport import verifyEvaluation
from futsal.teams.clusterRoles import assignMatch
from futsal.teams.sideEvidence import courtPositions, frameEvidence
from futsal.training.trainingRun import snapshotCode, utcNow

fractionFields = {"umapMinDistance", "goalkeeperZoneMin", "goalkeeperSideMin", "goalkeeperMaxShare", "matchIou"}
integerFields = {"umapComponents", "umapNeighbors", "clusterCount", "seed", "minCleanCrops", "minEvidence"}
homographyStatuses = ("verified", "minimal")


def loadTeamAssignmentConfig(path):
    config, evidence = readYaml(Path(path))
    if set(config) != {"schemaVersion", "featureKind", "reduction", "homographyStatuses", "goalRadiusMeters"} | \
            fractionFields | integerFields:
        raise ValueError("Unexpected or missing team assignment configuration fields.")
    if config["schemaVersion"] != 1:
        raise ValueError("Only team assignment schemaVersion 1 is supported.")
    if config["featureKind"] not in {"siglip", "colorHistogram"} or config["reduction"] not in {"umap", "none"}:
        raise ValueError("featureKind must be siglip or colorHistogram and reduction umap or none.")
    for key in fractionFields:
        if type(config[key]) not in {int, float} or not math.isfinite(config[key]) or not 0 < config[key] <= 1:
            raise ValueError(f"{key} must be finite and in (0, 1].")
    if any(type(config[key]) is not int or config[key] < (0 if key == "seed" else 1) for key in integerFields):
        raise ValueError("Integer team assignment fields must be in their supported range.")
    if config["clusterCount"] < 2 or config["umapNeighbors"] < 2:
        raise ValueError("clusterCount and umapNeighbors must be at least two.")
    radius = config["goalRadiusMeters"]
    if type(radius) not in {int, float} or not math.isfinite(radius) or radius <= 0:
        raise ValueError("goalRadiusMeters must be a positive finite number.")
    statuses = config["homographyStatuses"]
    if not isinstance(statuses, list) or not statuses or not set(statuses) <= set(homographyStatuses):
        raise ValueError(f"homographyStatuses must be a nonempty subset of {homographyStatuses}.")
    return config, evidence


def sealedKind(directory, kind, description):
    root = Path(directory).resolve()
    state, contract = verifyEvaluation(root)
    if state.get("kind") != kind or state["status"] != "completed" or state.get("testEvaluated") is not False:
        raise ValueError(f"{description}: a completed validation-only {kind} evaluation is required.")
    identity = {"directory": str(root), "kind": kind, "contractId": contract["contractId"],
                "evaluationSha256": contract["evaluationSha256"]}
    return root, state, identity


def matchCourt(geometry, matchId):
    dimensions = geometry["matchDimensions"].get(matchId) or geometry["defaultDimensions"]
    return {"length": dimensions["length"], "width": dimensions["width"]}


def assignCrops(crops, features, frames, geometry, config, progress):
    assignments, decisions = [], {}
    for matchId in sorted({row["matchId"] for row in crops}):
        rows = [row for row in crops if row["matchId"] == matchId and row["featureIndex"] is not None]
        court = matchCourt(geometry, matchId)
        positions = np.array([courtPositions([row["box"]], (frames.get(row["recordId"]) or {}).get("courtToImage"))[0]
                              for row in rows]).reshape(-1, 2)
        sideVotes = [(frames.get(row["recordId"]) or {}).get("sideVote") for row in rows]
        labels, roles, classNames, decisions[matchId] = assignMatch(
            features[[row["featureIndex"] for row in rows]], np.array([row["clean"] for row in rows]), positions,
            sideVotes, [row["recordId"] for row in rows], court, config)
        for row, label, role, className, position in zip(rows, labels, roles, classNames, positions, strict=True):
            projected = bool(np.isfinite(position).all())
            assignments.append({**{key: row[key] for key in ("recordId", "matchId", "detectionIndex", "box",
                                                               "confidence", "clean")},
                                "cluster": int(label), "role": role["role"], "roleSource": role["source"],
                                "className": className, "courtPosition": position.tolist() if projected else None})
        progress(f"Team assignment: {matchId} with {len(rows)} crops")
    unassigned = [row for row in crops if row["featureIndex"] is None]
    return assignments, decisions, unassigned


def assignedPredictions(references, assignments):
    byRecord = {}
    for row in assignments:
        byRecord.setdefault(row["recordId"], []).append(
            {"box": row["box"], "confidence": row["confidence"], "className": row["className"]})
    return [athleteRecord({"recordId": reference["recordId"], "dimensions": reference["dimensions"],
                           "detections": byRecord.get(reference["recordId"], [])}, "detections")
            for reference in references]


def evaluateTeams(featuresDir, homographyEvaluationDir, referenceEvaluationDir, outputDir, configPath, *,
                  progress=print):
    inputs = [Path(featuresDir), Path(homographyEvaluationDir), Path(referenceEvaluationDir)]
    outputDir = validateResearchOutput(outputDir, inputs, [Path(configPath)])
    config, configEvidence = loadTeamAssignmentConfig(configPath)
    featureRoot, featureState, featureIdentity = sealedKind(featuresDir, "teamFeatures", "Team features")
    homographyRoot, homographyState, homographyIdentity = sealedKind(
        homographyEvaluationDir, "homographyEvaluation", "Homography evaluation")
    if config["featureKind"] not in featureState["configuration"]["featureKinds"]:
        raise ValueError(f"The team features do not contain {config['featureKind']} embeddings.")
    referenceIdentity, references, baseline = sealedStage(
        referenceEvaluationDir, lambda classes: set(athleteClassNames) <= set(classes), "Detailed reference evaluation")
    crops = readJsonLines(featureRoot / "evidence" / "crops.jsonl")
    with np.load(featureRoot / "evidence" / "embeddings.npz") as archive:
        features = archive[config["featureKind"]]
    images = {row["recordId"]: row["imageSha256"]
              for row in readJsonLines(featureRoot / "evidence" / "frames.jsonl")}
    # Only the court homographies are used, so any player evaluation of the same images may supply the features.
    homographyRows = readJsonLines(homographyRoot / "evidence" / "frames.jsonl")
    if any(row["recordId"] in images and images[row["recordId"]] != row["imageSha256"] for row in homographyRows):
        raise ValueError("Team features and the homography evaluation disagree on a validation image.")
    frames = {row["recordId"]: frameEvidence(row, config["homographyStatuses"]) for row in homographyRows}
    pairs = [(reference, prediction) for reference, prediction in zip(references, baseline, strict=True)
             if reference["recordId"] in images]
    if any(images[reference["recordId"]] != reference["imageSha256"] for reference, _ in pairs):
        raise ValueError("Team features and detailed references disagree on a validation image.")
    evidenceDir = outputDir / "evidence"
    evidenceDir.mkdir(parents=True)
    state = {"schemaVersion": 1, "kind": "teamEvaluation", "status": "running", "startedAt": utcNow(),
             "purpose": "teamRoleValidation", "selection": "sealedValidation", "testEvaluated": False,
             "inputs": {"features": featureIdentity, "homography": homographyIdentity, "references": referenceIdentity},
             "configuration": config, "configurationEvidence": configEvidence,
             "featureConfiguration": featureState["configuration"]}
    saveEvaluation(outputDir, state)
    try:
        state["code"] = snapshotCode(evidenceDir)
        assignments, decisions, unassigned = assignCrops(crops, features, frames,
                                                         homographyState["courtGeometry"], config, progress)
        cohortReferences = [athleteRecord(reference, "targets") for reference, _ in pairs]
        scoring = {"operationalConfidence": featureState["configuration"]["detectionConfidence"],
                   "operationalIou": config["matchIou"]}
        summary = {"cohort": {"frames": len(pairs), "referenceFrames": len(references),
                              "featureFrames": len(images), "framesWithHomographyRow": sum(
                                  reference["recordId"] in frames for reference, _ in pairs),
                              "athleteReferences": sum(len(row["targets"]) for row in cohortReferences),
                              "unassignedDetections": len(unassigned)},
                   "pipeline": scoreAssignment(cohortReferences, assignedPredictions(cohortReferences, assignments),
                                               scoring),
                   "baseline": scoreAssignment(cohortReferences, [athleteRecord(prediction, "detections")
                                                                  for _, prediction in pairs], scoring)}
        writeJsonLines(evidenceDir / "assignments.jsonl", assignments)
        writeJson(evidenceDir / "matches.json", decisions)
        writeJson(evidenceDir / "summary.json", summary)
        state.update(status="completed", frameCount=len(pairs), assignmentCount=len(assignments))
    except (Exception, KeyboardInterrupt) as error:
        (outputDir / "pipelineFailure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     error={"type": type(error).__name__, "message": str(error)})
        sealEvaluation(outputDir, state)
        raise
    sealEvaluation(outputDir, state)
    return state
