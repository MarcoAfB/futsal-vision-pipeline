import copy
import shutil
from collections import Counter
from pathlib import Path

from futsal.datasets.datasetFiles import hashJson, readJson, writeJson
from futsal.evaluation.evaluationData import (
    readJsonLines,
    validatePredictions,
    validateResearchOutput,
)
from futsal.evaluation.evaluationRun import sealEvaluation
from futsal.inventory.catalog import hashFile
from futsal.reporting.researchReport import verifyEvaluation
from futsal.training.trainingRun import utcNow


def delta(baseline, candidate):
    return candidate - baseline if baseline is not None and candidate is not None else None


def validateComparison(baselineDir, candidateDir):
    try:
        return comparisonInputs(baselineDir, candidateDir)
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError(f"Incomplete or invalid scientific comparison metadata: {error}.") from error


def comparisonInputs(baselineDir, candidateDir):
    baseline, baselineSeal = verifyEvaluation(Path(baselineDir))
    candidate, candidateSeal = verifyEvaluation(Path(candidateDir))
    if baseline["status"] != "completed" or candidate["status"] != "completed" or (
        baseline["purpose"] != "scientificTraining" or candidate["purpose"] != "scientificTraining"
        or candidate.get("kind") != "configurationEvaluation"
    ):
        raise ValueError("Comparison requires completed scientific baseline and joint evaluations.")
    if baseline.get("integrityError") or candidate.get("integrityError") or baseline.get("testEvaluated") is not False or candidate.get("testEvaluated") is not False:
        raise ValueError("Comparison refuses integrity failures or test evaluations.")
    if baseline["protocol"] != candidate["protocol"] or baseline["protocolId"] != hashJson(baseline["protocol"]) or candidate["protocolId"] != hashJson(candidate["protocol"]):
        raise ValueError("Comparison protocols differ; evaluate with the same frozen protocol first.")
    if baseline["contractId"] != candidate["referenceContractId"]:
        raise ValueError("Baseline and candidate use different reference dataset contracts.")
    stages = [stage for stage in baseline["stages"] if stage.get("classes") == candidate["classes"] and stage["status"] == "completed"]
    if len(stages) != 1 or baseline["training"]["status"] != "completed":
        raise ValueError("Baseline has no unique completed canonical detector evaluation.")
    stage = stages[0]
    training = baseline["training"]
    expectedStages = ["genericObjects", "roleObjects", "unified25"]
    if [item["taskId"] for item in training["configuration"]["stages"]] != expectedStages or [item["taskId"] for item in training["stages"]] != expectedStages or any(item["status"] != "completed" for item in training["stages"]):
        raise ValueError("Baseline must be the completed 2-to-3-to-25 curriculum.")
    archivedRun = Path(baselineDir) / "evidence/trainingRun.json"
    archivedSelection = Path(baselineDir) / "evidence/trainingSelection.json"
    if readJson(archivedRun) != training or hashFile(archivedRun) != baseline["trainingRunSha256"] or hashFile(archivedSelection) != training["selectionSha256"]:
        raise ValueError("Baseline archived run/selection provenance is inconsistent.")
    if training["contractId"] != baseline["contractId"] or training["stages"][-1]["results"]["checkpoints"]["best"] != stage["checkpoint"]:
        raise ValueError("Baseline evaluation is not bound to its final curriculum checkpoint.")
    original = baseline["training"]["configuration"]
    current = candidate["components"][0]["run"]["configuration"]
    if any(original[key] != current[key] for key in ("modelSize", "seed")) or any(
        original["parameters"][key] != current["parameters"][key] for key in ("imgsz", "amp")
    ):
        raise ValueError("Comparison requires the same model scale, seed, resolution and precision.")
    if baseline["training"]["initialWeights"]["sha256"] != candidate["components"][0]["run"]["initialWeights"]["sha256"]:
        raise ValueError("Official initialization hashes differ between configurations.")
    oldRoot, newRoot = Path(baselineDir) / stage["evidenceDirectory"], Path(candidateDir) / "evidence/joint"
    refs = readJsonLines(oldRoot / "references.jsonl")
    newRefs = readJsonLines(newRoot / "references.jsonl")
    if refs != newRefs or [row["recordId"] for row in refs] != candidate["recordIds"] or stage["recordIds"] != candidate["recordIds"]:
        raise ValueError("Comparison requires identical canonical references and common record membership.")
    values = []
    for root in (oldRoot, newRoot):
        predictions = validatePredictions(refs, readJsonLines(root / "predictions.jsonl"), candidate["classes"], candidate["protocol"])
        if any("nativeOrder" not in row or "correctAtIou" not in row for row in predictions):
            raise ValueError("Comparison requires archived native matching/order evidence.")
        detection, points = readJson(root / "detection.json"), readJson(root / "points.json")
        counts = Counter(target["className"] for row in refs for target in row["targets"])
        if [row["className"] for row in detection["perClass"]] != candidate["classes"] or any(
            row["support"] != counts[row["className"]] for row in detection["perClass"]
        ) or any(row["support"] != counts[row["className"]] for row in points["perClass"]):
            raise ValueError("Archived metric support differs from canonical references.")
        if detection["imageCount"] != len(refs) or len(points["perClass"]) != len(candidate["pointClasses"]):
            raise ValueError("Archived metrics have incompatible cohort or point ontology.")
        values.append({"references": refs, "predictions": predictions, "detection": detection, "points": points})
    return baseline, candidate, baselineSeal, candidateSeal, values


def comparisonMetrics(values, protocol):
    first, second = values
    groupRows = []
    for group, before in first["detection"]["groups"].items():
        after = second["detection"]["groups"][group]
        if (before["classCount"], before["supportedClassCount"], before["support"]) != (after["classCount"], after["supportedClassCount"], after["support"]):
            raise ValueError("Group support differs between configurations.")
        for metric in ("ap50", "ap50To95", "precision", "recall", "f1"):
            groupRows.append({"group": group, "metric": metric, "support": before["support"],
                "baseline": before[metric], "candidate": after[metric], "delta": delta(before[metric], after[metric])})
    classRows = []
    for before, after in zip(first["detection"]["perClass"], second["detection"]["perClass"], strict=True):
        for metric in ("ap50To95", "precision", "recall", "f1", "falsePositives", "falseNegatives"):
            classRows.append({"className": before["className"], "support": before["support"], "metric": metric,
                "baseline": before[metric], "candidate": after[metric], "delta": delta(before[metric], after[metric])})
    tolerance = str(protocol["primaryPointTolerance"])
    pointRows = []
    for before, after in zip(first["points"]["perClass"], second["points"]["perClass"], strict=True):
        if before["className"] != after["className"]:
            raise ValueError("Point ontology order differs between configurations.")
        measures = {"successRate": (before["successRates"][tolerance], after["successRates"][tolerance]),
                    **{key: (before[key], after[key]) for key in ("localizedCount", "missingCount", "medianDistanceDiagonal", "p90DistanceDiagonal")}}
        for metric, (old, new) in measures.items():
            pointRows.append({"className": before["className"], "support": before["support"], "metric": metric,
                              "baseline": old, "candidate": new, "delta": delta(old, new)})
    return {"groups": groupRows, "classes": classRows, "points": pointRows,
            "pointSuccess": {"toleranceDiagonal": protocol["primaryPointTolerance"],
                "baseline": first["points"]["successRates"][tolerance],
                "candidate": second["points"]["successRates"][tolerance],
                "delta": delta(first["points"]["successRates"][tolerance], second["points"]["successRates"][tolerance])}}


def compareConfigurations(baselineEvaluationDir, candidateEvaluationDir, outputDir):
    first, second = Path(baselineEvaluationDir).resolve(), Path(candidateEvaluationDir).resolve()
    output = validateResearchOutput(outputDir, [first, second])
    baseline, candidate, oldSeal, newSeal, values = validateComparison(first, second)
    metrics = comparisonMetrics(values, candidate["protocol"])
    output.mkdir(parents=True)
    (output / "evidence").mkdir()
    for name, source in (("baseline", first), ("candidate", second)):
        target = output / "evidence" / name
        target.mkdir()
        shutil.copytree(source / "evidence", target / "evidence")
        for filename in ("evaluation.json", "evaluationContract.json"):
            shutil.copyfile(source / filename, target / filename)
    writeJson(output / "evidence/comparisonMetrics.json", metrics)
    state = {"schemaVersion": 1, "kind": "configurationComparison", "status": "completed",
        "startedAt": utcNow(), "purpose": "scientificTraining", "selection": "commonValidation",
        "classes": candidate["classes"], "pointClasses": candidate["pointClasses"], "protocol": candidate["protocol"],
        "protocolId": candidate["protocolId"], "recordIds": candidate["recordIds"], "testEvaluated": False,
        "baselineDirectory": str(first), "candidateDirectory": str(second),
        "baselineContractId": oldSeal["contractId"], "candidateContractId": newSeal["contractId"],
        "baselineEvaluationSha256": hashFile(first / "evaluation.json"),
        "candidateEvaluationSha256": hashFile(second / "evaluation.json"),
        "configurationChanges": ["detectorPartition", "publicCurriculumExposure", "taskEligibleTrainingCohorts"],
        "singleSeed": candidate["components"][0]["run"]["configuration"]["seed"]}
    try:
        archived = validateComparison(output / "evidence/baseline", output / "evidence/candidate")
        if oldSeal != archived[2] or newSeal != archived[3] or metrics != comparisonMetrics(archived[-1], candidate["protocol"]):
            raise ValueError("Comparison inputs changed while archiving evidence.")
        if verifyEvaluation(first)[1] != oldSeal or verifyEvaluation(second)[1] != newSeal:
            raise ValueError("Source evaluations changed during comparison.")
    except (OSError, ValueError) as error:
        state.update(status="failed", integrityError=str(error))
    sealEvaluation(output, state)
    return copy.deepcopy(state)
