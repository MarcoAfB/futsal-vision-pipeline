import shutil
from pathlib import Path

from futsal.datasets.datasetFiles import hashJson, readJson, writeJson
from futsal.datasets.syntheticCourt import validateSyntheticSnapshot
from futsal.evaluation.configurationComparison import comparisonMetrics
from futsal.evaluation.evaluationData import readJsonLines, validateResearchOutput
from futsal.evaluation.evaluationRun import sealEvaluation
from futsal.evaluation.modelComparison import (
    comparisonModels,
    modelChanges,
    validateComparisonMode,
    validateModelComparison,
)
from futsal.inventory.catalog import hashFile
from futsal.reporting.researchReport import verifyEvaluation
from futsal.training.trainingRun import utcNow


def courtEvaluation(root):
    state, seal = verifyEvaluation(root)
    run = state["training"]
    if (
        state["status"] != "completed"
        or state["purpose"] != "scientificTraining"
        or state["testEvaluated"] is not False
        or run["status"] != "completed"
        or run.get("integrityErrors")
        or state.get("integrityError")
    ):
        raise ValueError(
            "Court comparison requires completed, intact scientific validation evaluations."
        )
    if (
        [s["taskId"] for s in run["configuration"]["stages"]] != ["courtPoints"]
        or len(state["stages"]) != 1
        or len(run["stages"]) != 1
    ):
        raise ValueError(
            "Comparison requires two executions of the same independent court detector."
        )
    stage = state["stages"][0]
    if (
        stage["status"] != "completed"
        or stage["taskId"] != "courtPoints"
        or run["stages"][0]["status"] != "completed"
    ):
        raise ValueError("Court training or validation is incomplete.")
    if (
        readJson(root / "evidence/trainingRun.json") != run
        or hashFile(root / "evidence/trainingRun.json") != state["trainingRunSha256"]
        or hashFile(root / "evidence/trainingSelection.json") != run["selectionSha256"]
    ):
        raise ValueError("Court evaluation has inconsistent archived training provenance.")
    if (
        stage["checkpoint"] != run["stages"][0]["results"]["checkpoints"]["best"]
        or run["stages"][0]["inputWeights"]["sha256"] != run["initialWeights"]["sha256"]
        or run["contractId"] != state["contractId"]
    ):
        raise ValueError(
            "Court evaluation is not bound to independent initialization and its selected checkpoint."
        )
    directory = root / stage["evidenceDirectory"]
    values = {
        "references": readJsonLines(directory / "references.jsonl"),
        "predictions": readJsonLines(directory / "predictions.jsonl"),
        "detection": readJson(directory / "detection.json"),
        "points": readJson(directory / "points.json"),
    }
    selection = readJson(root / "evidence/trainingSelection.json")
    task = selection["tasks"]["courtPoints"]
    if (
        stage["recordIds"] != [r["recordId"] for r in values["references"]]
        or stage["recordIds"] != [r["recordId"] for r in task["validation"]]
        or any(r["origin"] != "own" for r in task["validation"])
    ):
        raise ValueError("Court evaluation must use its full real validation cohort.")
    if (
        selection["selectionMethod"] != "fullPerTask"
        or not run.get("bundleIntegrityAfterRun", {}).get("hashesVerified")
        or not run.get("runtimeIntegrityAfterRun")
    ):
        raise ValueError("Court comparison requires full selections and verified training inputs.")
    return state, seal, stage, task, values


def courtComparisonInputs(baselineDir, candidateDir, *, comparisonMode="controlled"):
    validateComparisonMode(comparisonMode)
    before, after = courtEvaluation(Path(baselineDir)), courtEvaluation(Path(candidateDir))
    baseline, candidate = before[0], after[0]
    first, second = baseline["training"], candidate["training"]
    oldConfig, newConfig = first["configuration"], second["configuration"]
    if (
        oldConfig["sourcePolicy"] != "realOnly"
        or newConfig["sourcePolicy"] != "realWithSyntheticCourt"
    ):
        raise ValueError("Compare real-only training against real plus synthetic court training.")
    validateModelComparison(first, second, comparisonMode)
    excluded = {"sourcePolicy", "modelSize"}
    if (
        {k: v for k, v in oldConfig.items() if k not in excluded}
        != {k: v for k, v in newConfig.items() if k not in excluded}
        or first["code"]["files"] != second["code"]["files"]
    ):
        raise ValueError(
            "Court runs must share architecture, initialization, code and training parameters."
        )
    if (
        baseline["contractId"] != candidate["contractId"]
        or baseline["protocol"] != candidate["protocol"]
        or baseline["protocolId"] != hashJson(baseline["protocol"])
        or candidate["protocolId"] != baseline["protocolId"]
        or before[2]["classes"] != after[2]["classes"]
        or before[4]["references"] != after[4]["references"]
    ):
        raise ValueError(
            "Court comparison requires identical real references, classes and evaluation protocol."
        )
    realBefore = before[3]["training"]
    realAfter = [row for row in after[3]["training"] if row["origin"] != "synthetic"]
    if realBefore != realAfter or before[3]["validation"] != after[3]["validation"]:
        raise ValueError("Only synthetic training membership may differ between court runs.")
    contract = readJson(Path(candidateDir) / "evidence/syntheticContract.json")
    rows = readJsonLines(Path(candidateDir) / "evidence/syntheticManifest.jsonl")
    validateSyntheticSnapshot(contract, rows, before[2]["classes"])
    if second.get("syntheticSource") != {
        "contractId": contract["contractId"],
        "imageCount": len(rows),
    } or not second.get("syntheticIntegrityAfterRun"):
        raise ValueError("Mixed court training source integrity is not verified.")
    expected = []
    for row in rows:
        expected.append(
            {
                **row,
                **{
                    f"{kind}Path": f"courtPoints/{row[f'{kind}Path']}"
                    for kind in ("image", "label")
                },
            }
        )
    actual = [row for row in after[3]["training"] if row["origin"] == "synthetic"]
    if sorted(actual, key=lambda r: r["recordId"]) != sorted(expected, key=lambda r: r["recordId"]):
        raise ValueError("Mixed training differs from the exact archived synthetic source.")
    return before, after


def compareCourtTraining(baselineEvaluationDir, candidateEvaluationDir, outputDir, *, comparisonMode="controlled"):
    first, second = Path(baselineEvaluationDir).resolve(), Path(candidateEvaluationDir).resolve()
    output = validateResearchOutput(outputDir, [first, second])
    before, after = courtComparisonInputs(first, second, comparisonMode=comparisonMode)
    models = comparisonModels(before[0]["training"], [after[0]["training"]])
    values = [before[4], after[4]]
    metrics = comparisonMetrics(values, before[0]["protocol"])
    metrics["byMatch"] = {
        matchId: comparisonMetrics(
            [
                {
                    "detection": value["detection"]["byMatch"][matchId],
                    "points": value["points"]["byMatch"][matchId],
                }
                for value in values
            ],
            before[0]["protocol"],
        )
        for matchId in values[0]["points"]["byMatch"]
    }
    output.mkdir(parents=True)
    for name, source in (("baseline", first), ("candidate", second)):
        target = output / "evidence" / name
        target.mkdir(parents=True)
        shutil.copytree(source / "evidence", target / "evidence")
        for filename in ("evaluation.json", "evaluationContract.json"):
            shutil.copyfile(source / filename, target / filename)
    writeJson(output / "evidence/comparisonMetrics.json", metrics)
    state = {
        "schemaVersion": 1,
        "kind": "courtTrainingComparison",
        "status": "completed",
        "startedAt": utcNow(),
        "purpose": "scientificTraining",
        "selection": "fullValidation",
        "testEvaluated": False,
        "singleSeed": before[0]["training"]["configuration"]["seed"],
        "comparisonMode": comparisonMode,
        "modelVariants": models,
        "configurationChanges": ["syntheticTrainingMembership", *modelChanges(models)],
        "baselineContractId": before[1]["contractId"],
        "candidateContractId": after[1]["contractId"],
        "classes": before[2]["classes"],
        "protocol": before[0]["protocol"],
        "trainingRunHashes": [v[0]["trainingRunSha256"] for v in (before, after)],
        "trainingCosts": [
            v[0]["training"]["stages"][0]["results"]["training"] for v in (before, after)
        ],
    }
    archived = courtComparisonInputs(output / "evidence/baseline", output / "evidence/candidate", comparisonMode=comparisonMode)
    if [v[1] for v in archived] != [before[1], after[1]] or [
        verifyEvaluation(p)[1] for p in (first, second)
    ] != [before[1], after[1]]:
        raise ValueError("Court evaluations changed during comparison publication.")
    sealEvaluation(output, state)
    return state
