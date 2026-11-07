from pathlib import Path

from futsal.datasets.datasetFiles import hashJson, readJson
from futsal.datasets.syntheticCourt import validateSyntheticSnapshot
from futsal.evaluation.evaluationData import readJsonLines
from futsal.inventory.catalog import hashFile
from futsal.reporting.researchReport import verifyEvaluation


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


def courtComparisonInputs(baselineDir, candidateDir):
    before, after = courtEvaluation(Path(baselineDir)), courtEvaluation(Path(candidateDir))
    baseline, candidate = before[0], after[0]
    first, second = baseline["training"], candidate["training"]
    oldConfig, newConfig = first["configuration"], second["configuration"]
    if (
        oldConfig["sourcePolicy"] != "realOnly"
        or newConfig["sourcePolicy"] != "realWithSyntheticCourt"
    ):
        raise ValueError("Compare real-only training against real plus synthetic court training.")
    if (
        {k: v for k, v in oldConfig.items() if k != "sourcePolicy"}
        != {k: v for k, v in newConfig.items() if k != "sourcePolicy"}
        or first["initialWeights"] != second["initialWeights"]
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
