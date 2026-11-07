import copy
from pathlib import Path

from futsal.datasets.datasetFiles import readJson, verifyEvidence, writeJson
from futsal.datasets.syntheticCourt import (
    readSyntheticManifest,
    validateSyntheticSnapshot,
    verifySyntheticCourt,
)


def requireCourtTraining(config):
    if (
        config.get("schemaVersion") != 2
        or config.get("purpose") != "scientificTraining"
        or [stage["taskId"] for stage in config["stages"]] != ["courtPoints"]
    ):
        raise ValueError(
            "Synthetic data are supported only for independent scientific courtPoints training."
        )


def archiveSyntheticSource(sourceDir, inputs, classes):
    contract = verifySyntheticCourt(sourceDir)
    rows = readSyntheticManifest(sourceDir)
    validateSyntheticSnapshot(contract, rows, classes)
    writeJson(inputs / "syntheticContract.json", contract)
    from futsal.evaluation.evaluationData import writeJsonLines

    writeJsonLines(inputs / "syntheticManifest.jsonl", rows)
    return contract, rows


def loadSyntheticSnapshot(runDir, config, classes):
    if config.get("sourcePolicy") == "realOnly":
        return None
    requireCourtTraining(config)
    if config.get("sourcePolicy") != "realWithSyntheticCourt":
        raise ValueError("Unknown scientific source policy.")
    root = Path(runDir)
    run = readJson(root / "run.json")
    inputs = root / "inputs"
    required = {"inputs/syntheticContract.json", "inputs/syntheticManifest.jsonl"}
    evidence = [item for item in run["inputEvidence"] if item["path"] in required]
    if {item["path"] for item in evidence} != required:
        raise ValueError("Mixed training has no archived synthetic source evidence.")
    verifyEvidence(root, evidence)
    contract = readJson(inputs / "syntheticContract.json")
    from futsal.evaluation.evaluationData import readJsonLines

    rows = readJsonLines(inputs / "syntheticManifest.jsonl")
    validateSyntheticSnapshot(contract, rows, classes)
    if run.get("syntheticSource") != {
        "contractId": contract["contractId"],
        "imageCount": len(rows),
    }:
        raise ValueError("Archived synthetic source identity differs from the run.")
    return contract, rows


def supplementSelection(selection, config, snapshot):
    requireCourtTraining(config)
    if snapshot is None:
        raise ValueError("Mixed court training requires a previously generated synthetic source.")
    contract, rows = snapshot
    task = selection["tasks"]["courtPoints"]
    validateSyntheticSnapshot(contract, rows, task["classes"])
    task["training"] = list(task["training"])
    realIds = {row["recordId"] for row in task["training"]}
    for source in rows:
        row = copy.deepcopy(source)
        if row["recordId"] in realIds:
            raise ValueError("Synthetic and real record IDs collide.")
        for kind in ("image", "label"):
            row[f"{kind}Path"] = f"courtPoints/{source[f'{kind}Path']}"
        task["training"].append(row)
    task["training"].sort(key=lambda row: row["recordId"])
    from futsal.training.trainingData import classSupport

    task["trainingSupport"] = classSupport(task["training"], task["classes"])
    selection["sourcePolicy"] = {
        **selection["sourcePolicy"],
        "policy": "realWithSyntheticCourt",
        "syntheticContractId": contract["contractId"],
        "syntheticImageCount": len(rows),
    }
    return selection
