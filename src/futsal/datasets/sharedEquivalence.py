import copy
from pathlib import Path

from futsal.datasets.datasetFiles import hashJson, readJson
from futsal.datasets.datasetPlan import loadDatasetPlan


def semanticRows(rows):
    return [{key: value for key, value in row.items() if key not in
             {"imagePath", "labelPath", "assetId"}} for row in rows]


def semanticSelection(selection):
    result = copy.deepcopy(selection)
    for task in result["tasks"].values():
        for split in ("training", "validation"):
            task[split] = semanticRows(task[split])
    return result


def verifyRunView(runDir, datasetDir, run):
    contract = readJson(Path(datasetDir) / "contract.json")
    if contract["contractId"] == run["contractId"]:
        return None
    if contract.get("kind") != "sharedTaskView":
        raise ValueError("A different dataset contract requires a verified shared view.")
    from futsal.datasets.sharedViews import verifyView
    from futsal.training.syntheticTraining import loadSyntheticSnapshot
    from futsal.training.trainingData import selectTrainingData

    verifyView(datasetDir)
    root = Path(runDir)
    archived = root / "inputs/datasetPlan.json"
    if not archived.is_file():
        previous = Path(run["datasetDirectory"])
        if readJson(previous / "contract.json")["contractId"] != run["contractId"]:
            raise ValueError("Original dataset identity is unavailable for semantic verification.")
        archived = previous / "plan.json"
    before = loadDatasetPlan(archived)
    after = loadDatasetPlan(Path(datasetDir) / "plan.json")
    if before["frozenSplit"] != after["frozenSplit"]:
        raise ValueError("Equivalent views must retain the identical frozen split.")
    for key in ("pointBoxes", "unsupportedValidationClasses"):
        if before["recipe"][key] != after["recipe"][key]:
            raise ValueError("Equivalent views changed annotation/evaluation policy.")
    oldTasks = {task["taskId"]: task for task in before["recipe"]["tasks"]}
    newTasks = {task["taskId"]: task for task in after["recipe"]["tasks"]}
    for stage in run["configuration"]["stages"]:
        taskId = stage["taskId"]
        if oldTasks[taskId] != newTasks[taskId]:
            raise ValueError("Equivalent views require identical class maps and source exposure.")
    savedPath = root / "selection.json"
    from futsal.inventory.catalog import hashFile

    if hashFile(savedPath) != run["selectionSha256"]:
        raise ValueError("Archived training selection changed.")
    classes = newTasks.get("courtPoints", {}).get("classes", [])
    snapshot = loadSyntheticSnapshot(root, run["configuration"], classes) if run["configuration"].get("schemaVersion") == 2 else None
    saved = semanticSelection(readJson(savedPath))
    current = semanticSelection(selectTrainingData(datasetDir, run["configuration"], snapshot))
    if saved != current:
        raise ValueError("Shared view changed ordered records, image hashes, targets or source policy.")
    return {"kind": "equivalentTaskSelection", "trainingContractId": run["contractId"],
            "evaluationContractId": contract["contractId"], "semanticSelectionId": hashJson(saved),
            "frozenSplitVerified": True, "nativeTargetsVerified": True}
