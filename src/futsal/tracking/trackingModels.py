from futsal.datasets.datasetFiles import readJson
from futsal.inventory.catalog import hashFile, resolveSourcePath


def loadTrackingModels(runDirs, selection, datasetDir, seed):
    plan = readJson(datasetDir / "plan.json")
    assignments = plan["frozenSplit"]["assignments"]
    expected = {(r["recordId"], r["imageSha256"], r["labelSha256"]) for r in selection["references"]}
    models = {}
    for size, root in runDirs.items():
        run = readJson(root / "run.json")
        config = run["configuration"]
        if run["status"] != "completed" or config["modelSize"] != size or config["seed"] != seed or (
            config["purpose"] != "scientificTraining" or config["sourcePolicy"] != "realOnly"
            or run.get("testEvaluated") is not False
        ):
            raise ValueError("Tracking requires completed, real-only scientific runs with matching size and seed.")
        stages = run["stages"]
        if len(stages) != 1 or stages[0]["taskId"] != "ballOnly" or stages[0]["status"] != "completed":
            raise ValueError("Tracking requires a completed standalone ballOnly stage.")
        if hashFile(root / "selection.json") != run["selectionSha256"]:
            raise ValueError("Training selection integrity mismatch.")
        saved = readJson(root / "selection.json")
        task = saved["tasks"].get("ballOnly")
        if task is None or task["classes"] != ["ball"] or set(saved["tasks"]) != {"ballOnly"}:
            raise ValueError("Training selection must contain only the ball ontology.")
        if {(r["recordId"], r["imageSha256"], r["labelSha256"]) for r in task["validation"]} != expected or len(task["validation"]) != len(expected):
            raise ValueError("Every model must retain the complete common validation cohort.")
        for split in ("training", "validation"):
            if any(row["origin"] != "own" or row["split"] != split or
                   assignments.get(row["matchId"]) != split for row in task[split]):
                raise ValueError("Training provenance violates the frozen own-match split.")
        best = stages[0]["results"]["checkpoints"]["best"]
        checkpoint = resolveSourcePath(root, best["path"])
        if hashFile(checkpoint) != best["sha256"]:
            raise ValueError("Checkpoint integrity mismatch.")
        models[size] = {"size": size, "runDirectory": str(root), "runSha256": hashFile(root / "run.json"),
                        "selectionSha256": run["selectionSha256"], "checkpointPath": str(checkpoint),
                        "checkpointSha256": best["sha256"], "trainingConfiguration": config}
    return models


def loadTrackedRuns(runDirs, selection, datasetDir):
    plan = readJson(datasetDir / "plan.json")
    assignments = plan["frozenSplit"]["assignments"]
    taskId, classes = selection["taskId"], selection["classes"]
    expected = {(r["recordId"], r["imageSha256"], r["labelSha256"]) for r in selection["references"]}
    models = {}
    for label, root in runDirs.items():
        run = readJson(root / "run.json")
        config = run["configuration"]
        if run["status"] != "completed" or config["purpose"] != "scientificTraining" or (
            config["sourcePolicy"] != "realOnly" or run.get("testEvaluated") is not False
        ):
            raise ValueError(f"{label}: tracking requires a completed real-only scientific training run.")
        stage = run["stages"][-1]
        if stage["taskId"] != taskId or stage["status"] != "completed" or config["stages"][-1]["taskId"] != taskId:
            raise ValueError(f"{label}: the final completed stage must be {taskId}.")
        if hashFile(root / "selection.json") != run["selectionSha256"]:
            raise ValueError(f"{label}: training selection integrity mismatch.")
        task = readJson(root / "selection.json")["tasks"].get(taskId)
        if task is None or task["classes"] != classes:
            raise ValueError(f"{label}: training selection does not contain the {taskId} ontology {classes}.")
        validation = {(r["recordId"], r["imageSha256"], r["labelSha256"]) for r in task["validation"]}
        if validation != expected or len(task["validation"]) != len(expected):
            raise ValueError(f"{label}: every model must retain the complete common validation cohort.")
        if any(row["origin"] != "own" or assignments.get(row["matchId"]) != "validation" for row in task["validation"]) or any(
            row["origin"] == "own" and assignments.get(row["matchId"]) != "training" for row in task["training"]
        ):
            raise ValueError(f"{label}: training provenance violates the frozen own-match split.")
        best = stage["results"]["checkpoints"]["best"]
        checkpoint = resolveSourcePath(root, best["path"])
        if hashFile(checkpoint) != best["sha256"]:
            raise ValueError(f"{label}: checkpoint integrity mismatch.")
        models[label] = {"label": label, "size": config["modelSize"], "taskId": taskId, "classes": classes,
                         "runDirectory": str(root), "runSha256": hashFile(root / "run.json"),
                         "selectionSha256": run["selectionSha256"], "checkpointPath": str(checkpoint),
                         "checkpointSha256": best["sha256"], "trainingConfiguration": config}
    return models


def inspectTrackedModels(models, config, device):
    from futsal.tracking.trackingBackend import NativeTracker

    for model in models.values():
        runner = NativeTracker(model["checkpointPath"], model["size"], model["classes"], config, device)
        del runner


def inspectTrackingModels(models, config, device):
    from futsal.tracking.trackingBackend import NativeBallTracker

    for size, model in models.items():
        runner = NativeBallTracker(model["checkpointPath"], size, config, device)
        del runner


def verifyTrackingModels(models):
    from pathlib import Path

    for model in models.values():
        root = Path(model["runDirectory"])
        for path, expected in ((root / "run.json", model["runSha256"]),
                               (root / "selection.json", model["selectionSha256"]),
                               (Path(model["checkpointPath"]), model["checkpointSha256"])):
            if hashFile(path) != expected:
                raise ValueError("Tracking model inputs changed during the campaign.")
