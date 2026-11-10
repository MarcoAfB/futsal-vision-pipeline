from futsal.training.modelWeights import validateModelSize

comparisonModes = ("controlled", "modelVariants")


def validateComparisonMode(comparisonMode):
    if comparisonMode not in comparisonModes:
        raise ValueError("comparisonMode must be controlled or modelVariants.")


def modelSummary(run):
    size = run["configuration"]["modelSize"]
    validateModelSize(size)
    weights = run["initialWeights"]
    if weights.get("modelFamily") != "yolo26" or weights.get("modelSize") != size:
        raise ValueError("Official initialization does not match the recorded model size.")
    return {"componentId": run["stages"][-1]["taskId"], "modelFamily": "yolo26", "modelSize": size,
            "initialWeightsSha256": weights["sha256"],
            "checkpointSha256": run["stages"][-1]["results"]["checkpoints"]["best"]["sha256"]}


def validateModelComparison(first, second, comparisonMode):
    validateComparisonMode(comparisonMode)
    before, after = modelSummary(first), modelSummary(second)
    if before["modelSize"] != after["modelSize"]:
        if comparisonMode == "controlled":
            raise ValueError("Controlled comparison requires the same model scale; use modelVariants explicitly.")
    elif first["initialWeights"] != second["initialWeights"]:
        raise ValueError("Official initialization differs for the same model scale.")


def comparisonModels(baseline, candidates):
    return {"baseline": [modelSummary(baseline)], "candidate": [modelSummary(run) for run in candidates]}


def modelChanges(models):
    size = models["baseline"][0]["modelSize"]
    return (["modelSize", "officialInitialization"]
            if any(item["modelSize"] != size for item in models["candidate"]) else [])
