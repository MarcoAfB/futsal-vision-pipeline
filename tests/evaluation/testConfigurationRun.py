import pytest

from futsal.datasets.datasetFiles import hashJson, readJson, writeJson
from futsal.evaluation.configurationRun import configurationInputs
from futsal.reporting.researchReport import verifyEvaluation


def testJointEvaluationBindsBothRunsAndExactCanonicalReferences(configurationEvaluation):
    output, runs, data, parent, state = configurationEvaluation
    verified, contract = verifyEvaluation(output)
    assert verified == state and contract["contractId"]
    assert len(state["recordIds"]) == 2 and state["testEvaluated"] is False
    assert state["components"][0]["run"]["initialWeights"] == state["components"][1]["run"]["initialWeights"]
    assert state["runtimeAfter"] and len(state["bundlesAfter"]) == 2
    assert readJson(output / "evidence/joint/points.json")["perClass"][-5]["className"] == "lowerCornerSideB"
    configurationInputs(*runs, data, parent)


@pytest.mark.parametrize("mutation", ["seed", "initialization", "artifact"])
def testJointEvaluationRefusesIncompatibleOrChangedComponents(configurationEvaluation, mutation):
    _, runs, data, parent, _ = configurationEvaluation
    root = runs[1]
    run = readJson(root / "run.json")
    if mutation == "seed":
        run["configuration"]["seed"] += 1
        run["configurationId"] = hashJson(run["configuration"])
        writeJson(root / "run.json", run)
    elif mutation == "initialization":
        run["stages"][0]["inputWeights"]["sha256"] = "0" * 64
        writeJson(root / "run.json", run)
    else:
        (root / run["stages"][0]["results"]["checkpoints"]["best"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError):
        configurationInputs(*runs, data, parent)


def testJointEvaluationAcceptsIndependentModelSizes(configurationEvaluation, tmp_path):
    from futsal.evaluation.configurationRun import evaluateConfiguration
    from futsal.training.modelWeights import prepareWeights
    from futsal.training.trainingRun import runTraining

    _, runs, data, parent, _ = configurationEvaluation
    prepareWeights("s", tmp_path / "weightsS")
    objects = tmp_path / "objectsS"
    run = runTraining(data, tmp_path / "detailedObjects.yaml", tmp_path / "weightsS/yolo26s.pt",
                      objects, modelSize="s", progress=lambda m: None)
    assert run["status"] == "completed", run.get("error")
    result = evaluateConfiguration(objects, runs[1], data, parent, "configs/evaluation/researchValidation.yaml",
                                   tmp_path / "mixedSizes", progress=lambda m: None)
    assert result["status"] == "completed", result.get("error")
    assert [item["run"]["configuration"]["modelSize"] for item in result["components"]] == ["s", "n"]
    run["configuration"]["parameters"]["batch"] += 1
    run["configurationId"] = hashJson(run["configuration"])
    writeJson(objects / "run.json", run)
    with pytest.raises(ValueError, match="shared protocol"):
        configurationInputs(objects, runs[1], data, parent)
